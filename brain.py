"""Whole-brain spiking model of the fruit fly on a connectome.

The leaky integrate-and-fire model of Shiu et al. (2024, Nature): every neuron
of a connectome, each synapse excitatory or inhibitory by its neurotransmitter.
Equations and constants are those of model.py in
https://github.com/philshiu/Drosophila_brain_model, which runs on Brian2;
here the same model runs in numba (Brian2 needs ~40 s per simulated second,
this ~0.7 s on 8 cores).

Connectomes (Brain(version)):
    "783"   FlyWire v783, the brain of a female fly: 138,639 neurons, as in the paper
    "mcns"  male CNS v1.0 (Janelia): brain and ventral nerve cord of a male fly,
            165,122 traced neurons, including the motor neurons of legs and wings
    "630"   FlyWire v630, only for the check below

The first use of a connectome downloads it (783: ~130 MB, mcns: ~1.1 GB) and
converts it to data/connectome_<version>.npz. `python brain.py` checks the model
against the published Brian2 results (downloads ~90 MB more).
"""

import csv
import time
import urllib.request
from pathlib import Path

import numpy as np
from numba import get_num_threads, njit, prange

DATA_DIR = Path("data")
MODEL_REPO = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/"
ANNOTATIONS = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/8587524c1748ce5ef2080822a2fc890fc03bf597/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
CONNECTOME_FILES = {  # version: (neuron list, connections)
    "783": ("Completeness_783.csv", "Connectivity_783.parquet"),
    "630": ("2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet"),
}
MCNS_DATA = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
MCNS_FILES = (  # neurons, their neurotransmitters, connections
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
)
# Shiu et al.: GABA and glutamate inhibit, everything else excites. FlyWire has no
# histamine; in flies it opens chloride channels (HisCl1, ort), so it inhibits too.
INHIBITORY = ("gaba", "glutamate", "histamine")
# On the same neurons the male CNS finds 1.8x as many synapses as FlyWire (median over
# the 4,669 cell types in both, for inputs and outputs alike), so this is how the
# synapses were detected rather than how the flies differ. W_SYNAPSE was fitted to
# FlyWire's counts; a male CNS synapse gets 1/1.8 of it.
MCNS_SYNAPSES_PER_FLYWIRE_SYNAPSE = 1.8

# Constants from model.py (default_params)
DT = 1e-4  # s, Brian2's default time step
V_REST, V_THRESHOLD = -52.0, -45.0  # mV; after a spike the neuron is reset to rest
T_MEMBRANE, T_SYNAPSE = 20e-3, 5e-3  # s
REFRACTORY = round(2.2e-3 / DT)  # steps
DELAY = round(1.8e-3 / DT)  # steps from a spike to its effect on the next neuron
W_SYNAPSE = 0.275  # mV per synapse
W_INPUT = W_SYNAPSE * 250  # mV: one input event always makes the neuron spike
QUIET_STEPS = round(0.05 / DT)  # see Brain.run

# Exact solution of the membrane equations over one step (Brian2's "linear" method):
#   dv/dt = (V_REST - v + g) / T_MEMBRANE,  dg/dt = -g / T_SYNAPSE
DECAY_V = np.exp(-DT / T_MEMBRANE)
DECAY_G = np.exp(-DT / T_SYNAPSE)
G_TO_V = T_SYNAPSE / (T_MEMBRANE - T_SYNAPSE) * (DECAY_V - DECAY_G)


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(parallel=True, cache=True)
def _run(n_steps, step0, v, g, ref_end, refractory, indptr, post, weight, ring, ring_n,
         input_ids, input_p, spikes, n_chunks):
    """Advance the network n_steps. v is relative to rest (mV). Adds spike counts to `spikes`.

    Per step, in Brian2's order: integrate the neurons, find spikes, deliver
    spikes emitted DELAY steps ago, add input events, reset the neurons that spiked.
    """
    n = v.shape[0]
    threshold = V_THRESHOLD - V_REST
    size = (n + n_chunks - 1) // n_chunks
    new = np.empty(n, np.int32)  # spiking neurons, each chunk writes into its own slice
    n_new = np.zeros(n_chunks, np.int64)
    for s in range(step0, step0 + n_steps):
        for c in prange(n_chunks):
            lo, m = c * size, 0
            for i in range(lo, min(n, lo + size)):
                if s < ref_end[i]:
                    # Refractory: the neuron is frozen at rest. In Brian2 synaptic
                    # input that arrives meanwhile is lost (checked against it).
                    g[i] = 0.0
                else:
                    v[i] = v[i] * DECAY_V + g[i] * G_TO_V
                    g[i] *= DECAY_G
                    if v[i] > threshold:
                        new[lo + m] = i
                        m += 1
            n_new[c] = m
        slot = s % DELAY  # holds the spikes of step s - DELAY, then those of step s
        for q in range(ring_n[slot]):
            pre = ring[slot, q]
            for e in range(indptr[pre], indptr[pre + 1]):
                g[post[e]] += weight[e]
        for q in range(input_ids.shape[0]):  # Poisson input
            if np.random.random() < input_p[q]:
                v[input_ids[q]] += W_INPUT
        k = 0
        for c in range(n_chunks):
            for q in range(n_new[c]):
                i = new[c * size + q]
                v[i] = 0.0
                g[i] = 0.0
                ref_end[i] = s + refractory[i]
                spikes[i] += 1
                ring[slot, k] = i
                k += 1
        ring_n[slot] = k


def _download(name, url):
    """Download once into data/ (kept if the conversion fails, deleted after it)."""
    path = DATA_DIR / name
    if not path.exists():
        print(f"Downloading {url} ...", flush=True)
        part = path.with_name(name + ".part")
        urllib.request.urlretrieve(url, part)
        part.replace(path)
    return path


def _grouped(ids, pre, post, synapses, **names):
    """Connections grouped by the sending neuron, the way the model reads them.
    synapses: signed synapse count of each connection (negative = inhibitory)."""
    order = np.argsort(pre, kind="stable")
    return dict(
        ids=ids,
        indptr=np.searchsorted(pre[order], np.arange(len(ids) + 1)),
        post=post[order].astype(np.int32),
        synapses=synapses[order].astype(np.int32),
        **names,
    )


def _annotations(version, ids):
    """Cell type, side and position of every neuron, from the connectome's annotation
    table (downloaded). Positions in µm, NaN where unknown. Returns them and the file."""
    index = {f: i for i, f in enumerate(ids.tolist())}
    cell_type, side = [""] * len(ids), [""] * len(ids)
    xyz = np.full((len(ids), 3), np.nan, dtype=np.float32)
    if version == "mcns":
        import pyarrow.feather as pf

        path = _download(MCNS_FILES[0], MCNS_DATA + MCNS_FILES[0])
        columns = ["bodyId", "type", "somaSide", "rootSide", "somaLocation", "tosomaLocation"]
        sides = {"L": "left", "R": "right", "M": "center"}
        for r in pf.read_table(path, columns=columns).to_pylist():
            i = index.get(r["bodyId"])
            if i is not None:
                cell_type[i], side[i] = r["type"] or "", sides.get(r["somaSide"] or r["rootSide"], "")
                location = r["somaLocation"] or r["tosomaLocation"]  # cell body, 8 nm voxels
                if location:
                    xyz[i] = np.array(location) * 0.008
    else:
        path = _download("annotations.tsv", ANNOTATIONS)
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="	"):
                i = index.get(int(row["root_id"]))
                if i is not None:
                    cell_type[i], side[i] = row["cell_type"], row["side"]
                    # the cell body if marked, else a point on the neuron; 4 x 4 x 40 nm voxels
                    at = "soma" if row["soma_x"] else "pos"
                    xyz[i] = [float(row[f"{at}_{a}"]) * nm / 1000 for a, nm in zip("xyz", (4, 4, 40))]
    return dict(cell_type=np.array(cell_type), side=np.array(side), xyz=xyz), path


def _read_flywire(version):
    """Shiu et al.'s files: neuron list and connections with signs already applied."""
    import pyarrow.parquet as pq

    neurons_csv, connections = (_download(f, MODEL_REPO + f) for f in CONNECTOME_FILES[version])
    with open(neurons_csv, newline="") as f:
        ids = np.array([int(row[0]) for row in list(csv.reader(f))[1:]], dtype=np.int64)
    table = pq.read_table(connections, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    data = _grouped(ids, *(table.column(k).to_numpy() for k in range(3)))
    files = [neurons_csv, connections]
    if version == "783":  # annotations (Schlegel et al. 2024) exist for this version
        names, tsv = _annotations(version, ids)
        data.update(names)
        files.append(tsv)
    return data, files


def _read_mcns():
    """Male CNS: traced neurons, their neurotransmitters, and synapse counts between
    all bodies; connections to untraced fragments are left out."""
    import pyarrow.feather as pf

    neurons_file, nt_file, connections = (_download(f, MCNS_DATA + f) for f in MCNS_FILES)
    neurons = pf.read_table(neurons_file, columns=["bodyId", "status"]).to_pylist()
    ids = np.array([r["bodyId"] for r in neurons if r["status"] == "Traced"], dtype=np.int64)
    index = {b: i for i, b in enumerate(ids.tolist())}
    sign = np.ones(len(ids), dtype=np.int64)  # no prediction: excitatory, as in Shiu et al.
    for r in pf.read_table(nt_file, columns=["body", "consensus_nt", "predicted_nt"]).to_pylist():
        nt = r["predicted_nt"] if r["consensus_nt"] == "unclear" else r["consensus_nt"]
        if nt in INHIBITORY and r["body"] in index:
            sign[index[r["body"]]] = -1

    table = pf.read_table(connections)
    order = np.argsort(ids)

    def to_index(bodies):  # -1 for bodies that are not traced neurons
        pos = order[np.searchsorted(ids, bodies, sorter=order).clip(max=len(ids) - 1)]
        return np.where(ids[pos] == bodies, pos, -1)

    pre, post = to_index(table["body_pre"].to_numpy()), to_index(table["body_post"].to_numpy())
    keep = (pre >= 0) & (post >= 0)
    pre, post = pre[keep], post[keep]
    names, _ = _annotations("mcns", ids)
    data = _grouped(ids, pre, post, table["weight"].to_numpy()[keep] * sign[pre], **names)
    return data, [neurons_file, nt_file, connections]


def prepare(version="783"):
    """Download a connectome once and store it as data/connectome_<version>.npz."""
    out = DATA_DIR / f"connectome_{version}.npz"
    if out.exists() and (version == "630" or "xyz" in np.load(out).files):
        return out
    DATA_DIR.mkdir(exist_ok=True)
    if out.exists():  # made before neuron positions were kept: add them, keep the rest
        data = dict(np.load(out))
        names, annotations = _annotations(version, data["ids"])
        data.update(names)
        files = [annotations]
    else:
        data, files = _read_mcns() if version == "mcns" else _read_flywire(version)
    np.savez(out, **data)
    for f in files:
        f.unlink()
    print(f"Saved {out}: {len(data['ids'])} neurons, {len(data['post'])} connections", flush=True)
    return out


class Brain:
    """The whole brain. Stimulate neurons with set_input(), advance with run().

    shuffle_seed: control experiment. Every neuron keeps its synapses and their
    signs, but they go to random neurons: same parts, random wiring.
    """

    def __init__(self, version="783", shuffle_seed=None):
        data = np.load(prepare(version))
        self.ids = data["ids"]  # FlyWire root IDs or male CNS body IDs
        self.n = len(self.ids)
        self.indptr, self.post = data["indptr"], data["post"]
        w_synapse = W_SYNAPSE / (MCNS_SYNAPSES_PER_FLYWIRE_SYNAPSE if version == "mcns" else 1)
        self.weight = data["synapses"].astype(np.float32) * np.float32(w_synapse)
        if shuffle_seed is not None:
            self.post = np.random.default_rng(shuffle_seed).permutation(self.post)
        if "cell_type" in data.files:  # not in "630"
            self.cell_type, self.side = data["cell_type"], data["side"]
            self.xyz = data["xyz"]  # position of each neuron's cell body, µm (NaN: unknown)
        self.n_chunks = 2 * get_num_threads()
        self.reset()
        self._advance(0, np.zeros(self.n, np.int32))  # compile the numba code now, not mid-run

    def neurons(self, cell_type, side):
        """Indices of the neurons of one cell type (e.g. "LC4") on one side ("left" / "right")."""
        return np.flatnonzero((self.cell_type == cell_type) & (self.side == side))

    def reset(self, seed=0):
        """All neurons at rest, no input."""
        _seed(seed)
        self.v = np.zeros(self.n)  # membrane potential relative to rest, mV
        self.g = np.zeros(self.n)  # synaptic input, mV
        self.ref_end = np.zeros(self.n, np.int64)  # first step after the refractory period
        self.refractory = np.full(self.n, REFRACTORY, np.int64)
        self.ring = np.zeros((DELAY, self.n), np.int32)  # spikes on their way, by step
        self.ring_n = np.zeros(DELAY, np.int64)
        self.step = 0
        self.silent = True  # no spikes in the last run() and none on the way
        self.set_input([], 0)

    def set_input(self, neurons, rate_hz):
        """Drive neurons with Poisson input at rate_hz (a number or one per neuron),
        like optogenetic activation in the paper. Replaces the previous input."""
        self.input_ids = np.asarray(neurons, np.int64)
        self.input_p = np.broadcast_to(np.asarray(rate_hz, float) * DT, self.input_ids.shape).copy()
        self.refractory[:] = REFRACTORY
        self.refractory[self.input_ids] = 0  # as in model.py: driven neurons have none

    def run(self, seconds):
        """Advance the brain; returns the number of spikes of every neuron."""
        n_steps = round(seconds / DT)
        spikes = np.zeros(self.n, np.int32)
        if self.silent and not self.input_p.any():
            # A silent brain without input stays silent: nothing can reach the
            # threshold. Decay the leftover potentials exactly instead of stepping.
            a, b = DECAY_V**n_steps, DECAY_G**n_steps
            self.v = self.v * a + self.g * (T_SYNAPSE / (T_MEMBRANE - T_SYNAPSE) * (a - b))
            self.g *= b
        else:
            self._advance(n_steps, spikes)
            # 50 ms without spikes: no input arrives any more and every potential
            # has passed its peak, so no neuron can spike again without new input
            self.silent = n_steps >= QUIET_STEPS and not spikes.any()
        self.step += n_steps
        return spikes

    def _advance(self, n_steps, spikes):
        _run(n_steps, self.step, self.v, self.g, self.ref_end, self.refractory, self.indptr, self.post,
             self.weight, self.ring, self.ring_n, self.input_ids, self.input_p, spikes, self.n_chunks)


def check():
    """Compare with the published Brian2 run of Shiu et al.'s example: 21 sugar-sensing
    neurons of the right side driven at 200 Hz, 30 trials of 1 s, connectome v630."""
    import pyarrow.parquet as pq

    sugar = [
        720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
        720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
        720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
        720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
        720575940611875570,
    ]
    mn9 = 720575940660219265  # proboscis motor neuron: the fly starts to feed
    brain = Brain("630")
    index = {f: i for i, f in enumerate(brain.ids)}
    reference = pq.read_table(_download("sugarR.parquet", MODEL_REPO + "results/example/sugarR.parquet"))
    brian2 = np.bincount([index[f] for f in reference.column("flywire_id").to_pylist()], minlength=brain.n) / 30
    ours = np.zeros(brain.n)
    start = time.perf_counter()
    for trial in range(30):
        brain.reset(seed=trial)
        brain.set_input([index[f] for f in sugar], 200.0)
        ours += brain.run(1.0) / 30
    active = (ours > 0) | (brian2 > 0)
    corr = np.corrcoef(ours[active], brian2[active])[0, 1]
    print(f"30 trials in {time.perf_counter() - start:.0f} s")
    print(f"Neurons that spiked: {np.count_nonzero(ours)} (Brian2: {np.count_nonzero(brian2)})")
    print(f"Correlation of firing rates with Brian2: {corr:.4f}")
    print(f"MN9: {ours[index[mn9]]:.1f} Hz (Brian2: {brian2[index[mn9]]:.1f} Hz)")
    assert corr > 0.99 and abs(ours[index[mn9]] - brian2[index[mn9]]) < 5, "does not match Brian2"
    print("OK: matches the published Brian2 results")


if __name__ == "__main__":
    check()
