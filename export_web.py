"""Export the brains for the web sandbox in web/.

Writes web/data/<version>/ for FlyWire ("783") and the male CNS ("mcns"):
    connectome.bin.gz.<k>  the wiring, gzip split into parts of <= 14 MB: indptr (int32),
                           then post (int32, sorted and delta-coded per sending neuron) and
                           signed synapse counts (int16), both split into byte planes
                           (FlyWire compresses to 30 MB instead of 50)
    map.bin                every neuron's cell body for the 3D brain map: x, y, z (int16, 0.1 µm,
                           -32768 = no position), rotated so that x points to the fly's right and
                           y down the screen in the starting view, z into the screen
    meta.json              sizes, parts, synapse weight, and the neuron groups the sandbox
                           stimulates / reads
    reference.json         spike rates of brain.py for web/check.mjs

Usage: python export_web.py [783|mcns]   (default: both)
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

from brain import MCNS_SYNAPSES_PER_FLYWIRE_SYNAPSE, W_SYNAPSE, Brain, prepare

OUT = Path("web/data")
PART_BYTES = 14_000_000  # stays under the 15 MB per-file limit of some static hosts

# Shiu et al. 2024 (figures.ipynb): labellar sugar-sensing neurons, FlyWire v630 IDs. Their
# "right" and "left" are image sides; the v783 annotations give the fly's sides, flipped.
# IDs that changed between v630 and v783 are skipped (1 of 21, 1 of 10).
SUGAR_783 = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
    720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
    720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570, 720575940620589838, 720575940631147148, 720575940608305161, 720575940629388135,
    720575940630968335, 720575940606801282, 720575940617398502, 720575940616167218, 720575940620296641,
    720575940627961104,
]
# Neuron groups by cell type; a string ending in "*" is a prefix. FlyWire types are matched
# to Shiu et al.'s neuron lists, male CNS types are the same cells under male CNS names.
GROUPS = {
    "783": dict(
        sugar=SUGAR_783,
        bitter=["LB1a,LB1d", "LB1b", "LB1c"],  # Shiu's bitter list is all of these, one side
        antenna=["JO-C*", "JO-E*", "JO-F*"],  # Johnston's organ JO-CE and JO-F -> grooming
        loom=["LC4", "LPLC2", "LC16"],  # as in run_escape.py
        GF=["DNp01"],  # giant fibre: escape
        DNa=["DNa01", "DNa02"],  # steering
        MDN=["MDN"],  # walking backwards
        MN9=["CB0701"],  # proboscis motor neuron (Shiu's MN9 ID has this type in v783)
        groom=["DNg62", "DNge078"],  # aDN1, aDN2 (Shiu's IDs): antennal grooming
    ),
    "mcns": dict(
        sugar=["LB3b", "LB3c"],  # labellar sugar neurons
        bitter=["LB1a", "LB1b", "LB1c", "LB1d"],
        antenna=["JO-C*", "JO-E*", "JO-F*"],
        loom=["LC4", "LPLC2", "LC16"],
        GF=["DNp01"],
        DNa=["DNa01", "DNa02"],
        MDN=["MDN"],
        MN9=["MN9"],
        groom=["DNg62", "DNge078"],
        TTMn=["TTMn"],  # jump muscle motor neuron, in the nerve cord: only the male CNS has it
    ),
}
INPUTS = ("sugar", "bitter", "antenna", "loom")
# Brain map, starting view: the axis that points down the screen. FlyWire is a brain, seen
# from the front (y is ventral); the male CNS also has the nerve cord, which hides behind the
# brain from the front, so it starts seen from above (z runs from head to tail).
DOWN = {"783": 1, "mcns": 2}
NO_POSITION = -32768


def group(brain, spec):
    """Indices of the neurons of these cell types (or IDs), per side."""
    if isinstance(spec[0], int):
        index = {f: i for i, f in enumerate(brain.ids.tolist())}
        match = np.zeros(brain.n, bool)
        match[[index[f] for f in spec if f in index]] = True
    else:
        types = brain.cell_type.astype(str)
        match = np.isin(types, [t for t in spec if not t.endswith("*")])
        for t in spec:
            if t.endswith("*"):
                match |= np.char.startswith(types, t[:-1])
    return {s: np.flatnonzero(match & (brain.side == s)).tolist() for s in ("left", "right")}


def planes(a):
    """Byte planes of a little-endian array: all lowest bytes, then the next, ..."""
    return a.view(np.uint8).reshape(-1, a.itemsize).T.tobytes()


def export(version):
    out = OUT / version
    out.mkdir(parents=True, exist_ok=True)
    brain = Brain(version)
    data = np.load(prepare(version))
    indptr, post, synapses = data["indptr"].astype(np.int32), data["post"], data["synapses"]
    assert np.abs(synapses).max() < 2**15

    # Sort the targets of each neuron and store differences: small numbers compress well
    rows = np.repeat(np.arange(brain.n), np.diff(indptr))
    order = np.lexsort((post, rows))
    post, synapses = post[order].astype(np.int32), synapses[order].astype(np.int16)
    delta = post.copy()
    delta[1:] -= post[:-1]
    first = indptr[:-1][np.diff(indptr) > 0]
    delta[first] = post[first]
    blob = gzip.compress(indptr.tobytes() + planes(delta) + planes(synapses), 6)
    for old in out.glob("connectome.bin.gz.*"):
        old.unlink()
    parts = []
    for k in range(0, len(blob), PART_BYTES):
        parts.append(f"connectome.bin.gz.{len(parts)}")
        (out / parts[-1]).write_bytes(blob[k:k + PART_BYTES])

    # Brain map: rotate (never mirror) into the starting view, the fly's left on the left
    down = DOWN[version]
    xyz = brain.xyz.astype(float)
    across = 1 if np.nanmean(xyz[brain.side == "right", 0]) > np.nanmean(xyz[brain.side == "left", 0]) else -1
    ex, ey = across * np.eye(3)[0], np.eye(3)[down]
    xyz = xyz @ np.array([ex, ey, np.cross(ex, ey)]).T
    xyz -= (np.nanmin(xyz, axis=0) + np.nanmax(xyz, axis=0)) / 2
    (out / "map.bin").write_bytes(np.where(np.isnan(xyz), NO_POSITION, np.round(xyz * 10)).astype(np.int16).tobytes())

    groups = {k: group(brain, spec) for k, spec in GROUPS[version].items()}
    w_synapse = W_SYNAPSE / (MCNS_SYNAPSES_PER_FLYWIRE_SYNAPSE if version == "mcns" else 1)
    meta = dict(
        version=version, n=brain.n, m=len(post), w_synapse=w_synapse, parts=parts, size_mb=round(len(blob) / 1e6),
        inputs=list(INPUTS), outputs=[k for k in groups if k not in INPUTS], groups=groups,
    )
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # What brain.py does with sugar input, for web/check.mjs to compare against
    trials, rates = 30, np.zeros(brain.n)
    for trial in range(trials):
        brain.reset(seed=trial)
        brain.set_input(groups["sugar"]["left"], 200.0)
        rates += brain.run(1.0) / trials
    top = np.argsort(-rates)[:300]
    reference = dict(input="sugar.left", hz=200.0, trials=trials, neurons=top.tolist(), rates=rates[top].tolist())
    (out / "reference.json").write_text(json.dumps(reference), encoding="utf-8")

    sizes = {k: f"{len(v['left'])}+{len(v['right'])}" for k, v in groups.items()}
    print(f"{version}: {len(blob) / 1e6:.1f} MB in {len(parts)} parts; groups (left+right): {sizes}", flush=True)


if __name__ == "__main__":
    for version in sys.argv[1:] or ["783", "mcns"]:
        export(version)
