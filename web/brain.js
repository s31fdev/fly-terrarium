// The whole-brain model of brain.py (Shiu et al. 2024 leaky integrate-and-fire), in JavaScript.
// Same equations, constants and order of events within a step; check.mjs compares it with brain.py.
//
// brain.py updates every neuron at every 0.1 ms step. Here a neuron is only touched when
// something happens to it: its state (potential v and synaptic input g at some step t0) is
// brought forward with the exact solution of the equations when input arrives, and right
// after, the step at which it will cross the threshold, if it will, is computed and put on
// a timetable. Same result, but the work follows the spikes, not the clock: a quiet brain
// costs nothing, a busy one a few times less than stepping every neuron.
//
// In a page this file runs as a module worker (see the message handler at the bottom).

export const DT = 1e-4; // s
const THRESHOLD = 7.0; // mV above rest (-45 vs -52)
const REFRACTORY = 22; // steps (2.2 ms)
const DELAY = 18; // steps (1.8 ms)
const W_INPUT = 0.275 * 250; // mV: one input event makes a neuron spike
const DECAY_V = Math.exp(-DT / 20e-3);
const DECAY_G = Math.exp(-DT / 5e-3);
const G_TO_V = (5e-3 / (20e-3 - 5e-3)) * (DECAY_V - DECAY_G);
// After k steps without input: v <- v a^k + g G_COEF (a^k - b^k), g <- g b^k
const G_COEF = G_TO_V / (DECAY_V - DECAY_G);
const TABLE = 1 << 16; // longer gaps: everything has decayed to < 1e-140
const A_POW = Float64Array.from({ length: TABLE }, (_, k) => DECAY_V ** k);
const B_POW = Float64Array.from({ length: TABLE }, (_, k) => DECAY_G ** k);
// The most one unit of g can still add to v: a neuron below THRESHOLD - g * H_MAX never spikes
const H_MAX = A_POW.reduce((max, a, k) => Math.max(max, G_COEF * (a - B_POW[k])), 0);
const WHEEL = 4096; // steps the timetable looks ahead; threshold crossings come within ~300

function random(seed) {
  // mulberry32: small, seeded, good enough for Poisson input
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Connectome from the parts written by export_web.py, unpacked while it downloads: the bytes go
// straight into the final arrays, and nothing else of that size is held (the male CNS peaks at
// ~220 MB, it was ~500 unpacking it whole). open(name) -> Promise<ReadableStream> of that part.
export async function loadConnectome(meta, open) {
  const n = meta.n, m = meta.m;
  const indptr = new Int32Array(n + 1), post = new Int32Array(m), synapses = new Int16Array(m);
  // The unpacked stream, in order: indptr, the 4 byte planes of post, the 2 of synapses. Plane b
  // holds byte b of every element, so it fills every itemsize-th byte of the array's memory from b
  // (typed arrays are little-endian on every platform browsers run on).
  const bytes = (a) => new Uint8Array(a.buffer);
  const planes = [[bytes(indptr), 1, 0, 4 * (n + 1)]];
  for (let b = 0; b < 4; b++) planes.push([bytes(post), 4, b, m]);
  for (let b = 0; b < 2; b++) planes.push([bytes(synapses), 2, b, m]);

  // All parts download at once and are read one after another as one gzip stream. Readers, not
  // for await: Safari cannot iterate a stream.
  const opened = meta.parts.map(open);
  opened.forEach((p) => p.catch(() => {})); // a failure is reported once, when its turn comes
  let k = 0, part = null;
  const gzipped = new ReadableStream({
    async pull(controller) {
      for (; k < opened.length; k++, part = null) {
        part ??= (await opened[k]).getReader();
        const { done, value } = await part.read();
        if (!done) return controller.enqueue(value);
      }
      controller.close();
    },
  });

  const unpacked = gzipped.pipeThrough(new DecompressionStream("gzip")).getReader();
  let plane = 0, at = 0; // where the next byte goes
  for (;;) {
    const { done, value: chunk } = await unpacked.read();
    if (done) break;
    for (let i = 0; i < chunk.length; ) {
      if (plane === planes.length) throw new Error("the connectome is longer than meta.json says");
      const [target, stride, first, length] = planes[plane];
      const take = Math.min(length - at, chunk.length - i);
      if (stride === 1) target.set(chunk.subarray(i, i + take), at);
      else for (let k = 0, p = first + stride * at; k < take; k++, p += stride) target[p] = chunk[i + k];
      i += take;
      at += take;
      if (at === length) (plane++), (at = 0);
    }
  }
  if (plane !== planes.length) throw new Error("the connectome is shorter than meta.json says");
  for (let i = 0; i < n; i++) for (let e = indptr[i] + 1; e < indptr[i + 1]; e++) post[e] += post[e - 1];
  return { n, indptr, post, synapses, wSynapse: meta.w_synapse };
}

// Control: every neuron keeps its synapses, but they go to random neurons (like brain.py --shuffled)
export function shuffle(post, seed = 7) {
  const rand = random(seed);
  for (let e = post.length - 1; e > 0; e--) {
    const k = Math.floor(rand() * (e + 1));
    const t = post[e]; post[e] = post[k]; post[k] = t;
  }
}

export class Brain {
  constructor({ n, indptr, post, synapses, wSynapse }) {
    Object.assign(this, { n, indptr, post, synapses, wSynapse });
    this.reset();
  }

  reset(seed = 0) {
    const n = this.n;
    this.v = new Float64Array(n); // mV above rest, at the end of step t0
    this.g = new Float64Array(n); // synaptic input, mV, at the end of step t0
    this.t0 = new Int32Array(n);
    this.refEnd = new Int32Array(n); // first step after the refractory period
    this.lastSpike = new Int32Array(n).fill(-1);
    this.version = new Int32Array(n); // changes with the state; older timetable entries are void
    this.driven = new Uint8Array(n); // driven neurons have no refractory period (as in model.py)
    this.touched = new Int32Array(n);
    this.touchedAt = new Int32Array(n).fill(-1);
    this.wheel = Array.from({ length: WHEEL }, () => []); // step -> [neuron, version, ...] to spike
    this.ring = Array.from({ length: DELAY }, () => []); // spikes on their way, by step
    this.step = 0;
    this.rand = random(seed + 1);
    this.setInput([], []);
  }

  // Poisson input: neurons[k] at rates[k] Hz. Replaces the previous input.
  setInput(neurons, rates) {
    if (this.inputIds) for (const i of this.inputIds) this.driven[i] = 0;
    this.inputIds = Int32Array.from(neurons);
    this.inputP = Float64Array.from(rates, (hz) => hz * DT);
    for (const i of this.inputIds) this.driven[i] = 1;
  }

  // Bring neuron i to the end of step s (no input in between).
  advance(i, s) {
    const k = s - this.t0[i];
    if (k === 0) return;
    const { v, g } = this;
    if (k < TABLE) {
      const a = A_POW[k], b = B_POW[k];
      v[i] = v[i] * a + g[i] * G_COEF * (a - b);
      g[i] *= b;
    } else v[i] = g[i] = 0;
    this.t0[i] = s;
  }

  // Advance nSteps; returns the neurons that spiked, once per spike.
  run(nSteps) {
    const { v, g, t0, refEnd, lastSpike, version, driven, touched, touchedAt, wheel, ring } = this;
    const { indptr, post, synapses, inputIds, inputP, rand } = this;
    const w = this.wSynapse;
    const out = [];
    for (let s = this.step; s < this.step + nSteps; s++) {
      // Neurons crossing the threshold at this step spike (brain.py: integrate, find spikes, ...)
      const due = wheel[s % WHEEL], fired = [];
      for (let q = 0; q < due.length; q += 2) {
        const i = due[q];
        if (version[i] !== due[q + 1]) continue;
        v[i] = g[i] = 0;
        t0[i] = lastSpike[i] = s;
        version[i]++;
        refEnd[i] = driven[i] ? s : s + REFRACTORY;
        fired.push(i);
        out.push(i);
      }
      due.length = 0;
      // ... deliver the spikes of DELAY steps ago, ... Input to a neuron that spiked now or
      // is refractory until after the next step is lost (Brian2 freezes and resets it)
      let nTouched = 0;
      const slot = s % DELAY;
      for (const pre of ring[slot]) {
        for (let e = indptr[pre]; e < indptr[pre + 1]; e++) {
          const j = post[e];
          if (s < refEnd[j] - 1 || lastSpike[j] === s) continue;
          this.advance(j, s);
          g[j] += synapses[e] * w;
          if (touchedAt[j] !== s) (touchedAt[j] = s), (touched[nTouched++] = j);
        }
      }
      ring[slot] = fired;
      // ... add input events ...
      for (let q = 0; q < inputIds.length; q++) {
        if (rand() >= inputP[q]) continue;
        const i = inputIds[q];
        if (lastSpike[i] === s) continue;
        this.advance(i, s);
        v[i] += W_INPUT;
        if (touchedAt[i] !== s) (touchedAt[i] = s), (touched[nTouched++] = i);
      }
      // ... and for each neuron that got input: will it cross the threshold, and when?
      for (let q = 0; q < nTouched; q++) {
        const i = touched[q];
        version[i]++;
        let vi = v[i], gi = g[i];
        if (Math.max(vi, 0) + Math.max(gi, 0) * H_MAX <= THRESHOLD) continue; // never: most of them
        for (let k = 1; k < WHEEL; k++) {
          const next = vi * DECAY_V + gi * G_TO_V; // brain.py's step
          gi *= DECAY_G;
          if (next > THRESHOLD) {
            wheel[(s + k) % WHEEL].push(i, version[i]);
            break;
          }
          if (next <= vi) break; // past its peak: only falls from here
          vi = next;
        }
      }
    }
    this.step += nSteps;
    return Int32Array.from(out);
  }
}

// Worker protocol. Page -> worker: {type: "load", base, meta, shuffled} once, then
// {type: "run", ms, input: {"<group>.<side>": Hz}} or {type: "reset"} (all neurons to rest).
// Worker -> page: {type: "progress", text}, {type: "ready"}, {type: "error", text},
// {type: "spikes", spikes (Int32Array), ms, wall}.
if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) {
  let brain, meta, inputKey = "";
  self.onmessage = async ({ data }) => {
    if (data.type === "reset") {
      brain?.reset();
      inputKey = "";
    } else if (data.type === "load") {
      // errors of an async handler never reach the page's worker.onerror: report them
      try {
        meta = data.meta;
        self.postMessage({ type: "progress", text: `loading the connectome (${meta.size_mb} MB)…` });
        const open = async (name) => {
          const response = await fetch(data.base + name);
          if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
          return response.body;
        };
        const connectome = await loadConnectome(meta, open);
        if (data.shuffled) shuffle(connectome.post);
        brain = new Brain(connectome);
        self.postMessage({ type: "ready" });
      } catch (error) {
        self.postMessage({ type: "error", text: error.message });
      }
    } else if (data.type === "run") {
      const key = JSON.stringify(data.input);
      if (key !== inputKey) {
        inputKey = key;
        const ids = [], rates = [];
        for (const [name, hz] of Object.entries(data.input)) {
          const [group, side] = name.split(".");
          if (hz > 0) for (const i of meta.groups[group][side]) ids.push(i), rates.push(hz);
        }
        brain.setInput(ids, rates);
      }
      const start = performance.now();
      const spikes = brain.run(Math.round(data.ms / 1000 / DT));
      self.postMessage({ type: "spikes", spikes, ms: data.ms, wall: performance.now() - start }, [spikes.buffer]);
    }
  };
}
