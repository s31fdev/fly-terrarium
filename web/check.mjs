// Checks brain.js against brain.py: sugar-sensing neurons driven at 200 Hz, 30 trials of 1 s,
// the same experiment export_web.py ran in Python (web/data/<version>/reference.json).
// Usage: node web/check.mjs [783|mcns]   (default: both)
import { readFile } from "node:fs/promises";
import { Brain, loadConnectome, DT } from "./brain.js";

let failed = false;
for (const version of process.argv.slice(2).length ? process.argv.slice(2) : ["783", "mcns"]) {
  const dir = new URL(`./data/${version}/`, import.meta.url);
  const meta = JSON.parse(await readFile(new URL("meta.json", dir)));
  const ref = JSON.parse(await readFile(new URL("reference.json", dir)));
  let start = performance.now();
  const brain = new Brain(await loadConnectome(meta, async (name) => new Uint8Array(await readFile(new URL(name, dir)))));
  console.log(`${version}: ${meta.n} neurons, ${meta.m} connections, loaded in ${((performance.now() - start) / 1000).toFixed(1)} s`);

  const [group, side] = ref.input.split(".");
  const rates = new Float64Array(meta.n);
  start = performance.now();
  for (let trial = 0; trial < ref.trials; trial++) {
    brain.reset(trial);
    const ids = meta.groups[group][side];
    brain.setInput(ids, ids.map(() => ref.hz));
    for (const i of brain.run(Math.round(1 / DT))) rates[i] += 1 / ref.trials;
  }
  const seconds = (performance.now() - start) / 1000;

  const ours = ref.neurons.map((i) => rates[i]);
  const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  const [ma, mb] = [mean(ours), mean(ref.rates)];
  const cov = ours.reduce((s, x, k) => s + (x - ma) * (ref.rates[k] - mb), 0);
  const corr = cov / Math.sqrt(ours.reduce((s, x) => s + (x - ma) ** 2, 0) * ref.rates.reduce((s, x) => s + (x - mb) ** 2, 0));
  const mn9 = meta.groups.MN9.left.concat(meta.groups.MN9.right);
  const hz = (r) => mn9.map((i) => r(i).toFixed(1)).join(" / ");
  const pyRate = (i) => ref.rates[ref.neurons.indexOf(i)] ?? 0;
  console.log(`  ${ref.trials} trials of 1 s in ${seconds.toFixed(1)} s (${(seconds / ref.trials).toFixed(2)} s per simulated second)`);
  console.log(`  correlation with brain.py over its ${ref.neurons.length} most active neurons: ${corr.toFixed(4)}`);
  console.log(`  MN9 left / right: ${hz((i) => rates[i])} Hz (brain.py: ${hz(pyRate)} Hz)`);
  if (!(corr > 0.99) || mn9.some((i) => Math.abs(rates[i] - pyRate(i)) > 5)) {
    console.error("  FAIL: does not match brain.py");
    failed = true;
  } else console.log("  OK: matches brain.py");
}
process.exit(failed ? 1 : 0);
