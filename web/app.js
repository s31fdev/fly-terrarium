// The terrarium: the world, what the fly senses, what its spikes make it do, and the page.
// The brain itself runs in brain.js (a worker) and is drawn by brainview.js. The world advances
// in steps of CHUNK of brain time: the page sends what the sensory neurons get, the worker
// answers with the spikes of all neurons, then the fly moves. If the brain is slower than real
// time, the world slows down with it.
import { BrainView, GROUPS } from "./brainview.js";

const CHUNK = 0.02; // s of brain time per round trip to the worker
const ARENA_R = 16; // mm, radius of the dish
// Senses (hand-chosen): Hz of Poisson input into the sensory neurons while touched
const TASTE_HZ = 150; // Shiu et al.: sugar neurons at 100-200 Hz make MN9 fire
const ANTENNA_HZ = 200;
const DROP_R = 2; // mm
const PUFF_REACH = 14, PUFF_S = 0.4; // mm, s
// Eyes, as run_escape.py: growth of the dark area in ommatidia/s -> Hz into LC4/LPLC2/LC16
const LOOM_MIN = 50, LOOM_GAIN = 0.5, LOOM_MAX = 150;
const OMMATIDIUM_SR = 0.0076; // solid angle of one ommatidium (~5 x 5 degrees)
const OMMATIDIA = 721; // per eye, as in NeuroMechFly
const SHADOW_R = 6, SHADOW_H = 60, SHADOW_FALL_S = 0.6; // mm, mm, s
// Motor side, as run_escape.py plus MN9, grooming and TTMn: Hz for a full-strength command
const FULL = { GF: 50, DNa: 40, MDN: 40, MN9: 40, groom: 20, TTMn: 50 };
const RUN_HOLD_S = 0.6, HOLD_S = 0.2; // s the commands linger after the neurons fall silent
// The fly jumps when the jump muscle motor neuron TTMn fires (male CNS). FlyWire ends at the
// neck and has no TTMn; there a giant fibre spike stands in for it (a made-up rule).
// 25 Hz (mean of both) = one spike of one neuron within CHUNK.
const JUMP_HZ = 25;
const FLIGHT_S = 0.5, FLIGHT_SPEED = 25; // s, mm/s: a hop across part of the dish
const TURN_RATE = 4; // rad/s at full DNa difference
// spikes/s: a brain this busy long after its last input is stuck in a self-sustaining wave;
// OVERLOAD_SPIKES is more than any stimulus here causes without one (loom on both eyes: ~150k)
const CALM_SPIKES = 20000, OVERLOAD_SPIKES = 250000;
const SENSE_MAX = { sugar: TASTE_HZ, bitter: TASTE_HZ, antenna: ANTENNA_HZ, loom: LOOM_MAX };
const SENSE_CELLS = { sugar: "sugar neurons", bitter: "bitter neurons", antenna: "Johnston’s organ neurons", loom: "looming detectors" };
const ACTION_CELLS = { GF: "giant fibre DNp01", TTMn: "jump motor neuron", DNa: "DNa01 + DNa02", MDN: "MDN", MN9: "proboscis MN9", groom: "aDN1 / aDN2" };
const STATUS = {
  wait: ["Waiting for the brain…", ""],
  jumpTTMn: ["Jumping — its jump neuron TTMn fired", "alarm"],
  jumpGF: ["Jumping — its giant fibre fired (our rule for the female)", "alarm"],
  run: ["Running away — its giant fibre fired", "alarm"],
  back: ["Backing up — its MDN neurons fired", "alarm"],
  groom: ["Cleaning its antennae — aDN1/aDN2 fired", "groom"],
  feed: ["Feeding — MN9 extended the proboscis", "feed"],
  stroll: ["Strolling — the brain is quiet, so this part is made up", ""],
  stand: ["Standing still — the brain is quiet", ""],
};

const $ = (id) => document.getElementById(id);
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const hold = (old, now, decay) => (Math.abs(now) >= Math.abs(old) * decay ? now : old * decay);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SIDES = ["left", "right"];

// ---------------------------------------------------------------- world
const fly = { x: 0, y: 0, heading: Math.PI / 2, z: 0, speed: 0, legs: 0, proboscis: 0, groom: 0, flightT: -1, doing: "wait" };
const cmd = { run: 0, turn: 0, back: 0, feed: 0, groom: 0 };
const wander = { walking: true, timer: 2, turn: 0 };
let drops = [], puffs = [], shadows = [];
let simTime = 0, eyesBefore = null, input = {};

function shadowHeight(sh) {
  const t = simTime - sh.t;
  if (t < SHADOW_FALL_S) return SHADOW_H - (SHADOW_H - 3) * (t / SHADOW_FALL_S);
  if (t < SHADOW_FALL_S + 0.3) return 3;
  if (t < SHADOW_FALL_S + 0.9) return 3 + (SHADOW_H - 3) * ((t - SHADOW_FALL_S - 0.3) / 0.6);
  return null;
}

// How much of each eye's view the shadows cover, in ommatidia. The eyes see all around
// except a blind wedge behind, overlap in front, and both look up.
function darkArea() {
  const area = [0, 0];
  for (const sh of shadows) {
    const h = shadowHeight(sh);
    if (h === null) continue;
    const dx = sh.x - fly.x, dy = sh.y - fly.y, flat = Math.hypot(dx, dy);
    const ommatidia = (2 * Math.PI * (1 - Math.cos(Math.atan(SHADOW_R / Math.hypot(flat, h))))) / OMMATIDIUM_SR;
    const azimuth = (wrap(Math.atan2(dy, dx) - fly.heading) * 180) / Math.PI; // + = left
    const up = (Math.atan2(h, flat) * 180) / Math.PI;
    if (up > 60 || (azimuth > -15 && azimuth < 165)) area[0] += ommatidia;
    if (up > 60 || (azimuth < 15 && azimuth > -165)) area[1] += ommatidia;
  }
  return area.map((a) => Math.min(a, OMMATIDIA));
}

// Input rate of each sensory group for the next CHUNK: {"group.side": Hz}
function senses() {
  const now = {};
  if (fly.flightT < 0) {
    const hx = fly.x + Math.cos(fly.heading) * 1.2, hy = fly.y + Math.sin(fly.heading) * 1.2;
    for (const d of drops) if (Math.hypot(d.x - hx, d.y - hy) < d.r) now[d.kind + ".left"] = now[d.kind + ".right"] = TASTE_HZ;
  }
  // a puff bends the antennae the more, the closer it is: ANTENNA_HZ right at the fly, 0 at PUFF_REACH
  let wind = 0;
  for (const p of puffs) {
    if (simTime - p.t < PUFF_S) wind = Math.max(wind, 1 - Math.hypot(p.x - fly.x, p.y - fly.y) / PUFF_REACH);
  }
  if (wind > 0) now["antenna.left"] = now["antenna.right"] = Math.round(ANTENNA_HZ * wind);
  const eyes = darkArea();
  if (eyesBefore) {
    SIDES.forEach((s, k) => {
      const hz = clamp(LOOM_GAIN * ((eyes[k] - eyesBefore[k]) / CHUNK - LOOM_MIN), 0, LOOM_MAX);
      if (hz > 0) now["loom." + s] = Math.round(hz);
    });
  }
  eyesBefore = eyes;
  if (Object.keys(now).length) lastInput = simTime;
  input = now;
  return now;
}

// Spike rates of the read-out neurons -> what the fly does for the next CHUNK
function behave(hz) {
  const dt = CHUNK, slow = Math.exp(-dt / RUN_HOLD_S), fast = Math.exp(-dt / HOLD_S);
  const gf = (hz.GF[0] + hz.GF[1]) / 2;
  cmd.run = hold(cmd.run, Math.min(gf / FULL.GF, 1), slow);
  cmd.turn = hold(cmd.turn, clamp((hz.DNa[1] - hz.DNa[0]) / FULL.DNa, -1, 1), fast); // + = right
  cmd.back = hold(cmd.back, Math.min((hz.MDN[0] + hz.MDN[1]) / FULL.MDN, 1), fast);
  cmd.feed = hold(cmd.feed, Math.min((hz.MN9[0] + hz.MN9[1]) / 2 / FULL.MN9, 1), fast);
  cmd.groom = hold(cmd.groom, Math.min(Math.max(...hz.groom) / 2 / FULL.groom, 1), fast);

  const turn = -cmd.turn * TURN_RATE;
  const jump = hz.TTMn ? (hz.TTMn[0] + hz.TTMn[1]) / 2 : gf;
  if (fly.flightT >= 0 || jump >= JUMP_HZ) {
    if (fly.flightT < 0) (fly.flightT = 0), (fly.doing = hz.TTMn ? "jumpTTMn" : "jumpGF");
    fly.flightT += dt;
    fly.heading += turn * dt;
    fly.speed = FLIGHT_SPEED;
    fly.z = 3 * Math.sin(Math.PI * Math.min(fly.flightT / FLIGHT_S, 1));
    if (fly.flightT >= FLIGHT_S) (fly.flightT = -1), (fly.z = 0);
  } else if (cmd.run > 0.2) {
    fly.speed = 18 * cmd.run;
    fly.heading += turn * dt;
    fly.doing = "run";
  } else if (cmd.back > 0.2) {
    fly.speed = -8 * cmd.back;
    fly.heading += turn * dt;
    fly.doing = "back";
  } else if (cmd.groom > 0.3) {
    fly.speed = 0;
    fly.groom += dt;
    fly.doing = "groom";
  } else if (cmd.feed > 0.3) {
    fly.speed = 0;
    fly.doing = "feed";
    const hx = fly.x + Math.cos(fly.heading) * 1.2, hy = fly.y + Math.sin(fly.heading) * 1.2;
    for (const d of drops) if (d.kind === "sugar" && Math.hypot(d.x - hx, d.y - hy) < d.r) d.r -= 0.25 * dt;
    drops = drops.filter((d) => d.r > 0.6);
  } else {
    // nothing from the brain: a made-up stroll (the model has no activity of its own)
    wander.timer -= dt;
    if (wander.timer <= 0) {
      wander.walking = !wander.walking;
      wander.timer = wander.walking ? 1.5 + 3 * Math.random() : 0.5 + 1.5 * Math.random();
    }
    const gauss = Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
    wander.turn += (-wander.turn * dt) / 0.4 + gauss * 2.5 * Math.sqrt(dt);
    fly.speed = wander.walking ? 6 : 0;
    fly.heading += (wander.walking ? wander.turn : 0) * dt + turn * dt;
    fly.doing = wander.walking ? "stroll" : "stand";
  }
  fly.proboscis += ((cmd.feed > 0.3 ? 1 : 0) - fly.proboscis) * 0.3;

  fly.x += Math.cos(fly.heading) * fly.speed * dt;
  fly.y += Math.sin(fly.heading) * fly.speed * dt;
  fly.legs += Math.abs(fly.speed) * dt * 5;
  const r = Math.hypot(fly.x, fly.y), wall = ARENA_R - 1.3;
  if (r > ARENA_R - 3 && fly.flightT < 0) {
    // the wall of the dish: turn along it (made up)
    const inward = wrap(Math.atan2(-fly.y, -fly.x) - fly.heading);
    fly.heading += clamp(inward, -3 * dt, 3 * dt) * (fly.speed >= 0 ? 1 : 0);
  }
  if (r > wall) {
    fly.x *= wall / r;
    fly.y *= wall / r;
    if (fly.flightT >= 0) (fly.flightT = -1), (fly.z = 0);
  }
  simTime += dt;
}

// ---------------------------------------------------------------- brain
// Everything that depends on the connectome is set by loadBrain()
let meta, OUT, outOf, sideOf, shown; // shown: smoothed output rates (Hz) for the page
let worker, ready = false, pending = false, wallRef = null, loads = 0;
let spikesPerS = 0, lastInput = 0; // lastInput: brain time of the last sensory input
let resetting = false; // the chunk on its way was computed before "Calm the brain"
const view = new BrainView($("brain"), $("brain-overlay"));
const NAMES = { 783: "female", mcns: "male" };

async function loadBrain(version) {
  const load = ++loads;
  stopBrain();
  const get = async (name) => {
    const response = await fetch(new URL(`data/${version}/${name}`, location.href));
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    return response;
  };
  let m, xyz;
  try {
    [m, xyz] = await Promise.all([get("meta.json").then((r) => r.json()), get("map.bin").then((r) => r.arrayBuffer())]);
  } catch (error) {
    if (load !== loads) return;
    // the old brain has stopped too: the buttons go back to it (the shuffle checkbox restarts it)
    if (meta) setPressed("[data-brain]", (b) => b.dataset.brain === meta.version);
    failed(version, error.message);
    return;
  }
  if (load !== loads) return; // another brain was picked meanwhile
  meta = m;
  OUT = meta.outputs;
  outOf = new Int8Array(meta.n).fill(-1);
  sideOf = new Uint8Array(meta.n);
  OUT.forEach((k, o) => SIDES.forEach((s, si) => meta.groups[k][s].forEach((i) => ((outOf[i] = o), (sideOf[i] = si)))));
  shown = Object.fromEntries(OUT.map((k) => [k, 0]));
  view.setBrain(meta, xyz);
  const count = meta.n.toLocaleString("en");
  $("brain-sub").textContent = `${NAMES[version]} · ${count} neurons`;
  $("intro-count").textContent = count;
  setPressed("[data-view]", (b) => b.dataset.view === meta.start_view);
  buildChips();
  buildMeters();
  startBrain();
}

// The old brain stops: no worker, no readings left over from it
function stopBrain() {
  worker?.terminate();
  ready = pending = resetting = false;
  spikesPerS = 0;
  rateHistory.fill(0);
  $("calm").hidden = true;
  setLoading("loading the brain…");
}

// A brain that did not load: which and why, and how to try again
function failed(version, why) {
  worker?.terminate(); // its other downloads stop too
  setLoading(`the ${NAMES[version]} brain failed to load: ${why}. Pick it again to retry.`);
}

function startBrain() {
  stopBrain();
  worker = new Worker("brain.js", { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "progress") setLoading(data.text);
    else if (data.type === "ready") {
      ready = true;
      wallRef = null;
      lastInput = simTime;
      speedMark = [performance.now(), simTime];
      setLoading(null);
    } else if (data.type === "spikes") onSpikes(data.spikes);
    else if (data.type === "error") failed(meta.version, data.text);
  };
  worker.onerror = (e) => failed(meta.version, e.message || "brain.js did not start");
  worker.postMessage({ type: "load", base: new URL(`data/${meta.version}/`, location.href).href, meta, shuffled: $("shuffled").checked });
}

function setLoading(text) {
  for (const id of ["loading", "brain-loading"]) {
    $(id).hidden = !text;
    if (text) $(id).textContent = text;
  }
}

function onSpikes(spikes) {
  pending = false;
  const hz = Object.fromEntries(OUT.map((k) => [k, [0, 0]]));
  for (const i of spikes) {
    const o = outOf[i];
    if (o >= 0) hz[OUT[o]][sideOf[i]] += 1 / CHUNK;
  }
  view.spike(spikes);
  behave(hz);
  const now = {
    GF: (hz.GF[0] + hz.GF[1]) / 2, DNa: hz.DNa[1] - hz.DNa[0], MDN: hz.MDN[0] + hz.MDN[1],
    MN9: (hz.MN9[0] + hz.MN9[1]) / 2, groom: Math.max(...hz.groom) / 2, TTMn: hz.TTMn && (hz.TTMn[0] + hz.TTMn[1]) / 2,
  };
  const k = 1 - Math.exp(-CHUNK / 0.15);
  for (const name of OUT) shown[name] += (now[name] - shown[name]) * k;
  if (resetting) resetting = false; // spikes from before the reset: not counted
  else spikesPerS += (spikes.length / CHUNK - spikesPerS) * k;
  view.setActivity(input, shown);
  // No fatigue in this model: a brain kicked into a self-sustaining wave never calms down
  $("calm").hidden = !(spikesPerS > OVERLOAD_SPIKES || (simTime - lastInput > 1.5 && spikesPerS > CALM_SPIKES));
  pump();
}

// Ask the brain for the next CHUNK once the world has caught up with the clock
function pump() {
  if (!ready || pending) return;
  const wall = performance.now() / 1000;
  if (wallRef === null) wallRef = wall - simTime;
  if (wall - wallRef - simTime > 0.2) wallRef = wall - simTime - 0.2; // brain behind: slow down, don't jump
  if (simTime + CHUNK <= wall - wallRef) {
    pending = true;
    worker.postMessage({ type: "run", ms: CHUNK * 1000, input: senses() });
  }
}

// ---------------------------------------------------------------- the brain card: chips and info
let selected = null, chipsFiring = [];
function buildChips() {
  selected = null;
  showInfo(null);
  for (const [row, kind] of [["chips-senses", "sense"], ["chips-actions", "action"]]) {
    const el = $(row);
    el.querySelectorAll("button").forEach((b) => b.remove());
    for (const key of Object.keys(GROUPS)) {
      if (GROUPS[key].kind !== kind || !meta.groups[key]) continue;
      const b = document.createElement("button");
      b.className = `chip ${kind}`;
      b.dataset.group = key;
      b.setAttribute("aria-pressed", "false");
      if (GROUPS[key].color) b.style.setProperty("--c", GROUPS[key].color);
      b.innerHTML = "<i></i>";
      b.append(GROUPS[key].name);
      b.addEventListener("click", () => select(selected === key ? null : key));
      b.addEventListener("pointerenter", (e) => e.pointerType === "mouse" && (view.highlight(key), showInfo(key)));
      b.addEventListener("pointerleave", () => (view.highlight(selected), showInfo(selected)));
      el.append(b);
    }
  }
  chipsFiring = [...document.querySelectorAll(".chip.action")];
}

function select(key) {
  selected = key;
  view.highlight(key);
  showInfo(key);
  setPressed(".chip", (b) => b.dataset.group === key);
}

function showInfo(key) {
  const info = $("info");
  info.hidden = !key;
  if (!key) return;
  const g = GROUPS[key], count = meta.groups[key].left.length + meta.groups[key].right.length;
  info.style.setProperty("--c", g.color ?? css("--action"));
  info.querySelector("h3").textContent = g.name;
  info.querySelector(".cells").textContent = `${g.cells} · ${count} neuron${count === 1 ? "" : "s"}`;
  info.querySelector("p").textContent = g.about;
}

// ---------------------------------------------------------------- senses -> brain -> actions
const rateHistory = new Array(100).fill(0); // spikes/s over the last 10 s
let meters = {};
function buildMeters() {
  meters = {};
  const row = (parent, key, name, detail, color) => {
    parent.insertAdjacentHTML("beforeend", `<div class="name"><b></b><small></small></div><div class="bar"><i></i></div><div class="hz">0 Hz</div>`);
    const [label, bar, value] = [...parent.children].slice(-3);
    label.firstChild.textContent = name;
    label.lastChild.textContent = detail;
    if (color) bar.style.setProperty("--c", color);
    meters[key] = { label: label.firstChild, bar: bar.firstChild, value };
  };
  $("senses").textContent = "";
  $("actions").textContent = "";
  for (const key of meta.inputs) {
    const count = meta.groups[key].left.length + meta.groups[key].right.length;
    row($("senses"), key, GROUPS[key].name, `${count} ${SENSE_CELLS[key]}`, GROUPS[key].color);
  }
  for (const key of ["GF", "TTMn", "DNa", "MDN", "MN9", "groom"]) if (OUT.includes(key)) row($("actions"), key, GROUPS[key].name, ACTION_CELLS[key]);
}

// Runs every frame: touch the page only where something changed (#status is a live region,
// and screen readers may read out every rewrite)
const setText = (el, text) => el.textContent !== text && (el.textContent = text);

function drawMeters() {
  if (!meta) return;
  for (const key of meta.inputs) {
    const m = meters[key], l = input[key + ".left"] ?? 0, r = input[key + ".right"] ?? 0;
    m.bar.style.width = `${(Math.max(l, r) / SENSE_MAX[key]) * 100}%`;
    setText(m.value, key === "loom" && (l || r) ? `L ${l} · R ${r}` : `${Math.max(l, r)} Hz`);
    m.value.classList.toggle("on", l > 0 || r > 0);
  }
  for (const key of OUT) {
    const m = meters[key], v = shown[key];
    m.bar.style.width = `${Math.min(100, (Math.abs(v) / FULL[key]) * 100)}%`;
    setText(m.value, `${Math.abs(v).toFixed(0)} Hz`);
    m.value.classList.toggle("on", Math.abs(v) >= 4);
    if (key === "DNa") setText(m.label, Math.abs(v) >= 4 ? (v > 0 ? "Turn right" : "Turn left") : "Turn");
  }
  for (const b of chipsFiring) b.classList.toggle("firing", Math.abs(shown[b.dataset.group] ?? 0) >= 4);
  const rate = Math.round(spikesPerS).toLocaleString("en"), lit = view.lit.length.toLocaleString("en");
  setText($("rate"), rate);
  setText($("lit"), `${lit} of ${meta.n.toLocaleString("en")} neurons lit`);
  setText($("brain-stats"), `${rate} spikes/s · ${lit} neurons lit`);
  const [text, tone] = STATUS[ready ? fly.doing : "wait"];
  setText($("status"), text);
  if ($("status").className !== `status ${tone}`) $("status").className = `status ${tone}`;
}

function drawSpark() {
  const top = Math.max(2000, ...rateHistory);
  $("spark").setAttribute("points", rateHistory.map((v, i) => `${(i * 220) / 99},${(42 - (v / top) * 38).toFixed(1)}`).join(" "));
}

// ---------------------------------------------------------------- the dish
const arena = $("arena"), ctx = arena.getContext("2d"), PANEL = css("--panel");
let scale = 1; // px per mm
function resize() {
  const size = Math.round(arena.getBoundingClientRect().width * Math.min(devicePixelRatio || 1, 2));
  arena.width = arena.height = size;
  scale = size / 2 / (ARENA_R + 1.5);
}
new ResizeObserver(resize).observe(arena);

function drawArena() {
  const W = arena.width;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const lamp = ctx.createRadialGradient(W / 2, W * 0.46, 0, W / 2, W * 0.46, W * 0.62);
  lamp.addColorStop(0, "#2A2A28");
  lamp.addColorStop(1, PANEL);
  ctx.fillStyle = lamp;
  ctx.fillRect(0, 0, W, W);
  ctx.setTransform(scale, 0, 0, -scale, W / 2, W / 2); // mm, y up
  ctx.fillStyle = "#8C8578";
  circle(0, 0, ARENA_R + 1);
  ctx.fillStyle = "#EDE8DD";
  circle(0, 0, ARENA_R);
  ctx.strokeStyle = "#DCD5C7";
  ctx.lineWidth = 0.1;
  ctx.beginPath();
  ctx.arc(0, 0, ARENA_R * 0.88, 0, 2 * Math.PI);
  ctx.stroke();

  for (const d of drops) {
    ctx.fillStyle = d.kind === "sugar" ? "rgba(227,169,59,0.5)" : "rgba(165,124,242,0.45)";
    circle(d.x, d.y, d.r);
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    circle(d.x - d.r * 0.35, d.y + d.r * 0.35, d.r * 0.2);
  }
  for (const p of puffs) {
    const age = (simTime - p.t) / PUFF_S;
    if (age > 1) continue;
    ctx.strokeStyle = `rgba(63,194,174,${1 - age})`;
    ctx.lineWidth = 0.3;
    for (const f of [1, 0.7, 0.4]) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, PUFF_REACH * age * f, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }
  puffs = puffs.filter((p) => simTime - p.t < PUFF_S);
  for (const sh of shadows) {
    const h = shadowHeight(sh);
    if (h === null) continue;
    const near = 1 - h / SHADOW_H, r = SHADOW_R * (1 + h / 25);
    const grad = ctx.createRadialGradient(sh.x, sh.y, 0, sh.x, sh.y, r);
    grad.addColorStop(0, `rgba(20,24,40,${0.12 + 0.55 * near})`);
    grad.addColorStop(1, "rgba(20,24,40,0)");
    ctx.fillStyle = grad;
    circle(sh.x, sh.y, r);
  }
  shadows = shadows.filter((sh) => shadowHeight(sh) !== null);

  if (fly.z > 0) {
    ctx.fillStyle = "rgba(0,0,0,0.18)";
    ellipse(fly.x - 0.3, fly.y - fly.z * 0.3, 1.6, 0.8, fly.heading);
  }
  ctx.save();
  ctx.translate(fly.x, fly.y + fly.z * 0.3);
  ctx.rotate(fly.heading);
  ctx.scale(1 + fly.z / 8, 1 + fly.z / 8);
  drawFly();
  ctx.restore();
}

function circle(x, y, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
}
function ellipse(x, y, rx, ry, angle) {
  ctx.beginPath();
  ctx.ellipse(x, y, rx, ry, angle, 0, 2 * Math.PI);
  ctx.fill();
}

// Top view, facing +x, left = +y, in mm (a fly is ~2.5 mm long)
function drawFly() {
  const flying = fly.flightT >= 0;
  ctx.strokeStyle = "#3b2a1c";
  ctx.lineCap = "round";
  ctx.lineWidth = 0.09;
  if (!flying) {
    for (const [k, base, at] of [[0, 0.9, 0.3], [1, 1.6, 0.1], [2, 2.3, -0.12]]) {
      for (const side of [1, -1]) {
        const tripod = (k + (side > 0 ? 0 : 1)) % 2; // L1 R2 L3 against R1 L2 R3
        let a = base + 0.3 * Math.sin(fly.legs + tripod * Math.PI);
        if (k === 0 && cmd.groom > 0.3) a = 0.35 + 0.25 * Math.sin(fly.groom * 25);
        const kx = at + Math.cos(a) * 0.8, ky = side * (0.3 + Math.sin(a) * 0.8);
        const b = a + (k === 2 ? 0.4 : -0.3);
        ctx.beginPath();
        ctx.moveTo(at, side * 0.25);
        ctx.lineTo(kx, ky);
        ctx.lineTo(kx + Math.cos(b) * 0.8, ky + side * Math.sin(b) * 0.8);
        ctx.stroke();
      }
    }
  }
  if (fly.proboscis > 0.02) {
    ctx.lineWidth = 0.14;
    ctx.strokeStyle = "#6b4a2a";
    ctx.beginPath();
    ctx.moveTo(1.05, 0);
    ctx.lineTo(1.05 + 0.6 * fly.proboscis, 0);
    ctx.stroke();
  }
  ctx.fillStyle = "#c9a36b";
  ellipse(-0.85, 0, 0.95, 0.55, 0);
  ctx.strokeStyle = "rgba(90,60,30,0.75)";
  ctx.lineWidth = 0.12;
  for (const x of [-0.55, -0.95, -1.35]) {
    ctx.beginPath();
    ctx.ellipse(x, 0, 0.1, 0.5 * Math.sqrt(1 - ((x + 0.85) / 0.95) ** 2), 0, 0, 2 * Math.PI);
    ctx.stroke();
  }
  ctx.fillStyle = "#9d7847";
  ellipse(0.15, 0, 0.55, 0.45, 0);
  ctx.fillStyle = "#a47b48";
  ellipse(0.85, 0, 0.3, 0.42, 0);
  ctx.fillStyle = "#b3261e";
  ellipse(0.88, 0.27, 0.22, 0.17, 0.3);
  ellipse(0.88, -0.27, 0.22, 0.17, -0.3);
  ctx.fillStyle = "rgba(225,232,240,0.5)";
  ctx.strokeStyle = "rgba(110,120,135,0.6)";
  ctx.lineWidth = 0.03;
  for (const side of [1, -1]) {
    const angle = flying ? side * (1.3 + 0.3 * Math.sin(fly.flightT * 120)) : side * 2.98;
    ctx.save();
    ctx.translate(0.2, side * 0.2);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.ellipse(1.05, 0, 1.05, 0.33, 0, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}

// ---------------------------------------------------------------- controls and the main loop
function setPressed(selector, test) {
  for (const b of document.querySelectorAll(selector)) b.setAttribute("aria-pressed", test(b));
}

let tool = "sugar";
for (const b of document.querySelectorAll("[data-tool]")) {
  b.addEventListener("click", () => {
    tool = b.dataset.tool;
    setPressed("[data-tool]", (o) => o === b);
  });
}
for (const b of document.querySelectorAll("[data-brain]")) {
  b.addEventListener("click", () => {
    setPressed("[data-brain]", (o) => o === b);
    loadBrain(b.dataset.brain);
  });
}
for (const b of document.querySelectorAll("[data-view]")) {
  b.addEventListener("click", () => {
    view.setView(b.dataset.view);
    setPressed("[data-view]", (o) => o === b);
  });
}
$("zoom-in").addEventListener("click", () => view.zoom(1.25));
$("zoom-out").addEventListener("click", () => view.zoom(0.8));
$("clear").addEventListener("click", () => {
  drops = [];
  puffs = [];
  shadows = [];
});
$("shuffled").addEventListener("change", () => meta && startBrain());
$("calm").querySelector("button").addEventListener("click", () => {
  worker.postMessage({ type: "reset" });
  // the rate is smoothed, and the chunk on its way was run before the reset (the worker takes
  // messages in order): counted, they would bring the button right back
  spikesPerS = 0;
  resetting = pending;
  $("calm").hidden = true;
});

function place(tool, x, y) {
  if (Math.hypot(x, y) > ARENA_R) return;
  if (tool === "sugar" || tool === "bitter") drops.push({ x, y, r: DROP_R, kind: tool });
  else if (tool === "puff") puffs.push({ x, y, t: simTime });
  else shadows.push({ x, y, t: simTime });
}
arena.addEventListener("pointerdown", (e) => {
  const rect = arena.getBoundingClientRect();
  const size = arena.width / scale; // mm across the canvas
  place(tool, ((e.clientX - rect.left) / rect.width - 0.5) * size, -((e.clientY - rect.top) / rect.height - 0.5) * size);
});

let lastFrame = performance.now(), speedMark = [performance.now(), 0], sampled = 0;
function frame(now) {
  const dt = Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  pump();
  if (now - sampled > 100) {
    sampled = now;
    rateHistory.push(spikesPerS);
    rateHistory.shift();
    drawSpark();
  }
  if (now - speedMark[0] > 1000) {
    const ratio = (simTime - speedMark[1]) / ((now - speedMark[0]) / 1000);
    $("speed").textContent = !ready ? "" : ratio > 0.93 ? "real time" : `slowed ×${ratio.toFixed(2)}`;
    speedMark = [now, simTime];
  }
  drawArena();
  view.draw(dt);
  drawMeters();
  requestAnimationFrame(frame);
}

resize();
loadBrain("783");
requestAnimationFrame(frame);
