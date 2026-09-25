// The terrarium: the world, what the fly senses, what its spikes make it do, and drawing.
// The brain itself runs in brain.js (a worker). The world advances in steps of CHUNK of
// brain time: the page sends what the sensory neurons get, the worker answers with the
// spikes of all neurons, then the fly moves. If the brain is slower than real time,
// the world slows down with it.

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
// Motor side, as run_escape.py plus MN9 and grooming: Hz for a full-strength command
const FULL = { GF: 50, DNa: 40, MDN: 40, MN9: 40, groom: 20 };
const RUN_HOLD_S = 0.6, HOLD_S = 0.2; // s the commands linger after the neurons fall silent
// The fly jumps when the jump muscle motor neuron TTMn fires (male CNS). FlyWire ends at the
// neck and has no TTMn; there a giant fibre spike stands in for it (a made-up rule).
// 25 Hz (mean of both) = one spike of one neuron within CHUNK.
const JUMP_HZ = 25;
const FLIGHT_S = 0.5, FLIGHT_SPEED = 25; // s, mm/s: a hop across part of the dish
const TURN_RATE = 4; // rad/s at full DNa difference
const METERS = [
  ["GF", "Giant fibre", "DNp01: escape"],
  ["TTMn", "Jump", "TTMn, motor neuron of the jump muscle (in the nerve cord)"],
  ["DNa", "Turn", "DNa01+DNa02, right minus left"],
  ["MDN", "Backwards", "MDN, the \"moonwalker\" neurons"],
  ["MN9", "Proboscis", "motor neuron MN9: feed"],
  ["groom", "Antennal grooming", "aDN1 / aDN2"],
];
FULL.TTMn = 50;
// spikes/s: a brain this busy long after its last input is stuck in a self-sustaining wave;
// OVERLOAD_SPIKES is more than any stimulus here causes without one (loom on both eyes: ~150k)
const CALM_SPIKES = 20000, OVERLOAD_SPIKES = 250000;

const $ = (id) => document.getElementById(id);
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const hold = (old, now, decay) => (Math.abs(now) >= Math.abs(old) * decay ? now : old * decay);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const rgb = (hex) => [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16));

const SIDES = ["left", "right"];

// ---------------------------------------------------------------- world
const fly = { x: 0, y: 0, heading: Math.PI / 2, z: 0, speed: 0, legs: 0, proboscis: 0, groom: 0, flightT: -1, doing: "" };
const cmd = { run: 0, turn: 0, back: 0, feed: 0, groom: 0 };
const wander = { walking: true, timer: 2, turn: 0 };
let drops = [], puffs = [], shadows = [];
let simTime = 0, eyesBefore = null;

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
  const input = {};
  if (fly.flightT < 0) {
    const hx = fly.x + Math.cos(fly.heading) * 1.2, hy = fly.y + Math.sin(fly.heading) * 1.2;
    for (const d of drops) if (Math.hypot(d.x - hx, d.y - hy) < d.r) input[d.kind + ".left"] = input[d.kind + ".right"] = TASTE_HZ;
  }
  // a puff bends the antennae the more, the closer it is: ANTENNA_HZ right at the fly, 0 at PUFF_REACH
  let wind = 0;
  for (const p of puffs) {
    if (simTime - p.t < PUFF_S) wind = Math.max(wind, 1 - Math.hypot(p.x - fly.x, p.y - fly.y) / PUFF_REACH);
  }
  if (wind > 0) input["antenna.left"] = input["antenna.right"] = Math.round(ANTENNA_HZ * wind);
  const eyes = darkArea();
  if (eyesBefore) {
    SIDES.forEach((s, k) => {
      const hz = clamp(LOOM_GAIN * ((eyes[k] - eyesBefore[k]) / CHUNK - LOOM_MIN), 0, LOOM_MAX);
      if (hz > 0) input["loom." + s] = Math.round(hz);
    });
  }
  eyesBefore = eyes;
  if (Object.keys(input).length) lastInput = simTime;
  return input;
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
    if (fly.flightT < 0) {
      fly.flightT = 0;
      fly.doing = hz.TTMn ? "jumped: TTMn, the jump motor neuron" : "jumped: a giant fibre spike (a made-up rule)";
    }
    fly.flightT += dt;
    fly.heading += turn * dt;
    fly.speed = FLIGHT_SPEED;
    fly.z = 3 * Math.sin(Math.PI * Math.min(fly.flightT / FLIGHT_S, 1));
    if (fly.flightT >= FLIGHT_S) (fly.flightT = -1), (fly.z = 0);
  } else if (cmd.run > 0.2) {
    fly.speed = 18 * cmd.run;
    fly.heading += turn * dt;
    fly.doing = "runs away: giant fibre";
  } else if (cmd.back > 0.2) {
    fly.speed = -8 * cmd.back;
    fly.heading += turn * dt;
    fly.doing = "backs up: MDN";
  } else if (cmd.groom > 0.3) {
    fly.speed = 0;
    fly.groom += dt;
    fly.doing = "cleans its antennae: aDN1/aDN2";
  } else if (cmd.feed > 0.3) {
    fly.speed = 0;
    fly.doing = "feeds: MN9 extended the proboscis";
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
    fly.doing = wander.walking ? "strolls (made up: the brain is silent)" : "stands (made up)";
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
let meta, OUT, outOf, sideOf, shown; // shown: smoothed Hz for the meters
let worker, ready = false, pending = false, wallRef = null, loads = 0;
let spikesPerS = 0, lastInput = 0; // lastInput: brain time of the last sensory input

async function loadBrain(version) {
  const load = ++loads;
  ready = pending = false;
  worker?.terminate();
  $("loading").style.display = "flex";
  $("loading").textContent = "loading the brain…";
  const base = new URL(`data/${version}/`, location.href).href;
  const [m, xy] = await Promise.all([
    fetch(base + "meta.json").then((r) => r.json()),
    fetch(base + "map.bin").then((r) => r.arrayBuffer()),
  ]);
  if (load !== loads) return; // another brain was picked meanwhile
  meta = m;
  OUT = meta.outputs;
  outOf = new Int8Array(meta.n).fill(-1);
  sideOf = new Uint8Array(meta.n);
  OUT.forEach((k, o) => SIDES.forEach((s, si) => meta.groups[k][s].forEach((i) => ((outOf[i] = o), (sideOf[i] = si)))));
  shown = Object.fromEntries(OUT.map((k) => [k, 0]));
  spikesPerS = 0;
  setupMap(new Uint16Array(xy));
  setupMeters();
  startBrain();
}

function startBrain() {
  worker?.terminate();
  ready = pending = false;
  $("loading").style.display = "flex";
  worker = new Worker("brain.js", { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "progress") $("loading").textContent = data.text;
    else if (data.type === "ready") {
      ready = true;
      wallRef = null;
      lastInput = simTime;
      speedMark = [performance.now(), simTime];
      $("loading").style.display = "none";
    } else if (data.type === "spikes") onSpikes(data.spikes);
  };
  worker.onerror = (e) => ($("loading").textContent = "the brain failed to load: " + e.message);
  worker.postMessage({ type: "load", base: new URL(`data/${meta.version}/`, location.href).href, shuffled: $("shuffled").checked });
}

function onSpikes(spikes) {
  pending = false;
  const hz = Object.fromEntries(OUT.map((k) => [k, [0, 0]]));
  for (const i of spikes) {
    const o = outOf[i];
    if (o >= 0) hz[OUT[o]][sideOf[i]] += 1 / CHUNK;
    light(i);
  }
  behave(hz);
  const now = {
    GF: (hz.GF[0] + hz.GF[1]) / 2, DNa: hz.DNa[1] - hz.DNa[0], MDN: hz.MDN[0] + hz.MDN[1],
    MN9: (hz.MN9[0] + hz.MN9[1]) / 2, groom: Math.max(...hz.groom) / 2, TTMn: hz.TTMn && (hz.TTMn[0] + hz.TTMn[1]) / 2,
  };
  const k = 1 - Math.exp(-CHUNK / 0.15);
  for (const name of OUT) shown[name] += (now[name] - shown[name]) * k;
  spikesPerS += (spikes.length / CHUNK - spikesPerS) * k;
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

// ---------------------------------------------------------------- brain map
const mapCanvas = $("map"), mctx = mapCanvas.getContext("2d");
const MAP_W = 356, MAX_MAP_H = 400, MARGIN = 8;
const ROLE_COLORS = ["#ffd27a", ...["--sugar", "--bitter", "--antenna", "--loom", "--readout"].map(css)].map(rgb);
// the starting view: FlyWire is a brain, seen from the front; the male CNS also has the nerve
// cord, which hides behind the brain from the front, so it starts seen from above
const MAP_TITLES = { 783: "Brain in 3D, from the front", mcns: "Brain and nerve cord in 3D, from above" };
const LABELS = { GF: "GF", DNa: "DNa", MDN: "MDN", MN9: "MN9", groom: "aDN", TTMn: "TTMn" };
const NO_POSITION = -32768;
let MAP_H, xyz, mapScale, px, py, role, background, frameImage, frame32, glow, lit, readouts, labelAt;
let yaw = 0, pitch = 0, turned = true; // rotation of the 3D map; turned: needs projecting again

// Cell bodies (3D, 0.1 µm, as exported) -> sizes, colours; the picture is made by project()
function setupMap(buffer) {
  xyz = new Int16Array(buffer);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < meta.n; i++) {
    if (xyz[3 * i] === NO_POSITION) continue;
    x0 = Math.min(x0, xyz[3 * i]), x1 = Math.max(x1, xyz[3 * i]);
    y0 = Math.min(y0, xyz[3 * i + 1]), y1 = Math.max(y1, xyz[3 * i + 1]);
  }
  // the starting view fits into MAP_W x MAX_MAP_H (the male CNS with its nerve cord is tall)
  mapScale = Math.min((MAP_W - 2 * MARGIN) / (x1 - x0), (MAX_MAP_H - 2 * MARGIN - 12) / (y1 - y0));
  MAP_H = Math.round((y1 - y0) * mapScale) + 2 * MARGIN + 12;
  mapCanvas.width = MAP_W;
  mapCanvas.height = MAP_H;
  $("map-title").textContent = MAP_TITLES[meta.version];
  role = new Uint8Array(meta.n);
  meta.inputs.forEach((k, r) => SIDES.forEach((s) => meta.groups[k][s].forEach((i) => (role[i] = r + 1))));
  readouts = [];
  for (let i = 0; i < meta.n; i++) if (outOf[i] >= 0) (role[i] = 5), readouts.push(i);
  px = new Int16Array(meta.n);
  py = new Int16Array(meta.n);
  background = mctx.createImageData(MAP_W, MAP_H);
  frameImage = mctx.createImageData(MAP_W, MAP_H);
  frame32 = new Uint32Array(frameImage.data.buffer);
  glow = new Float32Array(meta.n);
  lit = [];
  yaw = pitch = 0;
  turned = true;
}

// Turn the cells (yaw about the screen's vertical, then pitch about its horizontal axis), drop
// the depth: pixels, the density background and the labels of the read-out neurons
function project() {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const s = mapScale, midX = MAP_W / 2, midY = 12 + (MAP_H - 12) / 2;
  const count = new Float32Array(MAP_W * MAP_H);
  for (let i = 0; i < meta.n; i++) {
    const x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
    const X = Math.round(midX + (x * cy + z * sy) * s);
    const Y = Math.round(midY + (y * cp - (z * cy - x * sy) * sp) * s);
    const inside = x !== NO_POSITION && X > 0 && X < MAP_W - 1 && Y > 13 && Y < MAP_H - 1;
    px[i] = inside ? X : -1;
    py[i] = Y;
    if (inside) count[Y * MAP_W + X]++;
  }
  const max = count.reduce((a, b) => Math.max(a, b)), bg = rgb("#13151a");
  for (let p = 0; p < count.length; p++) {
    const f = Math.sqrt(count[p] / max) * 0.4;
    background.data.set([bg[0] + (150 - bg[0]) * f, bg[1] + (165 - bg[1]) * f, bg[2] + (190 - bg[2]) * f, 255], 4 * p);
  }
  labelAt = [];
  for (const k of OUT) {
    const ns = readouts.filter((i) => OUT[outOf[i]] === k && px[i] >= 0);
    if (!ns.length) continue;
    const i = ns.reduce((a, b) => (px[b] > px[a] ? b : a));
    let y = py[i] + 3;
    while (labelAt.some(([, x, ly]) => Math.abs(x - px[i] - 4) < 24 && Math.abs(ly - y) < 10)) y += 10; // no overlaps
    labelAt.push([LABELS[k], px[i] + 4, y]);
  }
  turned = false;
}

let drag = null;
mapCanvas.addEventListener("pointerdown", (e) => {
  drag = [e.clientX, e.clientY, yaw, pitch];
  mapCanvas.setPointerCapture(e.pointerId);
});
mapCanvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  yaw = drag[2] + (e.clientX - drag[0]) * 0.01;
  pitch = clamp(drag[3] + (e.clientY - drag[1]) * 0.01, -Math.PI / 2, Math.PI / 2);
  turned = true;
});
for (const type of ["pointerup", "pointercancel"]) mapCanvas.addEventListener(type, () => (drag = null));
mapCanvas.addEventListener("dblclick", () => ((yaw = pitch = 0), (turned = true)));

function light(i) {
  if (glow[i] < 0.05) lit.push(i);
  glow[i] += 1;
}
function drawMap(dt) {
  if (!frameImage) return;
  if (turned) project();
  frameImage.data.set(background.data);
  const fade = Math.exp(-dt / 0.3);
  const still = [];
  for (const i of lit) {
    glow[i] *= fade;
    if (glow[i] < 0.05) continue;
    still.push(i);
    if (px[i] < 0) continue; // no known position, or turned out of the picture
    const c = ROLE_COLORS[role[i]], f = Math.min(1, 0.45 + 0.3 * glow[i]);
    frame32[py[i] * MAP_W + px[i]] = (255 << 24) | ((c[2] * f) << 16) | ((c[1] * f) << 8) | (c[0] * f);
  }
  lit = still;
  for (const i of readouts) {
    if (px[i] < 0) continue;
    const c = ROLE_COLORS[5], f = glow[i] > 0.05 ? 1 : 0.35;
    const v = (255 << 24) | ((c[2] * f) << 16) | ((c[1] * f) << 8) | (c[0] * f);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) frame32[(py[i] + dy) * MAP_W + px[i] + dx] = v;
  }
  mctx.putImageData(frameImage, 0, 0);
  mctx.font = "10px ui-monospace, monospace";
  mctx.fillStyle = "rgba(235,235,235,0.85)";
  mctx.fillText(`${meta.n.toLocaleString("en")} neurons · ${Math.round(spikesPerS).toLocaleString("en")} spikes/s`, 6, 11);
  for (const [text, x, y] of labelAt) mctx.fillText(text, x, y);
}

// ---------------------------------------------------------------- meters
let meterEls = {};
function setupMeters() {
  $("meters").textContent = "";
  meterEls = {};
  for (const [key, title, what] of METERS.filter(([key]) => OUT.includes(key))) {
    $("meters").insertAdjacentHTML(
      "beforeend",
      `<span>${title}</span><span class="bar"><i id="bar-${key}"></i></span><span class="hz" id="hz-${key}"></span><span class="what">${what}</span>`,
    );
    meterEls[key] = [$("bar-" + key), $("hz-" + key)];
  }
}
function drawMeters() {
  for (const key in meterEls) {
    const [bar, text] = meterEls[key], v = shown[key];
    bar.style.width = `${Math.min(100, (Math.abs(v) / FULL[key]) * 100)}%`;
    text.textContent = key === "DNa" ? `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(0)} Hz` : `${v.toFixed(0)} Hz`;
  }
  $("doing").textContent = ready ? fly.doing : "waiting for the brain";
}

// ---------------------------------------------------------------- arena
const arena = $("arena"), ctx = arena.getContext("2d");
let scale = 1; // px per mm
function resize() {
  const size = Math.round(arena.getBoundingClientRect().width * devicePixelRatio);
  arena.width = arena.height = size;
  scale = size / 2 / (ARENA_R + 1.5);
}
new ResizeObserver(resize).observe(arena);

function drawArena() {
  const W = arena.width;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, W, W);
  ctx.setTransform(scale, 0, 0, -scale, W / 2, W / 2); // mm, y up
  ctx.fillStyle = css("--dish-rim");
  circle(0, 0, ARENA_R + 1);
  ctx.fillStyle = css("--dish");
  circle(0, 0, ARENA_R);

  for (const d of drops) {
    ctx.fillStyle = d.kind === "sugar" ? "rgba(224,165,38,0.45)" : "rgba(143,91,214,0.4)";
    circle(d.x, d.y, d.r);
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    circle(d.x - d.r * 0.35, d.y + d.r * 0.35, d.r * 0.2);
  }
  for (const p of puffs) {
    const age = (simTime - p.t) / PUFF_S;
    if (age > 1) continue;
    ctx.strokeStyle = `rgba(42,157,143,${1 - age})`;
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

// ---------------------------------------------------------------- input and the main loop
let tool = "sugar";
for (const b of document.querySelectorAll("[data-tool]")) {
  b.addEventListener("click", () => {
    tool = b.dataset.tool;
    for (const o of document.querySelectorAll("[data-tool]")) o.setAttribute("aria-pressed", o === b);
  });
}
$("clear").addEventListener("click", () => (drops = puffs = shadows = []));
$("shuffled").addEventListener("change", () => meta && startBrain());
for (const b of document.querySelectorAll("[data-brain]")) {
  b.addEventListener("click", () => {
    for (const o of document.querySelectorAll("[data-brain]")) o.setAttribute("aria-pressed", o === b);
    loadBrain(b.dataset.brain);
  });
}
$("calm").querySelector("button").addEventListener("click", () => {
  worker.postMessage({ type: "reset" });
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

let lastFrame = performance.now(), speedMark = [performance.now(), 0];
function frame(now) {
  const dt = Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  pump();
  if (now - speedMark[0] > 1000) {
    const ratio = (simTime - speedMark[1]) / ((now - speedMark[0]) / 1000);
    $("speed").textContent = !ready ? "" : ratio > 0.93 ? "the brain keeps up with real time" : `the brain lags: time ×${ratio.toFixed(2)}`;
    speedMark = [now, simTime];
  }
  drawArena();
  drawMap(dt);
  drawMeters();
  requestAnimationFrame(frame);
}

resize();
loadBrain("783");
requestAnimationFrame(frame);
