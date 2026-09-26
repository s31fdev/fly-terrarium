// The brain panel: every neuron's cell body in 3D, lit when it spikes. Drag to turn, wheel or
// pinch to zoom, double-click to go back; preset views; one group of neurons can be highlighted
// (the chips); the groups that are active get plain-language labels.

// The groups the page stimulates (senses) and reads (actions), in plain words
export const GROUPS = {
  sugar: { kind: "sense", name: "Sweet taste", color: "#E3A93B", cells: "sugar-sensing neurons of the proboscis",
    about: "They fire when the fly’s mouthparts touch sugar. Their signal reaches MN9, the motor neuron that extends the proboscis." },
  bitter: { kind: "sense", name: "Bitter taste", color: "#A57CF2", cells: "bitter-sensing neurons LB1a–d",
    about: "They fire on bitter food and silence the proboscis, even when there is sugar too." },
  antenna: { kind: "sense", name: "Antennae", color: "#3FC2AE", cells: "Johnston’s organ, JO-CE and JO-F",
    about: "They sense the antennae bending in the air. Their path leads to aDN1/aDN2, the command to clean the antennae." },
  loom: { kind: "sense", name: "Eyes", color: "#5B9BFF", cells: "looming detectors LC4, LPLC2, LC16",
    about: "They fire when something grows fast in one eye — the first step of an escape. The shadow tool drives them." },
  GF: { kind: "action", name: "Escape", cells: "giant fibre DNp01",
    about: "The escape command. A single spike of it can launch a jump." },
  TTMn: { kind: "action", name: "Jump", cells: "TTMn, in the nerve cord",
    about: "The motor neuron of the jump muscle. In the male’s map the giant fibre reaches it through the wiring." },
  DNa: { kind: "action", name: "Turn", cells: "DNa01 and DNa02",
    about: "The right ones minus the left ones decide which way the fly turns." },
  MDN: { kind: "action", name: "Back up", cells: "moonwalker neurons MDN",
    about: "They make the fly walk backwards." },
  MN9: { kind: "action", name: "Eat", cells: "proboscis motor neuron MN9",
    about: "It extends the proboscis to feed." },
  groom: { kind: "action", name: "Groom", cells: "aDN1 and aDN2",
    about: "Commands to clean the antennae with the front legs." },
};
const FIRE = "#FFD68C", ACTION = "#FF6B57", STAGE = "#07090D";
const NO_POSITION = -32768;
const FADE_S = 0.3; // a spike's flash fades with this time constant
const MARGIN = 28; // px around the brain in a preset view

const rgb = (hex) => [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16));

export class BrainView {
  constructor(canvas, overlay) {
    Object.assign(this, { canvas, overlay, ctx: canvas.getContext("2d") });
    this.bg = document.createElement("canvas"); // neuron density at CSS pixels
    this.haze = document.createElement("canvas"); // the same, blurred
    this.labels = new Map();
    this.tip = overlay.querySelector("[data-tip]");
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    this.listen();
  }

  // A new connectome: meta.json and map.bin of export_web.py
  setBrain(meta, buffer) {
    this.meta = meta;
    this.xyz = new Int16Array(buffer);
    const n = meta.n;
    this.X = new Float32Array(n); // turned, before scaling
    this.Y = new Float32Array(n);
    this.px = new Float32Array(n);
    this.py = new Float32Array(n);
    this.glow = new Float32Array(n);
    this.lit = [];
    this.role = new Uint8Array(n); // 0 other, 1.. a group (index into this.keys + 1)
    this.keys = Object.keys(GROUPS).filter((k) => meta.groups[k]);
    this.keys.forEach((k, r) => ["left", "right"].forEach((s) => meta.groups[k][s].forEach((i) => (this.role[i] = r + 1))));
    this.colors = [FIRE, ...this.keys.map((k) => GROUPS[k].color ?? ACTION)];
    this.readouts = this.keys.filter((k) => GROUPS[k].kind === "action").flatMap((k) => [...meta.groups[k].left, ...meta.groups[k].right]);
    this.selected = null;
    this.activity = { inputs: {}, outputs: {} };
    for (const el of this.labels.values()) el.remove();
    this.labels.clear();
    this.setView(meta.start_view);
  }

  setView(name) {
    if (!this.meta) return; // the buttons work before the brain has loaded
    [this.yaw, this.pitch] = this.meta.views[name];
    this.viewName = name;
    this.zoomBy = 1;
    this.offset = [0, 0];
    this.fitted = false;
    this.dirty = true;
  }

  zoom(factor, at = null) {
    if (!this.meta) return;
    this.nudge();
    const z = Math.min(8, Math.max(0.5, this.zoomBy * factor));
    if (at) {
      // keep the point under the cursor in place
      const [cx, cy] = [this.W / 2 + this.offset[0], this.H / 2 + this.offset[1]];
      const f = z / this.zoomBy;
      this.offset = [at[0] - (at[0] - cx) * f - this.W / 2, at[1] - (at[1] - cy) * f - this.H / 2];
    }
    this.zoomBy = z;
    this.dirty = true;
  }

  // Turn from (yaw, pitch) as by a drag of (right, down): the side facing you follows the pointer
  // (the depth axis points into the screen, so the yaw goes against the drag)
  turn(yaw, pitch, right, down) {
    this.yaw = yaw - right;
    this.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch + down));
  }

  // The blurred haze is the slow part of a redraw. It is left out while the brain is held, or
  // turned or zoomed in a quick series of steps (from the second step on), and comes back after.
  moving() {
    return this.held || performance.now() < this.stillAt;
  }

  nudge() {
    const now = performance.now();
    if (now - this.lastStep < 200) this.stillAt = now + 200;
    this.lastStep = now;
  }

  highlight(key) {
    this.selected = key;
    this.dirty = true;
  }

  spike(indices) {
    for (const i of indices) {
      if (this.glow[i] < 0.05) this.lit.push(i);
      this.glow[i] += 1;
    }
  }

  // inputs: {"loom.left": Hz, ...}; outputs: {GF: Hz, DNa: right-left Hz, ...}
  setActivity(inputs, outputs) {
    this.activity = { inputs, outputs };
  }

  resize() {
    const box = this.canvas.getBoundingClientRect();
    this.ratio = Math.min(devicePixelRatio || 1, 2);
    this.W = Math.max(1, Math.round(box.width));
    this.H = Math.max(1, Math.round(box.height));
    this.canvas.width = Math.round(this.W * this.ratio);
    this.canvas.height = Math.round(this.H * this.ratio);
    this.fitted = false;
    this.dirty = true;
  }

  // Turn (yaw about the vertical, then pitch about the horizontal axis), scale, drop the depth
  project() {
    const { xyz, meta, W, H, X, Y } = this;
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < meta.n; i++) {
      const x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
      if (x === NO_POSITION) {
        X[i] = NaN;
        continue;
      }
      X[i] = x * cy + z * sy;
      Y[i] = y * cp - (z * cy - x * sy) * sp;
      if (!this.fitted) (x0 = Math.min(x0, X[i])), (x1 = Math.max(x1, X[i])), (y0 = Math.min(y0, Y[i])), (y1 = Math.max(y1, Y[i]));
    }
    if (!this.fitted) {
      // a preset view fills the stage; turning by hand keeps the scale
      this.scale = Math.min((W - 2 * MARGIN) / (x1 - x0), (H - 2 * MARGIN) / (y1 - y0));
      this.center = [(x0 + x1) / 2, (y0 + y1) / 2];
      this.fitted = true;
    }
    const s = this.scale * this.zoomBy, ox = W / 2 + this.offset[0], oy = H / 2 + this.offset[1];
    // this runs on every frame of a drag: the buffers are kept while the size stays
    if (this.image?.width !== W || this.image?.height !== H) {
      this.count = new Float32Array(W * H);
      this.image = new ImageData(W, H);
      this.bg.width = this.haze.width = W;
      this.bg.height = this.haze.height = H;
    }
    const { count, image } = this, data = image.data;
    count.fill(0);
    data.fill(0);
    for (let i = 0; i < meta.n; i++) {
      if (Number.isNaN(X[i])) {
        this.px[i] = -1;
        continue;
      }
      const px = ox + (X[i] - this.center[0]) * s, py = oy + (Y[i] - this.center[1]) * s;
      const inside = px >= 0 && px < W && py >= 0 && py < H;
      this.px[i] = inside ? px : -1;
      this.py[i] = py;
      if (inside) count[(py | 0) * W + (px | 0)]++;
    }
    // density picture: how many cell bodies sit on each pixel
    let max = 1;
    for (let p = 0; p < count.length; p++) if (count[p] > max) max = count[p];
    for (let p = 0; p < count.length; p++) {
      if (!count[p]) continue;
      data[4 * p] = 150;
      data[4 * p + 1] = 165;
      data[4 * p + 2] = 190;
      data[4 * p + 3] = 40 + 190 * Math.sqrt(count[p] / max);
    }
    this.bg.getContext("2d").putImageData(image, 0, 0);
    const hz = this.haze.getContext("2d"), moving = this.moving();
    hz.clearRect(0, 0, W, H);
    if (!moving) {
      hz.filter = "blur(10px)";
      hz.drawImage(this.bg, 0, 0);
    }
    this.hazy = !moving;
    this.dirty = false;
  }

  // A soft round glow in one colour, drawn once and stamped under every flashing neuron
  sprite(color) {
    this.sprites ??= new Map();
    if (!this.sprites.has(color)) {
      const c = document.createElement("canvas"), [r, g, b] = rgb(color);
      c.width = c.height = 32;
      const x = c.getContext("2d"), grad = x.createRadialGradient(16, 16, 0, 16, 16, 16);
      grad.addColorStop(0, `rgba(${r},${g},${b},0.55)`);
      grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
      x.fillStyle = grad;
      x.fillRect(0, 0, 32, 32);
      this.sprites.set(color, c);
    }
    return this.sprites.get(color);
  }

  // where to put a group's label: its rightmost visible neuron, or the middle of a sense
  anchor(ids, middle) {
    let best = null, sx = 0, sy = 0, k = 0;
    for (const i of ids) {
      if (this.px[i] < 0) continue;
      if (middle) (sx += this.px[i]), (sy += this.py[i]), k++;
      else if (best === null || this.px[i] > this.px[best]) best = i;
    }
    if (middle) return k ? [sx / k, sy / k] : null;
    return best === null ? null : [this.px[best], this.py[best]];
  }

  draw(dt) {
    if (!this.meta || !this.W) return;
    if (!this.hazy && !this.moving()) this.dirty = true; // still again: the haze comes back
    if (this.dirty) this.project();
    const { ctx, px, py, glow, W, H } = this;
    ctx.setTransform(this.ratio, 0, 0, this.ratio, 0, 0);
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
    ctx.fillStyle = STAGE;
    ctx.fillRect(0, 0, W, H);
    const dim = this.selected ? 0.35 : 1;
    ctx.globalAlpha = 0.55 * dim;
    ctx.drawImage(this.haze, 0, 0);
    ctx.globalAlpha = 0.9 * dim;
    ctx.drawImage(this.bg, 0, 0);
    ctx.globalAlpha = 1;

    // the highlighted group: all of its neurons, in its colour
    const sel = this.selected && this.meta.groups[this.selected];
    if (sel && GROUPS[this.selected].kind === "sense") {
      ctx.fillStyle = GROUPS[this.selected].color;
      for (const i of [...sel.left, ...sel.right]) if (px[i] >= 0) ctx.fillRect(px[i] - 1.2, py[i] - 1.2, 2.4, 2.4);
    }

    // spikes: a soft glow, then a sharp dot, by colour and brightness in batches
    const fade = Math.exp(-dt / FADE_S), still = [], buckets = new Map();
    for (const i of this.lit) {
      glow[i] *= fade;
      if (glow[i] < 0.05) continue;
      still.push(i);
      if (px[i] < 0) continue;
      const key = this.role[i] * 4 + Math.min(3, Math.floor(glow[i] * 3));
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(i);
    }
    this.lit = still;
    ctx.globalCompositeOperation = "lighter";
    for (const [key, ids] of buckets) {
      const sprite = this.sprite(this.colors[key >> 2]);
      ctx.globalAlpha = 0.3 + 0.2 * (key & 3);
      for (const i of ids) ctx.drawImage(sprite, px[i] - 8, py[i] - 8, 16, 16);
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
    for (const [key, ids] of buckets) {
      const [r, g, b] = rgb(this.colors[key >> 2]), a = 0.55 + 0.15 * (key & 3);
      ctx.fillStyle = `rgba(${r},${g},${b},${a})`;
      for (const i of ids) ctx.fillRect(px[i] - 1, py[i] - 1, 2, 2);
    }

    // the neurons we read: rings, bright while they fire or when their group is highlighted
    ctx.lineWidth = 2;
    for (const i of this.readouts) {
      if (px[i] < 0) continue;
      const key = this.keys[this.role[i] - 1];
      const on = glow[i] > 0.05 || this.selected === key;
      ctx.strokeStyle = ctx.fillStyle = on ? ACTION : "rgba(255,107,87,0.4)";
      ctx.beginPath();
      ctx.arc(px[i], py[i], on ? 7 : 5.5, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(px[i], py[i], 2, 0, 2 * Math.PI);
      ctx.fill();
    }
    this.drawLabels();
  }

  // Labels for what is happening now: senses that get input, actions that fire, the highlighted group
  drawLabels() {
    const want = [];
    const { inputs, outputs } = this.activity;
    for (const [key, hz] of Object.entries(inputs)) {
      const [group, side] = key.split(".");
      if (!hz || (group !== "loom" && side === "right")) continue; // one label for both sides of taste and antennae
      const ids = group === "loom" ? this.meta.groups[group][side] : [...this.meta.groups[group].left, ...this.meta.groups[group].right];
      const name = group === "loom" ? (side === "left" ? "Left eye" : "Right eye") : GROUPS[group].name;
      want.push({ id: key, kind: "sense", color: GROUPS[group].color, title: name, text: `${Math.round(hz)} Hz in`, at: this.anchor(ids, true) });
    }
    for (const key of this.keys) {
      if (GROUPS[key].kind !== "action") continue;
      const hz = outputs[key] ?? 0;
      if (Math.abs(hz) < 4 && this.selected !== key) continue;
      const title = key === "DNa" && Math.abs(hz) >= 4 ? (hz > 0 ? "Turn right" : "Turn left") : GROUPS[key].name;
      want.push({ id: key, kind: "action", title, text: `${Math.round(Math.abs(hz))} Hz`, at: this.anchor([...this.meta.groups[key].left, ...this.meta.groups[key].right]) });
    }
    const seen = new Set(), placed = [];
    for (const w of want) {
      if (!w.at) continue;
      seen.add(w.id);
      let el = this.labels.get(w.id);
      if (!el) {
        el = document.createElement("div");
        el.className = `brain-label ${w.kind}`;
        el.innerHTML = "<b></b><span></span>";
        this.overlay.append(el);
        this.labels.set(w.id, el);
      }
      if (w.color) el.style.setProperty("--c", w.color);
      el.firstChild.textContent = w.title;
      el.lastChild.textContent = w.text;
      let [x, y] = [w.at[0] + 14, w.at[1] - 18];
      x = Math.min(x, this.W - 150);
      while (placed.some(([px, py]) => Math.abs(px - x) < 140 && Math.abs(py - y) < 40)) y += 42;
      placed.push([x, y]);
      el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
      el.hidden = false;
    }
    for (const [id, el] of this.labels) if (!seen.has(id)) el.hidden = true;
  }

  // The name of the read-out neuron under the pointer, if any
  hover(x, y) {
    if (!this.meta) return;
    let best = null, dist = 14;
    for (const i of this.readouts) {
      if (this.px[i] < 0) continue;
      const d = Math.hypot(this.px[i] - x, this.py[i] - y);
      if (d < dist) (dist = d), (best = i);
    }
    if (best === null) {
      this.tip.hidden = true;
      return;
    }
    const g = GROUPS[this.keys[this.role[best] - 1]];
    this.tip.textContent = `${g.name} · ${g.cells}`;
    this.tip.style.transform = `translate(${Math.round(Math.min(x + 12, this.W - 220))}px, ${Math.round(y + 14)}px)`;
    this.tip.hidden = false;
  }

  listen() {
    const c = this.canvas, pointers = new Map();
    let drag = null, pinch = null;
    const local = (e) => {
      const r = c.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    c.addEventListener("pointerdown", (e) => {
      // the left button moves the brain, the right one turns it; a finger turns it
      const mode = e.pointerType === "touch" ? "turn" : ["move", null, "turn"][e.button];
      if (!mode) return;
      c.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, local(e));
      this.held = this.armed = true;
      if (pointers.size === 1) drag = { mode, from: local(e), yaw: this.yaw, pitch: this.pitch, offset: this.offset };
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinch = Math.hypot(a[0] - b[0], a[1] - b[1]);
        drag = null;
      }
    });
    c.addEventListener("pointermove", (e) => {
      const p = local(e);
      if (!pointers.has(e.pointerId)) return this.hover(...p);
      pointers.set(e.pointerId, p);
      if (pinch && pointers.size === 2) {
        const [a, b] = [...pointers.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        this.zoom(d / pinch, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
        pinch = d;
      } else if (drag) {
        const dx = p[0] - drag.from[0], dy = p[1] - drag.from[1];
        if (drag.mode === "move") this.offset = [drag.offset[0] + dx, drag.offset[1] + dy];
        else this.turn(drag.yaw, drag.pitch, dx * 0.01, dy * 0.01);
        this.dirty = true;
        this.tip.hidden = true;
      }
    });
    const up = (e) => {
      if (!pointers.delete(e.pointerId)) return; // pointerup is followed by lostpointercapture
      if (pointers.size < 2) pinch = null;
      if (pointers.size) return;
      drag = null;
      this.held = false;
    };
    c.addEventListener("pointerup", up);
    c.addEventListener("pointercancel", up);
    c.addEventListener("lostpointercapture", up); // whatever else ends the press
    c.addEventListener("pointerleave", () => {
      this.tip.hidden = true;
      this.armed = false;
    });
    c.addEventListener("contextmenu", (e) => e.preventDefault()); // the right button turns
    c.addEventListener("dblclick", () => this.setView(this.viewName));
    c.addEventListener("wheel", (e) => {
      // a page scroll passing over the brain goes on; the wheel zooms with Ctrl / ⌘ (also a
      // trackpad pinch), or after a click on the brain until the pointer leaves it
      if (!e.ctrlKey && !e.metaKey && !this.armed) return;
      e.preventDefault();
      this.zoom(Math.exp(-e.deltaY * 0.0015), local(e));
    }, { passive: false });
    c.addEventListener("keydown", (e) => {
      const turn = { ArrowLeft: [-0.15, 0], ArrowRight: [0.15, 0], ArrowUp: [0, -0.15], ArrowDown: [0, 0.15] }[e.key];
      if (turn) {
        this.nudge();
        this.turn(this.yaw, this.pitch, ...turn);
        this.dirty = true;
      } else if (e.key === "+" || e.key === "=") this.zoom(1.25);
      else if (e.key === "-") this.zoom(0.8);
      else return;
      e.preventDefault();
    });
  }
}
