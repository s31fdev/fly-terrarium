"""A fly whose escape is decided by a whole-brain model built from the connectome.

A dark pillar flies at the standing fly. The eyes see it grow; how fast it
grows drives the brain's loom-detecting neurons (LC4, LPLC2, LC16) of that eye.
The brain (brain.py: every neuron of a connectome, Shiu et al. 2024) then
answers through its descending neurons, which are read out as the walking
signal:
    giant fibre (DNp01)       -> run (how vigorously)
    DNa01 + DNa02, right-left -> which way to turn
    moonwalker neurons (MDN)  -> walk backwards
Which of these fire, and on which side, comes from the wiring alone. The
sensory encoding and the conversion of spike rates to the walking signal are
hand-chosen (constants below). The male CNS also has the ventral nerve cord,
so there the jump motor neuron (TTMn) is logged too; the body cannot jump.

Usage: python run_escape.py [bearing] [--shuffled] [--mcns]
    bearing: where the pillar comes from, degrees from straight ahead,
    positive = left (default 60). --shuffled: control with the same neurons
    and synapses but random wiring. --mcns: the male CNS connectome instead
    of FlyWire. Results in output/: escape_<bearing>.mp4 (arena, what the fly
    sees, what its brain does), escape_<bearing>.png and .log (with _mcns and
    _shuffled appended).
"""

import logging
import sys
import time

import matplotlib.pyplot as plt  # run_route sets the headless backend
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action

from brain import Brain
from run_route import OUTPUT_DIR, PLAYBACK_SPEED, VIDEO_FPS, VIDEO_RES, make_simulation
from run_vision import VISION_HZ, add_pillar, darkness_above_horizon, fly_view, write_video

# Senses: loom -> input rate of the loom-detecting neurons of each eye
LOOM_NEURONS = ("LC4", "LPLC2", "LC16")  # all three respond to looming in real flies
LOOM_WINDOW = 2  # growth is measured over 2 eye readings (0.1 s): one stride, so the head
# bobbing of a walking fly (the object shifts across the horizon) cancels out
LOOM_MIN = 50.0  # ommatidia/s: slower growth of the dark area is not sensed. Like LC4,
# this responds to how fast the object expands, which peaks just before collision.
# The pillar thrown at 60 mm/s crosses it at ~17 mm.
LOOM_GAIN = 0.5  # Hz of input per ommatidium/s above LOOM_MIN
MAX_RATE = 150.0  # Hz, the paper's usual activation rate

# Motor side: descending neuron spike rates -> walking signal
GF_FULL = 50.0  # Hz (mean of both giant fibres) for full running speed
TURN_FULL = 40.0  # Hz of right-left difference of DNa01 + DNa02 for the sharpest turn
MDN_FULL = 40.0  # Hz (all four MDN) for full-speed walking backwards
RUN_HOLD_S = 0.6  # s; the fly keeps running for a while after the giant fibres fall silent,
TURN_HOLD_S = 0.2  # s; but turning and walking backwards stop soon after DNa / MDN do

# Scenario
START_DISTANCE = 30.0  # mm
STOP_DISTANCE = 6.0  # mm; the pillar is a ghost, it stops before reaching the fly
PILLAR_SPEED = 60.0  # mm/s
DURATION_S = 3.0

log = logging.getLogger("run_escape")


def hold(old, new, decay):
    """The larger (by magnitude) of the new value and the decaying old one."""
    return new if abs(new) >= abs(old) * decay else old * decay


class EscapeBrain:
    """Eyes -> loom-detecting neurons -> whole brain -> descending neurons -> [left, right] signal."""

    def __init__(self, version="783", shuffle_seed=None):
        self.brain = Brain(version, shuffle_seed=shuffle_seed)
        b = self.brain
        self.eyes = [np.concatenate([b.neurons(t, side) for t in LOOM_NEURONS]) for side in ("left", "right")]
        self.outputs = {
            "GF": [b.neurons("DNp01", s) for s in ("left", "right")],
            "DNa": [np.concatenate([b.neurons(t, s) for t in ("DNa01", "DNa02")]) for s in ("left", "right")],
            "MDN": [b.neurons("MDN", s) for s in ("left", "right")],
        }
        if len(b.neurons("TTMn", "left")):  # only the male CNS has the nerve cord
            self.outputs["TTMn"] = [b.neurons("TTMn", s) for s in ("left", "right")]
        self.past_areas = []  # dark area seen by each eye at the last LOOM_WINDOW readings
        self.run = self.turn = self.back = 0.0

    def step(self, readouts, dt):
        """What the eyes see now -> signal for the next dt seconds, plus what happened inside."""
        # How fast the dark area grows in each eye; it drives the loom neurons of that eye
        area = darkness_above_horizon(readouts).sum(axis=1)  # left, right eye
        past = self.past_areas[0] if self.past_areas else area
        growth = (area - past) / (dt * max(len(self.past_areas), 1))
        self.past_areas = (self.past_areas + [area])[-LOOM_WINDOW:]
        rate = np.clip(LOOM_GAIN * (growth - LOOM_MIN), 0, MAX_RATE)
        self.brain.set_input(np.concatenate(self.eyes), np.repeat(rate, [len(e) for e in self.eyes]))

        spikes = self.brain.run(dt)
        hz = {k: [spikes[n].sum() / dt for n in sides] for k, sides in self.outputs.items()}
        slow, fast = np.exp(-dt / RUN_HOLD_S), np.exp(-dt / TURN_HOLD_S)
        self.run = hold(self.run, min(np.mean(hz["GF"]) / GF_FULL, 1.0), slow)
        self.turn = hold(self.turn, np.clip((hz["DNa"][1] - hz["DNa"][0]) / TURN_FULL, -1, 1), fast)
        self.back = hold(self.back, min(sum(hz["MDN"]) / MDN_FULL, 1.0), fast)
        # positive turn = right: the left legs step harder
        ahead = 1 - 2 * self.back
        signal = self.run * np.clip([ahead + self.turn, ahead - self.turn], -1, 1)
        return signal, dict(input_hz=rate, spikes=spikes, **hz)


class BrainView:
    """A picture of what the brain is doing: every neuron is a dot where its cell body
    sits, seen from the front, the fly's left on the left. Neurons that spiked light up
    and fade within ~0.3 s: loom input blue, read-out neurons red (always shown as
    squares, dim while silent), all others yellow."""

    FADE_S = 0.3
    MARGIN = 6  # px
    COLORS = np.array([(1.0, 0.78, 0.25), (0.35, 0.65, 1.0), (1.0, 0.25, 0.25)], np.float32)
    LEGEND = ("spiked", "loom input", "read-out")  # one word per colour

    def __init__(self, escape, width=450):
        b = escape.brain
        self.known = ~np.isnan(b.xyz[:, 0])
        xy = b.xyz[:, :2].copy()  # x: side to side, y: top to bottom
        if np.nanmean(xy[b.side == "left", 0]) > np.nanmean(xy[b.side == "right", 0]):
            xy[:, 0] *= -1
        lo, hi = np.nanmin(xy, axis=0), np.nanmax(xy, axis=0)
        scale = (width - 2 * self.MARGIN) / (hi[0] - lo[0])
        height = -(-(int((hi[1] - lo[1]) * scale) + 2 * self.MARGIN + 14) // 16) * 16  # video wants 16 px steps
        self.px = np.zeros((b.n, 2), int)
        self.px[self.known] = ((xy[self.known] - lo) * scale).astype(int) + (self.MARGIN, self.MARGIN + 14)

        # Silhouette: how densely neurons sit at each pixel
        count = np.zeros((height, width), np.float32)
        np.add.at(count, (self.px[self.known, 1], self.px[self.known, 0]), 1)
        self.background = np.repeat((np.sqrt(count / count.max()) * 0.35)[..., None], 3, axis=2)

        role = np.zeros(b.n, int)
        role[np.concatenate(escape.eyes)] = 1
        self.read_out = [(name, n) for name, sides in escape.outputs.items() for n in np.concatenate(sides)]
        role[[n for _, n in self.read_out]] = 2
        self.color = self.COLORS[role]
        self.glow = np.zeros(b.n, np.float32)

        # Text drawn once: title, legend, and each read-out name next to its rightmost neuron
        text = Image.new("L", (width, height))
        draw = ImageDraw.Draw(text)
        draw.text((4, 1), f"{self.known.sum():,} neurons, front view", fill=255)
        self.chips, x = [], width - 4 - sum(draw.textlength(w) + 12 for w in self.LEGEND)
        for word in self.LEGEND:
            self.chips.append(int(x))
            draw.text((x + 8, 1), word, fill=255)
            x += draw.textlength(word) + 12
        for name, sides in escape.outputs.items():
            n = [i for i in np.concatenate(sides) if self.known[i]]
            if n:
                x, y = self.px[max(n, key=lambda i: self.px[i, 0])]
                draw.text((x + 5, y - 5), name, fill=255)
        self.text = np.asarray(text) > 0

    def render(self, spikes, dt):
        """RGB image (uint8) after dt seconds in which the neurons fired `spikes`."""
        self.glow = self.glow * np.float32(np.exp(-dt / self.FADE_S)) + spikes
        img = self.background.copy()
        on = np.flatnonzero((self.glow > 0.05) & self.known)
        np.maximum.at(img, (self.px[on, 1], self.px[on, 0]), np.minimum(1, 0.4 + 0.3 * self.glow[on, None]) * self.color[on])
        for _, n in self.read_out:
            if self.known[n]:
                x, y = self.px[n]
                img[y - 2:y + 3, x - 2:x + 3] = self.color[n] * (1.0 if self.glow[n] > 0.05 else 0.35)
        img[self.text] = 0.85
        for x, c in zip(self.chips, self.COLORS):
            img[4:9, x:x + 5] = c
        return (img * 255).astype(np.uint8)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    shuffled = "--shuffled" in sys.argv
    version = "mcns" if "--mcns" in sys.argv else "783"
    bearing = np.radians(float(args[0]) if args else 60.0)
    OUTPUT_DIR.mkdir(exist_ok=True)
    name = f"escape_{np.degrees(bearing):.0f}" + ("_mcns" if version == "mcns" else "") + ("_shuffled" if shuffled else "")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / f"{name}.log", mode="w")],
    )

    # The fly stands at the origin facing +x; the pillar comes from `bearing`.
    direction = np.array([np.cos(bearing), np.sin(bearing)])
    world = FlatGroundWorld()
    add_pillar(world, START_DISTANCE * direction)
    camera = world.mjcf_root.worldbody.add_camera(name="overview", pos=[0, 0, 50], fovy=45)
    fly, _, sim, controller = make_simulation(world=world, vision=True)
    sim.set_renderer(camera, camera_res=VIDEO_RES, playback_speed=PLAYBACK_SPEED, output_fps=VIDEO_FPS)
    pillar = sim.mj_data.mocap_pos[sim.mj_model.body("pillar").mocapid[0]]
    connectome = "male CNS" if version == "mcns" else "FlyWire"
    log.info(f"Loading the brain ({connectome})" + (" with shuffled wiring (control)" if shuffled else ""))
    escape = EscapeBrain(version, shuffle_seed=7 if shuffled else None)
    brain_view = BrainView(escape)
    jump = "TTMn" in escape.outputs
    log.info(f"Pillar comes from {np.degrees(bearing):+.0f} deg (positive = left) at {PILLAR_SPEED:.0f} mm/s")
    log.info("   t     dist  input L/R Hz    GF L/R Hz   DNa L/R Hz  MDN Hz   signal L/R" + ("  TTMn L/R Hz" if jump else ""))

    thorax_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{fly.name}/c_thorax")
    vision_every = round(1 / (VISION_HZ * sim.timestep))
    dt = vision_every * sim.timestep
    signal, views, brain_views, rows = np.zeros(2), [], [], []
    wall_start = time.perf_counter()
    for i in range(round(DURATION_S / sim.timestep)):
        t = i * sim.timestep
        pillar[:2] = max(START_DISTANCE - PILLAR_SPEED * t, STOP_DISTANCE) * direction
        if i % vision_every == 0:
            readouts = sim.get_ommatidia_readouts(fly.name)
            signal, info = escape.step(readouts, dt)
            view = fly_view(readouts)
            brain_img = brain_view.render(info["spikes"], dt)
            fly_pos = sim.mj_data.xpos[thorax_id][:2].copy()
            obs = HybridControllerObservation.from_sim(sim, fly.name)
            heading = np.degrees(np.arctan2(obs.fly_heading[1], obs.fly_heading[0]))
            dist = np.linalg.norm(pillar[:2] - fly_pos)
            rows.append(dict(
                t=t, fly_x=fly_pos[0], fly_y=fly_pos[1], heading=heading, pillar_x=pillar[0], pillar_y=pillar[1],
                input_l=info["input_hz"][0], input_r=info["input_hz"][1], gf=np.mean(info["GF"]),
                dna=info["DNa"][1] - info["DNa"][0], mdn=sum(info["MDN"]), ttmn=np.mean(info.get("TTMn", 0)),
            ))
            log.info(
                f"{t:4.2f} s {dist:5.1f} mm  {info['input_hz'][0]:5.0f} {info['input_hz'][1]:5.0f}  "
                f"{info['GF'][0]:5.0f} {info['GF'][1]:5.0f}  {info['DNa'][0]:5.0f} {info['DNa'][1]:5.0f}  "
                f"{sum(info['MDN']):5.0f}   {signal[0]:+.2f} {signal[1]:+.2f}"
                + (f"  {info['TTMn'][0]:5.0f} {info['TTMn'][1]:5.0f}" if jump else "")
                + f"  (wall {time.perf_counter() - wall_start:.0f} s)"
            )
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        apply_locomotion_action(sim, fly.name, controller.step(signal, obs))
        sim.step()
        if sim.render_as_needed():
            views.append(view)
            brain_views.append(brain_img)

    d = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    turned = np.degrees(np.unwrap(np.radians(d["heading"])))[-1] - d["heading"][0]
    start, end = np.array([d["fly_x"][0], d["fly_y"][0]]), np.array([d["fly_x"][-1], d["fly_y"][-1]])
    away = np.linalg.norm(end - STOP_DISTANCE * direction) - np.linalg.norm(start - STOP_DISTANCE * direction)
    log.info(
        f"The fly moved {np.linalg.norm(end - start):.1f} mm, {away:+.1f} mm away from where the pillar "
        f"stopped, and turned {turned:+.0f} deg (positive = left)"
    )
    write_video(OUTPUT_DIR / f"{name}.mp4", sim.renderer.frames[camera.name], views, brain_views)
    sim.close()  # release the OpenGL renderers now, not in whatever order Python exits

    # Left: paths from above. Right: what went into and came out of the brain.
    fig, (top, act) = plt.subplots(1, 2, figsize=(13, 5.5))
    top.plot(d["fly_x"], d["fly_y"], "k", lw=2, label="fly (thorax)")
    top.plot(d["pillar_x"], d["pillar_y"], color="0.6", lw=6, alpha=0.5, label="pillar")
    top.plot(*start, "ko", label="fly start")
    top.set_aspect("equal", adjustable="datalim")
    top.set_xlabel("x (mm)")
    top.set_ylabel("y (mm)")
    top.grid(alpha=0.3)
    top.legend()
    top.set_title("Paths from above")
    act.plot(d["t"], d["input_l"], "C0--", label="input, left eye")
    act.plot(d["t"], d["input_r"], "C1--", label="input, right eye")
    act.plot(d["t"], d["gf"], "C3", lw=2, label="giant fibre (mean)")
    act.plot(d["t"], d["dna"], "C2", lw=2, label="DNa01+02 right - left")
    act.plot(d["t"], d["mdn"], "C4", lw=2, label="MDN (all)")
    if jump:
        act.plot(d["t"], d["ttmn"], "k:", lw=2, label="jump motor neuron TTMn (mean)")
    act.set_xlabel("time (s)")
    act.set_ylabel("spike rate (Hz)")
    act.grid(alpha=0.3)
    act.legend()
    act.set_title("Brain: input and descending neurons")
    fig.suptitle(f"Loom from {np.degrees(bearing):+.0f} deg, {connectome}" + (", shuffled wiring" if shuffled else ""))
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=110, bbox_inches="tight")
    log.info(f"Saved {OUTPUT_DIR / (name + '.mp4')} and {OUTPUT_DIR / (name + '.png')}")


if __name__ == "__main__":
    main()
