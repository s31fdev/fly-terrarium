"""A fly that walks toward a dark pillar it sees with its compound eyes.

Each eye has 721 ommatidia: FlyGym renders what each eye camera sees and
averages it per ommatidium. Above the horizon the background is bright sky,
so dark ommatidia there can only be an object. A tiny "brain" turns toward the
eye that sees the dark spot (the more, the farther the spot is from straight
ahead), turns on the spot to look around when nothing is in sight, and stops
when the pillar fills a large part of the view. Like run_route.py it only
outputs the [left, right] descending signal.

Usage: python run_vision.py [x y]
    x y: pillar position in mm (default -10 25). The fly starts at 0 0 facing
    +x. Results in output/: vision.mp4 (top: arena, bottom: what the fly sees),
    vision.png and vision.log.
"""

import logging
import sys
import time

import imageio.v3 as iio
import matplotlib.pyplot as plt  # run_route sets the headless backend
import mujoco
import numpy as np
from flygym.compose import FlatGroundWorld
from flygym.vision.retina import Retina
from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action

from run_route import OUTPUT_DIR, PLAYBACK_SPEED, VIDEO_FPS, VIDEO_RES, descending_signal, make_simulation

MAX_TIME_S = 15.0  # give up after this long
DEFAULT_PILLAR = (-10.0, 25.0)  # mm, behind and to the left of the fly
PILLAR_RADIUS, PILLAR_HEIGHT = 1.5, 10.0  # mm
VISION_HZ = 20  # how often the eyes are read; rendering them is the costly part
HORIZON_ROW = 220  # ommatidia centred above this eye-image row see sky, not floor
SEEN = 0.3  # less total darkness than this: nothing in sight
ARRIVED = 100  # this much darkness: the pillar fills the view (~5 mm away)
TURN_GAIN = 1.0
MAX_TURN = 0.5  # sharpest turn when steering (0.5: inner legs stand still)
SEARCH_TURN = 0.5  # turn on the spot to look around when nothing is seen

log = logging.getLogger("run_vision")


# Which ommatidium each pixel of an eye image belongs to (0 = outside the eye).
# The layout is the same for both eyes.
ID_MAP = Retina().ommatidia_id_map.astype(int)


def eye_layout():
    """Where each ommatidium sits in the eye image: centre row and column (0..1).

    The fly's brain is wired the same way: each ommatidium always looks in the
    same direction.
    """
    ids = ID_MAP.ravel()
    rows, cols = np.indices(ID_MAP.shape)
    n_pixels = np.bincount(ids)[1:]
    row = np.bincount(ids, weights=rows.ravel())[1:] / n_pixels
    col = np.bincount(ids, weights=cols.ravel())[1:] / n_pixels / ID_MAP.shape[1]
    return row, col


ROW, COL = eye_layout()
ABOVE_HORIZON = ROW < HORIZON_ROW
# How far from straight ahead each ommatidium looks: 0 = the eye's front edge,
# 1 = its rear edge. The left eye's front is on the right of its image, the
# right eye's front on the left.
OFF_AHEAD = np.stack([1 - COL, COL])


def darkness_above_horizon(readouts):
    """Darkness (0 = sky, 1 = black) of each ommatidium above the horizon, shape (2, 721)."""
    brightness = readouts.max(axis=2)  # each ommatidium uses one of its two channels
    return np.clip(1 - brightness, 0, 1) * ABOVE_HORIZON


def brain(darkness):
    """Decide what to do from what the two eyes see."""
    size = darkness.sum()
    if size > ARRIVED:
        return "stop", 0.0
    if size < SEEN:
        return "left", SEARCH_TURN  # nothing in sight: turn and look around
    # > 0: the dark spot is on the left; larger when it is far from straight ahead
    steer = ((darkness[0] * OFF_AHEAD[0]).sum() - (darkness[1] * OFF_AHEAD[1]).sum()) / size
    strength = min(abs(steer) * TURN_GAIN, MAX_TURN)
    return ("left" if steer > 0 else "right"), strength


def add_pillar(world, pos):
    """Add a dark pillar. It is a mocap body, so it can be moved while the
    simulation runs (run_live.py vision mode). The fly does not collide with it."""
    body = world.mjcf_root.worldbody.add_body(
        name="pillar", mocap=True, pos=[pos[0], pos[1], PILLAR_HEIGHT / 2]
    )
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[PILLAR_RADIUS, PILLAR_HEIGHT / 2, 0],
        rgba=[0.1, 0.1, 0.1, 1],
        contype=0,
        conaffinity=0,
    )


def fly_view(readouts, step=2):
    """What the fly sees: each ommatidium painted as a hexagon of its brightness,
    left | right eye side by side, grayscale uint8. `step=2` gives half size.

    Same picture as FlyGym's retina.hex_pxls_to_human_readable, but one numpy
    lookup instead of a slow per-pixel loop.
    """
    id_map = ID_MAP[::step, ::step]
    eyes = [np.concatenate([[0.0], b])[id_map] for b in readouts.max(axis=2)]
    return (np.concatenate(eyes, axis=1) * 255).astype(np.uint8)


def write_video(path, arena_frames, *panels):
    """Video: the arena on top, below it each panel (gray or RGB images, one per
    frame), centred. The first panel is what the fly sees (left eye | right eye)."""
    frames = []
    for arena, *views in zip(arena_frames, *panels):
        rows = [arena]
        for view in views:
            if view.ndim == 2:
                view = np.repeat(view[:, :, None], 3, axis=2)  # gray -> RGB
            pad = (arena.shape[1] - view.shape[1]) // 2
            rows.append(np.pad(view, ((0, 0), (pad, arena.shape[1] - view.shape[1] - pad), (0, 0))))
        frames.append(np.concatenate(rows, axis=0))
    iio.imwrite(path, frames, fps=VIDEO_FPS, codec="libx264", quality=8)


def main():
    pillar = np.array([float(v) for v in sys.argv[1:3]] if len(sys.argv) > 2 else DEFAULT_PILLAR)
    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / "vision.log", mode="w")],
    )

    # Arena: the pillar and a camera looking straight down from high enough to
    # see both the fly's start and the pillar.
    world = FlatGroundWorld()
    add_pillar(world, pillar)
    half_fov = np.tan(np.radians(45 / 2))  # vertical; horizontal is 4/3 of it
    margin = 10  # mm; the tall pillar leans outward in perspective
    height = max((abs(pillar[1]) / 2 + margin) / half_fov, (abs(pillar[0]) / 2 + margin) / (half_fov * 4 / 3))
    camera = world.mjcf_root.worldbody.add_camera(
        name="overview", pos=[pillar[0] / 2, pillar[1] / 2, height], fovy=45
    )
    fly, _, sim, controller = make_simulation(world=world, vision=True)
    sim.set_renderer(camera, camera_res=VIDEO_RES, playback_speed=PLAYBACK_SPEED, output_fps=VIDEO_FPS)
    log.info(f"Pillar at {pillar} mm, fly starts at [0, 0] facing +x")

    thorax_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{fly.name}/c_thorax")
    n_steps = int(round(MAX_TIME_S / sim.timestep))
    vision_every = int(round(1 / (VISION_HZ * sim.timestep)))
    log_every = int(round(0.5 / sim.timestep))
    sample_every = int(round(0.01 / sim.timestep))
    path, views, arrived_at = [], [], None
    wall_start = time.perf_counter()
    for i in range(n_steps):
        if i % vision_every == 0:
            readouts = sim.get_ommatidia_readouts(fly.name)
            darkness = darkness_above_horizon(readouts)
            action, strength = brain(darkness)
            view = fly_view(readouts)
        if action == "stop" and arrived_at is None:
            arrived_at = i * sim.timestep
        if arrived_at is not None and i * sim.timestep > arrived_at + 1.0:
            break  # show the fly standing at the pillar for a second, then finish

        obs = HybridControllerObservation.from_sim(sim, fly.name)
        apply_locomotion_action(sim, fly.name, controller.step(descending_signal(action, strength), obs))
        sim.step()
        if sim.render_as_needed():
            views.append(view)

        position = sim.mj_data.xpos[thorax_id][:2]
        if i % sample_every == 0:
            path.append(position.copy())
        if i % log_every == 0:
            log.info(
                f"t={i * sim.timestep:4.1f} s  distance {np.linalg.norm(position - pillar):5.1f} mm  "
                f"sees L={darkness[0].sum():6.2f} R={darkness[1].sum():6.2f}  -> {action:<5} {strength:.2f}  "
                f"(wall {time.perf_counter() - wall_start:.0f} s)"
            )

    final_distance = np.linalg.norm(path[-1] - pillar)
    if arrived_at is None:
        log.info(f"Did not reach the pillar; {final_distance:.1f} mm from its centre")
    else:
        log.info(f"Reached the pillar at t={arrived_at:.1f} s; now {final_distance:.1f} mm from its centre")

    write_video(OUTPUT_DIR / "vision.mp4", sim.renderer.frames[camera.name], views)
    sim.close()  # release the OpenGL renderers now, not in whatever order Python exits

    # Top view: the fly's path, start and pillar.
    path = np.array(path)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(path[:, 0], path[:, 1], "k", lw=2, label="fly path (thorax)")
    ax.plot(*path[0], "ko", label="start")
    ax.add_patch(plt.Circle(pillar, PILLAR_RADIUS, color="0.2", label="pillar"))
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Vision-guided walking toward a dark pillar")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.savefig(OUTPUT_DIR / "vision.png", dpi=120, bbox_inches="tight")
    log.info(f"Saved {OUTPUT_DIR / 'vision.mp4'} and {OUTPUT_DIR / 'vision.png'}")


if __name__ == "__main__":
    main()
