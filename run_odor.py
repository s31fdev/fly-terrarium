"""A fly that walks toward an odor source.

The fly smells with its two antennae: the odor intensity at each antenna
depends on its distance to the source. A tiny "brain" compares left and right,
turns toward the stronger side and stops once the smell is strong enough (the
fly has reached the food). The brain only sees the two antenna readings and
outputs the same [left, right] descending signal as run_route.py.

Usage: python run_odor.py [x y]
    x y: odor source position in mm (default -10 25). The fly starts at 0 0
    facing +x. Results: output/odor.mp4, output/odor.png, output/odor.log.
"""

import logging
import sys
import time

import matplotlib.pyplot as plt  # run_route sets the headless backend
import mujoco
import numpy as np
from flygym.anatomy import BodySegment
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action

from run_route import OUTPUT_DIR, PLAYBACK_SPEED, VIDEO_FPS, VIDEO_RES, descending_signal, make_simulation

MAX_TIME_S = 15.0  # give up after this long
DEFAULT_SOURCE = (-10.0, 25.0)  # mm, behind and to the left of the fly
TURN_GAIN = 50  # antennae are only 0.22 mm apart, so the contrast is tiny
MAX_TURN = 0.5  # sharpest turn the brain asks for (0.5: inner legs stand still)
ARRIVED_INTENSITY = 0.25  # smell this strong means "at the food" (~1.7 mm away)

log = logging.getLogger("run_odor")


def odor_intensity(point, source):
    """Odor spreading from a point source: 1 at the source, ~1/distance^2 further away."""
    return 1.0 / (1.0 + np.sum((point[:2] - source) ** 2))


def brain(left, right):
    """Decide what to do from the two antenna readings alone."""
    if (left + right) / 2 > ARRIVED_INTENSITY:
        return "stop", 0.0
    contrast = (left - right) / (left + right)  # > 0: the odor comes from the left
    strength = min(abs(contrast) * TURN_GAIN, MAX_TURN)
    return ("left" if contrast > 0 else "right"), strength


def main():
    source = np.array([float(v) for v in sys.argv[1:3]] if len(sys.argv) > 2 else DEFAULT_SOURCE)
    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / "odor.log", mode="w")],
    )

    # Arena: a green disc marks the odor source (the fly can walk over it), and
    # a camera looks straight down from high enough to see the start and the source.
    world = FlatGroundWorld()
    world.mjcf_root.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[1.0, 0.02, 0],  # radius, half-height in mm
        pos=[source[0], source[1], 0.02],
        rgba=[0.2, 0.8, 0.2, 1],
        contype=0,
        conaffinity=0,
    )
    half_fov = np.tan(np.radians(45 / 2))  # vertical; horizontal is 4/3 of it
    height = max((abs(source[1]) / 2 + 5) / half_fov, (abs(source[0]) / 2 + 5) / (half_fov * 4 / 3))
    camera = world.mjcf_root.worldbody.add_camera(
        name="overview", pos=[source[0] / 2, source[1] / 2, height], fovy=45
    )
    fly, _, sim, controller = make_simulation(world=world)
    sim.set_renderer(camera, camera_res=VIDEO_RES, playback_speed=PLAYBACK_SPEED, output_fps=VIDEO_FPS)

    order = fly.get_bodysegs_order()
    # The funiculus (3rd antennal segment) carries the olfactory receptors.
    left_idx = order.index(BodySegment("l_funiculus"))
    right_idx = order.index(BodySegment("r_funiculus"))
    log.info(f"Odor source at {source} mm, fly starts at [0, 0] facing +x")

    n_steps = int(round(MAX_TIME_S / sim.timestep))
    log_every = int(round(0.5 / sim.timestep))
    sample_every = int(round(0.01 / sim.timestep))
    path, arrived_at = [], None
    wall_start = time.perf_counter()
    for i in range(n_steps):
        pos = sim.get_body_positions(fly.name)
        nose = (pos[left_idx] + pos[right_idx]) / 2
        left, right = odor_intensity(pos[left_idx], source), odor_intensity(pos[right_idx], source)
        action, strength = brain(left, right)
        if action == "stop" and arrived_at is None:
            arrived_at = i * sim.timestep
        if arrived_at is not None and i * sim.timestep > arrived_at + 1.0:
            break  # show the fly standing at the source for a second, then finish

        obs = HybridControllerObservation.from_sim(sim, fly.name)
        signal = descending_signal(action, strength)
        apply_locomotion_action(sim, fly.name, controller.step(signal, obs))
        sim.step()
        sim.render_as_needed()

        if i % sample_every == 0:
            path.append(nose[:2].copy())
        if i % log_every == 0:
            log.info(
                f"t={i * sim.timestep:4.1f} s  distance {np.linalg.norm(nose[:2] - source):5.1f} mm  "
                f"smell L={left:.5f} R={right:.5f}  -> {action:<5} {strength:.2f}  "
                f"(wall {time.perf_counter() - wall_start:.0f} s)"
            )

    final_distance = np.linalg.norm(path[-1] - source)
    if arrived_at is None:
        log.info(f"Did not reach the source; {final_distance:.1f} mm left")
    else:
        log.info(f"Reached the source at t={arrived_at:.1f} s; now {final_distance:.1f} mm from it")

    sim.renderer.save_video(OUTPUT_DIR / "odor.mp4")

    # Top view: the odor field (log scale), the fly's path, start and source.
    path = np.array(path)
    xs = np.linspace(min(path[:, 0].min(), source[0]) - 5, max(path[:, 0].max(), source[0]) + 5, 200)
    ys = np.linspace(min(path[:, 1].min(), source[1]) - 5, max(path[:, 1].max(), source[1]) + 5, 200)
    gx, gy = np.meshgrid(xs, ys)
    field = 1.0 / (1.0 + (gx - source[0]) ** 2 + (gy - source[1]) ** 2)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(gx, gy, np.log10(field), levels=20, cmap="Greens")
    ax.plot(path[:, 0], path[:, 1], "k", lw=2, label="fly path (antennae)")
    ax.plot(*path[0], "ko", label="start")
    ax.plot(*source, "r*", ms=15, label="odor source")
    ax.add_patch(plt.Circle(source, np.sqrt(1 / ARRIVED_INTENSITY - 1), fill=False, ls="--", color="r"))
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Odor-guided walking (green = odor intensity, dashed = 'arrived')")
    ax.legend(loc="lower right")
    fig.savefig(OUTPUT_DIR / "odor.png", dpi=120, bbox_inches="tight")
    log.info(f"Saved {OUTPUT_DIR / 'odor.mp4'} and {OUTPUT_DIR / 'odor.png'}")


if __name__ == "__main__":
    main()
