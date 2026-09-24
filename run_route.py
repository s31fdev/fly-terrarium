"""Walk a simulated fruit fly (NeuroMechFly v2, FlyGym 2.x) along a route.

The route is a list of steps in route.yaml (or a file given as the first
argument). Each step sets the two-value descending signal [left, right] of the
hybrid turning controller. Results are written to ./output:
route.mp4, trajectory.csv, trajectory.png and run.log.
"""

import csv
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # plots go to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import yaml
from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)

OUTPUT_DIR = Path("output")
ACTIONS = ("forward", "left", "right", "stop")
SAMPLE_PERIOD_S = 0.01  # how often the trajectory is recorded
VIDEO_RES = (480, 640)  # (height, width) in pixels
VIDEO_FPS = 25
PLAYBACK_SPEED = 0.5  # video plays at half speed so the legs are easy to follow

log = logging.getLogger("run_route")


def load_route(path):
    """Read route steps from YAML and check them."""
    steps = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(steps, list) or not steps:
        sys.exit(f"{path}: expected a non-empty list of steps")
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict) or step.get("action") not in ACTIONS:
            sys.exit(f"{path}, step {i}: action must be one of {ACTIONS}")
        step.setdefault("strength", 1.0)
        duration, strength = step.get("duration_s"), step["strength"]
        if not isinstance(duration, (int, float)) or duration <= 0:
            sys.exit(f"{path}, step {i}: duration_s must be a positive number")
        if not isinstance(strength, (int, float)) or not 0 <= strength <= 1:
            sys.exit(f"{path}, step {i}: strength must be between 0 and 1")
    return steps


def descending_signal(action, strength):
    """Translate one route step into the controller's [left, right] signal.

    Each value is the stepping amplitude of the legs on that side of the body;
    a negative value makes those legs step backwards.
    """
    if action == "forward":
        return np.array([strength, strength])
    if action == "stop":
        return np.zeros(2)
    # Turning: the outer side keeps full amplitude, the inner side slows down
    # (strength 0.5: inner legs stand still, 1.0: they step backwards).
    inner = 1.0 - 2.0 * strength
    if action == "left":
        return np.array([inner, 1.0])
    return np.array([1.0, inner])  # right


def make_simulation(timestep=None, control_every=1, world=None, vision=False):
    """Build the fly on flat ground and its walking controller.

    Returns (fly, camera, sim, controller) with the fly already standing on the
    ground. `timestep` is the physics step (None = FlyGym default, 0.1 ms);
    the controller must be called once every `control_every` physics steps.
    `world` lets the caller pass a FlatGroundWorld with extra objects in it;
    `vision=True` gives the fly its compound eyes.
    """
    # Fly with a camera that follows it and looks straight down.
    # "track" mode keeps the camera orientation fixed in the world frame,
    # so the video shows the fly turning, not the arena rotating.
    fly = make_locomotion_fly(name="fly", colorize=True)
    if vision:
        fly.add_vision()
    camera = fly.add_tracking_camera(
        name="top_cam",
        mode="track",
        pos_offset=(0, 0, 12),  # mm above the thorax
        rotation=Rotation3D("xyaxes", (1, 0, 0, 0, 1, 0)),  # image right = +x, up = +y
        fovy=40,
    )
    if world is None:
        world = FlatGroundWorld()
    world.add_fly(
        fly,
        [0, 0, 0.8],
        Rotation3D("quat", [1, 0, 0, 0]),  # fly faces +x
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=False,
    )
    sim = Simulation(world, timestep=timestep)

    controller = HybridTurningController(
        timestep=sim.timestep * control_every,
        preprogrammed_steps=PreprogrammedSteps(),
        output_dof_order=fly.get_actuated_jointdofs_order("position"),
    )
    stand_up(fly, sim, controller)
    return fly, camera, sim, controller


def stand_up(fly, sim, controller):
    """Reset the fly to the standing pose and let it settle on the ground."""
    sim.reset()
    controller.reset(seed=0)
    standing = LocomotionAction(
        joint_angles=controller.preprogrammed_steps.default_pose_by_dof_order(
            controller.output_dof_order
        ),
        adhesion_onoff=np.ones(6, dtype=bool),
    )
    apply_locomotion_action(sim, fly.name, standing)
    sim.warmup()


def main():
    route_file = sys.argv[1] if len(sys.argv) > 1 else "route.yaml"
    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "run.log", mode="w"),
        ],
    )
    route = load_route(route_file)
    step_ends = np.cumsum([step["duration_s"] for step in route])
    log.info(f"Route {route_file}: {len(route)} steps, {step_ends[-1]:.1f} s")

    fly, camera, sim, controller = make_simulation()
    sim.set_renderer(
        camera,
        camera_res=VIDEO_RES,
        playback_speed=PLAYBACK_SPEED,
        output_fps=VIDEO_FPS,
    )

    thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
    n_sim_steps = int(round(step_ends[-1] / sim.timestep))
    sample_every = int(round(SAMPLE_PERIOD_S / sim.timestep))
    log_every = int(round(1.0 / sim.timestep))
    rows = []  # (time_s, x_mm, y_mm, heading_deg, step_index)
    step_idx = 0
    wall_start = time.perf_counter()

    for i in range(n_sim_steps):
        t = i * sim.timestep
        while t >= step_ends[step_idx]:
            step_idx += 1
        step = route[step_idx]
        signal = descending_signal(step["action"], step["strength"])

        obs = HybridControllerObservation.from_sim(sim, fly.name)
        if i % sample_every == 0:
            x, y, _ = sim.get_body_positions(fly.name)[thorax_idx]
            # Heading = direction of the thorax x-axis, 0 deg = +x, CCW positive.
            heading = np.degrees(np.arctan2(obs.fly_heading[1], obs.fly_heading[0]))
            rows.append((t, x, y, heading, step_idx))
        if i % log_every == 0:
            log.info(
                f"t={t:5.1f} s  step {step_idx + 1} {step['action']:<7} "
                f"signal={np.round(signal, 2)}  wall {time.perf_counter() - wall_start:.0f} s"
            )

        apply_locomotion_action(sim, fly.name, controller.step(signal, obs))
        sim.step()
        sim.render_as_needed()

    log.info(f"Simulation done in {time.perf_counter() - wall_start:.0f} s wall time")

    video_path = OUTPUT_DIR / "route.mp4"
    sim.renderer.save_video(video_path)
    log.info(f"Saved {video_path} ({len(sim.renderer.frames[camera.name])} frames)")

    data = np.array(rows)
    with open(OUTPUT_DIR / "trajectory.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "x_mm", "y_mm", "heading_deg", "step", "action"])
        for t, x, y, heading, k in rows:
            action = route[k]["action"]
            writer.writerow([f"{t:.3f}", f"{x:.4f}", f"{y:.4f}", f"{heading:.2f}", k + 1, action])

    # Per-step summary: how far the fly went and how much it turned.
    # Positive heading change = counter-clockwise seen from above = turn left.
    # The heading wobbles with every stride, so each step's final heading is
    # averaged over its last 0.1 s (about one stride).
    heading = np.degrees(np.unwrap(np.radians(data[:, 3])))
    window = int(round(0.1 / SAMPLE_PERIOD_S))
    prev_heading, prev_pos = heading[0], data[0, 1:3]
    log.info("Step summary:")
    for k, step in enumerate(route):
        idx = np.flatnonzero(data[:, 4] == k)
        end_heading, end_pos = heading[idx[-window:]].mean(), data[idx[-1], 1:3]
        turn = end_heading - prev_heading
        dist = np.hypot(*(end_pos - prev_pos))
        prev_heading, prev_pos = end_heading, end_pos
        log.info(
            f"  {k + 1}. {step['action']:<7} strength={step['strength']:.1f}  "
            f"moved {dist:5.2f} mm  turned {turn:+7.1f} deg "
            f"({'left' if turn > 0 else 'right'})"
        )

    fig, ax = plt.subplots(figsize=(7, 7))
    for k, step in enumerate(route):
        idx = np.flatnonzero(data[:, 4] == k)
        idx = np.append(idx, min(idx[-1] + 1, len(data) - 1))  # connect to next step
        ax.plot(data[idx, 1], data[idx, 2], lw=2, label=f"{k + 1}. {step['action']}")
    arrows = data[:: int(round(1.0 / SAMPLE_PERIOD_S))]  # heading arrow every second
    ax.quiver(
        arrows[:, 1], arrows[:, 2],
        np.cos(np.radians(arrows[:, 3])), np.sin(np.radians(arrows[:, 3])),
        angles="xy", scale=25, width=0.004, color="0.3",
    )
    ax.plot(*data[0, 1:3], "ko", label="start")
    ax.plot(*data[-1, 1:3], "k*", ms=12, label="end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Fly trajectory (top view, arrows = heading every 1 s)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(OUTPUT_DIR / "trajectory.png", dpi=120, bbox_inches="tight")
    log.info(f"Saved {OUTPUT_DIR / 'trajectory.csv'} and {OUTPUT_DIR / 'trajectory.png'}")


if __name__ == "__main__":
    main()
