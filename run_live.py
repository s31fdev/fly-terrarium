"""Live fly in a MuJoCo window.

Runs natively on Windows (the window is drawn by the GPU), not in Docker.

    uv run run_live.py        you steer the fly with the arrow keys:
                              Up = walk, Down = stop, Left / Right = turn
                              (press again for a sharper turn)
    uv run run_live.py odor   the fly follows a smell (green disc) on its own;
                              arrow keys move the smell, relative to the view:
                              Up = away from you, Left = to the left, etc.

Mouse:  left drag = rotate view, right drag = move view, wheel = zoom,
        double-click the fly, then Ctrl + right drag = push it.
Close the window to quit.
"""

import functools
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import flygym_demo.complex_terrain.hybrid_controller as hybrid_controller
from flygym.anatomy import BodySegment
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action

from run_odor import add_odor_source, brain, odor_intensity
from run_route import descending_signal, make_simulation

# Speed settings. run_route.py uses 0.1 ms physics steps and runs the
# controller every step (10 kHz), ~16x slower than real time. These values
# give ~0.93x real time on an i7-9700F (walking) with nearly the same gait (see README).
# On a slower CPU the fly moves in slow motion.
PHYSICS_TIMESTEP = 2.5e-4  # s; 5e-4 is already unstable (the fly bounces)
CONTROL_EVERY = 8  # controller update every 8 physics steps = 500 Hz
FRAME_TIME = 1 / 60  # simulated seconds between screen updates
NOSLIP_ITERATIONS = 2  # FlyGym default is 5; 2 gives the same gait ~8% faster

ODOR_START = (15.0, 10.0)  # mm, ahead and to the left of the fly
ODOR_STEP = 3.0  # mm the smell moves per arrow key press

# The controller rebuilds 42 joint descriptors on every call. Caching them gives
# identical results about 15% faster.
hybrid_controller.dof_spec_to_jointdof = functools.lru_cache(maxsize=None)(
    hybrid_controller.dof_spec_to_jointdof
)

KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265  # GLFW key codes
ARROWS = {KEY_UP: "forward", KEY_DOWN: "stop", KEY_LEFT: "left", KEY_RIGHT: "right"}

odor_mode = sys.argv[1:] == ["odor"]
# Steering mode: current (action, strength). Replaced as a whole tuple, so the
# viewer thread (keyboard) and the main loop never see a half-updated command.
command = ("stop", 0.0)
# Odor mode: arrow presses waiting to move the smell (list.append is thread-safe).
odor_moves = []


def on_key(key):
    """Keyboard handler, called by the viewer thread on key press."""
    global command
    action = ARROWS.get(key)
    if action is None:
        return
    if odor_mode:
        odor_moves.append(key)
        return
    if action in ("left", "right"):
        # First press: smooth turn. Same arrow again: sharper, up to 1.0.
        strength = min(command[1] + 0.2, 1.0) if command[0] == action else 0.3
    else:
        strength = 1.0
    command = (action, strength)


def move_odor(key, source, azimuth_deg):
    """Shift the smell by one arrow press, relative to where the camera looks."""
    az = np.radians(azimuth_deg)
    away = np.array([np.cos(az), np.sin(az)])  # camera view direction on the ground
    left = np.array([-away[1], away[0]])
    shift = {KEY_UP: away, KEY_DOWN: -away, KEY_LEFT: left, KEY_RIGHT: -left}[key]
    source[:2] += ODOR_STEP * shift


def main():
    print(__doc__)
    world = FlatGroundWorld()
    if odor_mode:
        add_odor_source(world, ODOR_START)
    fly, _, sim, controller = make_simulation(PHYSICS_TIMESTEP, CONTROL_EVERY, world)
    sim.mj_model.opt.noslip_iterations = NOSLIP_ITERATIONS
    thorax_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{fly.name}/c_thorax")
    steps_per_frame = round(FRAME_TIME / sim.timestep)
    if odor_mode:
        order = fly.get_bodysegs_order()
        left_idx = order.index(BodySegment("l_funiculus"))
        right_idx = order.index(BodySegment("r_funiculus"))
        # Writing into this row of mocap_pos moves the green disc.
        source = sim.mj_data.mocap_pos[sim.mj_model.body("odor_source").mocapid[0]]

    with mujoco.viewer.launch_passive(
        sim.mj_model, sim.mj_data, key_callback=on_key, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Camera follows the fly from behind and above; the mouse can still move it.
        # In odor mode it is farther away so the smell is in view too.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = thorax_id
        viewer.cam.azimuth = 0
        viewer.cam.distance, viewer.cam.elevation = (40, -60) if odor_mode else (10, -40)

        i = 0
        wall_ref, sim_ref = time.perf_counter(), sim.mj_data.time
        speed, speed_wall, speed_sim = 1.0, wall_ref, sim_ref
        while viewer.is_running():
            if odor_mode:
                while odor_moves:
                    move_odor(odor_moves.pop(0), source, viewer.cam.azimuth)
                # The brain from run_odor.py decides from the two antennae alone.
                pos = sim.get_body_positions(fly.name)
                smell_left = odor_intensity(pos[left_idx], source[:2])
                smell_right = odor_intensity(pos[right_idx], source[:2])
                action, strength = brain(smell_left, smell_right)
                distance = np.linalg.norm((pos[left_idx] + pos[right_idx])[:2] / 2 - source[:2])
            else:
                action, strength = command

            signal = descending_signal(action, strength)
            for _ in range(steps_per_frame):
                if i % CONTROL_EVERY == 0:
                    obs = HybridControllerObservation.from_sim(sim, fly.name)
                    apply_locomotion_action(sim, fly.name, controller.step(signal, obs))
                sim.step()
                i += 1
            viewer.sync()

            # Keep simulated time in step with the wall clock: wait when ahead.
            # When behind, do not try to catch up later (it would look like
            # fast-forward), just run as fast as possible.
            now = time.perf_counter()
            ahead = (sim.mj_data.time - sim_ref) - (now - wall_ref)
            if ahead > 0:
                time.sleep(ahead)
            else:
                wall_ref, sim_ref = now, sim.mj_data.time

            # Simulation speed relative to real time, measured once per second.
            if now - speed_wall >= 1.0:
                speed = (sim.mj_data.time - speed_sim) / (now - speed_wall)
                speed_wall, speed_sim = now, sim.mj_data.time
                heading = np.degrees(np.arctan2(obs.fly_heading[1], obs.fly_heading[0]))
                smell_info = f"distance to smell={distance:5.1f} mm  " if odor_mode else ""
                print(
                    f"t={sim.mj_data.time:6.1f} s  {action:<7} strength={strength:.1f}  "
                    f"heading={heading:+6.1f} deg  {smell_info}speed={speed:.2f}x real time",
                    flush=True,
                )

            status = [("Command", f"{action} {strength:.1f}")]
            if odor_mode:
                status = [
                    ("Smell L / R", f"{smell_left:.4f} / {smell_right:.4f}"),
                    ("Brain", f"{action} {strength:.1f}"),
                    ("Distance", f"{distance:.1f} mm"),
                ]
                help_rows = [("Arrows", "move the smell (green disc)")]
            else:
                help_rows = [("Up", "walk"), ("Down", "stop"), ("Left / Right", "turn (again = sharper)")]
            status.append(("Speed", f"{speed:.2f}x real time"))
            viewer.set_texts([
                (None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 "\n".join(r[0] for r in status), "\n".join(r[1] for r in status)),
                (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 "\n".join(r[0] for r in help_rows), "\n".join(r[1] for r in help_rows)),
            ])


if __name__ == "__main__":
    main()
