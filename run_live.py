"""Live fly in a MuJoCo window, steered with the keyboard.

Runs natively on Windows (the window is drawn by the GPU), not in Docker.

Keys:   Up = walk   Down = stop   Left / Right = turn (press again: sharper turn)
Mouse:  left drag = rotate view, right drag = move view, wheel = zoom,
        double-click the fly, then Ctrl + right drag = push it.
Close the window to quit.
"""

import functools
import time

import mujoco
import mujoco.viewer
import numpy as np
import flygym_demo.complex_terrain.hybrid_controller as hybrid_controller
from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action

from run_route import descending_signal, make_simulation

# Speed settings. run_route.py uses 0.1 ms physics steps and runs the
# controller every step (10 kHz), ~16x slower than real time. These values
# give ~0.95x real time on an i7-9700F with nearly the same gait (see README).
# On a slower CPU the fly moves in slow motion.
PHYSICS_TIMESTEP = 2.5e-4  # s; 5e-4 is already unstable (the fly bounces)
CONTROL_EVERY = 8  # controller update every 8 physics steps = 500 Hz
FRAME_TIME = 1 / 60  # simulated seconds between screen updates
NOSLIP_ITERATIONS = 2  # FlyGym default is 5; 2 gives the same gait ~8% faster

# The controller rebuilds 42 joint descriptors on every call. Caching them gives
# identical results about 15% faster.
hybrid_controller.dof_spec_to_jointdof = functools.lru_cache(maxsize=None)(
    hybrid_controller.dof_spec_to_jointdof
)

KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265  # GLFW key codes
ARROWS = {KEY_UP: "forward", KEY_DOWN: "stop", KEY_LEFT: "left", KEY_RIGHT: "right"}

# Current (action, strength). Replaced as a whole tuple, so the viewer thread
# (keyboard) and the main loop (simulation) never see a half-updated command.
command = ("stop", 0.0)


def on_key(key):
    """Keyboard handler, called by the viewer thread on key press."""
    global command
    action = ARROWS.get(key)
    if action is None:
        return
    if action in ("left", "right"):
        # First press: smooth turn. Same arrow again: sharper, up to 1.0.
        strength = min(command[1] + 0.2, 1.0) if command[0] == action else 0.3
    else:
        strength = 1.0
    command = (action, strength)


def main():
    print(__doc__)
    fly, _, sim, controller = make_simulation(PHYSICS_TIMESTEP, CONTROL_EVERY)
    sim.mj_model.opt.noslip_iterations = NOSLIP_ITERATIONS
    thorax_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{fly.name}/c_thorax")
    steps_per_frame = round(FRAME_TIME / sim.timestep)

    with mujoco.viewer.launch_passive(
        sim.mj_model, sim.mj_data, key_callback=on_key, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Camera follows the fly from behind and above; the mouse can still move it.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = thorax_id
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 10, 0, -40

        i = 0
        wall_ref, sim_ref = time.perf_counter(), sim.mj_data.time
        speed, speed_wall, speed_sim = 1.0, wall_ref, sim_ref
        while viewer.is_running():
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
                print(
                    f"t={sim.mj_data.time:6.1f} s  {action:<7} strength={strength:.1f}  "
                    f"heading={heading:+6.1f} deg  speed={speed:.2f}x real time",
                    flush=True,
                )

            viewer.set_texts([
                (None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 "Command\nSpeed",
                 f"{action} {strength:.1f}\n{speed:.2f}x real time"),
                (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 "Up\nDown\nLeft / Right",
                 "walk\nstop\nturn (again = sharper)"),
            ])


if __name__ == "__main__":
    main()
