"""Live fly in a MuJoCo window.

    uv run run_live.py          you steer the fly with the arrow keys:
                                Up = walk, Down = stop, Left / Right = turn
                                (press again for a sharper turn)
    uv run run_live.py vision   the fly walks to a dark pillar it sees with its
                                eyes (its view is shown bottom right). Move the
                                pillar with the arrow keys, relative to the view
                                (Up = away from you), or hold Ctrl and drag it
                                with the right mouse button (if you double-clicked
                                something else, double-click the pillar first).
    uv run run_live.py escape   the fly stands still; throw the pillar at it (arrows
                                or Ctrl + right drag, as in vision mode) and a
                                whole-brain model of the connectome decides the
                                escape (see run_escape.py). Top right: which of its
                                neurons fire. `escape mcns`: the male CNS
                                connectome instead of FlyWire.

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
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action

from run_route import descending_signal, make_simulation, stand_up
from run_vision import VISION_HZ, add_pillar, brain, darkness_above_horizon, fly_view
from run_escape import BrainView, EscapeBrain

# Speed settings. run_route.py uses 0.1 ms physics steps and runs the
# controller every step (10 kHz), ~12x slower than real time. These values
# give ~0.93x real time on an i7-9700F (walking) with nearly the same gait (see README).
# On a slower CPU the fly moves in slow motion.
PHYSICS_TIMESTEP = 2.5e-4  # s; 5e-4 is already unstable (the fly bounces)
CONTROL_EVERY = 8  # controller update every 8 physics steps = 500 Hz
FRAME_TIME = 1 / 60  # simulated seconds between screen updates
NOSLIP_ITERATIONS = 2  # FlyGym default is 5; 2 gives the same gait ~8% faster

PILLAR_START = (15.0, 10.0)  # mm, ahead and to the left of the fly
PILLAR_STEP = 3.0  # mm the pillar moves per arrow key press

# The controller rebuilds 42 joint descriptors on every call. Caching them gives
# identical results about 15% faster.
hybrid_controller.dof_spec_to_jointdof = functools.lru_cache(maxsize=None)(
    hybrid_controller.dof_spec_to_jointdof
)

KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265  # GLFW key codes
ARROWS = {KEY_UP: "forward", KEY_DOWN: "stop", KEY_LEFT: "left", KEY_RIGHT: "right"}
ROTATE = int(mujoco.mjtPertBit.mjPERT_ROTATE)
# MuJoCo warnings raised when the physics blows up (bad position, velocity, acceleration)
BLOW_UPS = [int(w) for w in (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)]

escape_mode = sys.argv[1:2] == ["escape"]
vision_mode = sys.argv[1:] == ["vision"] or escape_mode  # both: a pillar and the fly's eyes
# Steering mode: current (action, strength). Replaced as a whole tuple, so the
# viewer thread (keyboard) and the main loop never see a half-updated command.
command = ("stop", 0.0)
# Vision mode: arrow presses waiting to move the pillar (list.append is thread-safe).
pillar_moves = []


def on_key(key):
    """Keyboard handler, called by the viewer thread on key press."""
    global command
    action = ARROWS.get(key)
    if action is None:
        return
    if vision_mode:
        pillar_moves.append(key)
        return
    if action in ("left", "right"):
        # First press: smooth turn. Same arrow again: sharper, up to 1.0.
        strength = min(command[1] + 0.2, 1.0) if command[0] == action else 0.3
    else:
        strength = 1.0
    command = (action, strength)


def move_pillar(key, pillar, azimuth_deg):
    """Shift the pillar by one arrow press, relative to where the camera looks."""
    az = np.radians(azimuth_deg)
    away = np.array([np.cos(az), np.sin(az)])  # camera view direction on the ground
    left = np.array([-away[1], away[0]])
    shift = {KEY_UP: away, KEY_DOWN: -away, KEY_LEFT: left, KEY_RIGHT: -left}[key]
    pillar[:2] += PILLAR_STEP * shift


def keep_drag_level(viewer, rest_z):
    """Make mouse dragging slide the pillar over the floor.

    MuJoCo's Ctrl + right drag moves the selected object in a vertical plane
    facing the camera, so dragging up would lift the pillar into the air. Turn
    that lift into "away from the camera" instead (like the Up arrow).
    """
    with viewer.lock():
        ref = viewer.perturb.refpos  # where the viewer puts the dragged object
        lift = ref[2] - rest_z
        az = np.radians(viewer.cam.azimuth)
        ref[0] += lift * np.cos(az)
        ref[1] += lift * np.sin(az)
        ref[2] = rest_z


def main():
    print(__doc__)
    world = FlatGroundWorld()
    if vision_mode:
        add_pillar(world, PILLAR_START)
    fly, _, sim, controller = make_simulation(PHYSICS_TIMESTEP, CONTROL_EVERY, world, vision=vision_mode)
    sim.mj_model.opt.noslip_iterations = NOSLIP_ITERATIONS
    thorax_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{fly.name}/c_thorax")
    steps_per_frame = round(FRAME_TIME / sim.timestep)
    if vision_mode:
        # Writing into this row of mocap_pos moves the pillar.
        pillar_body = sim.mj_model.body("pillar")
        pillar = sim.mj_data.mocap_pos[pillar_body.mocapid[0]]
        rest_z = pillar[2]
        vision_every = round(1 / (VISION_HZ * FRAME_TIME))  # in frames
        sim.get_ommatidia_readouts(fly.name)  # first call compiles the retina code (~7 s)
    if escape_mode:
        print("Loading the brain...", flush=True)
        escape = EscapeBrain(*sys.argv[2:3])  # connectome: "783" (default) or "mcns"
        brain_view = BrainView(escape)  # shown top right: which neurons fire
        escape_dt = vision_every * steps_per_frame * sim.timestep  # s of brain per eye reading

    with mujoco.viewer.launch_passive(
        sim.mj_model, sim.mj_data, key_callback=on_key, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Camera follows the fly from behind and above; the mouse can still move it.
        # In vision mode it is farther away so the pillar is in view too.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = thorax_id
        viewer.cam.azimuth = 0
        viewer.cam.distance, viewer.cam.elevation = (40, -45) if vision_mode else (10, -40)
        if vision_mode:
            # Select the pillar up front, so Ctrl + right drag moves it right away.
            with viewer.lock():
                viewer.perturb.select = pillar_body.id

        i, frame, blowups = 0, 0, 0
        wall_ref, sim_ref = time.perf_counter(), sim.mj_data.time
        speed, speed_wall, speed_sim = 1.0, wall_ref, sim_ref
        while viewer.is_running():
            if vision_mode:
                while pillar_moves:
                    move_pillar(pillar_moves.pop(0), pillar, viewer.cam.azimuth)
                if viewer.perturb.select == pillar_body.id and viewer.perturb.active:
                    keep_drag_level(viewer, rest_z)
                if frame % vision_every == 0:
                    # The brain decides from what the eyes see: the rule from run_vision.py,
                    # or in escape mode the model of the whole brain (run_escape.py).
                    readouts = sim.get_ommatidia_readouts(fly.name)
                    if escape_mode:
                        signal, info = escape.step(readouts, escape_dt)
                        brain_img = brain_view.render(info["spikes"], escape_dt)
                    else:
                        darkness = darkness_above_horizon(readouts)
                        action, strength = brain(darkness)
                    eyes = fly_view(readouts)  # left | right eye, 256 x 450
                    screen = viewer.viewport
                    if screen is not None and screen.width > eyes.shape[1] + 20:
                        corner = mujoco.MjrRect(screen.width - eyes.shape[1] - 10, 10, eyes.shape[1], eyes.shape[0])
                        images = [(corner, np.repeat(eyes[:, :, None], 3, axis=2))]
                        h, w = brain_img.shape[:2] if escape_mode else (0, 0)
                        if escape_mode and screen.height > eyes.shape[0] + h + 30 and screen.width > w + 20:
                            images.append((mujoco.MjrRect(screen.width - w - 10, screen.height - h - 10, w, h), brain_img))
                        viewer.set_images(images)
                distance = np.linalg.norm(sim.mj_data.xpos[thorax_id][:2] - pillar[:2])
            else:
                action, strength = command

            if not escape_mode:
                signal = descending_signal(action, strength)
            for _ in range(steps_per_frame):
                if i % CONTROL_EVERY == 0:
                    obs = HybridControllerObservation.from_sim(sim, fly.name)
                    apply_locomotion_action(sim, fly.name, controller.step(signal, obs))
                sim.step()
                i += 1

            # Twisting a body with the mouse (Ctrl + left drag) makes the physics
            # of this tiny model blow up, so rotation drags are ignored.
            if viewer.perturb.active & ROTATE:
                with viewer.lock():
                    viewer.perturb.active &= ~ROTATE
            viewer.sync()

            # If the physics blew up anyway, MuJoCo resets it to a default pose.
            # Put the fly back on its feet instead.
            if sum(sim.mj_data.warning[w].number for w in BLOW_UPS) > blowups:
                print("The physics became unstable; the fly stands up again.", flush=True)
                stand_up(fly, sim, controller)
                blowups = sum(sim.mj_data.warning[w].number for w in BLOW_UPS)
                wall_ref, sim_ref = time.perf_counter(), sim.mj_data.time
                speed_wall, speed_sim = wall_ref, sim_ref

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
                pillar_info = f"distance to pillar={distance:5.1f} mm  " if vision_mode else ""
                decision = f"legs L/R={signal[0]:+.2f} {signal[1]:+.2f}" if escape_mode else f"{action:<7} strength={strength:.1f}"
                print(
                    f"t={sim.mj_data.time:6.1f} s  {decision}  "
                    f"heading={heading:+6.1f} deg  {pillar_info}speed={speed:.2f}x real time",
                    flush=True,
                )

            if escape_mode:
                status = [
                    ("Loom input L / R", f"{info['input_hz'][0]:.0f} / {info['input_hz'][1]:.0f} Hz"),
                    ("Giant fibre L / R", f"{info['GF'][0]:.0f} / {info['GF'][1]:.0f} Hz"),
                    ("DNa01+02 L / R", f"{info['DNa'][0]:.0f} / {info['DNa'][1]:.0f} Hz"),
                    ("MDN (backwards)", f"{sum(info['MDN']):.0f} Hz"),
                    *([("TTMn (jump)", f"{info['TTMn'][0]:.0f} / {info['TTMn'][1]:.0f} Hz")] if "TTMn" in info else []),
                    ("Legs L / R", f"{signal[0]:+.1f} / {signal[1]:+.1f}"),
                    ("Distance", f"{distance:.1f} mm"),
                ]
            elif vision_mode:
                status = [
                    ("Eyes see L / R", f"{darkness[0].sum():.1f} / {darkness[1].sum():.1f}"),
                    ("Brain", f"{action} {strength:.1f}"),
                    ("Distance", f"{distance:.1f} mm"),
                ]
            else:
                status = [("Command", f"{action} {strength:.1f}")]
            if vision_mode:
                help_rows = [
                    ("Arrows", "move the pillar"),
                    ("Ctrl + right drag", "drag the pillar with the mouse"),
                ]
            else:
                help_rows = [("Up", "walk"), ("Down", "stop"), ("Left / Right", "turn (again = sharper)")]
            status.append(("Speed", f"{speed:.2f}x real time"))
            viewer.set_texts([
                (None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 "\n".join(r[0] for r in status), "\n".join(r[1] for r in status)),
                (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 "\n".join(r[0] for r in help_rows), "\n".join(r[1] for r in help_rows)),
            ])
            frame += 1


if __name__ == "__main__":
    main()
