#!/usr/bin/env python3
"""Automatic Zivid eye-to-hand calibration with the Zivid calibration board on the FR3.

One script for the whole procedure (old Part1 + Part2 + Part3):
  collect board captures + robot poses -> solve base_T_cam -> verify -> score.

Modes
-----
record : FIRST TIME. Robot in gravity compensation (this script publishes NO torques),
         you guide it by hand; ENTER = calibration pose, v+ENTER = verification pose,
         q+ENTER = finish. Each pose is captured immediately. At the end: calibration,
         score and, if the score is good, the joint configurations are written into the
         pose slots of the poses file (--poses).
run    : AUTOMATIC. Reads the joint configurations from the poses file, moves the robot
         there (joint-space impedance, minimum-jerk trajectory, low speed), captures,
         calibrates, verifies and prints the score. No key presses after the start.
check  : no robot, no camera: checks the poses file (joint limits, step size, duration).

Robot interface (same as H1-H3)
-------------------------------
- joint states  : sensor_msgs/JointState  on --joint-topic   (BEST_EFFORT)
- TCP pose      : geometry_msgs/PoseStamped on --pose-topic   (BEST_EFFORT, metres, Desk EE frame)
- torques       : std_msgs/Float64MultiArray on --torque-topic (BEST_EFFORT, 7 values, gravity
                  compensated by the robot: only the PD torques are sent)
All ROS work runs in a separate PROCESS, so the joint controller never stalls while the
Zivid camera is capturing.

Safety
------
- Joint goals are checked against the FR3 joint limits (with margin) before moving.
- Peak joint speed --vmax (default 0.2 rad/s), torques saturated, torque rate limited.
- If the joint states stop arriving, torques stop (the lab controller then applies zero
  torque: the robot stays in gravity compensation).
- Ctrl+C at any time: torques stop -> gravity compensation.
- The trajectory is a straight line in JOINT space between consecutive poses: record the
  poses in a sensible order and do the first automatic run with the emergency stop in hand.

Usage (PC GPU, venv zivid, ROS sourced, torque_subscriber_controller ACTIVE)
------------------------------------------------------------------------------
  python3 auto_calibration_board.py record --poses calibration_poses.yaml
  python3 auto_calibration_board.py run    --poses calibration_poses.yaml
  python3 auto_calibration_board.py check  --poses calibration_poses.yaml
Outputs go to --out-dir (default ~/zivid_ws/calib_board_<date_time>/).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]

# FR3 joint limits [rad] (Franka FR3 datasheet)
Q_MIN = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
Q_MAX = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])

# Joint impedance (half of the libfranka joint_impedance example) and torque limits
K_DEFAULT = np.array([300.0, 300.0, 300.0, 300.0, 125.0, 75.0, 25.0])   # Nm/rad
D_DEFAULT = np.array([25.0, 25.0, 25.0, 10.0, 6.0, 4.0, 2.0])           # Nm s/rad (low on the wrist: robust to jitter)
TAU_MAX = np.array([40.0, 40.0, 40.0, 40.0, 10.0, 10.0, 10.0])          # Nm (FR3: 87 / 12)
TAU_RATE = 1000.0                                                      # Nm/s (Franka limit)


# =============================================================================
# Small maths helpers
# =============================================================================
def min_jerk(q0: np.ndarray, q1: np.ndarray, T: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    '''Minimum-jerk reference: position and velocity at time t of a move q0 -> q1 of duration T.'''
    tau = float(np.clip(t / T, 0.0, 1.0))
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T if t < T else 0.0
    return q0 + (q1 - q0) * s, (q1 - q0) * ds


def move_duration(q0: np.ndarray, q1: np.ndarray, vmax: float, t_min: float = 3.0) -> float:
    '''Duration so that the peak joint speed (1.875 * dq / T for min-jerk) stays below vmax.'''
    return max(t_min, 1.875 * float(np.max(np.abs(q1 - q0))) / vmax)


def invert_se3(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rot_angle_deg(Rm: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0))))


def pose_msg_to_matrix_mm(msg) -> np.ndarray:
    '''PoseStamped (m) -> 4x4 with translation in mm.'''
    from scipy.spatial.transform import Rotation as R
    p, q = msg.pose.position, msg.pose.orientation
    T = np.eye(4)
    T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = 1000.0 * np.array([p.x, p.y, p.z])
    return T


def check_goal(q: np.ndarray, margin: float = 0.08) -> str | None:
    '''Return an error message if q is outside the FR3 limits minus margin, else None.'''
    low, high = Q_MIN + margin, Q_MAX - margin
    bad = [i for i in range(7) if not (low[i] <= q[i] <= high[i])]
    if bad:
        return "joints too close to the limits: " + ", ".join(
            f"q{i + 1}={q[i]:.3f} (range {low[i]:.3f}..{high[i]:.3f})" for i in bad)
    return None


# =============================================================================
# Motion server: separate process with all the ROS work (state, pose, joint PD)
# =============================================================================
def motion_server(conn, cfg: dict) -> None:
    '''
    Runs in its own process. Commands from the parent (via Pipe):
      ("state",)              -> dict with q, dq, state_age, pose_mm, pose_age, moving, active
      ("goto", q_goal, vmax)  -> start a min-jerk move from the measured q; reply move duration
      ("release",)            -> stop publishing torques (robot back to gravity compensation)
      ("quit",)               -> release and exit
    '''
    import threading
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float64MultiArray

    rclpy.init()
    node = rclpy.create_node("auto_calibration_motion")
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    pub = node.create_publisher(Float64MultiArray, cfg["torque_topic"], qos)

    K = np.asarray(cfg["K"]) * cfg["gain_scale"]
    D = np.asarray(cfg["D"]) * np.sqrt(cfg["gain_scale"])
    lock = threading.Lock()
    st = {"q": None, "dq": None, "t_state": 0.0, "pose": None, "t_pose": 0.0,
          "active": False, "traj": None, "tau_prev": np.zeros(7), "t_prev": None,
          "n_msgs": 0, "t_rate": time.monotonic(), "rate_hz": 0.0}

    def publish(tau: np.ndarray) -> None:
        msg = Float64MultiArray()
        msg.data = [float(x) for x in tau]
        pub.publish(msg)

    def on_joints(msg: JointState) -> None:
        try:
            idx = [msg.name.index(n) for n in JOINT_NAMES]
        except ValueError:
            return
        q = np.array([msg.position[i] for i in idx])
        dq = np.array([msg.velocity[i] for i in idx]) if len(msg.velocity) >= 7 else np.zeros(7)
        now = time.monotonic()
        with lock:
            st["q"], st["dq"], st["t_state"] = q, dq, now
            st["n_msgs"] += 1
            if now - st["t_rate"] >= 1.0:                     # joint state rate over the last second
                st["rate_hz"] = st["n_msgs"] / (now - st["t_rate"])
                st["n_msgs"], st["t_rate"] = 0, now
            if not st["active"] or st["traj"] is None or not cfg["publish"]:
                return
            q0, q1, T, t0 = st["traj"]
            q_ref, dq_ref = min_jerk(q0, q1, T, now - t0)
            tau = K * (q_ref - q) + D * (dq_ref - dq)
            tau = np.clip(tau, -TAU_MAX, TAU_MAX)
            # torque rate limit (Franka: 1000 Nm/s)
            dt = 0.001 if st["t_prev"] is None else max(1e-4, min(0.01, now - st["t_prev"]))
            step = TAU_RATE * dt
            tau = np.clip(tau, st["tau_prev"] - step, st["tau_prev"] + step)
            st["tau_prev"], st["t_prev"] = tau, now
        publish(tau)

    def on_pose(msg: PoseStamped) -> None:
        T = pose_msg_to_matrix_mm(msg)
        with lock:
            st["pose"], st["t_pose"] = T, time.monotonic()

    node.create_subscription(JointState, cfg["joint_topic"], on_joints, qos)
    node.create_subscription(PoseStamped, cfg["pose_topic"], on_pose, qos)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def release() -> None:
        with lock:
            st["active"], st["traj"] = False, None
            st["tau_prev"], st["t_prev"] = np.zeros(7), None
        if cfg["publish"]:
            for _ in range(5):
                publish(np.zeros(7))
                time.sleep(0.002)

    try:
        while True:
            # watchdog: joint states stopped while moving -> stop torques
            with lock:
                stale = st["active"] and (time.monotonic() - st["t_state"] > cfg["state_timeout"])
            if stale:
                release()
                print("[motion] joint states stopped arriving: torques released", flush=True)
            if not conn.poll(0.01):
                continue
            cmd = conn.recv()
            if cmd[0] == "state":
                with lock:
                    now = time.monotonic()
                    moving = False
                    if st["traj"] is not None:
                        moving = now - st["traj"][3] < st["traj"][2]
                    conn.send({
                        "q": None if st["q"] is None else st["q"].copy(),
                        "dq": None if st["dq"] is None else st["dq"].copy(),
                        "state_age": now - st["t_state"] if st["q"] is not None else float("inf"),
                        "pose_mm": None if st["pose"] is None else st["pose"].copy(),
                        "pose_age": now - st["t_pose"] if st["pose"] is not None else float("inf"),
                        "moving": moving, "active": st["active"], "rate_hz": st["rate_hz"],
                    })
            elif cmd[0] == "goto":
                q_goal, vmax = np.asarray(cmd[1], dtype=float), float(cmd[2])
                with lock:
                    if st["q"] is None:
                        conn.send(("error", "no joint state yet"))
                        continue
                    q0 = st["q"].copy()
                    T = move_duration(q0, q_goal, vmax)
                    st["traj"] = (q0, q_goal, T, time.monotonic())
                    if not st["active"]:
                        st["tau_prev"], st["t_prev"] = np.zeros(7), None
                    st["active"] = True
                conn.send(("ok", T))
            elif cmd[0] == "release":
                release()
                conn.send(("ok", 0.0))
            elif cmd[0] == "quit":
                release()
                conn.send(("ok", 0.0))
                break
    except (KeyboardInterrupt, EOFError):
        release()
    finally:
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class Robot:
    '''Parent-side handle of the motion server process.'''

    def __init__(self, args: argparse.Namespace, publish: bool):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        cfg = {"joint_topic": args.joint_topic, "pose_topic": args.pose_topic,
               "torque_topic": args.torque_topic, "K": K_DEFAULT.tolist(), "D": D_DEFAULT.tolist(),
               "gain_scale": args.gain_scale, "publish": publish, "state_timeout": 0.2}
        self.proc = ctx.Process(target=motion_server, args=(child, cfg), daemon=True)
        self.proc.start()

    def _call(self, *cmd):
        self.conn.send(cmd)
        return self.conn.recv()

    def state(self) -> dict:
        return self._call("state")

    def wait_ready(self, timeout: float = 10.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.state()
            if s["state_age"] < 0.2 and s["pose_age"] < 0.5:
                return s
            time.sleep(0.1)
        raise RuntimeError("No joint state / TCP pose received: check the driver, the topics and the controller.")

    def goto(self, q_goal: np.ndarray, vmax: float) -> float:
        status, value = self._call("goto", np.asarray(q_goal, dtype=float), vmax)
        if status != "ok":
            raise RuntimeError(value)
        return value

    def release(self) -> None:
        try:
            self._call("release")
        except (BrokenPipeError, EOFError):
            pass

    def close(self) -> None:
        try:
            self._call("quit")
        except (BrokenPipeError, EOFError):
            pass
        self.proc.join(timeout=3.0)
        if self.proc.is_alive():
            self.proc.terminate()

    def wait_still(self, vel_tol: float = 0.01, still_time: float = 0.8, timeout: float = 15.0,
                   max_state_age: float = 0.3) -> dict:
        '''Wait until the trajectory is over and all joint speeds stay below vel_tol for still_time.'''
        t0, t_still = time.time(), None
        while time.time() - t0 < timeout:
            s = self.state()
            if s["state_age"] > max_state_age:
                raise RuntimeError(f"Joint states stopped arriving (last one {s['state_age']:.2f} s ago)")
            if not s["moving"] and np.max(np.abs(s["dq"])) < vel_tol:
                t_still = t_still or time.time()
                if time.time() - t_still >= still_time:
                    return s
            else:
                t_still = None
            time.sleep(0.02)
        raise RuntimeError("Robot did not settle (still moving after the timeout)")


# =============================================================================
# Camera, calibration and verification
# =============================================================================
class Camera:
    def __init__(self):
        import zivid
        self.zivid = zivid
        self.app = zivid.Application()
        self.camera = self.app.connect_camera()
        print(f"[camera] connected: {self.camera.info.model_name} {self.camera.info.serial_number}")

    def capture_board(self):
        zivid = self.zivid
        frame = zivid.calibration.capture_calibration_board(self.camera)
        detection = zivid.calibration.detect_calibration_board(frame)
        return frame, detection


def capture_sample(cam: Camera, robot: Robot, still_check_mm: float, still_check_deg: float):
    '''Pose before + capture + pose after. Returns (frame, detection, base_T_hand_mm, q) or raises.'''
    s0 = robot.state()
    frame, detection = cam.capture_board()
    s1 = robot.state()
    if s0["pose_mm"] is None or s1["pose_mm"] is None or s1["pose_age"] > 0.5:
        raise RuntimeError("TCP pose not available")
    dT = invert_se3(s0["pose_mm"]) @ s1["pose_mm"]
    moved_mm, moved_deg = float(np.linalg.norm(dT[:3, 3])), rot_angle_deg(dT[:3, :3])
    if moved_mm > still_check_mm or moved_deg > still_check_deg:
        raise RuntimeError(f"robot moved during the capture ({moved_mm:.2f} mm, {moved_deg:.3f} deg)")
    if not detection.valid():
        raise RuntimeError(f"calibration board not detected: {detection.status_description()}"
                           if hasattr(detection, "status_description") else "calibration board not detected")
    return frame, detection, s1["pose_mm"], s1["q"]


def solve(samples: list[dict]) -> dict:
    '''Zivid eye-to-hand solve on the calibration samples.'''
    import zivid
    inputs = [zivid.calibration.HandEyeInput(zivid.calibration.Pose(s["base_T_hand"]), s["detection"])
              for s in samples]
    result = zivid.calibration.calibrate_eye_to_hand(inputs)
    if not result.valid():
        raise RuntimeError("Zivid hand-eye calibration FAILED (invalid result)")
    X = np.asarray(result.transform(), dtype=np.float64)
    res = [(float(r.translation()), float(r.rotation())) for r in result.residuals()]
    return {"base_T_cam": X, "residuals": res}


def verify(X: np.ndarray, samples: list[dict]) -> dict | None:
    '''Consistency of hand_T_board on samples NOT used in the solve (board rigid on the hand).'''
    if len(samples) < 2:
        return None
    Ys, centroids = [], []
    for s in samples:
        cam_T_board = np.asarray(s["detection"].pose().to_matrix(), dtype=np.float64)
        Ys.append(invert_se3(s["base_T_hand"]) @ X @ cam_T_board)
        c = np.append(np.asarray(s["detection"].centroid(), dtype=np.float64), 1.0)
        centroids.append((invert_se3(s["base_T_hand"]) @ X @ c)[:3])
    t_ref = np.mean([Y[:3, 3] for Y in Ys], axis=0)
    c_ref = np.mean(centroids, axis=0)
    from scipy.spatial.transform import Rotation as R
    R_ref = R.from_matrix(np.array([Y[:3, :3] for Y in Ys])).mean().as_matrix()
    pos = [float(np.linalg.norm(Y[:3, 3] - t_ref)) for Y in Ys]
    rot = [rot_angle_deg(R_ref.T @ Y[:3, :3]) for Y in Ys]
    cen = [float(np.linalg.norm(c - c_ref)) for c in centroids]
    return {"pos_mm": pos, "rot_deg": rot, "centroid_mm": cen}


def score(sol: dict, ver: dict | None, args: argparse.Namespace) -> tuple[bool, dict]:
    t = np.array([r[0] for r in sol["residuals"]])
    r = np.array([r[1] for r in sol["residuals"]])
    summary = {"n_calibration": len(t),
               "residual_translation_mm": {"mean": float(t.mean()), "max": float(t.max())},
               "residual_rotation_deg": {"mean": float(r.mean()), "max": float(r.max())}}
    ok = (t.mean() <= args.max_res_mean_mm and t.max() <= args.max_res_max_mm
          and r.mean() <= args.max_rot_mean_deg)
    if ver is not None:
        summary["verification"] = {"n": len(ver["pos_mm"]),
                                   "pos_mm_max": float(max(ver["pos_mm"])),
                                   "rot_deg_max": float(max(ver["rot_deg"])),
                                   "centroid_mm_max": float(max(ver["centroid_mm"]))}
        ok = ok and max(ver["pos_mm"]) <= args.max_ver_mm and max(ver["rot_deg"]) <= args.max_ver_deg
    summary["PASS"] = bool(ok)
    return ok, summary


# =============================================================================
# Poses file (slots)
# =============================================================================
def load_poses(path: Path) -> tuple[list, list]:
    import yaml
    if not path.exists():
        return [], []
    data = yaml.safe_load(path.read_text()) or {}
    clean = lambda lst: [np.asarray(q, dtype=float) for q in (lst or []) if q is not None]
    return clean(data.get("calibration")), clean(data.get("verification"))


def save_poses(path: Path, cal: list, ver: list, summary: dict) -> None:
    import yaml
    data = {"calibration": [[round(float(x), 5) for x in q] for q in cal],
            "verification": [[round(float(x), 5) for x in q] for q in ver]}
    header = (f"# FR3 joint configurations [rad], fr3_joint1..7\n"
              f"# saved {datetime.now():%Y-%m-%d %H:%M} by auto_calibration_board.py record\n"
              f"# score: residual mean {summary['residual_translation_mm']['mean']:.2f} mm, "
              f"max {summary['residual_translation_mm']['max']:.2f} mm\n")
    path.write_text(header + yaml.safe_dump(data, default_flow_style=None, sort_keys=False))


# =============================================================================
# Main procedures
# =============================================================================
def save_sample(out: Path, name: str, frame, base_T_hand: np.ndarray, q: np.ndarray) -> None:
    frame.save(str(out / "data" / f"{name}.zdf"))
    np.save(out / "data" / f"{name}_base_T_hand_mm.npy", base_T_hand)
    np.save(out / "data" / f"{name}_q.npy", q)


def finish(out: Path, cal: list[dict], ver: list[dict], args) -> tuple[bool, dict]:
    print(f"\n[calib] solving with {len(cal)} calibration samples...")
    sol = solve(cal)
    X = sol["base_T_cam"]
    np.save(out / "T_cam_to_base_Zivid_board.npy", X)
    print("[calib] base_T_cam (mm):")
    print(np.array2string(X, precision=4, suppress_small=True))
    for s, (t, r) in zip(cal, sol["residuals"]):
        print(f"  {s['name']}: translation {t:7.3f} mm   rotation {r:6.3f} deg")
    vres = verify(X, ver)
    if vres is not None:
        print(f"[verify] {len(ver)} verification samples:")
        for s, p, a in zip(ver, vres["pos_mm"], vres["rot_deg"]):
            print(f"  {s['name']}: {p:7.3f} mm   {a:6.3f} deg")
    ok, summary = score(sol, vres, args)
    summary["base_T_cam_mm"] = X.tolist()
    (out / "report.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print(f"SCORE: residuals mean {summary['residual_translation_mm']['mean']:.2f} mm, "
          f"max {summary['residual_translation_mm']['max']:.2f} mm, "
          f"rot mean {summary['residual_rotation_deg']['mean']:.3f} deg")
    if "verification" in summary:
        v = summary["verification"]
        print(f"       verification max {v['pos_mm_max']:.2f} mm, {v['rot_deg_max']:.3f} deg")
    print("RESULT: " + ("PASS" if ok else "FAIL (do not use this calibration)"))
    print(f"Saved in {out}")
    print("=" * 60)
    return ok, summary


def run_record(args, out: Path) -> None:
    robot = Robot(args, publish=False)          # NO torques: gravity compensation, hand guiding
    cam = Camera()
    cal, ver, q_cal, q_ver = [], [], [], []
    try:
        robot.wait_ready()
        print("\nRECORD mode: guide the robot by hand, keep it STILL, then:\n"
              "  ENTER = calibration pose   v+ENTER = verification pose   q+ENTER = finish\n")
        while True:
            key = input(f"[{len(cal)} cal / {len(ver)} ver] > ").strip().lower()
            if key == "q":
                break
            kind = "ver" if key == "v" else "cal"
            try:
                robot.wait_still(timeout=5.0, max_state_age=args.record_state_age)
                frame, det, T, q = capture_sample(cam, robot, args.still_mm, args.still_deg)
            except RuntimeError as exc:
                print(f"  NOT saved: {exc}")
                continue
            name = f"{kind}{(len(cal) if kind == 'cal' else len(ver)) + 1:02d}"
            save_sample(out, name, frame, T, q)
            sample = {"name": name, "detection": det, "base_T_hand": T}
            (cal if kind == "cal" else ver).append(sample)
            (q_cal if kind == "cal" else q_ver).append(q)
            print(f"  saved {name}")
    finally:
        robot.close()
    if len(cal) < 6:
        print("Too few calibration poses (need >= 6, better 15-20).")
        return
    ok, summary = finish(out, cal, ver, args)
    save_poses(out / "poses_used.yaml", q_cal, q_ver, summary)
    if ok and input(f"\nScore OK. Write these poses into {args.poses}? [y/N] ").strip().lower() == "y":
        save_poses(Path(args.poses), q_cal, q_ver, summary)
        print(f"Poses saved to {args.poses}: next time use 'run'.")


def run_auto(args, out: Path) -> None:
    q_cal, q_ver = load_poses(Path(args.poses))
    if len(q_cal) < 6:
        sys.exit(f"{args.poses}: only {len(q_cal)} calibration poses filled (need >= 6). Use 'record' first.")
    plan = [("cal", q) for q in q_cal] + [("ver", q) for q in q_ver]
    for kind, q in plan:
        err = check_goal(q)
        if err:
            sys.exit(f"Pose rejected ({kind}): {err}")

    robot = Robot(args, publish=True)
    cam = Camera()
    cal, ver = [], []
    try:
        s = robot.wait_ready()
        time.sleep(1.5)                          # measure the joint state rate
        rate = robot.state()["rate_hz"]
        print(f"Joint states received at {rate:.0f} Hz")
        if rate < args.min_rate and not args.force:
            sys.exit(f"Joint state rate too low for the controller (< {args.min_rate:.0f} Hz). "
                     "Check the network/driver, or use --force at your own risk.")
        s = robot.state()
        first_step = float(np.max(np.abs(plan[0][1] - s["q"])))
        print(f"\nAUTO mode: {len(q_cal)} calibration + {len(q_ver)} verification poses, vmax {args.vmax} rad/s.")
        print(f"Distance to the first pose: {first_step:.2f} rad (max joint).")
        input("Emergency stop in hand, workspace clear. ENTER to start (Ctrl+C to abort) ")
        counters = {"cal": 0, "ver": 0}
        for i, (kind, q_goal) in enumerate(plan, start=1):
            counters[kind] += 1
            name = f"{kind}{counters[kind]:02d}"
            T = robot.goto(q_goal, args.vmax)
            print(f"[{i}/{len(plan)}] {name}: moving ({T:.1f} s)...", flush=True)
            time.sleep(T)
            robot.wait_still()
            time.sleep(args.settle)
            try:
                frame, det, Th, q = capture_sample(cam, robot, args.still_mm, args.still_deg)
            except RuntimeError as exc:
                print(f"  skipped: {exc}")
                continue
            save_sample(out, name, frame, Th, q)
            (cal if kind == "cal" else ver).append({"name": name, "detection": det, "base_T_hand": Th})
            print(f"  captured {name}")
        if args.return_first:
            T = robot.goto(q_cal[0], args.vmax)
            time.sleep(T)
            robot.wait_still()
    except KeyboardInterrupt:
        print("\nInterrupted: torques released (gravity compensation).")
        return
    finally:
        robot.close()                           # releases the torques and stops the ROS process
    if len(cal) < 6:
        print(f"Only {len(cal)} valid calibration captures: not enough.")
        return
    finish(out, cal, ver, args)


def run_check(args) -> None:
    q_cal, q_ver = load_poses(Path(args.poses))
    plan = q_cal + q_ver
    print(f"{args.poses}: {len(q_cal)} calibration, {len(q_ver)} verification poses")
    total, prev = 0.0, None
    for i, q in enumerate(plan, start=1):
        err = check_goal(q)
        step = "" if prev is None else f"  step {np.max(np.abs(q - prev)):.2f} rad"
        if prev is not None:
            total += move_duration(prev, q, args.vmax) + args.settle + 2.5
        print(f"  {i:2d}: {'OK ' if err is None else 'BAD'}{step}{'  ' + err if err else ''}")
        prev = q
    print(f"Estimated duration: {total / 60:.1f} min (vmax {args.vmax} rad/s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["record", "run", "check"])
    p.add_argument("--poses", default=str(Path(__file__).with_name("calibration_poses.yaml")),
                   help="YAML file with the pose slots (joint configurations)")
    p.add_argument("--out-dir", default=None, help="Output folder (default ~/zivid_ws/calib_board_<date_time>)")
    p.add_argument("--joint-topic", default="/robot1/franka/joint_states")
    p.add_argument("--pose-topic", default="/robot1/franka_robot_state_broadcaster/current_pose")
    p.add_argument("--torque-topic", default="/robot1/torque_subscriber_controller/external_torques")
    p.add_argument("--vmax", type=float, default=0.2, help="Peak joint speed [rad/s]")
    p.add_argument("--gain-scale", type=float, default=1.0, help="Scale of the joint stiffness (damping scales with sqrt)")
    p.add_argument("--settle", type=float, default=1.0, help="Extra wait after the robot is still [s]")
    p.add_argument("--still-mm", type=float, default=0.3, help="Max TCP motion during a capture [mm]")
    p.add_argument("--still-deg", type=float, default=0.05, help="Max TCP rotation during a capture [deg]")
    p.add_argument("--min-rate", type=float, default=300.0, help="Minimum joint state rate to allow motion [Hz]")
    p.add_argument("--force", action="store_true", help="Move even if the joint state rate is below --min-rate")
    p.add_argument("--record-state-age", type=float, default=1.0,
                   help="record mode: max age of the last joint state [s] (no torques are sent, so it can be loose)")
    p.add_argument("--return-first", action="store_true", help="Go back to the first pose at the end")
    p.add_argument("--max-res-mean-mm", type=float, default=1.5)
    p.add_argument("--max-res-max-mm", type=float, default=3.0)
    p.add_argument("--max-rot-mean-deg", type=float, default=0.3)
    p.add_argument("--max-ver-mm", type=float, default=3.0)
    p.add_argument("--max-ver-deg", type=float, default=0.6)
    args = p.parse_args()

    if args.mode == "check":
        run_check(args)
        return
    out = Path(args.out_dir).expanduser() if args.out_dir else \
        Path.home() / "zivid_ws" / f"calib_board_{datetime.now():%Y-%m-%d_%H%M}"
    (out / "data").mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out}")
    if args.mode == "record":
        run_record(args, out)
    else:
        run_auto(args, out)


if __name__ == "__main__":
    main()
