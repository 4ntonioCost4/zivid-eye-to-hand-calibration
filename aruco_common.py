#!/usr/bin/env python3
"""Shared helpers for the Zivid ArUco eye-to-hand calibration scripts.

Kept in one place so that Part1, Part2, Part3 and QuickCalib always use the
same marker ID, dictionary, capture settings and transform maths.

The scripts import this module as `import aruco_common`, which works when they
are run directly with `python3 <path>/Zivid_Eye_to_Base_PartN.py` (the script
folder is on sys.path).

Units: Zivid poses are in millimetres; ROS poses are in metres. Every helper
here works in millimetres unless stated otherwise.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
import zivid

# --- Defaults of the printed marker -----------------------------------------
# Change them HERE (or on the command line) and every script follows.
DEFAULT_MARKER_ID = 20
DEFAULT_DICTIONARY = "aruco6x6_50"
DEFAULT_POSE_TOPIC = "/robot2/franka_robot_state_broadcaster/current_pose"


def add_common_args(parser: argparse.ArgumentParser, with_capture: bool = True) -> None:
    '''
    Inputs
    - `parser`: argparse parser of the calling script.
    - `with_capture`: also add the capture-settings options (not needed in Part2).

    Outputs
    - No return value. Adds the shared command-line options.
    '''
    parser.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID,
                        help=f"ArUco ID glued on the gripper (default {DEFAULT_MARKER_ID})")
    parser.add_argument("--dict", default=DEFAULT_DICTIONARY,
                        help=f"Zivid MarkerDictionary name (default {DEFAULT_DICTIONARY})")
    if with_capture:
        parser.add_argument("--settings", default=None,
                            help="Zivid settings .yml (e.g. saved from Zivid Studio). "
                                 "If omitted, Capture Assistant suggests settings ONCE at start-up.")
        parser.add_argument("--ambient", choices=["none", "hz50", "hz60"], default="hz50",
                            help="Ambient light frequency for Capture Assistant (UK mains = hz50)")
        parser.add_argument("--max-capture-ms", type=int, default=800,
                            help="Capture Assistant time budget in milliseconds")
        parser.add_argument("--pose-topic", default=DEFAULT_POSE_TOPIC,
                            help="PoseStamped topic with the robot EE pose (metres)")


def marker_dictionary(name: str):
    '''
    Inputs
    - `name`: dictionary name, e.g. "aruco6x6_50".

    Outputs
    - Returns the matching zivid.calibration.MarkerDictionary value.
    '''
    try:
        return getattr(zivid.calibration.MarkerDictionary, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown Zivid MarkerDictionary '{name}'") from exc


def make_settings(camera: zivid.Camera, args: argparse.Namespace, save_to: Path | None = None) -> zivid.Settings:
    '''
    Inputs
    - `camera`: connected Zivid camera.
    - `args`: parsed arguments (uses --settings, --ambient, --max-capture-ms).
    - `save_to`: optional .yml path where the settings actually used are saved.

    Outputs
    - Returns the zivid.Settings used for ALL captures of the session.
    '''
    if args.settings:
        settings = zivid.Settings.load(args.settings)
        print(f"[Settings] Loaded from {args.settings}")
    else:
        freq_enum = zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency
        params = zivid.capture_assistant.SuggestSettingsParameters(
            max_capture_time=timedelta(milliseconds=args.max_capture_ms),
            ambient_light_frequency=getattr(freq_enum, args.ambient),
        )
        print("[Settings] Running Capture Assistant once (keep the scene as during calibration)...")
        settings = zivid.capture_assistant.suggest_settings(camera, params)
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        settings.save(save_to)
        print(f"[Settings] Saved the settings used to {save_to}")
    return settings


def marker_pose_matrix(detection_result, marker_id: int) -> np.ndarray:
    '''
    Inputs
    - `detection_result`: output of zivid.calibration.detect_markers (valid).
    - `marker_id`: wanted ArUco ID.

    Outputs
    - Returns camera_T_marker as a 4x4 matrix in millimetres (marker-centred).
    '''
    markers = detection_result.detected_markers()
    if not markers:
        raise RuntimeError("detect_markers returned no markers")
    chosen = None
    for marker in markers:
        ident = getattr(marker, "identifier", None)
        ident = ident() if callable(ident) else ident
        if ident is None or int(ident) == int(marker_id):
            chosen = marker
            break
    if chosen is None:
        raise RuntimeError(f"Marker ID {marker_id} not among the detected markers")
    pose = chosen.pose
    pose = pose() if callable(pose) else pose
    try:
        return np.asarray(pose.to_matrix(), dtype=np.float64)
    except AttributeError as exc:
        raise RuntimeError(
            "Unexpected Zivid marker API: check `grep -rn detected_markers source/` "
            "in ~/zivid-python-samples (tag v2.17.2) and adapt marker_pose_matrix()."
        ) from exc


def ros_pose_to_matrix_mm(msg) -> np.ndarray:
    '''
    Inputs
    - `msg`: geometry_msgs/PoseStamped (metres).

    Outputs
    - Returns base_T_hand as a 4x4 matrix with translation in MILLIMETRES.
    '''
    p = msg.pose.position
    q = msg.pose.orientation
    q_xyzw = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    norm = np.linalg.norm(q_xyzw)
    if norm < 1e-12:
        raise ValueError("Received near-zero quaternion")
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q_xyzw / norm).as_matrix()
    T[:3, 3] = 1000.0 * np.array([p.x, p.y, p.z])
    return T


def invert_se3(T: np.ndarray) -> np.ndarray:
    '''Inverse of a rigid 4x4 transform.'''
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rot_angle_deg(Rm: np.ndarray) -> float:
    '''Rotation angle (degrees) of a 3x3 rotation matrix.'''
    c = float(np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def average_transforms(Ts: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    '''
    Inputs
    - `Ts`: list of 4x4 rigid transforms of the SAME physical quantity.

    Outputs
    - Mean transform (translation mean, rotation chordal mean via scipy),
      max translation deviation from the mean [same unit as input],
      max rotation deviation from the mean [deg].
    '''
    if not Ts:
        raise ValueError("No transforms to average")
    t = np.array([T[:3, 3] for T in Ts])
    rots = R.from_matrix(np.array([T[:3, :3] for T in Ts]))
    T_mean = np.eye(4)
    T_mean[:3, :3] = rots.mean().as_matrix()
    T_mean[:3, 3] = t.mean(axis=0)
    t_dev = float(np.max(np.linalg.norm(t - T_mean[:3, 3], axis=1)))
    r_dev = max(rot_angle_deg(T_mean[:3, :3].T @ T[:3, :3]) for T in Ts)
    return T_mean, t_dev, r_dev


def compare_transforms(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    '''Translation distance and rotation angle (deg) between two 4x4 transforms.'''
    d = invert_se3(A) @ B
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), rot_angle_deg(d[:3, :3])
