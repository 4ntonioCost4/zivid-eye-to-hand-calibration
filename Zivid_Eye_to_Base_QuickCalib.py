#!/usr/bin/env python3
"""QuickCalib: one-shot recalibration of base_T_cam with the ArUco on the gripper.

Requires Y = hand_T_marker (mm), saved once by Part3 (--y-out) after a full
Part1 -> Part2 -> Part3 calibration. Valid as long as the marker has NOT moved
on the gripper and the gripper has NOT been unmounted, and the Desk EE
configuration is the same as during the full calibration.

For each capture (robot standing still, marker visible):
    base_T_cam = base_T_hand (robot) · Y · inverse(cam_T_marker) (camera)
The captures are averaged; their spread tells how reliable the result is.

Usage (PC GPU, venv zivid, ROS sourced, robot still):
    python3 Zivid_Eye_to_Base_QuickCalib.py --y ./calibration_outputs/hand_T_marker_ArUco.npy \
        --compare ./calibration_outputs/T_cam_to_base_Zivid_ArUco.npy
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
import zivid

import aruco_common


class QuickCalib(Node):
    """Capture a few frames, compute base_T_cam from Y, average and save."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("zivid_quick_calib")
        self.args = args
        self.base_T_hand_mm: np.ndarray | None = None
        self.create_subscription(PoseStamped, args.pose_topic, self._pose_cb, aruco_common.pose_qos(args.reliable))
        self.get_logger().info(f"Subscribed to {args.pose_topic}")

        self.Y = np.load(args.y)
        if self.Y.shape != (4, 4):
            raise ValueError(f"{args.y} is not a 4x4 matrix")

        self.app = zivid.Application()
        self.camera = self.app.connect_camera()
        self.marker_id = args.marker_id
        self.marker_dict = aruco_common.marker_dictionary(args.dict)
        self.settings = aruco_common.make_settings(self.camera, args)

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.base_T_hand_mm = aruco_common.ros_pose_to_matrix_mm(msg)

    def _wait_for_pose(self, timeout_s: float = 10.0) -> None:
        t0 = time.time()
        while self.base_T_hand_mm is None and time.time() - t0 < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.base_T_hand_mm is None:
            raise RuntimeError(f"No robot pose received on {self.args.pose_topic}")

    def run(self) -> None:
        self._wait_for_pose()
        estimates: list[np.ndarray] = []
        print(f"[QuickCalib] Robot must stay STILL. Taking {self.args.shots} capture(s)...")
        for i in range(self.args.shots):
            # Latest robot pose just before the capture (robot is not moving).
            rclpy.spin_once(self, timeout_sec=0.05)
            base_T_hand = self.base_T_hand_mm.copy()
            frame = self.camera.capture_2d_3d(self.settings)
            detection = zivid.calibration.detect_markers(frame, [self.marker_id], self.marker_dict)
            if not detection.valid():
                print(f"  shot {i + 1}: marker ID {self.marker_id} NOT detected")
                continue
            cam_T_marker = aruco_common.marker_pose_matrix(detection, self.marker_id)
            base_T_cam = base_T_hand @ self.Y @ aruco_common.invert_se3(cam_T_marker)
            estimates.append(base_T_cam)
            print(f"  shot {i + 1}: t = {np.round(base_T_cam[:3, 3], 2)} mm")

        if not estimates:
            print("[QuickCalib] FAILED: marker never detected.")
            return

        base_T_cam, t_dev, r_dev = aruco_common.average_transforms(estimates)
        print(f"\n[QuickCalib] base_T_cam (mm), mean of {len(estimates)} capture(s):")
        print(base_T_cam)
        if len(estimates) > 1:
            print(f"[QuickCalib] spread: max {t_dev:.3f} mm, {r_dev:.4f} deg")

        if self.args.compare:
            ref = np.load(self.args.compare)
            dt, dr = aruco_common.compare_transforms(ref, base_T_cam)
            print(f"[QuickCalib] vs {self.args.compare}: {dt:.2f} mm, {dr:.3f} deg")
            print("            (a few mm / < 0.5 deg: camera unchanged; more: camera moved or Y invalid)")

        out = Path(self.args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, base_T_cam)
        stamped = out.with_name(f"{out.stem}_{datetime.now():%Y%m%d_%H%M%S}{out.suffix}")
        np.save(stamped, base_T_cam)
        print(f"[QuickCalib] Saved to {out} (and copy {stamped.name})")


def _options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--y", default="./calibration_outputs/hand_T_marker_ArUco.npy",
                        help="hand_T_marker (mm) saved by Part3")
    parser.add_argument("--shots", type=int, default=5,
                        help="Number of captures to average (robot still)")
    parser.add_argument("--compare", default=None,
                        help="Optional reference base_T_cam .npy (mm) to compare with")
    parser.add_argument("--out", default="./calibration_outputs/T_cam_to_base_Zivid_quick.npy",
                        help="Output base_T_cam (mm)")
    aruco_common.add_common_args(parser, with_capture=True)
    return parser.parse_args()


def main() -> None:
    args = _options()
    rclpy.init(args=None)
    node: QuickCalib | None = None
    try:
        node = QuickCalib(args)
        node.run()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
