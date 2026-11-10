#!/usr/bin/env python3
"""Part 1: collect image/robot-pose pairs for Zivid eye-to-hand calibration.

The camera is fixed relative to the robot base, and one ArUco marker is fixed
to the gripper. Each SPACE press captures a 2D+3D Zivid frame, detects the
configured marker, and saves the frame with the latest robot EE pose. Part 2
later uses these pairs to estimate the camera pose in the robot base frame.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import rclpy
from datetime import timedelta
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import zivid
from zividsamples.save_load_matrix import assert_affine_matrix_and_save

import aruco_common


class EyeToBasePart1(Node):
    """Collect valid ArUco captures and matching robot poses through ROS."""

    def __init__(self, data_dir: str, args: argparse.Namespace):
        '''
        Inputs
        - `self`: node instance.
        - `data_dir`: directory for paired ZDF frames and robot poses.

        Outputs
        - No return value.
        - Connects to Zivid and ROS, prepares the preview and capture directory.
        '''
        super().__init__("zivid_eye_to_base_part1")

        # The cached ROS pose stays in metres until a sample is written to disk.
        self.data_dir = Path(data_dir)
        self.preview_window = "Zivid Eye-to-Base Capture"
        self.base_T_hand: np.ndarray | None = None
        self.last_capture_image: np.ndarray | None = None

        # The topic reports the configured robot EE pose relative to its base.
        # Keep the robot still while capturing: this node uses the latest cached pose.
        self.create_subscription(
            PoseStamped,
            args.pose_topic,
            self._link8_pose_cb,
            10,
        )
        self.get_logger().info(f"Subscribed to {args.pose_topic}")

        # Reuse one camera connection for every manual capture.
        self.app = zivid.Application()
        self.camera = self.app.connect_camera()
        self.get_logger().info(f"Connected to Zivid camera: {self.camera.info.model}")
        # Explain that acquisition is triggered by SPACE, not by the preview loop.
        self.get_logger().info("Part 1 captures only when SPACE is pressed.")

        # Marker to detect: same ID/dictionary in every part (see aruco_common).
        self.marker_id = args.marker_id
        self.marker_dict = aruco_common.marker_dictionary(args.dict)
        self.get_logger().info(f"Marker: ID {self.marker_id}, dictionary {args.dict}")

        # Continue numbering from existing ZDF files; use a separate folder for ArUco data.
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Settings are fixed ONCE for the whole session (repeatable captures) and
        # saved next to the dataset, so the same .yml can be reused later.
        self.settings = aruco_common.make_settings(
            self.camera, args, save_to=self.data_dir / "settings_used.yml"
        )
        self.next_index = self._next_capture_index()
        cv2.namedWindow(self.preview_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.preview_window, 1280, 720)

    def _link8_pose_cb(self, msg: PoseStamped) -> None:
        '''
        Inputs
        - `self`: node instance.
        - `msg`: incoming `PoseStamped` for the robot hand pose.

        Outputs
        - No return value.
        - Caches base_T_hand from the configured robot EE pose topic, in metres.
        '''
        self.base_T_hand = self._pose_msg_to_matrix(msg)

    def _pose_msg_to_matrix(self, msg: PoseStamped) -> np.ndarray:
        '''
        Inputs
        - `self`: node instance.
        - `msg`: `PoseStamped` message to convert.

        Outputs
        - Returns a 4x4 base_T_hand matrix; its translation remains in metres.
        '''
        # Split the incoming ROS pose into translation and quaternion parts.
        p = msg.pose.position
        q = msg.pose.orientation
        q_xyzw = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        q_norm = np.linalg.norm(q_xyzw)
        if q_norm < 1e-12:
            raise ValueError("Received near-zero quaternion from /robot2/franka_robot_state_broadcaster/current_pose")

        # ROS supplies an xyzw quaternion. Normalise it before forming the rotation.
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_quat(q_xyzw / q_norm).as_matrix()
        T[:3, 3] = [p.x, p.y, p.z]
        return T

    def _next_capture_index(self) -> int:
        '''
        Inputs
        - `self`: node instance.

        Outputs
        - Returns the next available numeric dataset index based on the existing saved captures.
        '''
        # Only ZDF filenames determine the next index; each complete pair shares one index.
        pattern = re.compile(r"^img(\d+)\.zdf$")
        indices = []
        for name in os.listdir(self.data_dir):
            match = pattern.match(name)
            if match:
                indices.append(int(match.group(1)))
        return max(indices, default=0) + 1

    def _draw_status(self, image: np.ndarray | None) -> np.ndarray:
        '''
        Inputs
        - `self`: node instance.
        - `image`: optional image to annotate for the operator preview.

        Outputs
        - Returns the preview image with capture-state and robot-pose status text overlaid.
        '''
        # Fall back to a blank panel when there is no recent capture image to show.
        if image is None:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = image.copy()
        lines = [
            f"Dataset: {self.data_dir}",
            f"Next index: {self.next_index:02d}",
            "SPACE: capture one ArUco sample",
            "No continuous camera preview is running",
            "ESC: exit",
        ]
        # Show the cached robot position in millimetres for the operator's preview.
        if self.base_T_hand is None:
            lines.append("Robot pose: waiting for /robot2/franka_robot_state_broadcaster/current_pose")
            color = (0, 0, 255)
        else:
            xyz_mm = 1000.0 * self.base_T_hand[:3, 3]
            lines.append(
                f"Robot pose ready: x={xyz_mm[0]:.1f} mm y={xyz_mm[1]:.1f} mm z={xyz_mm[2]:.1f} mm"
            )
            color = (0, 255, 0)

        y = 32
        for line in lines:
            cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            y += 32
        return out

    def _save_dataset_pair(self, index: int, frame: zivid.Frame, robot_pose: np.ndarray) -> None:
        '''
        Inputs
        - `self`: node instance.
        - `index`: dataset index to save.
        - `frame`: captured Zivid frame.
        - `robot_pose`: matching base_T_hand, with translation already in mm.

        Outputs
        - No return value.
        - Writes the ZDF frame and corresponding robot pose files for one calibration sample.
        '''
        # Part 2 reads imgNN.zdf and posNN.yaml. The NPY pose is an extra copy.
        img_path = self.data_dir / f"img{index:02d}.zdf"
        pos_path = self.data_dir / f"pos{index:02d}.yaml"
        robot_pose_path = self.data_dir / f"base_T_hand_{index:02d}.npy"

        frame.save(img_path)
        assert_affine_matrix_and_save(robot_pose, pos_path)
        np.save(robot_pose_path, robot_pose)

    def _frame_to_bgr(self, frame: zivid.Frame) -> np.ndarray | None:
        '''
        Inputs
        - `self`: node instance.
        - `frame`: Zivid frame to convert for OpenCV display.

        Outputs
        - Returns a BGR preview image extracted from the Zivid frame, or `None` if conversion fails.
        '''
        # This preview conversion does not change the ZDF frame saved for calibration.
        try:
            bgra = frame.point_cloud().copy_data("bgra")
            if bgra is None or bgra.ndim != 3 or bgra.shape[2] < 3:
                return None
            return np.ascontiguousarray(bgra[:, :, :3])
        except Exception:
            return None

    def _capture_dataset_pair(self) -> bool:
        '''
        Inputs
        - `self`: node instance.

        Outputs
        - Returns True only when the ArUco is detected and the pair is saved.
        '''
        if self.base_T_hand is None:
            self.get_logger().warning("Robot pose not available yet. Waiting for /robot2/franka_robot_state_broadcaster/current_pose.")
            return False

        # Capture with the session settings fixed in __init__.
        frame = self.camera.capture_2d_3d(self.settings)

        # Same marker ID and dictionary as Parts 2 and 3.
        marker_id = self.marker_id
        marker_dict = self.marker_dict
        detection_result = zivid.calibration.detect_markers(frame, [marker_id], marker_dict)

        if not detection_result.valid():
            self.last_capture_image = self._draw_status(self._frame_to_bgr(frame))
            self.get_logger().warning(
                f"ArUco ID {marker_id} non rilevato (dizionario: {marker_dict})"
            )
            return False

        # ROS translation is in metres; Zivid marker poses and solver inputs use
        # millimetres. The rotation and the cached ROS pose are left unchanged.
        robot_pose = self.base_T_hand.copy()
        robot_pose[:3, 3] *= 1000.0
        # Advance the index only after a valid image/pose pair has been saved.
        self._save_dataset_pair(self.next_index, frame, robot_pose)
        self.last_capture_image = self._draw_status(self._frame_to_bgr(frame))
        self.get_logger().info(f"Saved sample {self.next_index:02d} to {self.data_dir}")
        self.next_index += 1
        return True

    def collect_data(self) -> None:
        '''
        Inputs
        - `self`: node instance.

        Outputs
        - No return value.
        - Runs the interactive loop until ESC or ROS shutdown.
        '''
        self.get_logger().info(
            f"Writing Zivid eye-to-hand dataset to {self.data_dir}. "
            "Each SPACE press saves one sample as imgNN.zdf and posNN.yaml."
        )
        try:
            # spin_once updates the cached pose; the loop itself does not capture.
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)
                # Redraw the last captured image and current ROS-pose status.
                display = self._draw_status(self.last_capture_image)
                cv2.imshow(self.preview_window, display)
                key = cv2.waitKey(1) & 0xFF

                # SPACE attempts one sample; a failed marker detection saves nothing.
                if key == 32:
                    self._capture_dataset_pair()
                elif key == 27:
                    break
        finally:
            cv2.destroyAllWindows()


def _options() -> argparse.Namespace:
    '''
    Inputs
    - No input parameters.

    Outputs
    - Returns the parsed command-line arguments for the Part 1 capture script.
    '''
    # The default path is relative to the process's working directory.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="./zivid_eye_to_base_calib_data",
        help="Directory where imgNN.zdf and posNN.yaml files will be stored",
    )
    aruco_common.add_common_args(parser, with_capture=True)
    return parser.parse_args()


def main(args: list[str] | None = None) -> None:
    '''
    Inputs
    - `args`: optional ROS command-line argument list.

    Outputs
    - No return value.
    - Parses arguments, initialises ROS, runs the Zivid capture node, and shuts it down cleanly.
    '''
    # Create and destroy the ROS node around the interactive capture session.
    cli_args = _options()
    rclpy.init(args=None)
    node: EyeToBasePart1 | None = None
    try:
        node = EyeToBasePart1(data_dir=cli_args.data_dir, args=cli_args)
        node.collect_data()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
