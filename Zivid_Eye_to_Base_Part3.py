#!/usr/bin/env python3
"""Part 3: check an ArUco eye-to-hand calibration on new robot poses.

Load the base_T_cam transform saved by Part 2, detect the marker in new Zivid
captures, and combine each detection with the latest robot EE pose. A marker
fixed to the gripper should keep the same hand_T_marker across robot poses.
The first valid capture is this session's comparison baseline. At the end,
the mean hand_T_marker (Y) over all valid captures is saved (--y-out): it is
the input of Zivid_Eye_to_Base_QuickCalib.py (one-shot recalibration).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from datetime import timedelta
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import zivid
from zividsamples.save_load_matrix import load_and_assert_affine_matrix

import aruco_common


class EyeToBasePart3(Node):
    """Monitor marker-pose consistency while the robot moves between poses."""

    def __init__(self, transform_path: str, capture_period_s: float, args: argparse.Namespace):
        '''
        Inputs
        - `self`: node instance.
        - `transform_path`: path to the saved Zivid calibration transform.
        - `capture_period_s`: seconds between verification captures.

        Outputs
        - No return value.
        - Prepares the display, robot-pose subscription, and Zivid camera.
        '''
        super().__init__("zivid_eye_to_base_part3")

        # Keep the Part 2 transform path and the interval for live checks.
        self.transform_path = Path(transform_path)
        self.capture_period_s = float(capture_period_s)
        self.base_T_cam: np.ndarray | None = None
        self.base_T_hand: np.ndarray | None = None
        self.last_detected_image: np.ndarray | None = None
        self.preview_window = "Zivid Eye-to-Base Verification"
        self.preview_size = (1280, 720)
        self.latest_status_line = "Waiting for first valid capture"
        self.latest_capture_time = 0.0

        # A fixed-size panel makes the status and last captured frame readable.
        cv2.namedWindow(self.preview_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.preview_window, self.preview_size[0], self.preview_size[1])

        # This topic supplies the configured EE pose relative to the robot base.
        # Captures should be taken only after the robot has stopped moving.
        self.create_subscription(
            PoseStamped,
            args.pose_topic,
            self._link8_pose_cb,
            10,
        )
        self.get_logger().info(f"Subscribed to {args.pose_topic}")

        # Use one camera connection throughout this verification session.
        self.app = zivid.Application()
        self.camera = self.app.connect_camera()
        self.get_logger().info(f"Connected to Zivid camera: {self.camera.info.model}")

        # Same marker as Parts 1 and 2; settings fixed once for the session.
        self.marker_id = args.marker_id
        self.marker_dict = aruco_common.marker_dictionary(args.dict)
        self.settings = aruco_common.make_settings(self.camera, args)
        self.y_out = Path(args.y_out)

    def _link8_pose_cb(self, msg: PoseStamped) -> None:
        '''
        Inputs
        - `self`: node instance.
        - `msg`: incoming `PoseStamped` for the robot hand pose.

        Outputs
        - No return value.
        - Caches base_T_hand with translation in metres.
        '''
        self.base_T_hand = self._pose_msg_to_matrix(msg)

    def _pose_msg_to_matrix(self, msg: PoseStamped) -> np.ndarray:
        '''
        Inputs
        - `self`: node instance.
        - `msg`: `PoseStamped` message to convert.

        Outputs
        - Returns base_T_hand as a 4x4 matrix with translation in metres.
        '''
        # Split the incoming ROS pose into translation and quaternion parts.
        p = msg.pose.position
        q = msg.pose.orientation
        q_xyzw = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        q_norm = np.linalg.norm(q_xyzw)
        if q_norm < 1e-12:
            raise ValueError("Received near-zero quaternion from /robot2/franka_robot_state_broadcaster/current_pose")

        # ROS uses xyzw quaternion ordering; normalise before conversion.
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_quat(q_xyzw / q_norm).as_matrix()
        T[:3, 3] = [p.x, p.y, p.z]
        return T

    def _invert_se3(self, T: np.ndarray) -> np.ndarray:
        '''
        Inputs
        - `self`: node instance.
        - `T`: homogeneous transform to invert.

        Outputs
        - Returns the inverse rigid transform, preserving its length unit.
        '''
        # For T=[R,t], T^-1=[R^T,-R^T t]. No general matrix inverse is needed.
        Rm = T[:3, :3]
        t = T[:3, 3]
        Ti = np.eye(4, dtype=np.float64)
        Ti[:3, :3] = Rm.T
        Ti[:3, 3] = -Rm.T @ t
        return Ti

    def _rot_err_deg(self, Rm: np.ndarray) -> float:
        '''
        Inputs
        - `self`: node instance.
        - `Rm`: relative rotation matrix to score.

        Outputs
        - Returns the angle of a relative rotation matrix in degrees.
        '''
        # Clamp round-off before arccos, whose argument must stay in [-1, 1].
        tr = float(np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0))
        return float(np.degrees(np.arccos(tr)))

    def _frame_to_bgr(self, frame: zivid.Frame) -> np.ndarray | None:
        '''
        Inputs
        - `self`: node instance.
        - `frame`: Zivid frame to convert for OpenCV display.

        Outputs
        - Returns a BGR preview image extracted from the Zivid frame, or `None` if conversion fails.
        '''
        # Preview conversion does not change the marker pose returned by Zivid.
        try:
            bgra = frame.point_cloud().copy_data("bgra")
            if bgra is None or bgra.ndim != 3 or bgra.shape[2] < 3:
                return None
            return np.ascontiguousarray(bgra[:, :, :3])
        except Exception:
            return None

    def _fit_to_panel(self, image: np.ndarray | None) -> np.ndarray:
        '''
        Inputs
        - `self`: node instance.
        - `image`: optional image to resize for the preview panel.

        Outputs
        - Returns the image resized and padded to the configured preview size.
        '''
        # Preserve aspect ratio and pad unused space in the preview panel.
        panel_w, panel_h = self.preview_size
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        if image is None:
            return panel

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return panel

        scale = min(panel_w / float(w), panel_h / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x0 = (panel_w - new_w) // 2
        y0 = (panel_h - new_h) // 2
        panel[y0:y0 + new_h, x0:x0 + new_w] = resized
        return panel

    def _draw_overlay(self, image: np.ndarray | None, seconds_remaining: float) -> np.ndarray:
        '''
        Inputs
        - `self`: node instance.
        - `image`: optional image to annotate.
        - `seconds_remaining`: time remaining before the next capture.

        Outputs
        - Returns the preview image with current verification status text overlaid.
        '''
        # Show which Part 2 transform is loaded and when the next capture is due.
        panel = self._fit_to_panel(image)
        pose_state = "ready" if self.base_T_hand is not None else "waiting for /robot2/franka_robot_state_broadcaster/current_pose"
        lines = [
            f"Transform: {self.transform_path}",
            f"Robot pose: {pose_state}",
            f"Next capture in: {max(0.0, seconds_remaining):.1f} s",
            self.latest_status_line,
            "ESC: exit",
        ]
        y = 34
        for line in lines:
            cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            y += 34
        return panel

    def _show_preview(self, image: np.ndarray | None, seconds_remaining: float) -> int:
        '''
        Inputs
        - `self`: node instance.
        - `image`: optional image to display.
        - `seconds_remaining`: time remaining before the next capture.

        Outputs
        - Returns the pressed key code from the preview window.
        '''
        # ESC is handled by the calling verification loop.
        preview = self._draw_overlay(image, seconds_remaining)
        cv2.imshow(self.preview_window, preview)
        return cv2.waitKey(1) & 0xFF

    def _load_transform_matrix(self) -> np.ndarray:
        '''
        Inputs
        - `self`: node instance.

        Outputs
        - Returns the saved 4x4 base_T_cam transform, with translation in mm.
        '''
        # Part 2 writes both formats; select the loader by file extension.
        if not self.transform_path.exists():
            raise FileNotFoundError(f"Transform file not found: {self.transform_path}")

        if self.transform_path.suffix.lower() == ".yaml":
            T = load_and_assert_affine_matrix(self.transform_path)
        elif self.transform_path.suffix.lower() == ".npy":
            T = np.load(self.transform_path)
        else:
            raise ValueError(f"Unsupported transform file type: {self.transform_path}")

        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 transform in {self.transform_path}, got {T.shape}")
        return T

    def _draw_detected_points(self, image: np.ndarray, detection_result: zivid.calibration.DetectionResult) -> np.ndarray:
        '''
        Inputs
        - `self`: node instance.
        - `image`: image to annotate.
        - `detection_result`: a legacy calibration-board result.

        Outputs
        - Returns a board-feature preview; this helper is unused for ArUco.
        '''
        # Retained from the board workflow; ArUco captures do not call it.
        out = image.copy()
        try:
            pts = np.asarray(detection_result.feature_points_2d(), dtype=np.float64)
            valid_mask = np.isfinite(pts).all(axis=2)
            for uv in pts[valid_mask]:
                u = int(round(float(uv[0])))
                v = int(round(float(uv[1])))
                cv2.circle(out, (u, v), 4, (0, 255, 255), -1, cv2.LINE_AA)
        except Exception:
            pass
        return out

    def _capture_measurement(self) -> tuple[np.ndarray, np.ndarray] | None:
        '''
        Inputs
        - `self`: node instance.

        Outputs
        - Returns camera_T_marker (mm) and cached base_T_hand (m), or None.
        '''
        # A camera observation needs a corresponding robot EE pose.
        if self.base_T_hand is None:
            self.latest_status_line = "Waiting for robot pose"
            return None

        # Capture with the session settings fixed in __init__.
        frame = self.camera.capture_2d_3d(self.settings)

        # Same printed marker ID and dictionary as Parts 1 and 2.
        detection_result = zivid.calibration.detect_markers(frame, [self.marker_id], self.marker_dict)
        image = self._frame_to_bgr(frame)
        if image is None:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)

        if not detection_result.valid():
            self.last_detected_image = image
            self.latest_status_line = f"ArUco not detected"
            return None
        
        # A MarkerShape pose is centred on the marker. Only one ID is allowed,
        # so the first detected marker is the intended one.
        camera_T_marker = aruco_common.marker_pose_matrix(detection_result, self.marker_id)
        self.last_detected_image = image
        return camera_T_marker, self.base_T_hand.copy()

    def _print_error_status(self, sample_count: int, pos_mean_mm: float, pos_latest_mm: float, rot_mean_deg: float, rot_latest_deg: float) -> None:
        '''
        Inputs
        - `self`: node instance.
        - `sample_count`: number of accumulated comparison samples.
        - `pos_mean_mm`: mean marker-centre drift relative to the first capture.
        - `pos_latest_mm`: most recent marker-centre drift.
        - `rot_mean_deg`: mean orientation drift relative to the first capture.
        - `rot_latest_deg`: most recent orientation drift.

        Outputs
        - No return value.
        - Prints and caches the rolling verification status line.
        '''
        # Use one updating terminal line when interactive; throttle output to 5 Hz.
        status = (
            f"N={sample_count} | "
            f"pos_mean={pos_mean_mm:.3f} mm | pos_latest={pos_latest_mm:.3f} mm | "
            f"rot_mean={rot_mean_deg:.4f} deg | rot_latest={rot_latest_deg:.4f} deg"
        )
        self.latest_status_line = status
        now_s = time.time()
        last_any_s = getattr(self, "_reach_status_last_any_emit_s", 0.0)
        if now_s - last_any_s < 0.2:
            return
        self._reach_status_last_any_emit_s = now_s

        if sys.stdout.isatty():
            prev_len = getattr(self, "_reach_status_prev_len", 0)
            pad = " " * max(0, prev_len - len(status))
            sys.stdout.write("\r" + status + pad)
            sys.stdout.flush()
            self._reach_status_prev_len = len(status)
        else:
            print(status, flush=True)

    def verify_found_transforms(self) -> None:
        '''
        Inputs
        - `self`: node instance.

        Outputs
        - No return value.
        - Reports whether hand_T_marker stays constant across new robot poses.
        '''
        # Keep the camera-to-base estimate fixed during the whole check.
        if self.base_T_cam is None:
            self.base_T_cam = self._load_transform_matrix()
            print(f"[Verification] Loaded base_T_cam from: {self.transform_path}")
            print(self.base_T_cam)

        # The first valid observation defines a baseline, not a ground-truth
        # hand-to-marker transform. This script does not persist that baseline.
        baseline_hand_T_marker = None
        all_hand_T_marker: list[np.ndarray] = []  # every valid estimate of Y (mm)
        pos_errors_mm: list[float] = []
        rot_errors_deg: list[float] = []

        print(
            "\n[Verification] Capturing one Zivid ArUco frame every "
            f"{self.capture_period_s:.1f} seconds. Press ESC to stop."
        )

        try:
            # Process ROS callbacks between captures and leave the preview responsive.
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)
                now_s = time.time()
                seconds_remaining = self.capture_period_s - (now_s - self.latest_capture_time)

                if self._show_preview(self.last_detected_image, seconds_remaining) == 27:
                    break

                if now_s - self.latest_capture_time < self.capture_period_s:
                    continue

                self.latest_capture_time = now_s
                # Failed detection or missing robot telemetry produces no sample.
                measurement = self._capture_measurement()
                if measurement is None:
                    continue

                camera_T_marker, base_T_hand = measurement
                base_T_hand_mm = base_T_hand.copy()
                base_T_hand_mm[:3, 3] *= 1000.0

                # ROS pose uses metres while Zivid base_T_cam and camera_T_marker
                # use millimetres. Convert only translation before composition.
                base_T_marker = self.base_T_cam @ camera_T_marker
                # hand_T_marker = inverse(base_T_hand) @ base_T_cam @ camera_T_marker.
                hand_T_marker = self._invert_se3(base_T_hand_mm) @ base_T_marker
                all_hand_T_marker.append(hand_T_marker.copy())

                if baseline_hand_T_marker is None:
                    baseline_hand_T_marker = hand_T_marker.copy()
                    self.latest_status_line = "Baseline captured from first valid Zivid detection"
                    print("[Verification] Baseline hand_T_marker captured from first valid estimate")
                    continue

                # The translation norm measures marker-centre drift (mm); the
                # relative rotation angle measures orientation drift (degrees).
                dT = self._invert_se3(baseline_hand_T_marker) @ hand_T_marker
                pos_err_mm = float(np.linalg.norm(dT[:3, 3]))
                rot_err_deg = self._rot_err_deg(dT[:3, :3])

                pos_errors_mm.append(pos_err_mm)
                rot_errors_deg.append(rot_err_deg)

                self._print_error_status(
                    sample_count=len(pos_errors_mm),
                    pos_mean_mm=float(np.mean(pos_errors_mm)),
                    pos_latest_mm=pos_err_mm,
                    rot_mean_deg=float(np.mean(rot_errors_deg)),
                    rot_latest_deg=rot_err_deg,
                )
        finally:
            # Release the preview even on interruption and summarize valid pairs.
            cv2.destroyAllWindows()
            if sys.stdout.isatty() and getattr(self, "_reach_status_prev_len", 0) > 0:
                sys.stdout.write("\n")
                sys.stdout.flush()

            if pos_errors_mm:
                print(
                    "[Verification Summary] "
                    f"samples={len(pos_errors_mm)} | "
                    f"pos mean={np.mean(pos_errors_mm):.3f} mm, median={np.median(pos_errors_mm):.3f} mm, max={np.max(pos_errors_mm):.3f} mm | "
                    f"rot mean={np.mean(rot_errors_deg):.4f} deg, median={np.median(rot_errors_deg):.4f} deg, max={np.max(rot_errors_deg):.4f} deg"
                )
            else:
                print("[Verification Summary] No valid comparison samples were accumulated.")

            # Save Y = mean hand_T_marker for the one-shot recalibration.
            # Only meaningful if the robot was MOVED between captures.
            if len(all_hand_T_marker) >= 3:
                Y, t_dev, r_dev = aruco_common.average_transforms(all_hand_T_marker)
                self.y_out.parent.mkdir(parents=True, exist_ok=True)
                np.save(self.y_out, Y)
                print(f"[Y] Saved mean hand_T_marker (mm) from {len(all_hand_T_marker)} captures to {self.y_out}")
                print(f"[Y] max deviation from mean: {t_dev:.3f} mm, {r_dev:.4f} deg")
                print(Y)
            else:
                print("[Y] Not saved: need at least 3 valid captures.")


def _options() -> argparse.Namespace:
    '''
    Inputs
    - No input parameters.

    Outputs
    - Returns the parsed command-line arguments for the Part 3 verification script.
    '''
    # The default YAML path is relative to the working directory. Override it
    # when Part 1 used a different dataset folder.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transform-path",
        type=str,
        default="./zivid_eye_to_base_calib_data/handEyeTransform_ArUco.yaml",
        help="Path to the base_T_cam transform saved by Zivid_Eye_to_Base_Part2",
    )
    parser.add_argument(
        "--capture-period",
        type=float,
        default=2.0,
        help="Seconds between Zivid verification captures",
    )
    parser.add_argument(
        "--y-out",
        default="./calibration_outputs/hand_T_marker_ArUco.npy",
        help="Where to save the mean hand_T_marker (Y, mm) used by QuickCalib",
    )
    aruco_common.add_common_args(parser, with_capture=True)
    return parser.parse_args()


def main(args: list[str] | None = None) -> None:
    '''
    Inputs
    - `args`: optional ROS command-line argument list.

    Outputs
    - No return value.
    - Parses arguments, initialises ROS, runs the verification node, and shuts it down cleanly.
    '''
    # Run the ROS node and release resources on normal exit or interruption.
    cli_args = _options()
    rclpy.init(args=None)
    node: EyeToBasePart3 | None = None
    try:
        node = EyeToBasePart3(
            transform_path=cli_args.transform_path,
            capture_period_s=cli_args.capture_period,
            args=cli_args,
        )
        node.verify_found_transforms()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
