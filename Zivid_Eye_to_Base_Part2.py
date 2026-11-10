#!/usr/bin/env python3
"""Part 2: estimate the fixed Zivid camera pose from saved ArUco samples.

This stage uses the image/robot-pose pairs collected by Part 1. It re-detects
the marker in each ZDF frame, constructs Zivid HandEyeInput pairs, and runs
eye-to-hand calibration. It does not subscribe to ROS or capture live images.
The result base_T_cam is written only after Zivid reports a valid solution.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import zivid
from zividsamples.save_load_matrix import assert_affine_matrix_and_save, load_and_assert_affine_matrix

import aruco_common


class EyeToBasePart2:
    """Solve eye-to-hand calibration using saved ArUco frames and robot poses."""

    def __init__(self, data_dir: str, marker_id: int = aruco_common.DEFAULT_MARKER_ID, dict_name: str = aruco_common.DEFAULT_DICTIONARY):
        '''
        Inputs
        - `self`: solver instance.
        - `data_dir`: directory containing the saved Zivid calibration captures.

        Outputs
        - No return value.
        - Defines the input dataset and distinct ArUco output filenames.
        '''
        # The NPY output is relative to the working directory; YAML stays with the dataset.
        self.data_dir = Path(data_dir)
        self.output_dir = Path("calibration_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_transform_npy = self.output_dir / "T_cam_to_base_Zivid_ArUco.npy"
        self.output_transform_yaml = self.data_dir / "handEyeTransform_ArUco.yaml"
        # Same marker ID and dictionary as Parts 1 and 3.
        self.marker_id = marker_id
        self.marker_dict = aruco_common.marker_dictionary(dict_name)

    def _paired_dataset_indices(self) -> list[int]:
        '''
        Inputs
        - `self`: solver instance.

        Outputs
        - Returns the sorted dataset indices that have both a saved ZDF frame and a matching robot-pose YAML file.
        '''
        # Ignore incomplete captures: only indices in both filename sets are usable.
        img_pattern = re.compile(r"^img(\d+)\.zdf$")
        pos_pattern = re.compile(r"^pos(\d+)\.yaml$")
        img_indices = set()
        pos_indices = set()

        if not self.data_dir.is_dir():
            return []

        for path in self.data_dir.iterdir():
            img_match = img_pattern.match(path.name)
            if img_match:
                img_indices.add(int(img_match.group(1)))
                continue

            pos_match = pos_pattern.match(path.name)
            if pos_match:
                pos_indices.add(int(pos_match.group(1)))

        return sorted(img_indices & pos_indices)

    def _build_hand_eye_input(self) -> tuple[list[int], list[zivid.calibration.HandEyeInput]]:
        '''
        Inputs
        - `self`: solver instance.

        Outputs
        - Returns valid sample indices and their Zivid HandEyeInput objects.
        '''
        # Part 1 uses the same numeric index for each image/robot-pose pair.
        indices = self._paired_dataset_indices()
        if not indices:
            raise RuntimeError(f"No imgNN.zdf / posNN.yaml pairs found in {self.data_dir}")

        valid_indices: list[int] = []
        hand_eye_input: list[zivid.calibration.HandEyeInput] = []
        skipped = 0
        # Re-run ArUco detection on each ZDF frame to rebuild the detection
        # objects that Zivid's HandEyeInput needs; Part 1 saved only frames/poses.
        for index in indices:
            img_path = self.data_dir / f"img{index:02d}.zdf"
            pos_path = self.data_dir / f"pos{index:02d}.yaml"

            frame = zivid.Frame(img_path)
            # Match the ID and dictionary configured in Parts 1 and 3.
            marker_id = self.marker_id
            marker_dict = self.marker_dict
            detection_result = zivid.calibration.detect_markers(frame, [marker_id], marker_dict)

            if not detection_result.valid():
                skipped += 1
                print(
                    f"ArUco ID {marker_id} non rilevato (dizionario: {marker_dict})"
                )
                continue

            # Part 1 saved base_T_hand in millimetres. Pass the entire marker
            # detection result to Zivid; there is no manual camera-pose averaging.
            robot_pose_matrix = np.asarray(load_and_assert_affine_matrix(pos_path), dtype=np.float64)
            robot_pose = zivid.calibration.Pose(robot_pose_matrix)
            valid_indices.append(index)
            hand_eye_input.append(zivid.calibration.HandEyeInput(robot_pose, detection_result))

        print(
            f"Loaded {len(hand_eye_input)} valid calibration pairs from {len(indices)} dataset samples"
            + (f" ({skipped} skipped)" if skipped else "")
        )
        return valid_indices, hand_eye_input

    def run(self) -> None:
        '''
        Inputs
        - `self`: solver instance.

        Outputs
        - No return value.
        - Solves base_T_cam, saves it when valid, and reports diagnostics.
        '''
        # Keep the SDK application alive while loading ZDF files and solving.
        _app = zivid.Application()
        valid_indices, hand_eye_input = self._build_hand_eye_input()
        # Enforce the six-pair minimum here; Zivid checks pose diversity during
        # the solve, so six nearly identical poses may still be insufficient.
        if len(hand_eye_input) < 6:
            raise RuntimeError("Need at least 6 valid ArUco detections for eye-to-hand calibration.")

        # The camera is fixed in the scene, hence eye-to-hand calibration.
        calibration_result = zivid.calibration.calibrate_eye_to_hand(hand_eye_input)
        if not calibration_result.valid():
            print("Hand-eye calibration FAILED")
            print(calibration_result)
            return

        base_T_cam = np.asarray(calibration_result.transform(), dtype=np.float64)

        # Never overwrite the previous board calibration: these names identify
        # the ArUco result. Part 3 can load either output format.
        np.save(self.output_transform_npy, base_T_cam)
        assert_affine_matrix_and_save(base_T_cam, self.output_transform_yaml)



        print("Hand-eye calibration OK")
        print("Result is the camera pose in robot base frame (base_T_cam):")
        print(base_T_cam)
        print(f"Saved NumPy transform to {self.output_transform_npy}")
        print(f"Saved YAML transform to {self.output_transform_yaml}")

        # Residuals diagnose the fitted samples; verification on new poses is
        # still performed separately by Part 3.
        residuals = calibration_result.residuals()
        if residuals:
            print("Residuals:")
            for sample_index, residual in zip(valid_indices, residuals):
                print(
                    f"  {sample_index:02d}: "
                    f"translation={residual.translation()} rotation={residual.rotation()}"
                )


def _options() -> argparse.Namespace:
    '''
    Inputs
    - No input parameters.

    Outputs
    - Returns the parsed command-line arguments for the Part 2 solve script.
    '''
    # Point this at the same dataset directory used for Part 1.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./zivid_eye_to_base_calib_data",
        help="Directory containing imgNN.zdf and posNN.yaml calibration captures",
    )
    aruco_common.add_common_args(parser, with_capture=False)
    return parser.parse_args()


def main() -> None:
    '''
    Inputs
    - No input parameters.

    Outputs
    - No return value.
    - Parses arguments and runs the Zivid eye-to-base calibration solver.
    '''
    # This stage runs once on saved data and needs no ROS node.
    args = _options()
    EyeToBasePart2(data_dir=args.data_dir, marker_id=args.marker_id, dict_name=args.dict).run()


if __name__ == "__main__":
    main()
