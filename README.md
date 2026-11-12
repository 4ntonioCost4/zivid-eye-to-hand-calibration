# Zivid eye-to-hand calibration with an ArUco on the gripper

Calibration of a **fixed Zivid camera** (Zivid 2+ MR60) with respect to the **base of a Franka FR3**, using one ArUco marker glued permanently on the gripper instead of the Zivid calibration board. No need to unmount the gripper.

Result: `base_T_cam`, a 4x4 transform in **millimetres** (camera pose in the robot base frame).

Based on the Zivid hand-eye API (`detect_markers` + `calibrate_eye_to_hand`) and on the PaCMate `camera_calibration` scripts.

## Requirements

- Zivid SDK + `zivid` Python package **with the same version** (lab: 2.17.2), `zividsamples` (from `zivid-python-samples`, same tag), numpy < 2, scipy, opencv-python.
- ROS 2 Humble (`rclpy`), robot EE pose published as `geometry_msgs/PoseStamped` in metres (default topic `/robot1/franka_robot_state_broadcaster/current_pose`, change with `--pose-topic`; subscribed BEST_EFFORT, `--reliable` to change).
- Lab PC GPU: `source /opt/ros/humble/setup.bash && source ~/venvs/zivid/bin/activate`

## Automatic calibration with the Zivid board (`auto_calibration_board.py`)

One script = collection + solve + verification + score, with the **Zivid calibration board** mounted on the gripper (no marker needed). Pose slots are in `calibration_poses.yaml` (FR3 joint configurations).

```bash
# PC GPU, venv zivid, ROS sourced, driver running with torque_subscriber_controller ACTIVE
python3 $S/auto_calibration_board.py check  --poses $S/calibration_poses.yaml   # no robot: limits, steps, duration
python3 $S/auto_calibration_board.py record --poses $S/calibration_poses.yaml   # 1st time: hand-guided, ENTER per pose
python3 $S/auto_calibration_board.py run    --poses $S/calibration_poses.yaml   # next times: fully automatic
```

- `record`: no torques are published (gravity compensation). ENTER = calibration pose, `v`+ENTER = verification pose, `q`+ENTER = finish. If the score is PASS, the script asks to write the joint configurations into the slots.
- `run`: moves through the saved poses (joint impedance, min-jerk, `--vmax` 0.2 rad/s), captures, solves, verifies, prints the score. Ctrl+C = torques released.
- Score (PASS): residual mean <= 1.5 mm, max <= 3 mm, rotation mean <= 0.3 deg, verification <= 3 mm / 0.6 deg (all configurable).
- Each capture is rejected if the TCP moved during it (> 0.3 mm or 0.05 deg).
- Outputs in `~/zivid_ws/calib_board_<date_time>/`: `T_cam_to_base_Zivid_board.npy` (mm), `report.json`, `poses_used.yaml`, `data/`.
- First automatic run: `--gain-scale 0.5 --vmax 0.1`, emergency stop in hand. The robot moves in a straight line in **joint** space between consecutive poses.

## Marker

- Dictionary `aruco6x6_50`, **ID 20** (defaults in `aruco_common.py`, or `--marker-id` / `--dict`).
- Side **40 mm** + 5 mm white border, matte paper, rigid flat support.
- Glued on a **fixed part of the gripper** (not on the fingers).
- Generator: https://chev.me/arucogen/ (6x6, size in mm, print at 100%).

## Files

| File | Role |
|---|---|
| `aruco_common.py` | shared defaults (marker, topic), settings, marker pose, transform maths |
| `Zivid_Eye_to_Base_Part1.py` | data collection: SPACE = capture + robot pose (robot still) |
| `Zivid_Eye_to_Base_Part2.py` | solve `base_T_cam`, print residuals |
| `Zivid_Eye_to_Base_Part3.py` | verification on new poses + saves `Y = hand_T_marker` |
| `Zivid_Eye_to_Base_QuickCalib.py` | one-shot recalibration using `Y` |
| `auto_calibration_board.py` | automatic board calibration (record / run / check) |
| `calibration_poses.yaml` | pose slots (joint configurations) for the automatic run |

## Full calibration (first time, or after moving the marker/gripper)

Run from an empty working folder (outputs are written relative to it):

```bash
S=<path to this repo>
mkdir -p ~/zivid_ws/calib_aruco_$(date +%F) && cd ~/zivid_ws/calib_aruco_$(date +%F)

python3 $S/Zivid_Eye_to_Base_Part1.py --data-dir ./data          # 10-15 varied poses, ESC to finish
python3 $S/Zivid_Eye_to_Base_Part2.py --data-dir ./data          # residuals ~1 mm, < 0.5 deg
python3 $S/Zivid_Eye_to_Base_Part3.py \
    --transform-path ./calibration_outputs/T_cam_to_base_Zivid_ArUco.npy \
    --settings ./data/settings_used.yml --capture-period 8      # MOVE the robot between captures
python3 $S/Zivid_Eye_to_Base_QuickCalib.py --settings ./data/settings_used.yml \
    --compare ./calibration_outputs/T_cam_to_base_Zivid_ArUco.npy   # check: few mm, < 0.5 deg
```

Outputs:
- `calibration_outputs/T_cam_to_base_Zivid_ArUco.npy` — `base_T_cam` (mm)
- `calibration_outputs/hand_T_marker_ArUco.npy` — `Y` (mm), input of QuickCalib
- `data/settings_used.yml` — capture settings to reuse

## Quick recalibration (camera moved, marker untouched)

```bash
python3 $S/Zivid_Eye_to_Base_QuickCalib.py --y <path>/hand_T_marker_ArUco.npy \
    --settings <path>/settings_used.yml --compare <previous base_T_cam .npy>
```

Robot still, marker visible. Averages 5 captures (`--shots`).

## Validity

- `base_T_cam` is valid while **camera and robot base do not move**.
- `Y` (and so QuickCalib) is valid while **the marker does not move on the gripper**, the gripper is not remounted, and the **Desk end-effector configuration** is unchanged (the pose topic reports the Desk EE frame).
- Units: Zivid in mm, ROS in m. Changing the end-effector does not invalidate `base_T_cam`.
