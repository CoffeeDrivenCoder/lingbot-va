# LingBot-VA G1 Offline Client

This folder contains the G1-side offline inference client for LingBot-VA.
It is read-only on the robot: it captures one G1 head RGB-D frame and current
arm state, calls the LingBot-VA websocket server, converts the predicted
camera-frame EEF poses into G1 link7 targets, and exports an IK joint sequence
for inspection. It does not send robot motion commands.

## Run

From the `lingbot-va` repository root on the G1 machine:

```bash
G1_LINGBOT_PYTHON="$(which python)" \
LINGBOT_SERVER_HOST=<server-ip> \
LINGBOT_SERVER_PORT=30002 \
LINGBOT_PROMPT="serve bread" \
G1_LINGBOT_SIDE=right \
bash evaluation/g1/run_g1_lingbot_offline_policy_check.sh
```

If using an SSH tunnel to the server, set:

```bash
LINGBOT_SERVER_HOST=127.0.0.1
LINGBOT_SERVER_PORT=29536
```

## Outputs

The client writes results to:

```text
evaluation/g1/artifacts/lingbot_offline_policy_check/<timestamp>_lingbot_g1_offline/
```

Important files:

```text
summary.json
joint_summary.json
initial_pose_check in summary.json
ik_joint_trajectory.csv
offline_policy_check_report.json
input_state.json
lingbot_ego_rgb_320x256.png
g1_rgb_raw_bgr.jpg
raw_action.npy
```

`initial_pose_check` compares the current G1 hand-in-camera pose with
fine-tuning data starts:

```text
episode_start_close: close to task-start demonstrations
segment_start_close: close to training-window starts
```
