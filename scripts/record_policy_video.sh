#!/usr/bin/env bash
# Record an MP4 of a trained policy, headless, without opening a viewer.
#
#   ./scripts/record_policy_video.sh <args_file> <model.pt> <out_dir> [episodes] [num_envs]
#
# Example:
#   ./scripts/record_policy_video.sh \
#       args/mcwamp_g1_steering_300m_s2_e1b_args.txt \
#       output/s2_e1b_seed2/model.pt \
#       output/videos/s2_e1b_seed2
#
# Run from the htwk-gym root. Writes <out_dir>/playback.mp4 (854x480, 30 fps).
#
# Two non-obvious requirements, both learned the hard way:
#
#   1. DISPLAY + the X socket must reach the container. Isaac Gym's camera
#      rendering needs a graphics context, and create_sim() hangs forever
#      (no error, no timeout) when there is no display to attach to - even
#      though nothing is drawn on screen. `who` shows which :N is live.
#
#   2. --test_episodes is REQUIRED. run.py defaults it to int64 max, which is
#      right for the interactive viewer (you watch until you close it) and
#      wrong here: the run never ends and never writes the file.
#
# Recording and the viewer are mutually exclusive by design
# (isaac_gym_engine.py: `if visualize: record_video = False`), so this always
# passes --visualize false.

set -euo pipefail

IMAGE="mimickit-isaacgym:p4"

if [ $# -lt 3 ]; then
    sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

ARGS_FILE="$1"; MODEL="$2"; OUT_DIR="$3"
EPISODES="${4:-64}"; NUM_ENVS="${5:-64}"

if [ ! -f "MimicKit/${ARGS_FILE}" ]; then
    echo "error: MimicKit/${ARGS_FILE} not found (run from the htwk-gym root)" >&2
    exit 1
fi
if [ ! -f "MimicKit/${MODEL}" ]; then
    echo "error: MimicKit/${MODEL} not found" >&2
    exit 1
fi

# pick a live X display: $DISPLAY if set, else the first socket in /tmp/.X11-unix
DISP="${DISPLAY:-}"
if [ -z "${DISP}" ]; then
    SOCK=$(ls /tmp/.X11-unix/X* 2>/dev/null | head -1 || true)
    if [ -z "${SOCK}" ]; then
        echo "error: no X display found. Isaac Gym cannot render without one -" >&2
        echo "       create_sim() will hang silently. Open a session (AnyDesk," >&2
        echo "       physical login, or an Xvfb) and retry." >&2
        exit 1
    fi
    DISP=":${SOCK##*/X}"
fi

NAME="rec_$(basename "${OUT_DIR}")"
docker rm -f "${NAME}" >/dev/null 2>&1 || true

echo "model     ${MODEL}"
echo "args      ${ARGS_FILE}"
echo "display   ${DISP}"
echo "episodes  ${EPISODES} (num_envs ${NUM_ENVS})"
echo "out       ${OUT_DIR}/playback.mp4"
echo

docker run --rm --name "${NAME}" --gpus all --shm-size=2g \
    -v "$PWD/MimicKit:/workspace/MimicKit" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY="${DISP}" \
    -e WANDB_MODE=disabled \
    "${IMAGE}" python3.8 mimickit/run.py \
    --arg_file "${ARGS_FILE}" \
    --mode test --devices cuda:0 \
    --num_envs "${NUM_ENVS}" --test_episodes "${EPISODES}" \
    --visualize false --video true \
    --model_file "${MODEL}" \
    --out_dir "${OUT_DIR}/" 2>&1 | grep -E "Saved video|Mean Return|Mean Episode Length|Error|error" || true

if [ -f "MimicKit/${OUT_DIR}/playback.mp4" ]; then
    echo "ok: MimicKit/${OUT_DIR}/playback.mp4"
else
    echo "FAILED: no MP4 written" >&2
    exit 1
fi
