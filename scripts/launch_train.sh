#!/usr/bin/env bash
# Launch a MimicKit training run in the project's canonical W&B location.
#
# The entity and project live here, in one place, on purpose: they were once
# passed by hand on the command line and a single typo ("akcit-r-humanoids")
# cost a run. Everything about where a run lands is now decided by this file.
#
#   ./scripts/launch_train.sh <exp_name> <args_file> <seed> [tag ...]
#
# Example:
#   ./scripts/launch_train.sh s2_e1 args/mcwamp_g1_steering_300m_s2_e1_args.txt 2 s2 e1
#
# Run from the htwk-gym root. Starts detached; follow with `docker logs -f <exp>_seed<N>`.

set -euo pipefail

# --- canonical W&B location for the whole S1-S5 ladder ----------------------
WANDB_ENTITY_="akcit-rl-humanoids"
WANDB_PROJECT_="tsinghua_policy"
IMAGE="mimickit-isaacgym:p4"

if [ $# -lt 3 ]; then
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

EXP="$1"; ARGS_FILE="$2"; SEED="$3"; shift 3
EXTRA_TAGS="$*"

RUN="${EXP}_seed${SEED}"
TAGS="seed${SEED}"
if [ -n "${EXTRA_TAGS}" ]; then
    TAGS="${TAGS},$(echo "${EXTRA_TAGS}" | tr ' ' ',')"
fi

if [ ! -f "MimicKit/${ARGS_FILE}" ]; then
    echo "error: MimicKit/${ARGS_FILE} not found (run from the htwk-gym root)" >&2
    exit 1
fi

if [ ! -f "${HOME}/.netrc" ]; then
    echo "error: ~/.netrc missing - the container has no W&B credentials" >&2
    exit 1
fi

# --- video: one clip per output_iter, uploaded to W&B alongside the curves ---
# MimicKit already wires this end to end (engine camera -> Sim_Recording ->
# wandb_logger._write_video); --video true is all it takes. Two catches:
#
#   * Isaac Gym's camera needs a graphics context, so DISPLAY and the X socket
#     must reach the container. With no display, create_sim() hangs FOREVER
#     with no error - a 2h run that silently never starts. Hence the hard check
#     below instead of letting it hang.
#   * The X session must survive the whole run. Closing AnyDesk is fine (the
#     Xorg seat stays up); logging out of the desktop is not.
#
# Set RECORD_VIDEO=0 to opt out.
RECORD_VIDEO="${RECORD_VIDEO:-1}"
VIDEO_ARGS=()
VIDEO_STATUS="off"

if [ "${RECORD_VIDEO}" != "0" ]; then
    DISP="${DISPLAY:-}"
    if [ -z "${DISP}" ]; then
        SOCK=$(ls /tmp/.X11-unix/X* 2>/dev/null | head -1 || true)
        [ -n "${SOCK}" ] && DISP=":${SOCK##*/X}"
    fi
    if [ -z "${DISP}" ]; then
        echo "error: RECORD_VIDEO is on but no X display was found." >&2
        echo "       Isaac Gym would hang silently instead of training." >&2
        echo "       Open a session (AnyDesk / physical login), or rerun with" >&2
        echo "       RECORD_VIDEO=0 to train without video." >&2
        exit 1
    fi
    VIDEO_ARGS=(-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY="${DISP}")
    VIDEO_STATUS="on (DISPLAY=${DISP}, one clip per output_iter)"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${RUN}"; then
    echo "error: a container named ${RUN} already exists" >&2
    exit 1
fi

echo "run       ${RUN}"
echo "args      ${ARGS_FILE}"
echo "wandb     ${WANDB_ENTITY_}/${WANDB_PROJECT_}  tags=[${TAGS}]"
echo "video     ${VIDEO_STATUS}"
echo "out_dir   output/${RUN}/"
echo

docker run -d --rm --name "${RUN}" --gpus all --shm-size=2g \
    -v "$PWD/MimicKit:/workspace/MimicKit" \
    -v "$HOME/.netrc:/root/.netrc:ro" \
    "${VIDEO_ARGS[@]}" \
    -e WANDB_ENTITY="${WANDB_ENTITY_}" \
    -e WANDB_PROJECT="${WANDB_PROJECT_}" \
    -e WANDB_NAME="${RUN}" \
    -e WANDB_TAGS="${TAGS}" \
    "${IMAGE}" python3.8 mimickit/run.py \
    --arg_file "${ARGS_FILE}" \
    --mode train --devices cuda:0 --rand_seed "${SEED}" \
    --video "$([ "${RECORD_VIDEO}" != "0" ] && echo true || echo false)" \
    --out_dir "output/${RUN}/"

echo
echo "started. follow with:  docker logs -f ${RUN}"
echo "stop with:             docker stop ${RUN}"
