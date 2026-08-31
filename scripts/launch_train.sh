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

if docker ps -a --format '{{.Names}}' | grep -qx "${RUN}"; then
    echo "error: a container named ${RUN} already exists" >&2
    exit 1
fi

echo "run       ${RUN}"
echo "args      ${ARGS_FILE}"
echo "wandb     ${WANDB_ENTITY_}/${WANDB_PROJECT_}  tags=[${TAGS}]"
echo "out_dir   output/${RUN}/"
echo

docker run -d --rm --name "${RUN}" --gpus all --shm-size=2g \
    -v "$PWD/MimicKit:/workspace/MimicKit" \
    -v "$HOME/.netrc:/root/.netrc:ro" \
    -e WANDB_ENTITY="${WANDB_ENTITY_}" \
    -e WANDB_PROJECT="${WANDB_PROJECT_}" \
    -e WANDB_NAME="${RUN}" \
    -e WANDB_TAGS="${TAGS}" \
    "${IMAGE}" python3.8 mimickit/run.py \
    --arg_file "${ARGS_FILE}" \
    --mode train --devices cuda:0 --rand_seed "${SEED}" \
    --out_dir "output/${RUN}/"

echo
echo "started. follow with:  docker logs -f ${RUN}"
echo "stop with:             docker stop ${RUN}"
