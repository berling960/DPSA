#!/usr/bin/env bash
set -euo pipefail

# Example DPSA attack run.
# Override variables from the shell, e.g.:
#   DATASET=CUB MODEL=dinov3 NUM_ATTACKS=100 GPU_ID=0 ./run.sh

PYTHON=${PYTHON:-python}
GPU_ID=${GPU_ID:-0}
DATASET=${DATASET:-ImageNet}
MODEL=${MODEL:-dinov3}
ATTACK=${ATTACK:-dpsa}
INPUT_DIR=${INPUT_DIR:-./data}
OUTPUT_DIR=${OUTPUT_DIR:-./results_dpsa}
PRETRAINED_MODELS_ROOT=${PRETRAINED_MODELS_ROOT:-./pretrained_models}
NUM_ATTACKS=${NUM_ATTACKS:--1}
EPOCH=${EPOCH:-15}
NUM_OP_SAMPLES=${NUM_OP_SAMPLES:-30}
NUM_NEIGHBOR=${NUM_NEIGHBOR:-15}
POOL_CHAIN_LENGTH=${POOL_CHAIN_LENGTH:-3}
LOCAL_PROBE_SAMPLES=${LOCAL_PROBE_SAMPLES:-1}

echo "Generate adversarial examples: ${DATASET} ${MODEL} ${ATTACK}"
"${PYTHON}" main.py \
  --GPU_ID "${GPU_ID}" \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --attack "${ATTACK}" \
  --num_attacks "${NUM_ATTACKS}" \
  --epoch "${EPOCH}" \
  --num_op_samples "${NUM_OP_SAMPLES}" \
  --num_neighbor "${NUM_NEIGHBOR}" \
  --pool_chain_length "${POOL_CHAIN_LENGTH}" \
  --local_probe_samples "${LOCAL_PROBE_SAMPLES}" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --pretrained_models_root "${PRETRAINED_MODELS_ROOT}"

echo "Evaluate transferability"
"${PYTHON}" main.py \
  --GPU_ID "${GPU_ID}" \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --attack "${ATTACK}" \
  --num_attacks "${NUM_ATTACKS}" \
  --epoch "${EPOCH}" \
  --num_op_samples "${NUM_OP_SAMPLES}" \
  --num_neighbor "${NUM_NEIGHBOR}" \
  --pool_chain_length "${POOL_CHAIN_LENGTH}" \
  --local_probe_samples "${LOCAL_PROBE_SAMPLES}" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --pretrained_models_root "${PRETRAINED_MODELS_ROOT}" \
  -e
