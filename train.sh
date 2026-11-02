#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAMERA_NAME="${1:-${CAMERA_NAME:-FLIR-align}}"
GPU_ID="${2:-${GPU_ID:-0}}"
DATA_DIR="${D3F_IR_DATA_DIR:-${PROJECT_DIR}/data/${CAMERA_NAME}}"
DINO_CKPT="${D3F_IR_DINO_CKPT:-${PROJECT_DIR}/ckpt/DINOv3/model.safetensors}"
PRETRAIN_CKPT="${D3F_IR_PRETRAIN_CKPT:-}"
RESUME_CKPT="${D3F_IR_RESUME_CKPT:-}"
DEFAULT_PRETRAIN_CKPT="${PROJECT_DIR}/ckpt/pretrain/best.pt"
DEFAULT_RESUME_CKPT="${PROJECT_DIR}/ckpt/${CAMERA_NAME}/latest.pt"
if [[ -z "${PRETRAIN_CKPT}" && -f "${DEFAULT_PRETRAIN_CKPT}" ]]; then
    PRETRAIN_CKPT="${DEFAULT_PRETRAIN_CKPT}"
fi
if [[ -z "${RESUME_CKPT}" && -f "${DEFAULT_RESUME_CKPT}" ]]; then
    RESUME_CKPT="${DEFAULT_RESUME_CKPT}"
fi

if [[ ! -d "${DATA_DIR}" ]]; then
    printf 'Dataset directory does not exist: %s\n' "${DATA_DIR}" >&2
    printf 'Set D3F_IR_DATA_DIR to the dataset root.\n' >&2
    exit 1
fi

if [[ ! -f "${DINO_CKPT}" ]]; then
    printf 'DINO checkpoint does not exist: %s\n' "${DINO_CKPT}" >&2
    printf 'Set D3F_IR_DINO_CKPT to the DINOv3 safetensors checkpoint.\n' >&2
    exit 1
fi

TRAIN_FILE=""
VAL_FILE=""
if [[ -f "${DATA_DIR}/annotations_train.json" ]]; then
    TRAIN_FILE="${DATA_DIR}/annotations_train.json"
    VAL_FILE="${DATA_DIR}/annotations_validation.json"
fi

OUTPUT_DIR="${D3F_IR_OUTPUT_DIR:-${PROJECT_DIR}/ckpt/${CAMERA_NAME}}"
VIS_DIR="${D3F_IR_VIS_DIR:-${PROJECT_DIR}/outputs/visualizations/${CAMERA_NAME}}"
mkdir -p "${OUTPUT_DIR}" "${VIS_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

CMD=(
    python3 "${PROJECT_DIR}/train.py"
    --data_dir "${DATA_DIR}"
    --train_file "${TRAIN_FILE}"
    --val_file "${VAL_FILE}"
    --camera_name "${CAMERA_NAME}"
    --batch_size "${BATCH_SIZE:-1}"
    --patch_size "${PATCH_SIZE:-16}"
    --epochs "${EPOCHS:-500}"
    --lr "${LR:-4e-4}"
    --num_workers "${NUM_WORKERS:-8}"
    --seed "${SEED:-42}"
    --output_dir "${OUTPUT_DIR}"
    --vis_dir "${VIS_DIR}"
    --time_token_count "${TIME_TOKENS:-2}"
    --latent_dim "${LATENT_DIM:-256}"
    --dit_depth "${DIT_DEPTH:-14}"
    --dit_heads "${DIT_HEADS:-16}"
    --dit_hidden "${DIT_HIDDEN:-128}"
    --dit_mlp_ratio "${DIT_MLP:-4.0}"
    --dit_head_depth "${DIT_HEAD_DEPTH:-2}"
    --dit_head_hidden "${DIT_HEAD_HIDDEN:-256}"
    --dit_head_heads "${DIT_HEAD_HEADS:-16}"
    --dit_head_mlp_ratio "${DIT_HEAD_MLP:-4.0}"
    --decoder_layers "${DEC_LAYERS:-6}"
    --decoder_hidden "${DEC_HIDDEN:-384}"
    --decoder_heads "${DEC_HEADS:-16}"
    --decoder_mlp_ratio "${DEC_MLP:-4.0}"
    --fusion_heads "${FUSION_HEADS:-4}"
    --shallow_channels "${SHALLOW_CH:-80}"
    --val_flow_steps "${VAL_STEPS:-10}"
    --align_weight "${ALIGN_WEIGHT:-1.0}"
    --dino_ckpt "${DINO_CKPT}"
    --device cuda
)

if [[ -n "${ALIGN_LAYER:-}" ]]; then
    CMD+=(--align_layer "${ALIGN_LAYER}")
fi
if [[ -n "${PRETRAIN_CKPT}" ]]; then
    if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
        printf 'Pretrained checkpoint does not exist: %s\n' "${PRETRAIN_CKPT}" >&2
        exit 1
    fi
    CMD+=(--pretrain_ckpt "${PRETRAIN_CKPT}")
fi
if [[ -n "${RESUME_CKPT}" ]]; then
    if [[ ! -f "${RESUME_CKPT}" ]]; then
        printf 'Resume checkpoint does not exist: %s\n' "${RESUME_CKPT}" >&2
        exit 1
    fi
    CMD+=(--resume "${RESUME_CKPT}")
fi

"${CMD[@]}"
