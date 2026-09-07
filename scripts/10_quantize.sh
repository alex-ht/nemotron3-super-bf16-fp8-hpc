#!/usr/bin/env bash
# BF16 → FP8 PTQ via Megatron-Bridge examples/quantization/quantize.py (Model Optimizer).
# Official flags only: --hf-model-id --export-quant-cfg --megatron-save-path --pp --tp --ep --trust-remote-code
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
fail_fast_preflight
check_sif

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/quantize_${EXPORT_QUANT_CFG}_${TS}.log"
info "Logging to ${LOG_FILE}"

# shellcheck disable=SC2129
{
  echo "=== quantize start $(date -Is) ==="
  echo "HF_MODEL=${HF_MODEL}"
  echo "EXPORT_QUANT_CFG=${EXPORT_QUANT_CFG}"
  echo "MEGATRON_SAVE_PATH=${MEGATRON_SAVE_PATH}"
  echo "TP=${TP} PP=${PP} EP=${EP} ETP=${ETP} CALIB_SIZE=${CALIB_SIZE}"
} | tee -a "${LOG_FILE}"

EXTRA_ARGS=()
if [[ -n "${CALIB_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--calib-size "${CALIB_SIZE}")
fi
if [[ -n "${ETP:-}" && "${ETP}" != "1" ]]; then
  EXTRA_ARGS+=(--etp "${ETP}")
fi

set -x
launch_torchrun examples/quantization/quantize.py \
  --hf-model-id "${HF_MODEL}" \
  --export-quant-cfg "${EXPORT_QUANT_CFG}" \
  --megatron-save-path "${MEGATRON_SAVE_PATH}" \
  --pp "${PP}" \
  --tp "${TP}" \
  --ep "${EP}" \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
set +x

info "Quantize finished. Megatron ckpt: ${MEGATRON_SAVE_PATH}"
