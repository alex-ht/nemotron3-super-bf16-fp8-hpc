#!/usr/bin/env bash
# Export quantized Megatron ckpt → Hugging Face via examples/quantization/export.py
# Official: --hf-model-id --megatron-load-path --export-dir --pp --dtype --trust-remote-code
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
fail_fast_preflight
check_sif

[[ -d "${MEGATRON_SAVE_PATH}" ]] || die "MEGATRON_SAVE_PATH missing: ${MEGATRON_SAVE_PATH}"
mkdir -p "${EXPORT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/export_hf_${TS}.log"
info "Logging to ${LOG_FILE}"
{
  echo "=== export start $(date -Is) ==="
  echo "EXPORT_DIR=${EXPORT_DIR}"
  echo "EXPORT_TP=${EXPORT_TP} EXPORT_PP=${EXPORT_PP} EXPORT_EP=${EXPORT_EP} DTYPE=${EXPORT_DTYPE}"
} | tee -a "${LOG_FILE}"

# Temporarily override parallelism for export torchrun world size if needed.
# Export docs often use a different --pp than quantize.
export TP="${EXPORT_TP}"
export PP="${EXPORT_PP}"
export EP="${EXPORT_EP}"
export ETP="${EXPORT_ETP:-1}"

EXTRA_ARGS=()
if [[ -n "${ETP:-}" && "${ETP}" != "1" ]]; then
  EXTRA_ARGS+=(--etp "${ETP}")
fi
if [[ "${EXPORT_EXTRA_MODULES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--export-extra-modules)
fi

set -x
launch_torchrun examples/quantization/export.py \
  --hf-model-id "${HF_MODEL}" \
  --megatron-load-path "${MEGATRON_SAVE_PATH}" \
  --export-dir "${EXPORT_DIR}" \
  --pp "${EXPORT_PP}" \
  --tp "${EXPORT_TP}" \
  --ep "${EXPORT_EP}" \
  --dtype "${EXPORT_DTYPE}" \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
set +x

info "Export finished. HF dir: ${EXPORT_DIR}"
