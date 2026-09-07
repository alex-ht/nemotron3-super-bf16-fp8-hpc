#!/usr/bin/env bash
# Optional smoke test: examples/quantization/ptq_generate.py on quantized Megatron ckpt.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
fail_fast_preflight
check_sif

[[ -d "${MEGATRON_SAVE_PATH}" ]] || die "MEGATRON_SAVE_PATH missing: ${MEGATRON_SAVE_PATH}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/ptq_generate_${TS}.log"
PROMPTS="${PROMPTS:-Hello!|Born in California, Soyer trained as a}"
OSL="${OSL:-32}"

info "Logging to ${LOG_FILE}"
{
  echo "=== ptq_generate start $(date -Is) ==="
  echo "MEGATRON_SAVE_PATH=${MEGATRON_SAVE_PATH}"
  echo "TP=${TP} PP=${PP} EP=${EP}"
} | tee -a "${LOG_FILE}"

EXTRA_ARGS=()
if [[ -n "${ETP:-}" && "${ETP}" != "1" ]]; then
  EXTRA_ARGS+=(--etp "${ETP}")
fi

set -x
launch_torchrun examples/quantization/ptq_generate.py \
  --hf-model-id "${HF_MODEL}" \
  --megatron-load-path "${MEGATRON_SAVE_PATH}" \
  --pp "${PP}" \
  --tp "${TP}" \
  --ep "${EP}" \
  --prompts "${PROMPTS}" \
  --osl "${OSL}" \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
set +x

info "ptq_generate finished."
