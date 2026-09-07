#!/usr/bin/env bash
# Create workspace directories and print a path checklist.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
fail_fast_preflight

cat << SUMMARY
Prepared directories:
  REPO_ROOT           = ${REPO_ROOT}
  WORKSPACE           = ${WORKSPACE}
  MODEL_DIR           = ${MODEL_DIR}
  HF_HOME             = ${HF_HOME}
  MEGATRON_SAVE_PATH  = ${MEGATRON_SAVE_PATH}
  EXPORT_DIR          = ${EXPORT_DIR}
  LOG_DIR             = ${LOG_DIR}
  SCRATCH_DIR         = ${SCRATCH_DIR}
  SIF_PATH            = ${SIF_PATH}
  HF_MODEL            = ${HF_MODEL}
  EXPORT_QUANT_CFG    = ${EXPORT_QUANT_CFG}
  Parallelism         = NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} TP=${TP} PP=${PP} EP=${EP}
SUMMARY

if [[ ! -f "${SIF_PATH}" ]]; then
  warn "SIF missing — run: CONFIG=... ./scripts/00_pull_or_build_sif.sh"
fi
