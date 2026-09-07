#!/usr/bin/env bash
# Submit quantize -> (optional) generate -> export with Slurm job dependencies.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

CONFIG="${CONFIG:-${REPO_ROOT}/configs/fp8_conservative.env}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
export CONFIG REPO_ROOT

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/common.sh"
load_env

EXTRA_SBATCH=()
[[ -n "${SLURM_PARTITION}" ]] && EXTRA_SBATCH+=(--partition="${SLURM_PARTITION}")
[[ -n "${SLURM_ACCOUNT}" ]] && EXTRA_SBATCH+=(--account="${SLURM_ACCOUNT}")
[[ -n "${SLURM_QOS:-}" ]] && EXTRA_SBATCH+=(--qos="${SLURM_QOS}")
[[ -n "${SLURM_TIME}" ]] && EXTRA_SBATCH+=(--time="${SLURM_TIME}")
[[ -n "${SLURM_CONSTRAINT:-}" ]] && EXTRA_SBATCH+=(--constraint="${SLURM_CONSTRAINT}")
[[ -n "${SLURM_RESERVATION:-}" ]] && EXTRA_SBATCH+=(--reservation="${SLURM_RESERVATION}")
if [[ "${SLURM_EXCLUSIVE:-0}" == "1" ]]; then
  EXTRA_SBATCH+=(--exclusive)
fi
[[ -n "${SLURM_MEM:-}" ]] && EXTRA_SBATCH+=(--mem="${SLURM_MEM}")
[[ -n "${SLURM_CPUS_PER_TASK:-}" ]] && EXTRA_SBATCH+=(--cpus-per-task="${SLURM_CPUS_PER_TASK}")

EXTRA_SBATCH+=(--nodes="${NNODES}" --gpus-per-node="${GPUS_PER_NODE}")

EXPORT_VARS="ALL,CONFIG=${CONFIG},REPO_ROOT=${REPO_ROOT}"

info "Submitting pipeline with CONFIG=${CONFIG}"
info "NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} TP=${TP} PP=${PP} EP=${EP}"

QJOB=$(sbatch --parsable "${EXTRA_SBATCH[@]}" \
  --job-name="${SLURM_JOB_NAME}-quantize" \
  --export="${EXPORT_VARS}" \
  "${REPO_ROOT}/slurm/quantize.sbatch")
info "Queued quantize job: ${QJOB}"

DEP_AFTER="${QJOB}"
GJOB="skipped"
if [[ "${SKIP_GENERATE}" != "1" ]]; then
  GJOB=$(sbatch --parsable "${EXTRA_SBATCH[@]}" \
    --job-name="${SLURM_JOB_NAME}-generate" \
    --dependency=afterok:"${DEP_AFTER}" \
    --export="${EXPORT_VARS}" \
    "${REPO_ROOT}/slurm/generate.sbatch")
  info "Queued generate job: ${GJOB} (afterok:${DEP_AFTER})"
  DEP_AFTER="${GJOB}"
else
  info "Skipping generate (SKIP_GENERATE=1)"
fi

EJOB=$(sbatch --parsable "${EXTRA_SBATCH[@]}" \
  --job-name="${SLURM_JOB_NAME}-export" \
  --dependency=afterok:"${DEP_AFTER}" \
  --export="${EXPORT_VARS}" \
  "${REPO_ROOT}/slurm/export.sbatch")
info "Queued export job: ${EJOB} (afterok:${DEP_AFTER})"

cat << SUMMARY
Pipeline submitted:
  quantize : ${QJOB}
  generate : ${GJOB}
  export   : ${EJOB}
Monitor: squeue -u \$USER
Logs:    ${REPO_ROOT}/logs/
SUMMARY
