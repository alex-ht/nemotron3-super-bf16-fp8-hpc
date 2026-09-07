#!/bin/bash
#SBATCH --job-name=nemotron3-fp8-generate
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-generate-%j.out
#SBATCH --error=logs/slurm-generate-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
fi
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

CONFIG="${CONFIG:-${REPO_ROOT}/configs/fp8_conservative.env}"
export CONFIG REPO_ROOT
export NNODES="${SLURM_NNODES:-${NNODES:-2}}"
export GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${GPUS_PER_NODE:-8}}"

echo "[sbatch generate] CONFIG=${CONFIG} job=${SLURM_JOB_ID}"
bash "${REPO_ROOT}/scripts/20_ptq_generate.sh"
