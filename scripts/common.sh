#!/usr/bin/env bash
# Shared helpers for Nemotron 3 Super BF16→FP8 PTQ (Singularity/Apptainer + Slurm).
# Official flow: Megatron-Bridge super-v3 + Model Optimizer PTQ (NOT a naive dtype cast).
set -euo pipefail

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${_COMMON_DIR}/.." && pwd)"
export REPO_ROOT="${REPO_ROOT:-$REPO_ROOT_DEFAULT}"

die() { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO]  $*"; }
warn() { echo "[WARN]  $*" >&2; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

# Load CONFIG (.env style) then optional REPO_ROOT/.env overlays.
load_env() {
  local cfg="${CONFIG:-}"
  if [[ -n "${cfg}" ]]; then
    [[ -f "${cfg}" ]] || die "CONFIG file not found: ${cfg}"
    info "Loading CONFIG=${cfg}"
    set -a
    # shellcheck disable=SC1090
    source "${cfg}"
    set +a
  fi
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    info "Loading ${REPO_ROOT}/.env"
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
  fi

  export REPO_ROOT="${REPO_ROOT:-$REPO_ROOT_DEFAULT}"
  export WORKSPACE="${WORKSPACE:-${REPO_ROOT}/scratch/workspace}"
  export MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/scratch/models}"
  export HF_HOME="${HF_HOME:-${REPO_ROOT}/scratch/hf_home}"
  export MEGATRON_SAVE_PATH="${MEGATRON_SAVE_PATH:-${REPO_ROOT}/scratch/megatron_ckpt}"
  export EXPORT_DIR="${EXPORT_DIR:-${REPO_ROOT}/scratch/hf_export}"
  export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
  export SCRATCH_DIR="${SCRATCH_DIR:-${REPO_ROOT}/scratch}"
  export SIF_PATH="${SIF_PATH:-${SCRATCH_DIR}/nemo_26.02.nemotron_3_super.sif}"

  export NGC_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/nemo:26.02.nemotron_3_super}"
  export MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-/opt/Megatron-Bridge}"
  export SINGULARITY_BIN="${SINGULARITY_BIN:-}"
  if [[ -z "${SINGULARITY_BIN}" ]]; then
    if command -v apptainer >/dev/null 2>&1; then
      SINGULARITY_BIN=apptainer
    elif command -v singularity >/dev/null 2>&1; then
      SINGULARITY_BIN=singularity
    else
      die "Neither apptainer nor singularity found in PATH"
    fi
    export SINGULARITY_BIN
  fi

  export HF_MODEL="${HF_MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}"
  export EXPORT_QUANT_CFG="${EXPORT_QUANT_CFG:-mamba_moe_fp8_conservative}"
  export CALIB_SIZE="${CALIB_SIZE:-256}"

  export NNODES="${NNODES:-1}"
  export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
  export TP="${TP:-8}"
  export PP="${PP:-1}"
  export EP="${EP:-8}"
  export ETP="${ETP:-1}"

  export EXPORT_TP="${EXPORT_TP:-1}"
  export EXPORT_PP="${EXPORT_PP:-8}"
  export EXPORT_EP="${EXPORT_EP:-1}"
  export EXPORT_ETP="${EXPORT_ETP:-1}"
  export EXPORT_DTYPE="${EXPORT_DTYPE:-bfloat16}"

  export MASTER_PORT="${MASTER_PORT:-29500}"
  export SLURM_PARTITION="${SLURM_PARTITION:-}"
  export SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
  export SLURM_TIME="${SLURM_TIME:-24:00:00}"
  export SLURM_JOB_NAME="${SLURM_JOB_NAME:-nemotron3-fp8-ptq}"
}

ensure_dirs() {
  mkdir -p "${LOG_DIR}" "${SCRATCH_DIR}" "${WORKSPACE}" "${MODEL_DIR}" \
    "${HF_HOME}" "${MEGATRON_SAVE_PATH}" "${EXPORT_DIR}"
}

check_sif() {
  [[ -f "${SIF_PATH}" ]] || die "SIF not found: ${SIF_PATH}. Run scripts/00_pull_or_build_sif.sh first."
}

# Build --bind list for model, ckpt, HF_HOME, workspace, repo, optional Megatron host tree.
build_bind_args() {
  local -a binds=()
  local p
  for p in "${REPO_ROOT}" "${WORKSPACE}" "${MODEL_DIR}" "${HF_HOME}" \
           "${MEGATRON_SAVE_PATH}" "${EXPORT_DIR}" "${LOG_DIR}" "${SCRATCH_DIR}"; do
    mkdir -p "${p}"
    binds+=(--bind "${p}:${p}")
  done
  if [[ -n "${MEGATRON_BRIDGE_HOST:-}" ]]; then
    [[ -d "${MEGATRON_BRIDGE_HOST}" ]] || die "MEGATRON_BRIDGE_HOST not a directory: ${MEGATRON_BRIDGE_HOST}"
    binds+=(--bind "${MEGATRON_BRIDGE_HOST}:${MEGATRON_BRIDGE_ROOT}")
  fi
  # Extra user binds: EXTRA_BINDS="/a:/a,/b:/b"
  if [[ -n "${EXTRA_BINDS:-}" ]]; then
    local IFS=','
    local b
    for b in ${EXTRA_BINDS}; do
      [[ -n "${b}" ]] && binds+=(--bind "${b}")
    done
  fi
  printf '%s\n' "${binds[@]}"
}

# Resolve MASTER_ADDR for multi-node torchrun (first host in SLURM allocation).
resolve_master_addr() {
  if [[ -n "${MASTER_ADDR:-}" ]]; then
    echo "${MASTER_ADDR}"
    return
  fi
  if [[ -n "${SLURM_NODELIST:-}" ]]; then
    if command -v scontrol >/dev/null 2>&1; then
      scontrol show hostnames "${SLURM_NODELIST}" | head -n 1
      return
    fi
  fi
  hostname -s
}

# Export torchrun / distributed env from Slurm when present.
setup_distributed_env() {
  export NNODES="${SLURM_NNODES:-${NNODES}}"
  export GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${GPUS_PER_NODE}}"
  if [[ -n "${SLURM_NTASKS_PER_NODE:-}" && -z "${SLURM_GPUS_ON_NODE:-}" ]]; then
    # Prefer explicit GPUS_PER_NODE from config when Slurm GPU count unknown
    :
  fi
  export NODE_RANK="${SLURM_NODEID:-0}"
  export MASTER_ADDR="$(resolve_master_addr)"
  export MASTER_PORT="${MASTER_PORT:-29500}"
  export WORLD_SIZE="$((NNODES * GPUS_PER_NODE))"

  info "Distributed: NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} NODE_RANK=${NODE_RANK}"
  info "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} WORLD_SIZE=${WORLD_SIZE}"
}

# Run a command inside the container with --nv and binds.
# Usage: run_in_container [--workdir DIR] -- CMD...
run_in_container() {
  check_sif
  local workdir="${MEGATRON_BRIDGE_ROOT}"
  local -a extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --workdir) workdir="$2"; shift 2 ;;
      --) shift; break ;;
      *) extra+=("$1"); shift ;;
    esac
  done
  [[ $# -gt 0 ]] || die "run_in_container: missing command"

  local -a binds=()
  mapfile -t binds < <(build_bind_args)

  export HF_HOME
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  export TORCH_HOME="${TORCH_HOME:-${HF_HOME}/torch}"

  info "Exec: ${SINGULARITY_BIN} exec --nv (workdir=${workdir})"
  "${SINGULARITY_BIN}" exec --nv \
    --pwd "${workdir}" \
    "${binds[@]}" \
    --env "HF_HOME=${HF_HOME}" \
    --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
    --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" \
    --env "TORCH_HOME=${TORCH_HOME}" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    --env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
    --env "MEGATRON_BRIDGE_ROOT=${MEGATRON_BRIDGE_ROOT}" \
    --env "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}" \
    "${extra[@]}" \
    "${SIF_PATH}" \
    "$@"
}

# Multi-node / multi-GPU launcher: prefer srun+torchrun under Slurm.
# Inner command is relative to MEGATRON_BRIDGE_ROOT (examples/...).
launch_torchrun() {
  local script_rel="$1"
  shift
  [[ -n "${script_rel}" ]] || die "launch_torchrun: missing script path"

  setup_distributed_env
  ensure_dirs
  check_sif

  local nproc="${GPUS_PER_NODE}"
  local wrap="${SCRATCH_DIR}/torchrun_wrap_$$.sh"
  mkdir -p "${SCRATCH_DIR}"

  # Write a wrapper so argument quoting stays correct under srun + singularity.
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "export NODE_RANK=\${SLURM_NODEID:-${NODE_RANK}}"
    echo "export MASTER_ADDR=${MASTER_ADDR}"
    echo "export MASTER_PORT=${MASTER_PORT}"
    printf "exec torchrun --nnodes=%q --nproc_per_node=%q --node_rank=\"\${NODE_RANK}\" --master_addr=%q --master_port=%q %q" \
      "${NNODES}" "${nproc}" "${MASTER_ADDR}" "${MASTER_PORT}" "${script_rel}"
    local a
    for a in "$@"; do
      printf " %q" "${a}"
    done
    echo
  } > "${wrap}"
  chmod +x "${wrap}"
  info "Wrote launcher wrapper: ${wrap}"

  local -a binds=()
  mapfile -t binds < <(build_bind_args)

  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    info "Launching via srun + container + torchrun"
    srun --ntasks="${NNODES}" --ntasks-per-node=1 --gpus-per-node="${GPUS_PER_NODE}" \
      "${SINGULARITY_BIN}" exec --nv \
      --pwd "${MEGATRON_BRIDGE_ROOT}" \
      "${binds[@]}" \
      --env "HF_HOME=${HF_HOME}" \
      --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}" \
      --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}" \
      --env "HF_TOKEN=${HF_TOKEN:-}" \
      --env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
      --env "MEGATRON_BRIDGE_ROOT=${MEGATRON_BRIDGE_ROOT}" \
      --env "MASTER_ADDR=${MASTER_ADDR}" \
      --env "MASTER_PORT=${MASTER_PORT}" \
      "${SIF_PATH}" \
      bash "${wrap}"
  else
    info "Launching torchrun inside container (no Slurm srun)"
    run_in_container -- bash "${wrap}"
  fi
}

fail_fast_preflight() {
  load_env
  require_cmd "${SINGULARITY_BIN}"
  ensure_dirs
  case "${EXPORT_QUANT_CFG}" in
    mamba_moe_fp8_conservative|mamba_moe_fp8_aggressive|mamba_moe_nvfp4_conservative|mamba_moe_nvfp4_aggressive) ;;
    *) warn "Unusual EXPORT_QUANT_CFG=${EXPORT_QUANT_CFG} (expected mamba_moe_fp8_* or mamba_moe_nvfp4_*)" ;;
  esac
  info "Preflight OK: REPO_ROOT=${REPO_ROOT} SIF_PATH=${SIF_PATH} CFG=${EXPORT_QUANT_CFG}"
}
