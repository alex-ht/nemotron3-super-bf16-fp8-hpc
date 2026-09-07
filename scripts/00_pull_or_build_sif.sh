#!/usr/bin/env bash
# Pull (preferred) or build the Nemotron 3 Super NGC container as a .sif
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_env
ensure_dirs

MODE="${1:-pull}"  # pull | build
require_cmd "${SINGULARITY_BIN}"

if [[ -f "${SIF_PATH}" ]]; then
  info "SIF already exists: ${SIF_PATH}"
  ls -lh "${SIF_PATH}"
  exit 0
fi

mkdir -p "$(dirname "${SIF_PATH}")"

case "${MODE}" in
  pull)
    info "Pulling ${NGC_IMAGE} -> ${SIF_PATH}"
    info "NGC auth tip: export SINGULARITY_DOCKER_USERNAME='\$oauthtoken' and SINGULARITY_DOCKER_PASSWORD=<NGC_API_KEY>"
    "${SINGULARITY_BIN}" pull "${SIF_PATH}" "docker://${NGC_IMAGE#docker://}"
    ;;
  build)
    DEF="${REPO_ROOT}/singularity/nemotron3-super.def"
    [[ -f "${DEF}" ]] || die "Definition file missing: ${DEF}"
    info "Building from ${DEF} -> ${SIF_PATH}"
    "${SINGULARITY_BIN}" build --nv "${SIF_PATH}" "${DEF}"
    ;;
  *)
    die "Usage: $0 [pull|build]"
    ;;
esac

[[ -f "${SIF_PATH}" ]] || die "SIF not created"
ls -lh "${SIF_PATH}"
info "Done. Do NOT commit the .sif file."
