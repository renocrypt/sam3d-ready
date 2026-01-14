#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

default_src_root="$ROOT_DIR/.."
if [[ -d "$ROOT_DIR/../sources/MV-SAM3D" ]]; then
  default_src_root="$ROOT_DIR/../sources"
fi

SRC_ROOT="${SRC_ROOT:-$default_src_root}"
SAM3D_OBJECTS_DIR="${SAM3D_OBJECTS_DIR:-$SRC_ROOT/MV-SAM3D}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$ROOT_DIR/micromamba}"
ENV_NAME="${ENV_NAME:-mv-sam3d-unified}"
TAG="${TAG:-hf}"
FORCE="${FORCE:-0}"
HF_REPO_ID="${HF_REPO_ID:-facebook/sam-3d-objects}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$ROOT_DIR/bin/micromamba}"

log() {
  printf "[sam3d-ready/mv-sam3d] %s\n" "$*"
}

ensure_micromamba() {
  if command -v micromamba >/dev/null 2>&1; then
    MICROMAMBA="$(command -v micromamba)"
  elif [[ -x "$MICROMAMBA_BIN" ]]; then
    MICROMAMBA="$MICROMAMBA_BIN"
  else
    log "Installing micromamba into $ROOT_DIR/bin"
    mkdir -p "$ROOT_DIR/bin"
    arch="$(uname -m)"
    case "$arch" in
      x86_64) platform="linux-64" ;;
      aarch64|arm64) platform="linux-aarch64" ;;
      *)
        echo "Unsupported architecture: $arch" >&2
        exit 1
        ;;
    esac
    curl -Ls "https://micro.mamba.pm/api/micromamba/${platform}/latest" \
      | tar -xvj -C "$ROOT_DIR/bin" --strip-components=1 bin/micromamba
    MICROMAMBA="$MICROMAMBA_BIN"
  fi

  export MAMBA_ROOT_PREFIX
  log "micromamba: $MICROMAMBA"
  log "MAMBA_ROOT_PREFIX: $MAMBA_ROOT_PREFIX"
}

run_in_env() {
  "$MICROMAMBA" run -n "$ENV_NAME" "$@"
}

ensure_env_exists() {
  if ! "$MICROMAMBA" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Missing micromamba env: $ENV_NAME" >&2
    echo "Run: $ROOT_DIR/setup_env_mv_sam3d.sh" >&2
    exit 1
  fi
}

maybe_login() {
  local token=""
  token="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
  if [[ -z "$token" ]]; then
    return 0
  fi

  local whoami
  whoami="$(run_in_env hf auth whoami 2>/dev/null || true)"
  if [[ "$whoami" == "Not logged in" ]]; then
    log "Logging into Hugging Face (token provided via env)"
    run_in_env env HF_TOKEN="$token" python - <<'PY'
from huggingface_hub import login
import os

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("Missing HF_TOKEN")

login(token=token, add_to_git_credential=False)
print("Logged in")
PY
  fi
}

download_checkpoints() {
  if [[ ! -d "$SAM3D_OBJECTS_DIR" ]]; then
    echo "SAM3D_OBJECTS_DIR does not exist: $SAM3D_OBJECTS_DIR" >&2
    exit 1
  fi

  local dest_root dest_dir download_dir
  dest_root="$SAM3D_OBJECTS_DIR/checkpoints"
  dest_dir="$dest_root/$TAG"
  download_dir="$dest_root/${TAG}-download"

  if [[ -f "$dest_dir/pipeline.yaml" && "$FORCE" != "1" ]]; then
    log "Checkpoints already present: $dest_dir/pipeline.yaml"
    return 0
  fi

  mkdir -p "$dest_root"
  if [[ -d "$download_dir" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      rm -rf "$download_dir"
    else
      echo "Download dir already exists: $download_dir (set FORCE=1 to overwrite)" >&2
      exit 1
    fi
  fi

  log "Downloading checkpoints from: $HF_REPO_ID"
  log "Destination: $dest_dir"
  set +e
  if [[ "$FORCE" == "1" ]]; then
    run_in_env hf download --repo-type model --local-dir "$download_dir" --max-workers 1 --force-download "$HF_REPO_ID"
  else
    run_in_env hf download --repo-type model --local-dir "$download_dir" --max-workers 1 "$HF_REPO_ID"
  fi
  local status=$?
  set -e

  if [[ $status -ne 0 ]]; then
    rm -rf "$download_dir" || true
    log "Download failed."
    log "If you see a 401/GatedRepoError, request access and authenticate:"
    log "  - Get access: https://huggingface.co/$HF_REPO_ID"
    log "  - Then run: HF_TOKEN=... $ROOT_DIR/setup_checkpoints_mv_sam3d.sh"
    exit $status
  fi

  if [[ ! -d "$download_dir/checkpoints" ]]; then
    echo "Unexpected download layout (missing $download_dir/checkpoints)." >&2
    echo "Inspect: $download_dir" >&2
    exit 1
  fi

  if [[ -d "$dest_dir" ]]; then
    rm -rf "$dest_dir"
  fi
  mv "$download_dir/checkpoints" "$dest_dir"
  rm -rf "$download_dir"

  if [[ ! -f "$dest_dir/pipeline.yaml" ]]; then
    echo "Download completed but pipeline.yaml missing: $dest_dir/pipeline.yaml" >&2
    exit 1
  fi

  log "Done. Found: $dest_dir/pipeline.yaml"
}

main() {
  ensure_micromamba
  ensure_env_exists
  maybe_login
  download_checkpoints
}

main "$@"
