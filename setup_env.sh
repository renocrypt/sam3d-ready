#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="${SRC_ROOT:-$ROOT_DIR/../sources}"
SAM3D_OBJECTS_DIR="${SAM3D_OBJECTS_DIR:-$SRC_ROOT/sam-3d-objects}"
SAM3D_BODY_DIR="${SAM3D_BODY_DIR:-$SRC_ROOT/sam-3d-body}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$ROOT_DIR/micromamba}"
ENV_NAME="${ENV_NAME:-sam3d-unified}"
ENV_YAML="${ENV_YAML:-$ROOT_DIR/env-unified.yml}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$ROOT_DIR/bin/micromamba}"

INSTALL_BODY_DEPS="${INSTALL_BODY_DEPS:-1}"
INSTALL_DETECTRON2="${INSTALL_DETECTRON2:-1}"
INSTALL_MOGE="${INSTALL_MOGE:-0}"
INSTALL_SAM3="${INSTALL_SAM3:-0}"
SKIP_NVIDIA_PYINDEX="${SKIP_NVIDIA_PYINDEX:-0}"

PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121}"
PIP_FIND_LINKS="${PIP_FIND_LINKS:-https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0+PTX}"
PACKAGING_VERSION="${PACKAGING_VERSION:-<25}"
IPYKERNEL_VERSION="${IPYKERNEL_VERSION:-<7}"
JUPYTER_CLIENT_VERSION="${JUPYTER_CLIENT_VERSION:-<8}"

log() {
  printf "[sam3d-ready] %s\n" "$*"
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

ensure_env_yaml() {
  if [[ ! -f "$ENV_YAML" ]]; then
    log "Creating env yaml at $ENV_YAML"
    cp "$SAM3D_OBJECTS_DIR/environments/default.yml" "$ENV_YAML"
  fi
  # Always enforce env name
  if grep -q '^name:' "$ENV_YAML"; then
    sed -i "s/^name:.*/name: ${ENV_NAME}/" "$ENV_YAML"
  else
    printf "name: %s\n" "$ENV_NAME" | cat - "$ENV_YAML" >"${ENV_YAML}.tmp"
    mv "${ENV_YAML}.tmp" "$ENV_YAML"
  fi
}

ensure_env() {
  if "$MICROMAMBA" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    log "Env exists: $ENV_NAME (updating)"
    "$MICROMAMBA" env update -f "$ENV_YAML" -n "$ENV_NAME"
  else
    log "Creating env: $ENV_NAME"
    "$MICROMAMBA" env create -f "$ENV_YAML"
  fi
}

run_in_env() {
  "$MICROMAMBA" run -n "$ENV_NAME" "$@"
}

pip_in_env() {
  run_in_env env \
    PIP_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL" \
    PIP_FIND_LINKS="$PIP_FIND_LINKS" \
    python -m pip "$@"
}

install_core() {
  log "Installing core PyTorch (cu121)"
  pip_in_env install --upgrade pip
  run_in_env python -m pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

  log "Installing SAM 3D Objects"
  install_sam3d_objects

  log "Patching hydra for SAM 3D Objects"
  run_in_env python "$SAM3D_OBJECTS_DIR/patching/hydra"
}

install_sam3d_objects() {
  if [[ "$SKIP_NVIDIA_PYINDEX" == "1" ]]; then
    log "Skipping nvidia-pyindex (SKIP_NVIDIA_PYINDEX=1)"
    install_sam3d_objects_manual
    return
  fi

  log "Installing nvidia-pyindex (sdist)"
  if ! pip_in_env install --no-build-isolation nvidia-pyindex==1.0.9; then
    log "nvidia-pyindex install failed; falling back to manual requirements install"
    install_sam3d_objects_manual
    return
  fi

  pip_in_env install -e "$SAM3D_OBJECTS_DIR[dev]"
  pip_in_env install -e "$SAM3D_OBJECTS_DIR[p3d]"
  pip_in_env install -e "$SAM3D_OBJECTS_DIR[inference]"
}

install_sam3d_objects_manual() {
  local req_src req_filtered
  req_src="$SAM3D_OBJECTS_DIR/requirements.txt"
  req_filtered="$ROOT_DIR/requirements.objects.filtered.txt"
  grep -v '^nvidia-pyindex' "$req_src" > "$req_filtered"

  pip_in_env install -r "$req_filtered"
  pip_in_env install --no-build-isolation -r "$SAM3D_OBJECTS_DIR/requirements.p3d.txt"
  run_in_env env \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    PIP_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL" \
    PIP_FIND_LINKS="$PIP_FIND_LINKS" \
    python -m pip install --no-build-isolation -r "$SAM3D_OBJECTS_DIR/requirements.inference.txt"
  pip_in_env install -r "$SAM3D_OBJECTS_DIR/requirements.dev.txt"
  pip_in_env install -e "$SAM3D_OBJECTS_DIR" --no-deps
}

install_body() {
  if [[ "$INSTALL_BODY_DEPS" != "1" ]]; then
    log "Skipping SAM 3D Body deps (INSTALL_BODY_DEPS=0)"
    return
  fi

  log "Installing SAM 3D Body python deps"
  pip_in_env install \
    pytorch-lightning pyrender opencv-python yacs scikit-image einops timm \
    dill pandas rich hydra-colorlog pyrootutils chump networkx==3.2.1 \
    joblib appdirs appnope ffmpeg cython xtcocotools huggingface_hub

  log "Pinning packaging for lightning compatibility"
  pip_in_env install "packaging${PACKAGING_VERSION}"

  if [[ "$INSTALL_DETECTRON2" == "1" ]]; then
    log "Installing Detectron2 (source build)"
    pip_in_env install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
      --no-build-isolation --no-deps
  else
    log "Skipping Detectron2 (INSTALL_DETECTRON2=0)"
  fi

  if [[ "$INSTALL_MOGE" == "1" ]]; then
    log "Installing MoGe (optional)"
    pip_in_env install git+https://github.com/microsoft/MoGe.git
  else
    log "Skipping MoGe (INSTALL_MOGE=0)"
  fi

  if [[ "$INSTALL_SAM3" == "1" ]]; then
    log "Installing SAM3 (optional)"
    if [[ ! -d "$ROOT_DIR/sam3" ]]; then
      git clone https://github.com/facebookresearch/sam3.git "$ROOT_DIR/sam3"
    fi
    pip_in_env install -e "$ROOT_DIR/sam3"
    pip_in_env install decord psutil
  else
    log "Skipping SAM3 (INSTALL_SAM3=0)"
  fi
}

install_jupyter_kernel() {
  log "Installing Jupyter kernel support"
  pip_in_env install "ipykernel${IPYKERNEL_VERSION}" "jupyter-client${JUPYTER_CLIENT_VERSION}"
  run_in_env python -m ipykernel install --user \
    --name "$ENV_NAME" \
    --display-name "${ENV_NAME}"
}

main() {
  ensure_micromamba
  ensure_env_yaml
  ensure_env
  install_core
  install_body
  install_jupyter_kernel
  log "Done. Activate with: $MICROMAMBA run -n $ENV_NAME python -V"
}

main "$@"
