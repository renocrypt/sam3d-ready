# sam3d-ready

Goal: one micromamba environment that can run SAM 3D Objects and SAM 3D Body.

## Expected layout (default)
This folder sits next to `sources/`:
- `../sources/sam-3d-objects`
- `../sources/sam-3d-body`

Override paths with:
- `SRC_ROOT`
- `SAM3D_OBJECTS_DIR`
- `SAM3D_BODY_DIR`

## Quickstart
Use the automation script (recommended):
```
./setup_env.sh
```

The script is idempotent and will:
- install micromamba if missing (into `./bin`)
- create/update the env from `env-unified.yml`
- install SAM 3D Objects deps (torch 2.5.1 + cu121) and patch hydra
- install SAM 3D Body deps + Detectron2 (optional)
- register a Jupyter kernel named `ENV_NAME`

Optional flags:
```
SRC_ROOT=../sources \
SAM3D_OBJECTS_DIR=../sources/sam-3d-objects \
SAM3D_BODY_DIR=../sources/sam-3d-body \
ENV_NAME=sam3d-unified \
ENV_YAML=./env-unified.yml \
INSTALL_BODY_DEPS=1 \
INSTALL_DETECTRON2=1 \
INSTALL_MOGE=0 \
INSTALL_SAM3=0 \
SKIP_NVIDIA_PYINDEX=0 \
TORCH_CUDA_ARCH_LIST=7.0+PTX \
PACKAGING_VERSION=<25 \
IPYKERNEL_VERSION=<7 \
JUPYTER_CLIENT_VERSION=<8 \
./setup_env.sh
```

## Jupyter kernel support
The script installs `ipykernel` and registers a kernel named `sam3d-unified`. You can use it from Jupyter Lab/Notebook for development.

## Notes
- The unified stack follows SAM 3D Objects' CUDA 12.1 / torch 2.5.1 requirements.
- SAM 3D Body's data prep uses a separate env upstream (python 3.9 + torch 2.4.0 + cu118).
- Heavy source builds: pytorch3d, flash-attn, gsplat, detectron2.

## If the unified env fails
- Keep separate envs (objects cu121; body separate).
- Downgrade torch to 2.4.1 for pytorch3d.
- Containerize each project.
