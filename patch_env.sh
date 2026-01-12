#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="/usr/lib64-nvidia"
CONF_FILE="/etc/ld.so.conf.d/nvidia.conf"
IPY_CONF="/etc/ipython/ipython_config.py"
IPY_MARKER="# sam3d-ready override (disable colab kernel)"
PATCH_LDCONFIG="${PATCH_LDCONFIG:-1}"
PATCH_IPYTHON="${PATCH_IPYTHON:-1}"

log() {
  printf "[patch-env] %s\n" "$*"
}

require_root() {
  if [[ "$(id -u)" != "0" ]]; then
    log "Error: must run as root"
    exit 1
  fi
}

file_has_line() {
  local needle="$1"
  local file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -q --fixed-strings --line-regexp "$needle" "$file"
  else
    grep -Fxq "$needle" "$file"
  fi
}

patch_ldconfig() {
  if [[ ! -d "${TARGET_DIR}" ]]; then
    log "Error: ${TARGET_DIR} not found"
    exit 1
  fi

  if [[ ! -e "${TARGET_DIR}/libnvidia-ml.so.1" && ! -e "${TARGET_DIR}/libnvidia-ml.so" ]]; then
    log "Error: libnvidia-ml.so not found in ${TARGET_DIR}"
    exit 1
  fi

  local changed=0
  if [[ -f "${CONF_FILE}" ]]; then
    if file_has_line "${TARGET_DIR}" "${CONF_FILE}"; then
      log "${TARGET_DIR} already present in ${CONF_FILE}"
    else
      echo "${TARGET_DIR}" >> "${CONF_FILE}"
      log "Appended ${TARGET_DIR} to ${CONF_FILE}"
      changed=1
    fi
  else
    printf "%s\n" "${TARGET_DIR}" > "${CONF_FILE}"
    log "Created ${CONF_FILE}"
    changed=1
  fi

  if [[ "${changed}" -eq 1 ]]; then
    ldconfig
    log "ldconfig updated"
  else
    log "ldconfig skipped (no changes)"
  fi
}

patch_ipython() {
  if [[ ! -f "${IPY_CONF}" ]]; then
    log "Error: ${IPY_CONF} not found"
    exit 1
  fi

  if command -v rg >/dev/null 2>&1; then
    if ! rg -q "google\\.colab" "${IPY_CONF}"; then
      log "No google.colab config detected; skipping IPython patch"
      return
    fi
  else
    if ! grep -q "google\\.colab" "${IPY_CONF}"; then
      log "No google.colab config detected; skipping IPython patch"
      return
    fi
  fi

  if command -v rg >/dev/null 2>&1; then
    if rg -qF "${IPY_MARKER}" "${IPY_CONF}"; then
      log "IPython override already present in ${IPY_CONF}"
      return
    fi
  else
    if grep -qF "${IPY_MARKER}" "${IPY_CONF}"; then
      log "IPython override already present in ${IPY_CONF}"
      return
    fi
  fi

  cat >> "${IPY_CONF}" <<'EOF'

# sam3d-ready override (disable colab kernel)
c.IPKernelApp.kernel_class = 'ipykernel.ipkernel.IPythonKernel'
c.InteractiveShellApp.extensions = []
c.InteractiveShellApp.exec_lines = []
c.InteractiveShellApp.reraise_ipython_extension_failures = False
EOF

  log "Appended IPython override to ${IPY_CONF}"
}

main() {
  require_root
  if [[ "${PATCH_LDCONFIG}" == "1" ]]; then
    patch_ldconfig
  else
    log "Skipping ldconfig patch (PATCH_LDCONFIG=0)"
  fi

  if [[ "${PATCH_IPYTHON}" == "1" ]]; then
    patch_ipython
  else
    log "Skipping IPython patch (PATCH_IPYTHON=0)"
  fi
  log "Done"
}

main "$@"
