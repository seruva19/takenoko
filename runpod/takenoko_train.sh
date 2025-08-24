#!/usr/bin/env bash
set -euo pipefail

# Minimal training launcher for RunPod using runpod/takenoko.runpod
CONF_FILE="runpod/takenoko.runpod"

# Load configuration
if [ -f "${CONF_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${CONF_FILE}"
fi

# Defaults
PERSIST_ROOT="${PERSIST_ROOT:-/runpod-volume/takenoko}"
DATA_DIR="${DATA_DIR:-${PERSIST_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PERSIST_ROOT}/output}"
MODELS_DIR="${MODELS_DIR:-${PERSIST_ROOT}/models}"
VENV_DIR="${VENV_DIR:-${PERSIST_ROOT}/venv}"
USE_TORCH_CU126="${USE_TORCH_CU126:-true}"
CONFIG="${CONFIG:-}"
URLS="${URLS:-}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Resolve paths relative to the config file when not absolute
CONF_DIR="$(cd "$(dirname "${CONF_FILE}")" && pwd)"
if [[ -n "${CONFIG}" && ! -f "${CONFIG}" && "${CONFIG}" != /* && -f "${CONF_DIR}/${CONFIG}" ]]; then
  CONFIG="${CONF_DIR}/${CONFIG}"
fi

if [ -z "${CONFIG}" ]; then
  echo "CONFIG is not set in ${CONF_FILE} (path to your training TOML)." >&2
  exit 1
fi

echo "Preparing persistent directories at ${PERSIST_ROOT}"
mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}" "${MODELS_DIR}" "${VENV_DIR}"

cd "${REPO_ROOT}"
if [ ! -e data ]; then ln -s "${DATA_DIR}" data; fi
if [ ! -e output ]; then ln -s "${OUTPUT_DIR}" output; fi
if [ ! -e models ]; then ln -s "${MODELS_DIR}" models; fi

echo "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  build-essential ca-certificates git wget curl ffmpeg unzip tar libgl1 libglib2.0-0 python3-venv
sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

echo "Creating Python venv and installing deps"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [ "${USE_TORCH_CU126}" = "true" ]; then
  pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch==2.6.0+cu126 torchvision==0.21.0+cu126
fi

pip install -e .

# Optional dataset download (direct URLs only)
if [ -n "${URLS}" ]; then
  echo "Downloading dataset into ${DATA_DIR}"
  cd "${DATA_DIR}"
  declare -a _URLS
  read -r -a _URLS <<< "${URLS}"
  for u in "${_URLS[@]:-}"; do
    fname=$(basename "$u")
    echo "-- Fetching: $u"
    if command -v aria2c >/dev/null 2>&1; then
      aria2c -x 8 -s 8 -k 1M -o "$fname" "$u"
    else
      wget -c -O "$fname" "$u"
    fi
    lower="${fname,,}"
    if [[ "$lower" =~ \.zip$ ]]; then
      unzip -o "$fname"
    elif [[ "$lower" =~ \.tar\.gz$ ]] || [[ "$lower" =~ \.tgz$ ]]; then
      tar -xzf "$fname"
    elif [[ "$lower" =~ \.tar$ ]]; then
      tar -xf "$fname"
    fi
  done
fi

echo "Caching latents"
cd "${REPO_ROOT}"
python -u src/takenoko.py --non-interactive --cache-latents --config "${CONFIG}"

echo "Caching text-encoder outputs"
python -u src/takenoko.py --non-interactive --cache-text-encoder --config "${CONFIG}"

echo "Training"
python -u src/takenoko.py --non-interactive --train --config "${CONFIG}"

echo "Done. Outputs in: ${OUTPUT_DIR}"


