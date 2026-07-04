#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINGBOT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${LINGBOT_ROOT}"

SERVER_HOST="${LINGBOT_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${LINGBOT_SERVER_PORT:-29536}"
PROMPT="${LINGBOT_PROMPT:-serve bread}"
SIDE="${G1_LINGBOT_SIDE:-right}"
TAG="${G1_LINGBOT_TAG:-lingbot_g1_offline}"
OUT_DIR="${G1_LINGBOT_OUT_DIR:-./evaluation/g1/artifacts/lingbot_offline_policy_check}"
CFG="${G1_LINGBOT_CFG:-./evaluation/g1/cfg/g1_serve_bread_right.yaml}"
PYTHON_BIN="${G1_LINGBOT_PYTHON:-python3}"

"${PYTHON_BIN}" evaluation/g1/g1_lingbot_offline_policy_check.py \
  --server-host "${SERVER_HOST}" \
  --server-port "${SERVER_PORT}" \
  --prompt "${PROMPT}" \
  --side "${SIDE}" \
  --tag "${TAG}" \
  --out-dir "${OUT_DIR}" \
  --cfg "${CFG}" \
  --no-close-camera \
  --no-close-arm
