#!/usr/bin/env bash
set -euo pipefail

if [[ "${LEXNLP_USE_STANFORD:-false}" == "true" ]]; then
    SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"
    python3 "${REPOSITORY_ROOT}/scripts/bootstrap_assets.py" \
        --stanford \
        --stanford-dir "${STANFORD_NLP_PATH:-${REPOSITORY_ROOT}/libs/stanford_nlp}"
fi
