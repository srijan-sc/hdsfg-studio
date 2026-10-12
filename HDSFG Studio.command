#!/bin/bash
# Double-click to start HDSFG Studio (macOS). Opens the last data folder used.
# Needs uv (https://docs.astral.sh/uv/); it sets up Python and the libraries on the first run.
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed: run   curl -LsSf https://astral.sh/uv/install.sh | sh   and try again."
    read -r -p "Press Return to close."
    exit 1
fi
exec uv run --python 3.12 hdsfg-studio "$@"
