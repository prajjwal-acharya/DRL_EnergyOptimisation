#!/bin/bash
# Mission verification wrapper: run the full pytest suite from the repo root.
# Exists because conductor verification commands cannot contain shell operators
# and are split on spaces, so this script must be reachable via a space-free
# symlink path (e.g. /Volumes/code/rp/automation/check_pytest.sh).
cd "$(cd "$(dirname "$0")" && pwd -P)/.." || exit 1
exec .venv/bin/python -m pytest -q
