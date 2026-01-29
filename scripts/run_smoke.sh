#!/usr/bin/env bash
set -euo pipefail
python3 -m pytest -q
python3 scripts/run_experiment.py --config configs/experiment.yaml
