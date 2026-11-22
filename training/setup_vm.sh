#!/usr/bin/env bash
# Run ON the VM, inside ~/forecasttrust. Creates the environment and runs the CPU unit test.
set -euo pipefail
cd ~/forecasttrust
python3 -m venv .venv
source .venv/bin/activate
pip install -q -U pip
# requirements.txt ends with "-e .", which installs OpenTSLM itself, so run it from that folder
(cd OpenTSLM && pip install -q -r requirements.txt)
pip install -q -e TimeNet/packages/timenet xgboost pytest
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
cd training && python -m pytest test_scoring.py -q -p no:cacheprovider
python baselines_timenet.py --registry ../registry --out ../results/baselines.json
