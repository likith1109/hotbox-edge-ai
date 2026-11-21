#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python train.py --team 25 --hidden 32,32 --epochs 1500 --skip-plots --output-dir artifacts/team_25/mission2/32-32_raw --model-dir artifacts/team_25/mission2/32-32_raw/model --no-normalise
python train.py --team 25 --hidden 4 --epochs 1500 --skip-plots --output-dir artifacts/team_25/mission2/4_normalised --model-dir artifacts/team_25/mission2/4_normalised/model
python train.py --team 25 --hidden 4,4 --epochs 1500 --skip-plots --output-dir artifacts/team_25/mission2/4-4_normalised --model-dir artifacts/team_25/mission2/4-4_normalised/model
python train.py --team 25 --hidden 8 --epochs 1500 --skip-plots --output-dir artifacts/team_25/mission2/8_normalised --model-dir artifacts/team_25/mission2/8_normalised/model
python train.py --team 25 --hidden 8,8 --epochs 1500 --skip-plots --output-dir artifacts/team_25/mission2/8-8_normalised --model-dir artifacts/team_25/mission2/8-8_normalised/model
python train.py --team 25 --hidden 16 --epochs 1500 --skip-plots --output-dir artifacts/team_25/mission2/16_normalised --model-dir artifacts/team_25/mission2/16_normalised/model
python train.py --team 25 --hidden 16,16 --epochs 1500 --skip-plots --output-dir artifacts/team_25/mission2/16-16_normalised --model-dir artifacts/team_25/mission2/16-16_normalised/model
python train.py --team 25 --hidden 32 --epochs 1500 --skip-plots --output-dir artifacts/team_25/mission2/32_normalised --model-dir artifacts/team_25/mission2/32_normalised/model
python train.py --team 25 --hidden 32,32 --epochs 1500 --skip-plots --output-dir artifacts/team_25/mission2/32-32_normalised --model-dir artifacts/team_25/mission2/32-32_normalised/model
python train.py --team 25 --hidden 64 --epochs 1500 --skip-plots --output-dir artifacts/team_25/mission2/64_normalised --model-dir artifacts/team_25/mission2/64_normalised/model
python train.py --team 25 --hidden 64,64 --epochs 1500 --skip-plots --output-dir artifacts/team_25/mission2/64-64_normalised --model-dir artifacts/team_25/mission2/64-64_normalised/model
python train.py --team 25 --hidden 16,16 --epochs 1500 --skip-plots --output-dir artifacts/team_25/mission2/selected_retrain --model-dir model
