#!/usr/bin/env bash
# Train the encoder checkpoints with the same hyperparameters as the original runs.
# Requires a GPU environment (see requirements.gpu.txt) and CE_DATA with
# train_*.pkl, dev.pkl, tune.pkl plus CE_PREP prepared_gate0_full_{dev,tune}_v1.
# Fitted files are also published at $BER_HF_REPO; scripts/fetch_encoders.py
# downloads them. This script rebuilds all eight members. kD–kG: hard_mix then mine_oof.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/env.sh"
N="${HERE}/../src/neural"
PY="${PYTHON:-python}"
export CE_OUT="${CE_OUT:-$CE_DATA}"

wt() {
  local tag="$1"
  if [[ -f "${CE_OUT%/}/$tag/model.pt" ]]; then
    echo "${CE_OUT%/}/$tag/model.pt"
  elif [[ -f "${BER_WEIGHTS%/}/$tag/model.pt" ]]; then
    echo "${BER_WEIGHTS%/}/$tag/model.pt"
  else
    echo "missing $tag/model.pt in CE_OUT or BER_WEIGHTS" >&2
    exit 1
  fi
}

echo "=== e5_cont1300k (multilingual-e5-small, 1.3M pairs) ==="
"$PY" -u "$N/ce_train.py" --model "$BER_E5" --tag e5_cont1300k --max-train 1300000 --bs 64 --lr 5e-5 --ckpt-every 4000

echo "=== kA (e5-base, seed 7, 1.9M pairs) ==="
"$PY" -u "$N/ce_train.py" --model "$BER_E5_BASE" --tag kA --seed 7 --max-train 1900000 --bs 64 --lr 3e-5 --min-avail-gb 4 --ckpt-every 4000

echo "=== kB (e5-base, seed 8, excludes kA sample) ==="
"$PY" -u "$N/ce_train.py" --model "$BER_E5_BASE" --tag kB --seed 8 --exclude 7:1900000 --max-train 1900000 --bs 64 --lr 3e-5 --min-avail-gb 4 --ckpt-every 4000

echo "=== kC (e5-base, seed 9) ==="
"$PY" -u "$N/ce_train.py" --model "$BER_E5_BASE" --tag kC --seed 9 --max-train 1900000 --bs 64 --lr 3e-5 --min-avail-gb 4 --ckpt-every 4000

echo "=== hard_mix.pkl for kD/kE ==="
"$PY" -u "$N/hard_mix.py" "${CE_DATA%/}/hard_mix.pkl"
HD="${CE_DATA%/}/hard_dir"
mkdir -p "$HD"
ln -sfn "${CE_DATA%/}/hard_mix.pkl" "$HD/train_00.pkl"
ln -sfn "${CE_DATA%/}/dev.pkl" "$HD/dev.pkl"
ln -sfn "${CE_DATA%/}/tune.pkl" "$HD/tune.pkl"

echo "=== kD (hard mix, init kA, seed 21) ==="
CE_DATA="$HD/" CE_OUT="$CE_OUT" "$PY" -u "$N/ce_train.py" --model "$BER_E5_BASE" --tag kD --seed 21 \
  --init "$(wt kA)" --max-train 0 --bs 64 --lr 1e-5 --min-avail-gb 4 --eval-bs 128 --ckpt-every 1000

echo "=== kE (hard mix, init kB, seed 22) ==="
CE_DATA="$HD/" CE_OUT="$CE_OUT" "$PY" -u "$N/ce_train.py" --model "$BER_E5_BASE" --tag kE --seed 22 \
  --init "$(wt kB)" --max-train 0 --bs 64 --lr 1e-5 --min-avail-gb 4 --eval-bs 128 --ckpt-every 1000

echo "=== mine_oof mix for kF/kG ==="
"$PY" -u "$N/mine_oof.py"
MX="${CE_DATA%/}/mine_mix"

echo "=== kF (mined errors, init kA, seed 31) ==="
CE_DATA="$MX/" CE_OUT="$CE_OUT" "$PY" -u "$N/ce_train.py" --model "$BER_E5_BASE" --tag kF --seed 31 \
  --init "$(wt kA)" --max-train 0 --bs 64 --lr 1e-5 --min-avail-gb 4 --eval-bs 128 --ckpt-every 1000

echo "=== kG (mined errors, init kB, seed 32) ==="
CE_DATA="$MX/" CE_OUT="$CE_OUT" "$PY" -u "$N/ce_train.py" --model "$BER_E5_BASE" --tag kG --seed 32 \
  --init "$(wt kB)" --max-train 0 --bs 64 --lr 1e-5 --min-avail-gb 4 --eval-bs 128 --ckpt-every 1000

echo "done. checkpoints under $CE_OUT/{e5_cont1300k,kA,kB,kC,kD,kE,kF,kG}/model.pt"
echo "Distributed training wrappers live in src/kaggle/ and call the same ce_train.py / ce_score.py."
