# Source from the repository root after cloning.
# Point BER_DATA at a dataset/ directory that contains train/ and test/.

_slash() { case "$1" in */) printf '%s' "$1";; *) printf '%s/' "$1";; esac; }

export BER_ROOT="${BER_ROOT:-$PWD}"
export BER_DATA="$(_slash "${BER_DATA:-$BER_ROOT/data}")"
export CE_DATA="$(_slash "${CE_DATA:-$BER_ROOT/work/ce}")"
export CE_OUT="$(_slash "${CE_OUT:-$CE_DATA}")"
export BER_PKG="${BER_PKG:-$BER_ROOT}"
PKG="$BER_PKG"
export CE_PREP="$(_slash "${CE_PREP:-$PKG/model/prep}")"
export BER_OWNER="${BER_OWNER:-$PKG/model/owner_assignment}"
export BER_WEIGHTS="${BER_WEIGHTS:-$PKG/model/encoders}"
export BER_HF_REPO="${BER_HF_REPO:-DeeAxe/business-entity-matching-encoders}"
export BER_HF_REVISION="${BER_HF_REVISION:-9b04e5a37a28c9d73cd3dfa9953a399b8e18a81e}"
export BER_STORE="${BER_STORE:-$BER_ROOT/work}"
export BER_V2="${BER_V2:-$BER_ROOT/work/v2}"
export BER_E5="${BER_E5:-intfloat/multilingual-e5-small}"
export BER_E5_BASE="${BER_E5_BASE:-intfloat/multilingual-e5-base}"
export BER_E5_REVISION="${BER_E5_REVISION:-614241f622f53c4eeff9890bdc4f31cfecc418b3}"
export BER_E5_BASE_REVISION="${BER_E5_BASE_REVISION:-d128750597153bb5987e10b1c3493a34e5a4502a}"
export BER_TEST_PARTITIONS="${BER_TEST_PARTITIONS:-128}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$CE_DATA"

# Local retrains write CE_OUT. Fitted checkpoints live in BER_WEIGHTS; if a
# file is missing, fetch it from $BER_HF_REPO (scripts/fetch_encoders.py).
ber_encoder_pt() {
  local tag="$1"
  if [[ -f "${CE_OUT%/}/$tag/model.pt" ]]; then
    printf '%s\n' "${CE_OUT%/}/$tag/model.pt"
    return 0
  fi
  if [[ -f "${BER_WEIGHTS%/}/$tag/model.pt" ]]; then
    printf '%s\n' "${BER_WEIGHTS%/}/$tag/model.pt"
    return 0
  fi
  "${PYTHON:-python3}" "$PKG/scripts/fetch_encoders.py" --tag "$tag" --weights "${BER_WEIGHTS%/}" >&2 || true
  if [[ -f "${BER_WEIGHTS%/}/$tag/model.pt" ]]; then
    printf '%s\n' "${BER_WEIGHTS%/}/$tag/model.pt"
    return 0
  fi
  echo "missing ${tag}/model.pt (train, or python $PKG/scripts/fetch_encoders.py --tag $tag)" >&2
  return 1
}

echo "BER_ROOT=$BER_ROOT"
echo "BER_DATA=$BER_DATA"
echo "CE_DATA=$CE_DATA"
echo "CE_OUT=$CE_OUT"
echo "CE_PREP=$CE_PREP"
echo "BER_WEIGHTS=$BER_WEIGHTS"
echo "BER_HF_REPO=$BER_HF_REPO"
echo "BER_HF_REVISION=$BER_HF_REVISION"
echo "BER_E5_REVISION=$BER_E5_REVISION"
echo "BER_E5_BASE_REVISION=$BER_E5_BASE_REVISION"
echo "BER_TEST_PARTITIONS=$BER_TEST_PARTITIONS"
