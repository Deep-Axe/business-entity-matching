# Pipeline

End-to-end path from raw TSVs to exclusive matches. Source `scripts/env.sh` first. Encoder weights belong in `BER_WEIGHTS` (`python scripts/fetch_encoders.py`). Pair pickles and score arrays go under `CE_DATA` (`work/ce` by default).

## Data

`BER_DATA` should contain:

```
train/train_source1.tsv
train/train_source2.tsv
train/train_matches.tsv          # labelled owners, when training
test/test_source1.tsv
test/test_source2.tsv
```

Each row is a business record (`id`, `name`, `address`, `city`, `country`, …). Matches are pairs `(source1_id, source2_id)`.

## 1. Store and BM25 index

```bash
python -m ber build-store --data-dir "$BER_DATA" --output work/store
python -m ber build-index --store work/store --output work/index
```

The store is a SQLite snapshot of both catalogs. The index is a Tantivy BM25 index over normalized name and address text, with optional Indic transliteration.

## 2. Candidate features and CatBoost gate

```bash
python -m ber features --store work/store --index work/index \
  --references "$BER_DATA/test/test_source1.tsv" \
  --output work/features --unlabeled --top-k 20

python -m ber predict-shards --features work/features \
  --model model/gate0_100k/enhanced.cbm \
  --threshold 0.76 --output work/gate
```

Training the gate (100k labelled owners, enhanced 76-feature view) lives in `src/tools/v2_fit_cpu.py`. Thresholds for later stages are chosen on the **full** source-2 gallery: omit `--gallery-fold` when you featurize tune/dev for calibration. Identity-only training registries may stay fold-restricted.

## 3. Cross-encoder scores

Pair pickles for the uncertain band are produced by `src/neural/ce_data.py`. Score each member, then average logits:

```bash
python src/neural/ce_score.py --tag e5_cont1300k
python src/neural/ce_score.py --tag kA   # … kB–kG
python src/neural/make_c8.py
```

`make_c8.py` concatenates band chunks in **numeric** order (`band_chunk_paths`). Lexicographic `0,1,10,11,2,…` is wrong and will silently misalign the stacker.

Hub bases are pinned:

- `intfloat/multilingual-e5-small` @ `614241f622f53c4eeff9890bdc4f31cfecc418b3`
- `intfloat/multilingual-e5-base` @ `d128750597153bb5987e10b1c3493a34e5a4502a`

`src/neural/ber_paths.py` `pretrained_kwargs` uses those revisions whenever the path is a Hub id rather than a local directory.

## 4. Stacker

```bash
NAMEF=1 NAMEF2=1 NOCTY=1 python src/neural/stack_eval.py c8 --test
```

GroupKFold is by source-1 id. `NOCTY=1` drops the country one-hot; `NAMEF` / `NAMEF2` add character-level name features. Threshold `t = 0.75` is selected on tune out-of-fold scores.

## 5. Recovery heads

Dense neighbours that BM25 never proposed:

```bash
python src/neural/dense_search.py
```

Exact-key channels (India / US identifiers) are built by `src/neural/keych_test_rows.py` and scored with the same stacker under `TESTGLOB`. France is not admitted through the key head; assignment uses `France=0.9`.

## 6. Exclusive assignment

```bash
python src/neural/assign.py --w 0 --t 0.75 \
  --score stack_c8_nocty_nf2 \
  --country-t France=0.9 \
  --extra-glob 'work/ce/dense/*.pkl;work/ce/keych/*.pkl' \
  --extra-t 0.9 --strict --out work/matches
```

`--strict` checks **each** extra glob on its own, and checks that the number of test partitions equals `BER_TEST_PARTITIONS` (default 128) rather than “however many files happen to be on disk”. Empty routes must still publish `complete.json` with `rows=0`.

`winner()` in `src/tools/v3_owner_assignment.py` is deletion-only: it never steals a claimed source-2 record from a higher-scoring owner.

## Retraining encoders

```bash
source scripts/env.sh
bash scripts/train_encoders.sh
```

That script is the record of the published hyperparameters: E5-small at 1.3M pairs; `kA`–`kC` at 1.9M pairs on e5-base; `kD`/`kE` on `hard_mix.pkl`; `kF`/`kG` on mined OOF errors. Fitted files are also the contents of the Hugging Face repo.
