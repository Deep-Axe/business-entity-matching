---
license: mit
library_name: transformers
base_model:
  - intfloat/multilingual-e5-small
  - intfloat/multilingual-e5-base
tags:
  - entity-matching
  - record-linkage
  - cross-encoder
  - multilingual
---

# Business entity matching encoders

Fine-tuned multilingual-E5 checkpoints used by
[Deep-Axe/business-entity-matching](https://github.com/Deep-Axe/business-entity-matching)
to score candidate business-record pairs.

Each directory contains a `model.pt` state dict for a classification head on
top of E5. Load them through the matching repo (`scripts/fetch_encoders.py`,
`src/neural/ce_score.py`); they are not drop-in `AutoModel` folders.

Pinned snapshot: `9b04e5a37a28c9d73cd3dfa9953a399b8e18a81e`.

## Members

| Path | Base | Training |
| --- | --- | --- |
| `e5_cont1300k/model.pt` | multilingual-e5-small | 1.3M labelled pairs |
| `kA/model.pt` | multilingual-e5-base | 1.9M pairs, seed 7 |
| `kB/model.pt` | multilingual-e5-base | 1.9M pairs, seed 8, excludes kA sample |
| `kC/model.pt` | multilingual-e5-base | 1.9M pairs, seed 9 |
| `kD/model.pt` | multilingual-e5-base | hard-mix continuation of kA |
| `kE/model.pt` | multilingual-e5-base | hard-mix continuation of kB |
| `kF/model.pt` | multilingual-e5-base | mined-error continuation of kA |
| `kG/model.pt` | multilingual-e5-base | mined-error continuation of kB |

The published ensemble score `c8` is the mean logit of all eight members on
the uncertain band (E5-small probability outside the band).

`gate0_100k/enhanced.cbm` is the CatBoost pairwise gate that runs *before*
the encoders. It is small enough to live in the GitHub repo as well.

## Checksums

See `SHA256SUMS` in this repository. Bases are pinned at

- `intfloat/multilingual-e5-small` @ `614241f622f53c4eeff9890bdc4f31cfecc418b3`
- `intfloat/multilingual-e5-base` @ `d128750597153bb5987e10b1c3493a34e5a4502a`

## License

MIT for these fine-tunes. The E5 bases keep their own MIT terms.
