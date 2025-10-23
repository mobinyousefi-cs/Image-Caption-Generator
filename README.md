# Image Caption Generator (CNN + LSTM)

A clean, reproducible implementation of an image caption generator using **Xception** image features and an **LSTM** decoder, built with TensorFlow/Keras. The repository is structured for research and production with a `src/` layout, tests, CI, and packaging.

<p align="center">
  <img alt="Image Caption Generator" src="https://user-images.githubusercontent.com/placeholder/captioner-demo.png" width="70%" />
</p>

---

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Evaluation (Optional)](#evaluation-optional)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Author](#author)

---

## Features
- Preprocess **Flickr8k** captions with `<start>/<end>` tokens and basic normalization
- Extract 2048‑d image vectors using **Xception** (ImageNet) with global average pooling
- Train a fusion **LSTM** decoder (image embedding + prefix tokens → next token)
- Save artifacts: **tokenizer**, **max length**, **cached features**, and **Keras model** under `artifacts/`
- Minimal **CLI** for training and captioning
- Production‑friendly repo: `src/` package, tests (`pytest`), Ruff + Black linting, GitHub Actions CI

---

## Repository Structure
```
image-caption-generator/
├─ src/
│  └─ image_captioner/
│     ├─ __init__.py          # package init + convenience imports
│     ├─ config.py            # paths, hyperparams, env overrides
│     ├─ data.py              # parse/clean captions, dataset splits
│     ├─ vocab.py             # tokenizer build/save/load, padding
│     ├─ features.py          # Xception feature extraction (NPZ cache)
│     ├─ model.py             # CNN-encoder + LSTM-decoder (Keras)
│     ├─ train.py             # end-to-end training loop
│     ├─ infer.py             # greedy decode; Captioner API
│     └─ utils.py             # seeding, small helpers
├─ tests/
│  └─ test_sanity.py          # smoke tests
├─ notebooks/
│  └─ 00_quick_start.ipynb    # starter notebook (placeholder)
├─ .github/workflows/ci.yml   # lint + test CI
├─ .editorconfig              # consistent editors
├─ .gitignore                 # ignore data/artifacts, etc.
├─ LICENSE                    # MIT
├─ pyproject.toml             # packaging + tooling config
├─ requirements.txt           # pinned baseline dependencies
└─ README.md                  # this file
```

---

## Quickstart
```bash
# 1) Create & activate a virtual environment
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 2) Install deps and project (editable)
pip install -r requirements.txt
pip install -e .

# 3) Put the Flickr8k data under ./data (see next section)

# 4) Train
python -m image_captioner.main train

# 5) Caption an image
python -m image_captioner.main caption path/to/image.jpg
```

> **Python**: 3.10+ recommended (tested with 3.11).  
> **Hardware**: A CUDA‑enabled GPU is recommended for feature extraction and training.

---

## Dataset
Download and extract the Flickr8k archives under `./data/`:

- **Images (~1GB)**  
  https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
- **Captions + splits**  
  https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

Expected layout after extraction:
```
data/
├─ Flickr8k_Dataset/
│  └─ Flicker8k_Dataset/
│     ├─ 1000268201_693b08cb0e.jpg
│     └─ ... (8k images)
└─ Flickr8k_text/
   ├─ Flickr8k.token.txt
   ├─ Flickr_8k.trainImages.txt
   ├─ Flickr_8k.devImages.txt
   └─ Flickr_8k.testImages.txt
```

> **Note**: The file is sometimes named `Flickr8k.token.txt` or `Flickr8k.token`. This repo expects `Flickr8k.token.txt` (you can rename if needed).

---

## Configuration
Most paths and hyperparameters are in `src/image_captioner/config.py`. They can also be overridden via environment variables with the same **UPPERCASE** names as the dataclass fields, e.g.:

```bash
export EPOCHS=10
export BATCH_SIZE=128
export FEATURES_FILE=artifacts/features/xception_features.npz
```

Key options:
- `vocab_size_cap`: cap vocabulary size (e.g., 10k top words)
- `embedding_dim`, `lstm_units`: model sizes
- `epochs`, `batch_size`: training knobs
- `beam_size`: reserved for future beam search (currently greedy decoding)

Directories `artifacts/features`, `artifacts/vocab`, `artifacts/model` are created automatically.

---

## Training
```bash
python -m image_captioner.main train
```
This will:
1. Parse & normalize captions; wrap with `<start>` and `<end>` tokens
2. Build a Keras `Tokenizer` (capped by `vocab_size_cap`)
3. Compute and save `max_length`
4. Extract Xception features to NPZ (cached for subsequent runs)
5. Train the caption model and save to `artifacts/model/captioner.h5`

To restart training from scratch, delete `artifacts/` and rerun.

---

## Inference
```bash
python -m image_captioner.main caption path/to/your_image.jpg
```
- Loads the saved model, tokenizer, and max length
- Encodes the image via Xception → 2048‑d vector
- Uses **greedy decoding** to generate a caption

**Programmatic usage**:
```python
from image_captioner.infer import Captioner
c = Captioner()
print(c.caption("path/to/image.jpg"))
```

---

## Evaluation (Optional)
Basic intrinsic metrics like **BLEU** can be added easily. Example snippet (extend in `tests/` or a notebook):
```python
from nltk.translate.bleu_score import corpus_bleu
# references: list[list[tokenized_ref_caption]], candidates: list[tokenized_hyp]
score = corpus_bleu(references, candidates)
print({"BLEU": score})
```
> BLEU scores on Flickr8k with a simple CNN+LSTM baseline are typically modest (e.g., BLEU‑1 ≈ 0.5–0.6), and improve with attention or transformer decoders.

---

## Troubleshooting
- **TensorFlow not seeing GPU**: verify CUDA/cuDNN versions match TensorFlow; install the correct `tensorflow` build.
- **Out of memory**: lower `batch_size`, or run feature extraction on CPU and training on GPU separately.
- **Missing `Flickr8k.token.txt`**: ensure the captions file name matches `config.py` or set `CAPTIONS_FILE` env var.
- **NPZ not found**: the first training run generates `artifacts/features/xception_features.npz`; don’t forget to extract images correctly.

---

## Roadmap
- [ ] Beam search decoding
- [ ] Add attention mechanism (Bahdanau/Luong)
- [ ] Transformer decoder variant
- [ ] Proper evaluation scripts (BLEU, METEOR, CIDEr)
- [ ] Gradio demo app for interactive captioning

---

## Acknowledgements
- **Flickr8k** dataset split/text files provided via links curated by Jason Brownlee.
- Built with **TensorFlow/Keras** and **Xception** (ImageNet weights).

---

## License
This project is released under the **MIT License**. See [LICENSE](LICENSE).

---

## Author
**Mobin Yousefi** — <https://github.com/mobinyousefi-cs>

