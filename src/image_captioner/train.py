#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
Project: Image Caption Generator (CNN + LSTM)
File: train.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-10-23
Updated: 2025-10-23
License: MIT License (see LICENSE file for details)
===================================================================

Description:
End-to-end training script: builds tokenizer, prepares sequences,
loads Xception features, trains LSTM decoder, and saves artifacts.

Usage:
python -m image_captioner.train

Notes:
- Uses simple many-to-one training (predict next word given prefix + image).

===================================================================
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm

from .config import load_config
from .data import build_pairs, load_splits
from .features import load_features, extract_and_save
from .vocab import build_tokenizer, save_tokenizer, seqs_and_maxlen, pad_texts
from .model import build_caption_model
from .utils import set_seed


def _create_training_samples(captions: List[str], tok, maxlen: int) -> Tuple[np.ndarray, np.ndarray]:
    X_seq, y = [], []
    seq = tok.texts_to_sequences(captions)
    # Flatten per caption into many prefix -> next token pairs
    for s in seq:
        for i in range(1, len(s)):
            in_seq, out_tok = s[:i], s[i]
            X_seq.append(in_seq)
            y.append(out_tok)
    X_seq = pad_texts(tok, X_seq, maxlen)
    y = np.array(y)
    return X_seq, y


def main() -> None:
    cfg = load_config()
    set_seed(cfg.seed)

    pairs = build_pairs(cfg)
    train, dev, _ = load_splits(cfg)

    # Flatten captions for tokenizer
    all_caps = []
    for c_list in pairs.values():
        all_caps.extend(c_list)

    tok = build_tokenizer(all_caps, num_words=cfg.vocab_size_cap)
    save_tokenizer(tok, cfg)

    # Compute max length if not fixed
    _, maxlen = seqs_and_maxlen(tok, all_caps)
    if cfg.max_length_cap:
        maxlen = min(maxlen, cfg.max_length_cap)
    Path(cfg.max_length_file).write_text(str(maxlen))

    # Features
    if not Path(cfg.features_file).exists():
        extract_and_save(cfg)
    feats = load_features(cfg)

    # Build dataset indices
    def is_split(img_name: str, split_set) -> bool:
        return img_name in split_set

    X_img_train, X_seq_train, y_train = [], [], []
    X_img_dev, X_seq_dev, y_dev = [], [], []

    for img_path, c_list in tqdm(pairs.items(), desc="prepare"):
        img_name = Path(img_path).name
        if img_name not in feats:
            continue
        # Generate (seq_prefix -> next_token) samples
        Xs, ys = _create_training_samples(c_list, tok, maxlen)
        Ximg = np.repeat(feats[img_name][None, :], repeats=len(ys), axis=0)
        if is_split(img_name, train):
            X_img_train.append(Ximg)
            X_seq_train.append(Xs)
            y_train.append(ys)
        elif is_split(img_name, dev):
            X_img_dev.append(Ximg)
            X_seq_dev.append(Xs)
            y_dev.append(ys)

    X_img_train = np.vstack(X_img_train)
    X_seq_train = np.vstack(X_seq_train)
    y_train = np.array(np.concatenate(y_train))

    X_img_dev = np.vstack(X_img_dev)
    X_seq_dev = np.vstack(X_seq_dev)
    y_dev = np.array(np.concatenate(y_dev))

    model, _ = build_caption_model(
        vocab_size=min(len(tok.word_index) + 1, tok.num_words or (len(tok.word_index) + 1)),
        max_length=maxlen,
        embedding_dim=cfg.embedding_dim,
        lstm_units=cfg.lstm_units,
    )

    model.summary()

    history = model.fit(
        [X_img_train, X_seq_train], y_train,
        validation_data=([X_img_dev, X_seq_dev], y_dev),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=1,
    )

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    model.save(cfg.model_file)
    print(f"Saved model -> {cfg.model_file}")


if __name__ == "__main__":
    main()