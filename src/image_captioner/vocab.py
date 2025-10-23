#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
Project: Image Caption Generator (CNN + LSTM)
File: vocab.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-10-23
Updated: 2025-10-23
License: MIT License (see LICENSE file for details)
===================================================================

Description:
Tokenizer building, saving/loading, and sequence utilities (pad,
compute max length). Uses Keras Tokenizer.

Usage:
from image_captioner.vocab import build_tokenizer, save_tokenizer

Notes:
- Keeps <start>/<end> tokens; oov replaced with <unk>.

===================================================================
"""
from __future__ import annotations
from pathlib import Path
import pickle
from typing import Iterable, Tuple

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from .config import load_config


SPECIAL_TOKENS = {"<pad>": 0}


def build_tokenizer(captions: Iterable[str], num_words: int | None = 10000) -> Tokenizer:
    tok = Tokenizer(num_words=num_words, oov_token="<unk>", filters="")
    tok.fit_on_texts(list(captions))
    # Reserve index 0 for <pad>
    return tok


def save_tokenizer(tok: Tokenizer, cfg=None) -> None:
    cfg = cfg or load_config()
    Path(cfg.vocab_dir).mkdir(parents=True, exist_ok=True)
    with open(cfg.tokenizer_file, "wb") as f:
        pickle.dump(tok, f)


def load_tokenizer(cfg=None) -> Tokenizer:
    cfg = cfg or load_config()
    with open(cfg.tokenizer_file, "rb") as f:
        return pickle.load(f)


def seqs_and_maxlen(tok: Tokenizer, texts: Iterable[str]) -> Tuple[np.ndarray, int]:
    seqs = tok.texts_to_sequences(list(texts))
    maxlen = max(len(s) for s in seqs) if seqs else 0
    padded = pad_sequences(seqs, maxlen=maxlen, padding="post")
    return padded, maxlen


def pad_texts(tok: Tokenizer, texts: Iterable[str], maxlen: int) -> np.ndarray:
    seqs = tok.texts_to_sequences(list(texts))
    return pad_sequences(seqs, maxlen=maxlen, padding="post")