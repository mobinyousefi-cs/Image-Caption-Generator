#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
Project: Image Caption Generator (CNN + LSTM)
File: infer.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-10-23
Updated: 2025-10-23
License: MIT License (see LICENSE file for details)
===================================================================

Description:
Inference utilities: greedy/beam search decoding to generate captions.

Usage:
from image_captioner.infer import Captioner

Notes:
- Uses saved tokenizer, max length, and trained model.

===================================================================
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from typing import List

from keras.models import load_model

from .config import load_config
from .features import _image_model, _preprocess
from .vocab import load_tokenizer


@dataclass
class Captioner:
    model_path: Path | None = None

    def __post_init__(self):
        self.cfg = load_config()
        self.model_path = Path(self.model_path or self.cfg.model_file)
        self.model = load_model(self.model_path)
        self.encoder = _image_model()
        self.tok = load_tokenizer(self.cfg)
        self.maxlen = int(Path(self.cfg.max_length_file).read_text())
        # Build reverse map
        self.index_word = {i: w for w, i in self.tok.word_index.items()}
        self.index_word[0] = "<pad>"

    def _encode_image(self, image_path: str) -> np.ndarray:
        x = _preprocess(image_path)
        vec = self.encoder.predict(x, verbose=0)[0]
        return vec[None, :]

    def _greedy(self, img_vec: np.ndarray) -> str:
        seq = [self.tok.word_index.get("<start>", 1)]
        for _ in range(self.maxlen):
            padded = self._pad_seq(seq)
            yhat = self.model.predict([img_vec, padded], verbose=0)
            next_id = int(np.argmax(yhat))
            word = self.index_word.get(next_id, "<unk>")
            if word == "<end>":
                break
            seq.append(next_id)
        words = [self.index_word.get(i, "<unk>") for i in seq[1:]]
        return " ".join(words)

    def _pad_seq(self, seq: List[int]) -> np.ndarray:
        arr = np.zeros((1, self.maxlen), dtype=np.int32)
        arr[0, : len(seq)] = seq
        return arr

    def caption(self, image_path: str) -> str:
        img_vec = self._encode_image(image_path)
        return self._greedy(img_vec)