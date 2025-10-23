#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
Project: Image Caption Generator (CNN + LSTM)
File: features.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-10-23
Updated: 2025-10-23
License: MIT License (see LICENSE file for details)
===================================================================

Description:
Feature extraction using Xception pre-trained on ImageNet. Produces a
fixed 2048-d vector per image (global average pooled) and stores as NPZ.

Usage:
from image_captioner.features import extract_and_save

Notes:
- Lazily loads model; caches features to NPZ for faster training.

===================================================================
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
from keras.applications.xception import Xception, preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array

from .config import load_config
from .data import build_pairs
from .utils import set_seed


_IMG_SIZE = (299, 299)


def _image_model() -> Model:
    base = Xception(include_top=False, weights="imagenet")
    x = GlobalAveragePooling2D()(base.output)
    return Model(inputs=base.input, outputs=x)


def _preprocess(path: str) -> np.ndarray:
    img = load_img(path, target_size=_IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


def extract_and_save(cfg=None) -> Path:
    cfg = cfg or load_config()
    set_seed(cfg.seed)
    pairs = build_pairs(cfg)
    model = _image_model()

    feats: Dict[str, np.ndarray] = {}
    for img_path in pairs.keys():
        x = _preprocess(img_path)
        vec = model.predict(x, verbose=0)[0]
        feats[Path(img_path).name] = vec.astype(np.float32)

    Path(cfg.features_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.features_file, **feats)
    return cfg.features_file


def load_features(cfg=None) -> dict[str, np.ndarray]:
    cfg = cfg or load_config()
    data = np.load(cfg.features_file, allow_pickle=False)
    return {k: data[k] for k in data.files}