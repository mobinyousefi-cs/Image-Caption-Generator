#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
Project: Image Caption Generator (CNN + LSTM)
File: config.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-10-23
Updated: 2025-10-23
License: MIT License (see LICENSE file for details)
===================================================================

Description:
Configuration utilities and default constants. The values can be overridden
by environment variables or a user-supplied YAML/JSON file in the future.

Usage:
from image_captioner.config import load_config
cfg = load_config()

Notes:
- Paths are relative to the repository root by default.

===================================================================
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Config:
    # Dataset
    data_root: Path = Path("data")
    images_dir: Path = Path("data/Flickr8k_Dataset/Flicker8k_Dataset")
    text_dir: Path = Path("data/Flickr8k_text")
    captions_file: Path = Path("data/Flickr8k_text/Flickr8k.token.txt")
    train_list: Path = Path("data/Flickr8k_text/Flickr_8k.trainImages.txt")
    dev_list: Path = Path("data/Flickr8k_text/Flickr_8k.devImages.txt")
    test_list: Path = Path("data/Flickr8k_text/Flickr_8k.testImages.txt")

    # Features
    features_dir: Path = Path("artifacts/features")
    features_file: Path = Path("artifacts/features/xception_features.npz")

    # Vocabulary
    vocab_dir: Path = Path("artifacts/vocab")
    tokenizer_file: Path = Path("artifacts/vocab/tokenizer.pkl")
    max_length_file: Path = Path("artifacts/vocab/max_length.txt")

    # Model
    model_dir: Path = Path("artifacts/model")
    model_file: Path = Path("artifacts/model/captioner.h5")

    # Training
    seed: int = 42
    batch_size: int = 64
    epochs: int = 20
    embedding_dim: int = 256
    lstm_units: int = 256
    vocab_size_cap: int | None = 10000  # cap top-k words; None = all
    max_length_cap: int | None = None   # if None computed from data
    beam_size: int = 3


def _env_path(key: str, default: Path) -> Path:
    return Path(os.getenv(key, str(default)))


def load_config() -> Config:
    cfg = Config()
    # Allow simple override via environment variables
    for field in cfg.__dataclass_fields__:
        val = getattr(cfg, field)
        if isinstance(val, Path):
            setattr(cfg, field, _env_path(field.upper(), val))
        else:
            env_val = os.getenv(field.upper())
            if env_val is not None:
                cast = type(val)
                try:
                    setattr(cfg, field, cast(env_val))
                except Exception:
                    pass
    # Ensure directories exist
    for p in [cfg.features_dir, cfg.vocab_dir, cfg.model_dir]:
        Path(p).mkdir(parents=True, exist_ok=True)
    return cfg