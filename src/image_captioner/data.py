#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
Project: Image Caption Generator (CNN + LSTM)
File: data.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-10-23
Updated: 2025-10-23
License: MIT License (see LICENSE file for details)
===================================================================

Description:
Data loading & preprocessing for Flickr8k: parse captions, split sets,
clean text, and build (image_path, caption) pairs.

Usage:
from image_captioner.data import load_captions_index

Notes:
- Captions are wrapped with start/end tokens <start>, <end>.
- Basic text normalization: lowercase, remove punct, digits.

===================================================================
"""
from __future__ import annotations
from pathlib import Path
import re
from collections import defaultdict
from typing import Dict, List

from .config import load_config


_PUNCT_RE = re.compile(r"[^a-z ]+")


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"'s", "", text)
    text = re.sub(r"[-/,()!?.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = _PUNCT_RE.sub("", text)
    return text.strip()


def load_splits(cfg=None) -> tuple[set[str], set[str], set[str]]:
    cfg = cfg or load_config()
    train = set(Path(cfg.train_list).read_text().split())
    dev = set(Path(cfg.dev_list).read_text().split())
    test = set(Path(cfg.test_list).read_text().split())
    return train, dev, test


def load_captions_index(cfg=None) -> Dict[str, List[str]]:
    """Return dict: image_name -> list[captions]."""
    cfg = cfg or load_config()
    captions_index: Dict[str, List[str]] = defaultdict(list)
    with open(cfg.captions_file, "r", encoding="utf-8") as f:
        for line in f:
            # format: image#id\tcaption
            if "\t" not in line:
                continue
            key, caption = line.strip().split("\t")
            image_name = key.split("#")[0]
            caption = _normalize(caption)
            if caption:
                captions_index[image_name].append(f"<start> {caption} <end>")
    return captions_index


def build_pairs(cfg=None) -> dict[str, list[str]]:
    cfg = cfg or load_config()
    splits = load_splits(cfg)
    train, dev, test = splits
    caps = load_captions_index(cfg)
    # Filter to files that exist in images_dir
    images_dir = Path(cfg.images_dir)
    result: dict[str, list[str]] = {}
    for img_name, caption_list in caps.items():
        img_path = images_dir / img_name
        if not img_path.exists():
            continue
        if img_name in train or img_name in dev or img_name in test:
            result[str(img_path)] = caption_list
    return result