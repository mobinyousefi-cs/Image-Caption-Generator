#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
Project: Image Caption Generator (CNN + LSTM)
File: main.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-10-23
Updated: 2025-10-23
License: MIT License (see LICENSE file for details)
===================================================================

Description:
CLI entrypoints for training and captioning.

Usage:
python -m image_captioner.main train
python -m image_captioner.main caption path/to/image.jpg

Notes:
- Keeps CLI minimal; extend as needed.

===================================================================
"""
from __future__ import annotations
import sys

from .train import main as train_main
from .infer import Captioner


def _usage() -> None:
    print("Usage: python -m image_captioner.main [train|caption] [image_path]")


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if not argv:
        _usage()
        return
    cmd = argv[0]
    if cmd == "train":
        train_main()
    elif cmd == "caption":
        if len(argv) < 2:
            _usage()
            return
        c = Captioner()
        print(c.caption(argv[1]))
    else:
        _usage()


if __name__ == "__main__":
    main()