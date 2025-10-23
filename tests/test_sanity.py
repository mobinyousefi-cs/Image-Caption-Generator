#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
Project: Image Caption Generator (CNN + LSTM)
File: test_sanity.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-10-23
Updated: 2025-10-23
License: MIT License (see LICENSE file for details)
===================================================================

Description:
Basic smoke tests to ensure modules import and config loads.

Usage:
pytest -q

Notes:
- Extend with dataset-specific checks when data is available.

===================================================================
"""
from image_captioner import load_config


def test_config_paths_exist():
    cfg = load_config()
    assert cfg is not None