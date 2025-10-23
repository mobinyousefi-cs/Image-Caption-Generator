#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
Project: Image Caption Generator (CNN + LSTM)
File: __init__.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-10-23
Updated: 2025-10-23
License: MIT License (see LICENSE file for details)
===================================================================

Description:
Package initializer for image_captioner. Exposes top-level convenience imports.

Usage:
from image_captioner import load_config

Notes:
- Centralizes common imports to keep notebooks/scripts tidy.

===================================================================
"""
from .config import load_config
from .infer import Captioner