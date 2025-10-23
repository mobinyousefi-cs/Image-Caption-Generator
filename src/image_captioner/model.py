#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
Project: Image Caption Generator (CNN + LSTM)
File: model.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-10-23
Updated: 2025-10-23
License: MIT License (see LICENSE file for details)
===================================================================

Description:
Defines the CNN-Encoder + LSTM-Decoder captioning model using Keras.

Usage:
from image_captioner.model import build_caption_model

Notes:
- Simple additive fusion of image vector and word embedding.
- Teacher forcing training with categorical cross-entropy.

===================================================================
"""
from __future__ import annotations
from typing import Tuple

from keras import Input, Model
from keras.layers import Dense, Embedding, LSTM, Dropout, Add
from keras.layers import Activation
from keras.optimizers import Adam


def build_caption_model(vocab_size: int, max_length: int, embedding_dim: int = 256, lstm_units: int = 256) -> Tuple[Model, Model]:
    # Image feature branch (2048)
    img_input = Input(shape=(2048,), name="image_feat")
    x_img = Dropout(0.5)(img_input)
    x_img = Dense(embedding_dim, activation="relu", name="img_dense")(x_img)

    # Text branch
    seq_input = Input(shape=(max_length,), name="seq_input")
    x_txt = Embedding(vocab_size, embedding_dim, mask_zero=True, name="emb")(seq_input)
    x_txt = Dropout(0.5)(x_txt)
    x_txt = LSTM(lstm_units, name="lstm")(x_txt)

    # Fusion
    x = Add(name="fusion")([x_img, x_txt])
    x = Dense(lstm_units, activation="relu")(x)
    out = Dense(vocab_size, activation="softmax", name="softmax")(x)

    model = Model(inputs=[img_input, seq_input], outputs=out, name="captioner")
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(1e-3))

    # Encoder for inference (image -> embedding)
    encoder = Model(inputs=img_input, outputs=x_img, name="encoder")
    return model, encoder