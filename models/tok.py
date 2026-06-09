#!/usr/bin/env python3.12
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026  Anton Maurer

"""Tokenize text using dots.tts Mistral tokenizer, save token IDs as binary."""
import sys, numpy as np
sys.path.insert(0, '/tmp/dots_tts_py')
from transformers import AutoTokenizer

tok_path = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
tok = AutoTokenizer.from_pretrained(tok_path)

text = sys.argv[1] if len(sys.argv) > 1 else "Hello world"
ids = tok(text, return_tensors='pt').input_ids[0].numpy().astype(np.int32)
ids.tofile('tokens.bin')
print(f'Tokenized "{text}" -> {len(ids)} tokens: {ids.tolist()}')
