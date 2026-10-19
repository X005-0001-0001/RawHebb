# -*- coding: utf-8 -*-
"""v121 diagnostic runner: fresh 0.6.0.0 baseline (no brain.pt warm start)."""
import argparse
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('rh060', os.path.join(HERE, 'RawHebb+bio+0.6.0.0.py'))
rh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rh)

ap = argparse.ArgumentParser()
ap.add_argument('--limit', type=int, default=500)
ap.add_argument('--epochs', type=int, default=1)
ap.add_argument('--val-max', type=int, default=40)
ap.add_argument('--threads', type=int, default=3)
ap.add_argument('--save', default='_v121_baseline.pt')
args = ap.parse_args()

rh.DEVICE = 'cpu'
torch.set_num_threads(args.threads)
rh.deterministic(rh.SEED)
tok = rh.RawHebbTokenizer(os.path.join(HERE, rh.TOK_BIN), os.path.join(HERE, rh.TOK_JSON))
model = rh.build_model(tok, torch.device('cpu'))
rh.train_hebbian(model, tok, 'train_dsr1.jsonl', epochs=args.epochs, limit=args.limit,
                 val_file='val_dsr1.jsonl', val_max=args.val_max)
rh.save_model(model, args.save)
