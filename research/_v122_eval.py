# -*- coding: utf-8 -*-
"""Evaluate one RawHebb branch file + checkpoint on the held-out val set."""
import argparse
import importlib.util as ilu
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(fname):
    spec = ilu.spec_from_file_location('m_eval', os.path.join(HERE, fname))
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--val-max', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    m = load_module(args.file)
    tok = m.RawHebbTokenizer(os.path.join(HERE, m.TOK_BIN), os.path.join(HERE, m.TOK_JSON))
    model = m.build_model(tok, torch.device('cpu'))
    sd = torch.load(os.path.join(HERE, args.ckpt), map_location='cpu', weights_only=False)
    cur = model.state_dict()
    sd = {k: v for k, v in sd.items() if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
    model.load_state_dict(sd, strict=False)
    rows = list(m.iter_jsonl(os.path.join(HERE, 'val_dsr1.jsonl')))
    if args.val_max:
        rows = rows[:args.val_max]
    v = m.evaluate(model, tok, rows)
    print('EVAL file=%s ckpt=%s n=%d CE=%.4f top1=%.4f top5=%.4f'
          % (args.file, args.ckpt, v['n'], v['ce'], v['top1'], v['top5']))


if __name__ == '__main__':
    main()
