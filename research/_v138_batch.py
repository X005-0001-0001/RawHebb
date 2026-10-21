# -*- coding: utf-8 -*-
"""Create a batched screening branch from 0.14.0.0.

The online update is kept per dialogue (sliced caches), but the forward pass can
run B dialogues at once.  This is an experiment: within a batch, later forwards
see the memory state from the batch start.
"""
import io

src = r'D:\explore\wav\RawHebb+bio+0.14.0.0.py'
dst = r'D:\explore\wav\RawHebb+batch+bag+0.9.192.0.py'
t = io.open(src, encoding='utf-8').read()

# Name / version.
t = t.replace('RawHebb+bio+0.14.0.0', 'RawHebb+batch+bag+0.9.192.0', 1)
t = t.replace("APP_NAME = 'RawHebb'", "APP_NAME = 'RawHebb'", 1)
t = t.replace("APP_BRANCH = 'bio'", "APP_BRANCH = 'batch+bag'", 1)
t = t.replace("APP_VERSION = '0.14.0.0'", "APP_VERSION = '0.9.192.0'", 1)

# ---- forward: make tri / bag / four batch-aware ----
old = '''            if TRIGRAM_MEMORY:
                _ids = input_ids[0]
                _pad = torch.full((1,), int(getattr(self, 'pad_id', 0)),
                                  dtype=_ids.dtype, device=dev)
                _prev1 = torch.cat([_pad, _ids[:-1]])
                _prev2 = torch.cat([_pad, _pad, _ids[:-2]])
                _keys = ((_prev2 * self._tri_base + _prev1) % TRIGRAM_BUCKETS)
                _trow = self.tri_count[_keys]
                _trs = _trow.sum(dim=-1, keepdim=True)
                _tp = (_trow + TRI_PRIOR_STRENGTH * _prior) / (_trs + TRI_PRIOR_STRENGTH)
                _experts.append(torch.log(_tp).unsqueeze(0))
            if FOUR_BAG_PRIOR:
                _bagrows = self.pair_count[input_ids[0]]
                _bagcum = torch.cumsum(_bagrows, dim=0)
                _bagprev = torch.cat(
                    [torch.zeros(1, self.vocab_size, dtype=_bagcum.dtype, device=_bagcum.device),
                     _bagcum[:-1]], dim=0)
                _bagsum = _bagprev.sum(dim=-1, keepdim=True)
                _bagp = ((_bagprev + PAIR_PRIOR_STRENGTH * _prior)
                         / (_bagsum + PAIR_PRIOR_STRENGTH))
            if FOURGRAM_MEMORY:
                _f1 = _prev1
                _f2 = _prev2
                _p3 = torch.cat([_pad, _pad, _pad, _ids[:-3]])
                _fkeys = (((_p3 * self._tri_base + _f2) * self._tri_base + _f1)
                          % FOURGRAM_BUCKETS)
                _frow = self.four_count[_fkeys]
                _frs = _frow.sum(dim=-1, keepdim=True)
                _fp = ((_frow + FOUR_PRIOR_STRENGTH * _bagp)
                       / (_frs + FOUR_PRIOR_STRENGTH))
                _experts.append(torch.log(_fp.clamp(min=1e-12)).unsqueeze(0))'''
new = '''            if TRIGRAM_MEMORY:
                _ids = input_ids
                _pad = torch.full((B, 1), int(getattr(self, 'pad_id', 0)),
                                  dtype=_ids.dtype, device=dev)
                _prev1 = torch.cat([_pad, _ids[:, :-1]], dim=1)
                _prev2 = torch.cat([_pad, _pad, _ids[:, :-2]], dim=1)
                _keys = ((_prev2 * self._tri_base + _prev1) % TRIGRAM_BUCKETS)
                _trow = self.tri_count[_keys]
                _trs = _trow.sum(dim=-1, keepdim=True)
                _tp = (_trow + TRI_PRIOR_STRENGTH * _prior) / (_trs + TRI_PRIOR_STRENGTH)
                _experts.append(torch.log(_tp))
            if FOUR_BAG_PRIOR:
                _bagrows = self.pair_count[input_ids]
                _bagcum = torch.cumsum(_bagrows, dim=1)
                _bagprev = torch.cat(
                    [torch.zeros(B, 1, self.vocab_size, dtype=_bagcum.dtype,
                                 device=_bagcum.device), _bagcum[:, :-1]], dim=1)
                _bagsum = _bagprev.sum(dim=-1, keepdim=True)
                _bagp = ((_bagprev + PAIR_PRIOR_STRENGTH * _prior)
                         / (_bagsum + PAIR_PRIOR_STRENGTH))
            if FOURGRAM_MEMORY:
                _f1 = _prev1
                _f2 = _prev2
                _p3 = torch.cat([_pad, _pad, _pad, _ids[:, :-3]], dim=1)
                _fkeys = (((_p3 * self._tri_base + _f2) * self._tri_base + _f1)
                          % FOURGRAM_BUCKETS)
                _frow = self.four_count[_fkeys]
                _frs = _frow.sum(dim=-1, keepdim=True)
                _fp = ((_frow + FOUR_PRIOR_STRENGTH * _bagp)
                       / (_frs + FOUR_PRIOR_STRENGTH))
                _experts.append(torch.log(_fp.clamp(min=1e-12)))'''
if old not in t:
    raise SystemExit('forward block not found')
t = t.replace(old, new, 1)

# Cache unflattened readouts so a per-dialogue update can slice them.
old = "        self._buf['lm_flat'] = [lm[:, :T - 1, :].reshape(-1, self.vocab_size) for lm in lms]"
new = ("        self._buf['lm_seq'] = [lm[:, :T - 1, :] for lm in lms]\n"
       "        self._buf['lm_flat'] = [lm[:, :T - 1, :].reshape(-1, self.vocab_size) for lm in lms]")
if old not in t:
    raise SystemExit('lm_flat not found')
t = t.replace(old, new, 1)
old = "        lm_flat = self._buf['lm_flat']"
new = "        lm_flat = [x.reshape(-1, self.vocab_size) for x in self._buf['lm_seq']]"
if old not in t:
    raise SystemExit('lm_flat use not found')
t = t.replace(old, new, 1)

# ---- train signature + batch loop ----
old = "def train_hebbian(model, tok, path, epochs=EPOCHS, limit=LIMIT_SAMPLES, val_file=None,\n                  val_max=40, eval_every=0, logfile=None):"
new = "def train_hebbian(model, tok, path, epochs=EPOCHS, limit=LIMIT_SAMPLES, val_file=None,\n                  val_max=40, eval_every=0, logfile=None, batch_size=1):"
if old not in t:
    raise SystemExit('signature not found')
t = t.replace(old, new, 1)

old = '''        ce_sum, n_ce, t_ep = 0.0, 0, time.time()
        for k, rec in enumerate(rows):
            ids, labels = tok.chat_to_ids(get_messages(rec))
            if (labels != -100).sum() == 0:
                continue
            ids = ids.unsqueeze(0).to(dev)
            labels = labels.unsqueeze(0).to(dev)
            model(ids, labels=labels)
            model.hebbian_update(ids)
            n = int((labels[:, :-1] != -100).sum())
            ce_sum += float(model._buf.get('ce', float('nan'))) * n
            n_ce += n
            if eval_every and (k + 1) % eval_every == 0 and val_rows:
                v = evaluate(model, tok, val_rows)
                log_and_print('[MID] ep%d %d/%d ｜ 训练CE=%.4f ｜ **验证CE=%.4f** top1=%.4f ｜ %.0fs'
                              % (ep, k + 1, len(rows), ce_sum / max(n_ce, 1), v['ce'], v['top1'],
                                 time.time() - t0))
            if (k + 1) % LOG_INTERVAL == 0:
                log_and_print('[TRAIN] ep%d %d/%d ｜ CE(累计)=%.4f ｜ %.1fs'
                              % (ep, k + 1, len(rows), ce_sum / max(n_ce, 1), time.time() - t_ep))'''
new = '''        ce_sum, n_ce, t_ep = 0.0, 0, time.time()

        def _slice_buf(buf, i):
            out = {}
            for key, val in buf.items():
                if torch.is_tensor(val):
                    out[key] = val[i:i + 1]
                elif isinstance(val, list):
                    out[key] = [x[i:i + 1] if torch.is_tensor(x) else x for x in val]
                else:
                    out[key] = val
            return out

        for start in range(0, len(rows), max(1, batch_size)):
            chunk = rows[start:start + max(1, batch_size)]
            pairs = [tok.chat_to_ids(get_messages(rec)) for rec in chunk]
            pairs = [(x, y) for x, y in pairs if int((y != -100).sum()) > 0]
            if not pairs:
                continue
            _T = max(int(x.shape[0]) for x, _ in pairs)
            _B = len(pairs)
            ids = torch.full((_B, _T), model.pad_id, dtype=torch.long)
            labels = torch.full((_B, _T), -100, dtype=torch.long)
            for i, (x, y) in enumerate(pairs):
                ids[i, :x.shape[0]] = x
                labels[i, :y.shape[0]] = y
            ids = ids.to(dev)
            labels = labels.to(dev)
            model(ids, labels=labels)
            _all = model._buf
            for i in range(_B):
                model._buf = _slice_buf(_all, i)
                model.hebbian_update(ids[i:i + 1])
            model._buf = _all
            n = int((labels[:, :-1] != -100).sum())
            ce_sum += float(_all.get('ce', float('nan'))) * n
            n_ce += n
            k = start + len(chunk) - 1
            if eval_every and (k + 1) % eval_every == 0 and val_rows:
                v = evaluate(model, tok, val_rows)
                log_and_print('[MID] ep%d %d/%d ｜ 训练CE=%.4f ｜ **验证CE=%.4f** top1=%.4f ｜ %.0fs'
                              % (ep, k + 1, len(rows), ce_sum / max(n_ce, 1), v['ce'], v['top1'],
                                 time.time() - t0))
            if (k + 1) % LOG_INTERVAL == 0 or k + 1 == len(rows):
                log_and_print('[TRAIN] ep%d %d/%d ｜ CE(累计)=%.4f ｜ %.1fs'
                              % (ep, k + 1, len(rows), ce_sum / max(n_ce, 1), time.time() - t_ep))'''
if old not in t:
    raise SystemExit('train loop not found')
t = t.replace(old, new, 1)

# CLI.
old = "    ap.add_argument('--threads', type=int, default=THREADS)"
new = ("    ap.add_argument('--threads', type=int, default=THREADS)\n"
       "    ap.add_argument('--batch-size', type=int, default=1, help='screening batch; default 1 = online')\n"
       "    ap.add_argument('--device', default='', help=\"'' keeps compile-time DEVICE; 'cpu'/'cuda' overrides\")")
if old not in t:
    raise SystemExit('args not found')
t = t.replace(old, new, 1)

old = '''    elif args.train:
        dev = setup_device()'''
new = '''    elif args.train:
        global DEVICE
        if args.device:
            DEVICE = args.device
        dev = setup_device()'''
if old not in t:
    raise SystemExit('device block not found')
t = t.replace(old, new, 1)

old = '''        train_hebbian(model, tok, args.train_file, epochs=args.epochs, limit=args.limit,
                      val_file=args.val_file, val_max=args.val_max, eval_every=args.eval_every)'''
new = '''        train_hebbian(model, tok, args.train_file, epochs=args.epochs, limit=args.limit,
                      val_file=args.val_file, val_max=args.val_max, eval_every=args.eval_every,
                      batch_size=args.batch_size)'''
if old not in t:
    raise SystemExit('call not found')
t = t.replace(old, new, 1)

io.open(dst, 'w', encoding='utf-8', newline='\n').write(t)
print('wrote', dst)
