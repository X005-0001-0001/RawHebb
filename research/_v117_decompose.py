# -*- coding: utf-8 -*-
"""分解 v115 的 logits：底板(W_out) 与记忆支路各自值多少 CE。
   目的：模型卡在 unigram 水平（5.96 vs unigram 5.88），要弄清是**哪一支**在拖后腿。"""
import os, io, json, importlib, math
os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['V112_L0_READ']='1'; os.environ['V116_BASE_SRC']='last'
import torch, torch.nn.functional as F
torch.set_num_threads(6)
wh = importlib.import_module('wavesnn_hebbian_v116'); wh.DEVICE=torch.device('cpu')

class Tok(wh.FastBPE):
    def load(self, path):
        d=torch.load(path,map_location='cpu',weights_only=False)
        from tokenizers import Tokenizer
        self.initial_vocab_size=d.get('initial_vocab_size',wh.VOCAB_SIZE)
        self.special_tokens_list=d.get('special_tokens_list',wh.SPECIAL_TOKENS_LIST.copy())
        jp=os.path.join(os.path.dirname(path),os.path.splitext(os.path.basename(path))[0]+'_tokenizer.json')
        self._tokenizer=Tokenizer.from_file(jp); self._update_special_ids(); self._loaded_json=jp; return True
tok=Tok(wh.VOCAB_SIZE); tok.load('tokenizer_v63b4000.bin'); V=tok.vocab_size
m=wh.WaveSNN(vocab_size=V,hidden_size=128,num_layers=4,pad_token_id=tok.pad_token_id,
             use_residual=wh.USE_RESIDUAL,freeze_seed=43,freeze_layers=wh.FREEZE_LAYERS,
             freeze_embedding=True,seed_embedding=43).to('cpu')
m.load_state_dict(torch.load('v115_sd.pt',map_location='cpu',weights_only=False),strict=False); m.eval()
G=float(m.v111_gamma.data)

rows=[json.loads(l) for l in io.open('val_dsr1.jsonl',encoding='utf-8') if l.strip()][:40]
ds=wh.StreamingJSONLDataset('val_dsr1.jsonl',tok,max_seq_len=200)
# ★再加一档：不训练、只把底板特征从「最后一层(τ=80)」换成「L0(τ=10)」，用同一份 W_out
acc={'full':[0.0,0],'base_only':[0.0,0],'mem_only':[0.0,0],'full_L0base':[0.0,0],'base_L0':[0.0,0]}
top1={k:0 for k in acc}
Wn=torch.nn.functional.normalize(m.W_out, dim=-1)
Widx=torch.nn.functional.normalize(m.W_index, dim=-1) if hasattr(m,'W_index') else None
with torch.no_grad():
    for rec in rows:
        ids,lab=ds._tokenize_conversation(rec['messages'])
        if ids.numel()>200: ids,lab=ids[-200:],lab[-200:]
        ids=ids.unsqueeze(0); lab=lab.unsqueeze(0)
        out=m(ids,labels=lab)
        full=(out[0] if isinstance(out,tuple) else out)
        lms=[]
        for li in list(m._v112_idx_buf):
            lm=m._cached_layer_outputs[li]@m.W_m_vocab[li]
            lm=lm/lm.pow(2).mean().sqrt().clamp(min=1e-6); lms.append(lm)
        w=m._v111_layer_weights[list(m._v112_idx_buf)]
        merged=sum(w[k]*lms[k] for k in range(len(lms)))/w.pow(2).sum().sqrt().clamp(min=1e-6)
        mem=G*merged
        boo=full-mem          # 底板支路 = 完整 − 记忆支路
        # 用 L0 的膜电位重新算一遍底板（权重完全不动，只换特征层）
        l0=m._cached_layer_outputs[0]
        l0n=l0/l0.norm(dim=-1,keepdim=True).clamp(min=1e-6)
        baseL0=(l0n*(1.0))@Wn.T
        if Widx is not None:
            dg_raw=m.embedding(ids)@m.W_dg.T
            tv,ti=dg_raw.abs().topk(m.k_index,dim=-1)
            dg_sp=torch.zeros_like(dg_raw).scatter_(-1,ti,dg_raw.gather(-1,ti))
            dg_sp=dg_sp/dg_sp.norm(dim=-1,keepdim=True).clamp(min=1e-6)
            ema=torch.zeros_like(dg_sp[:,0,:]); ctx=[]
            for _t in range(dg_sp.shape[1]):
                ema=m._dg_decay*ema+dg_sp[:,_t,:]; ctx.append(ema.clone())
            ctx=torch.stack(ctx,1); ctx=ctx/ctx.norm(dim=-1,keepdim=True).clamp(min=1e-6)
            baseL0=baseL0+ctx@Widx.T
        fullL0=baseL0+mem
        tm=lab[:,1:]!=-100
        lg=full[:,:-1,:].reshape(-1,V); tg=lab[:,1:].reshape(-1); mk=tm.reshape(-1)
        for key,L in (('full',full),('base_only',boo),('mem_only',mem),
                      ('full_L0base',fullL0),('base_L0',baseL0)):
            l2=L[:,:-1,:].reshape(-1,V)
            acc[key][0]+=float(F.cross_entropy(l2[mk],tg[mk],reduction='sum')); acc[key][1]+=int(mk.sum())
            top1[key]+=int((l2[mk].argmax(-1)==tg[mk]).sum())
n=acc['full'][1]
print('[DEC] 计分位置 %d ｜ γ=%.3f'%(n,G))
for k in acc:
    print('[DEC] %-10s CE=%.4f ｜ top1=%.4f'%(k,acc[k][0]/acc[k][1],top1[k]/max(n,1)))
print('[DEC] 解读：base_only ≈ unigram(5.88) 说明底板只学到了词频；mem_only 若接近 full，说明记忆支路没在出力。')
