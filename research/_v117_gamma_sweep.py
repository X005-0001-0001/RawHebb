# -*- coding: utf-8 -*-
"""γ 扫描（零训练，只换记忆支路的音量）：γ 由局部规则长到 1.43，可能是它把底板的排序淹了。"""
import os, io, json, importlib
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

rows=[json.loads(l) for l in io.open('val_dsr1.jsonl',encoding='utf-8') if l.strip()][:40]
ds=wh.StreamingJSONLDataset('val_dsr1.jsonl',tok,max_seq_len=200)
GAMMAS=[0.0,0.1,0.2,0.3,0.5,0.7,1.0,1.434]
acc={g:[0.0,0,0] for g in GAMMAS}   # ce_sum, top1, top5
with torch.no_grad():
    for rec in rows:
        ids,lab=ds._tokenize_conversation(rec['messages'])
        if ids.numel()>200: ids,lab=ids[-200:],lab[-200:]
        ids=ids.unsqueeze(0); lab=lab.unsqueeze(0)
        out=m(ids,labels=lab); full=(out[0] if isinstance(out,tuple) else out)
        lms=[]
        for li in list(m._v112_idx_buf):
            lm=m._cached_layer_outputs[li]@m.W_m_vocab[li]
            lms.append(lm/lm.pow(2).mean().sqrt().clamp(min=1e-6))
        w=m._v111_layer_weights[list(m._v112_idx_buf)]
        merged=sum(w[k]*lms[k] for k in range(len(lms)))/w.pow(2).sum().sqrt().clamp(min=1e-6)
        mk=(lab[:,1:]!=-100).reshape(-1); tg=lab[:,1:].reshape(-1)
        base=full-float(m.v111_gamma.data)*merged
        for g in GAMMAS:
            lg=(base+g*merged)[:,:-1,:].reshape(-1,V)
            acc[g][0]+=float(F.cross_entropy(lg[mk],tg[mk],reduction='sum')); acc[g][1]+=int(mk.sum())
            acc[g][2]+=int((lg[mk].argmax(-1)==tg[mk]).sum())
print('[GAMMA] 计分位置 %d ｜ 学到的 γ=%.3f'%(acc[1.0][1],float(m.v111_gamma.data)))
for g in GAMMAS:
    print('[GAMMA] γ=%.3f  CE=%.4f  top1=%.4f'%(g,acc[g][0]/max(acc[g][1],1),acc[g][2]/max(acc[g][1],1)))
