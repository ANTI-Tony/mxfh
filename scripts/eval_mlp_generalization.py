"""自包含版：用 Sonnet 训练的 MLP 门控预测 DeepSeek reward>0。
只依赖 numpy + torch（读缓存 embedding，手写 scaler/AUROC），绕开 venv/sklearn/yaml。"""
import json
from pathlib import Path
import numpy as np, torch, torch.nn as nn

import os
RD = Path(os.environ.get("DATA_DIR", "data"))
B = ("gos_original","delete_top","add_irrelevant","replace_similar")
EMB = 384

def load_emb(p):
    d = np.load(p, allow_pickle=True)
    return {k: d["E"][i] for i,k in enumerate(d["ids"])}
qemb = load_emb(RD/"surrogate"/"dyn_query_emb.npz")
semb = load_emb(RD/"surrogate"/"dyn_skill_emb.npz")

rows = [json.loads(l) for l in (RD/"runs.jsonl").read_text().splitlines() if l.strip()]
def mdl(r): return r.get("model_name") or r.get("model") or ""
def er(r): v=r.get("error_type"); return v if v is not None else r.get("error")
def rw(r): v=r.get("reward"); return v if isinstance(v,(int,float)) else None
son, ds = {}, {}
for r in rows:
    k=(r["query_id"], r["bundle_type"])
    if mdl(r)=="claude-sonnet-4.6" and er(r) is None and rw(r) is not None: son[k]=r
    elif "deepseek" in mdl(r).lower() and er(r) is None and rw(r) is not None: ds[k]=r

def feat(r):
    q=r["query_id"]; b=r["bundle_type"]
    qe=qemb.get(q, np.zeros(EMB,np.float32))
    es=[semb[s] for s in r["skill_ids"] if s in semb]
    be=np.mean(es,axis=0) if es else np.zeros(EMB,np.float32)
    oh=np.zeros(4,np.float32); oh[B.index(b)]=1.0
    p=np.array(r.get("ppr_scores") or [0.0],dtype=np.float32)
    pf=np.array([p.sum(),p.mean(),p.max(),p.std()],np.float32)
    return np.concatenate([qe,be,oh,pf]).astype(np.float32)

son_k=list(son); ds_k=list(ds)
Xtr=np.array([feat(son[k]) for k in son_k],np.float32); Ytr=np.array([1 if son[k]["reward"]>0 else 0 for k in son_k])
Xte=np.array([feat(ds[k]) for k in ds_k],np.float32);  Yte=np.array([1 if ds[k]["reward"]>0 else 0 for k in ds_k])
Rte=np.array([ds[k]["reward"] for k in ds_k],np.float32)

mu=Xtr.mean(0); sd=Xtr.std(0)+1e-8
Xtr_s=((Xtr-mu)/sd).astype(np.float32); Xte_s=((Xte-mu)/sd).astype(np.float32)

class MLP(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.enc=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(0.3),
                            nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.3),
                            nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.3))
        s.head=nn.Linear(64,1)
    def forward(s,x): return s.head(s.enc(x))

torch.manual_seed(0)
m=MLP(Xtr_s.shape[1]); pos=max(1,int(Ytr.sum())); neg=max(1,len(Ytr)-pos)
crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],dtype=torch.float32))
opt=torch.optim.Adam(m.parameters(),lr=1e-3)
Xt=torch.tensor(Xtr_s); Yt=torch.tensor(Ytr,dtype=torch.float32).view(-1,1)
best=1e9;bad=0;bs=None; m.train()
for ep in range(100):
    opt.zero_grad(); loss=crit(m(Xt),Yt); loss.backward(); opt.step()
    if loss.item()<best-1e-4: best=loss.item();bad=0;bs={k:v.clone() for k,v in m.state_dict().items()}
    else:
        bad+=1
        if bad>=15: break
if bs: m.load_state_dict(bs)
m.eval()
with torch.no_grad(): prob=1/(1+np.exp(-m(torch.tensor(Xte_s)).view(-1).numpy()))
pred=(prob>=0.5).astype(int)

def auroc(y,s):
    y=np.asarray(y); s=np.asarray(s); P=(y==1).sum(); N=(y==0).sum()
    if P==0 or N==0: return float("nan")
    order=np.argsort(s); ranks=np.empty(len(s)); ranks[order]=np.arange(1,len(s)+1)
    return (ranks[y==1].sum()-P*(P+1)/2)/(P*N)
def prec(y,p): tp=((p==1)&(y==1)).sum(); fp=((p==1)&(y==0)).sum(); return tp/(tp+fp) if tp+fp else 0.0
def rec(y,p): tp=((p==1)&(y==1)).sum(); fn=((p==0)&(y==1)).sum(); return tp/(tp+fn) if tp+fn else 0.0

son_lab=np.array([1 if son[k]["reward"]>0 else 0 for k in ds_k])
print("="*60)
print("用【Sonnet 训练的 MLP 门控】预测 DeepSeek 的 reward>0")
print("="*60)
print(f"训练(Sonnet)干净={len(Ytr)} (正{int(Ytr.sum())})  |  测试(DeepSeek)={len(Yte)} (正{int(Yte.sum())}/负{int((Yte==0).sum())})")
print(f"\n门控预测 vs DeepSeek 真实:")
print(f"  AUROC     = {auroc(Yte,prob):.3f}   ← ≈0.5=不迁移")
print(f"  Accuracy  = {(pred==Yte).mean():.3f}")
print(f"  Precision = {prec(Yte,pred):.3f}   Recall = {rec(Yte,pred):.3f}")
call=pred.astype(bool); tot=Rte.sum() or 1.0
print(f"\n若用它决定要不要调 DeepSeek:")
print(f"  省调用={ (~call).sum()/len(call):.0%}  保留DeepSeek-reward={Rte[call].sum()/tot:.0%}  reward/调用={Rte[call].sum()/max(1,call.sum()):.3f} (Always={Rte.mean():.3f})")
print(f"\n对照: 直接迁移 Sonnet 的 label 预测 DeepSeek → 命中 {(son_lab==Yte).mean():.3f}；DeepSeek 正类基率={Yte.mean():.3f}")
json.dump({"n_train":int(len(Ytr)),"n_test":int(len(Yte)),"auroc":float(auroc(Yte,prob)),
           "acc":float((pred==Yte).mean()),"precision":float(prec(Yte,pred)),"recall":float(rec(Yte,pred)),
           "sonnet_label_acc_on_deepseek":float((son_lab==Yte).mean()),"deepseek_pos_rate":float(Yte.mean())},
          open(RD/"results"/"mlp_on_deepseek.json","w"), indent=2, ensure_ascii=False)
print(f"\n[written] {RD/'results'/'mlp_on_deepseek.json'}")
