"""跨模型泛化三表:门控AUROC / 时间 / 金额。
用 Sonnet 训的 MLP 门控预测每个执行模型的 reward>0（AUROC 需正负样本）。
只依赖 numpy+torch，读缓存 embedding，手写 scaler/AUROC。

用法: python scripts/build_gen_tables.py
数据目录默认 ./data，可用 DATA_DIR 环境变量覆盖。
"""
import os, json, statistics
from pathlib import Path
import numpy as np, torch, torch.nn as nn

RD = Path(os.environ.get("DATA_DIR", "data"))
BASE_MODEL = os.environ.get("BASE_MODEL", "claude-sonnet-4.6")   # 门控训练用的基座
B = ("gos_original", "delete_top", "add_irrelevant", "replace_similar")
EMB = 384
# 各模型每 1M token 定价 (input, output) USD — 按需补充；未列则不算金额
PRICE = {
    "deepseek": (0.27, 1.10), "gpt-4o": (2.50, 10.0),
    "gemini-2.5-pro": (1.25, 10.0), "gemini-2.5-flash": (0.30, 2.50),
    "qwen3-max": (1.20, 6.0), "glm-4.6": (0.60, 2.0), "kimi-k2": (0.60, 2.5),
    "llama-4-maverick": (0.20, 0.60), "claude-sonnet-4": (3.0, 15.0),
    "mistral-large": (2.0, 6.0),
}
def price_of(m):
    for k, v in PRICE.items():
        if k in m: return v
    return None

def load_emb(p):
    d = np.load(p, allow_pickle=True); return {k: d["E"][i] for i, k in enumerate(d["ids"])}
qemb = load_emb(RD / "surrogate" / "dyn_query_emb.npz")
semb = load_emb(RD / "surrogate" / "dyn_skill_emb.npz")
rows = [json.loads(l) for l in (RD / "runs.jsonl").read_text().splitlines() if l.strip()]
def mdl(r): return r.get("model_name") or r.get("model") or ""
def er(r): v = r.get("error_type"); return v if v is not None else r.get("error")
def rw(r): v = r.get("reward"); return v if isinstance(v, (int, float)) else None

son = {}; models = {}
for r in rows:
    k = (r["query_id"], r["bundle_type"]); m = mdl(r)
    if er(r) is not None or rw(r) is None: continue
    if m == BASE_MODEL: son[k] = r
    else: models.setdefault(m, {})[k] = r

def feat(r):
    qe = qemb.get(r["query_id"], np.zeros(EMB, np.float32))
    es = [semb[s] for s in r["skill_ids"] if s in semb]
    be = np.mean(es, axis=0) if es else np.zeros(EMB, np.float32)
    oh = np.zeros(4, np.float32); oh[B.index(r["bundle_type"])] = 1.0
    p = np.array(r.get("ppr_scores") or [0.0], dtype=np.float32)
    return np.concatenate([qe, be, oh, np.array([p.sum(), p.mean(), p.max(), p.std()], np.float32)]).astype(np.float32)

# ---- 训练 MLP 门控(全 Sonnet)----
sk = list(son)
Xtr = np.array([feat(son[k]) for k in sk], np.float32)
Ytr = np.array([1 if son[k]["reward"] > 0 else 0 for k in sk])
mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8; Xtr_s = ((Xtr - mu) / sd).astype(np.float32)
class MLP(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
                              nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                              nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        s.head = nn.Linear(64, 1)
    def forward(s, x): return s.head(s.enc(x))
torch.manual_seed(0)
m = MLP(776); pos = max(1, int(Ytr.sum())); neg = max(1, len(Ytr) - pos)
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
Xt = torch.tensor(Xtr_s); Yt = torch.tensor(Ytr, dtype=torch.float32).view(-1, 1)
best = 1e9; bad = 0; bs = None; m.train()
for ep in range(100):
    opt.zero_grad(); loss = crit(m(Xt), Yt); loss.backward(); opt.step()
    if loss.item() < best - 1e-4: best = loss.item(); bad = 0; bs = {k: v.clone() for k, v in m.state_dict().items()}
    else:
        bad += 1
        if bad >= 15: break
if bs: m.load_state_dict(bs)
m.eval()
def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s); P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0: return None
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))

print(f"门控训练基座 = {BASE_MODEL} ({len(Ytr)} 样本, 正 {int(Ytr.sum())})\n")
print("=" * 78)
print("表1 · 泛化模型 (Sonnet 训的门控 → 预测各模型 reward>0)")
print("=" * 78)
print(f"{'模型':<34}{'n':>5}{'正类':>5}{'AUROC':>8}{'省调用':>8}{'保留R':>8}{'R/调用':>8}")
for name, dd in sorted(models.items()):
    keys = list(dd)
    if not keys: continue
    X = ((np.array([feat(dd[k]) for k in keys], np.float32) - mu) / sd).astype(np.float32)
    Y = np.array([1 if dd[k]["reward"] > 0 else 0 for k in keys])
    R = np.array([dd[k]["reward"] for k in keys], np.float32)
    with torch.no_grad(): prob = 1 / (1 + np.exp(-m(torch.tensor(X)).view(-1).numpy()))
    pred = (prob >= 0.5).astype(int); call = pred.astype(bool); tot = R.sum() or 1.0
    au = auroc(Y, prob)
    aus = f"{au:.3f}" if au is not None else "N/A"
    print(f"{name[:33]:<34}{len(keys):>5}{int(Y.sum()):>5}{aus:>8}"
          f"{100*(~call).sum()/len(call):>7.0f}%{100*R[call].sum()/tot:>7.0f}%{R[call].sum()/max(1,call.sum()):>8.3f}")

print("\n" + "=" * 78); print("表2 · 花费时间"); print("=" * 78)
print(f"{'模型':<34}{'n':>5}{'均值/条':>10}{'中位':>8}{'总时长':>10}")
for name, dd in sorted(models.items()):
    ts = [r["execution_time"] for r in dd.values() if isinstance(r.get("execution_time"), (int, float))]
    if ts: print(f"{name[:33]:<34}{len(ts):>5}{statistics.mean(ts):>9.0f}s{statistics.median(ts):>7.0f}s{sum(ts)/3600:>8.2f}h")

print("\n" + "=" * 78); print("表3 · 花费金额"); print("=" * 78)
print(f"{'模型':<34}{'输入tok':>14}{'输出tok':>12}{'金额$':>10}")
for name, dd in sorted(models.items()):
    it = sum(r.get("input_tokens") or 0 for r in dd.values())
    ot = sum(r.get("output_tokens") or 0 for r in dd.values())
    p = price_of(name)
    cost = f"{it/1e6*p[0]+ot/1e6*p[1]:.2f}" if p else "?"
    print(f"{name[:33]:<34}{it:>14,}{ot:>12,}{cost:>10}")
