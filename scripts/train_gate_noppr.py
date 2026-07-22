"""训练【去-PPR】执行门控 gate_noppr.pt —— 用于检索器无关(retrieval-agnostic)验证。
非 GoS 检索器(bm25/embedding/...)没有 PPR 特征,故用这个不依赖 PPR 的门控。
协议与 train_gate.py 完全一致,唯一区别:训练/推理都把 ppr 特征块[772:776]置零。
消融已证 w/o-ppr AUROC 0.742 ≈ 全特征 0.744,去 PPR 不掉分。
输出: models/gate_noppr.pt / scaler_noppr.npz / gate_noppr_meta.json
"""
import os, sys, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gate import GateMLP, load_embeddings, make_feat, load_rollouts, FEAT_DIM

DATA_DIR = os.environ.get("DATA_DIR", "data")
BASE_MODEL = os.environ.get("BASE_MODEL", "claude-sonnet-4.6")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
PPR = slice(772, 776)   # ppr 特征块

qemb, semb = load_embeddings(DATA_DIR); feat = make_feat(qemb, semb)
base, _ = load_rollouts(DATA_DIR, BASE_MODEL); keys = list(base)
X = np.array([feat(base[k]) for k in keys], np.float32); X[:, PPR] = 0.0   # 去 PPR
Y = np.array([1 if base[k]["reward"] > 0 else 0 for k in keys])
mu = X.mean(0); sd = X.std(0) + 1e-8; Xs = ((X - mu) / sd).astype(np.float32)

torch.manual_seed(0)
model = GateMLP(FEAT_DIM)
pos = max(1, int(Y.sum())); neg = max(1, len(Y) - pos)
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
Xt = torch.tensor(Xs); Yt = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
best = 1e9; bad = 0; bs = None; model.train()
for _ in range(100):
    opt.zero_grad(); loss = crit(model(Xt), Yt); loss.backward(); opt.step()
    if loss.item() < best - 1e-4: best = loss.item(); bad = 0; bs = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= 15: break
if bs: model.load_state_dict(bs)
model.eval()

md = Path(MODELS_DIR)
torch.save(model.state_dict(), md / "gate_noppr.pt")
np.savez(md / "scaler_noppr.npz", mu=mu, sd=sd)
(md / "gate_noppr_meta.json").write_text(json.dumps({
    "base_model": BASE_MODEL, "feat_dim": FEAT_DIM, "ppr": "zeroed (retrieval-agnostic)",
    "n_train": int(len(Y)), "n_pos": int(Y.sum()), "n_neg": int((Y == 0).sum()),
    "final_train_loss": float(best),
    "note": "去PPR门控,用于非GoS检索器的检索器无关验证。",
}, indent=2, ensure_ascii=False))
print(f"✅ gate_noppr 已保存到 {MODELS_DIR}/ (训练 {len(Y)} 正{int(Y.sum())}/负{int((Y==0).sum())}, loss={best:.4f})")
