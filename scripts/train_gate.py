"""训练执行门控并保存到 models/。一次训好,后续直接加载,不再现训现用。

用法: python scripts/train_gate.py
输出: models/gate.pt (权重), models/scaler.npz (标准化), models/gate_meta.json (元信息)
数据目录默认 ./data(DATA_DIR 覆盖),基座默认 claude-sonnet-4.6(BASE_MODEL 覆盖)。
"""
import os, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gate import GateMLP, load_embeddings, make_feat, load_rollouts, save_gate, FEAT_DIM

DATA_DIR = os.environ.get("DATA_DIR", "data")
BASE_MODEL = os.environ.get("BASE_MODEL", "claude-sonnet-4.6")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")

qemb, semb = load_embeddings(DATA_DIR)
feat = make_feat(qemb, semb)
base, _ = load_rollouts(DATA_DIR, BASE_MODEL)
keys = list(base)
X = np.array([feat(base[k]) for k in keys], np.float32)
Y = np.array([1 if base[k]["reward"] > 0 else 0 for k in keys])
mu = X.mean(0); sd = X.std(0) + 1e-8
Xs = ((X - mu) / sd).astype(np.float32)

torch.manual_seed(0)
model = GateMLP(FEAT_DIM)
pos = max(1, int(Y.sum())); neg = max(1, len(Y) - pos)
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
Xt = torch.tensor(Xs); Yt = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
best = 1e9; bad = 0; bs = None; model.train()
for ep in range(100):
    opt.zero_grad(); loss = crit(model(Xt), Yt); loss.backward(); opt.step()
    if loss.item() < best - 1e-4:
        best = loss.item(); bad = 0; bs = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= 15: break
if bs: model.load_state_dict(bs)
model.eval()

meta = {
    "base_model": BASE_MODEL, "feat_dim": FEAT_DIM,
    "arch": "776-256-128-64-1, ReLU+Dropout(0.3), BCEWithLogitsLoss(pos_weight=neg/pos), Adam lr=1e-3, full-batch, early-stop patience 15, seed 0",
    "n_train": int(len(Y)), "n_pos": int(Y.sum()), "n_neg": int((Y == 0).sum()),
    "final_train_loss": float(best),
    "note": "门控输入不含执行模型身份,可零样本套到任意执行模型预测 P(reward>0)。",
}
save_gate(model, mu, sd, meta, MODELS_DIR)
print(f"✅ 门控已训练并保存到 {MODELS_DIR}/")
print(f"   训练样本 {meta['n_train']} (正 {meta['n_pos']} / 负 {meta['n_neg']}), final_loss={best:.4f}")
print(f"   文件: gate.pt, scaler.npz, gate_meta.json")
