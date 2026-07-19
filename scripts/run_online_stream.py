"""在线持续更新(核心实验)——只有一个模型,一直原地更新它的权重。

从原始 MLP 门控(models/gate.pt,Sonnet 训练)出发,按固定顺序流过每个执行模型的 rollout。
**同一个模型实例**:每验证一条任务(或攒 --update-every 条)就更新一次它的头(编码器冻结),
跨模型不重置、连续累积。见的反馈越多越准。**全程只有一个模型**,跑完存成一个
models/gate_online.pt(不分阶段另存)。

复现:初始 gate.pt(train_gate.py 确定性生成) + 本脚本确定性重放这条流 → 同一个在线模型。

用法: python scripts/run_online_stream.py [--update-every 1] [--lr 0.02] [--order gpt-4o,...]
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gate import load_embeddings, make_feat, load_rollouts, load_gate

DATA_DIR = os.environ.get("DATA_DIR", "data")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
BASE_MODEL = os.environ.get("BASE_MODEL", "claude-sonnet-4.6")
DEFAULT_ORDER = ["gpt-4o", "gemini-2.5-pro", "gemini-2.5-flash", "claude-sonnet-4",
                 "llama-4-maverick", "mistral-large",           # 国外强模型 先
                 "deepseek", "qwen3-max", "glm-4.6", "kimi-k2"]  # 国产 后

ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--update-every", type=int, default=1, help="每验证 N 条更新一次(1=逐条;也可 10/20)")
ap.add_argument("--order", default="")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
order = [s.strip() for s in a.order.split(",") if s.strip()] or DEFAULT_ORDER

# —— 一个模型:编码器冻结 + 持续更新的头(从初始门控的头出发)——
gate, mu, sd = load_gate(MODELS_DIR)
enc = gate.enc
for p in enc.parameters(): p.requires_grad = False
opt = torch.optim.SGD(gate.head.parameters(), lr=a.lr); crit = nn.BCEWithLogitsLoss()

qemb, semb = load_embeddings(DATA_DIR); feat = make_feat(qemb, semb)
_, models = load_rollouts(DATA_DIR, BASE_MODEL)

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s); P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0: return None
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))

print(f"一个持续更新的门控 | 编码器冻结, 头 lr={a.lr}, 每 {a.update_every} 条更新一次")
print(f"顺序: {order}\n")
print(f"{'阶段':<4}{'模型':<26}{'n':>5}{'正类':>5}{'该模型在线AUROC':>16}{'累计已见':>10}")

seen = 0; buf_h = []; buf_y = []; results = []
gate.eval()
for stage, sub in enumerate(order, 1):
    name = next((n for n in models if sub in n and models[n]), None)
    if not name:
        print(f"{stage:<4}{sub:<26}{'(无数据,跳过)':>10}"); continue
    dd = models[name]; keys = list(dd)
    rng = np.random.default_rng(a.seed); idx = rng.permutation(len(keys))
    X = np.array([feat(dd[keys[i]]) for i in idx], np.float32); Xs = ((X - mu) / sd).astype(np.float32)
    Y = np.array([1 if dd[keys[i]]["reward"] > 0 else 0 for i in idx])
    with torch.no_grad(): H = enc(torch.tensor(Xs)).numpy()
    probs = np.zeros(len(keys))
    for t in range(len(keys)):
        h = torch.tensor(H[t:t+1])
        with torch.no_grad(): probs[t] = 1/(1+np.exp(-gate.head(h).item()))  # 先预测(当前权重)
        buf_h.append(h); buf_y.append(float(Y[t])); seen += 1
        if len(buf_h) >= a.update_every:                                     # 攒够 N 条,更新一次
            gate.head.train(); opt.zero_grad()
            hb = torch.cat(buf_h); yb = torch.tensor(buf_y).view(-1, 1)
            loss = crit(gate.head(hb), yb); loss.backward(); opt.step(); gate.head.eval()
            buf_h.clear(); buf_y.clear()
    au = auroc(Y, probs); aus = f"{au:.3f}" if au is not None else "N/A(无正负)"
    print(f"{stage:<4}{name.split('/')[-1][:25]:<26}{len(keys):>5}{int(Y.sum()):>5}{aus:>16}{seen:>10}")
    results.append({"stage": stage, "model": name, "n": len(keys), "pos": int(Y.sum()), "online_auroc": au, "seen": seen})

# 只存这一个持续更新后的模型
torch.save(gate.state_dict(), Path(MODELS_DIR) / "gate_online.pt")
(Path(MODELS_DIR) / "online_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\n只有一个模型 → 存为 {MODELS_DIR}/gate_online.pt(持续更新后的当前状态)")
print("复现:train_gate.py 生成初始 gate.pt + 本脚本确定性重放 → 同一个在线模型。")
