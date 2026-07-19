"""在线更新评估(核心):固定门控 gate.pt 作为起点,在每个执行模型的 rollout 上
【prequential 在线更新】——逐条:先预测→揭晓真实reward→1步SGD更新头(编码器冻结)。
越往后见过的反馈越多,门控对该模型越准。输出:静态 vs 在线 的 AUROC 对比 + 在线曲线。

这就是"用得越久越好"的机制:换个新执行模型,初始可能不准,但边跑边更新,后面变好。
增量更新极便宜(每条 1 步 SGD 更新一个小 logistic 头),不是重训。

用法: python scripts/eval_online.py [--model <子串筛选>] [--lr 0.05] [--seeds 5]
"""
import os, sys, json, argparse, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gate import load_embeddings, make_feat, load_rollouts, load_gate, predict

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="", help="只评估名字含此子串的模型(默认全部)")
ap.add_argument("--lr", type=float, default=0.05, help="在线头 SGD 学习率")
ap.add_argument("--seeds", type=int, default=5, help="打乱顺序的种子数(取平均,减小顺序噪声)")
a = ap.parse_args()

DATA_DIR = os.environ.get("DATA_DIR", "data")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
BASE_MODEL = os.environ.get("BASE_MODEL", "claude-sonnet-4.6")

gate, mu, sd = load_gate(MODELS_DIR)
enc = gate.enc
for p in enc.parameters(): p.requires_grad = False   # 编码器永久冻结(固定表示)
qemb, semb = load_embeddings(DATA_DIR)
feat = make_feat(qemb, semb)
_, models = load_rollouts(DATA_DIR, BASE_MODEL)

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s); P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0: return None
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))

def eval_model(dd):
    keys = list(dd)
    X = np.array([feat(dd[k]) for k in keys], np.float32)
    Xs = ((X - mu) / sd).astype(np.float32)
    Y = np.array([1 if dd[k]["reward"] > 0 else 0 for k in keys], np.int64)
    with torch.no_grad(): H = enc(torch.tensor(Xs)).numpy()   # 冻结编码器表示

    # 静态基线(冻结头,不更新)
    static_prob = predict(gate, mu, sd, X)
    static_au = auroc(Y, static_prob)

    # 在线:每 seed 打乱顺序,prequential 先预测后更新,取平均
    online_aus = []; curves = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed); idx = rng.permutation(len(keys))
        head = nn.Linear(64, 1); head.load_state_dict(gate.head.state_dict())  # warm-start 自固定门控的头
        opt = torch.optim.SGD(head.parameters(), lr=a.lr); crit = nn.BCEWithLogitsLoss()
        probs = np.zeros(len(keys)); ys = np.zeros(len(keys))
        for t, i in enumerate(idx):
            h = torch.tensor(H[i:i+1])
            head.eval()
            with torch.no_grad(): probs[t] = 1/(1+np.exp(-head(h).item()))   # 先预测(更新前)
            ys[t] = Y[i]
            head.train(); opt.zero_grad()                                    # 揭晓真值后 1 步 SGD
            loss = crit(head(h), torch.tensor([[float(Y[i])]])); loss.backward(); opt.step()
        online_aus.append(auroc(ys, probs))
        # 累计 AUROC 曲线(前 t 条上的在线预测)
        n = len(keys); pts = list(range(max(6, n//6), n+1, max(1, n//6)))
        curve = [auroc(ys[:p], probs[:p]) for p in pts]
        curves.append((pts, curve))
    online_aus = [x for x in online_aus if x is not None]
    return len(keys), int(Y.sum()), static_au, (np.mean(online_aus) if online_aus else None), curves[0]

print(f"固定门控起点=models/gate.pt | 编码器冻结, 在线更新头(lr={a.lr}, {a.seeds}seed平均)\n")
print(f"{'模型':<34}{'n':>5}{'正类':>5}{'静态AUROC':>10}{'在线AUROC':>10}{'提升':>8}")
for name, dd in sorted(models.items()):
    if a.model and a.model not in name: continue
    if not dd: continue
    n, pos, sau, oau, curve = eval_model(dd)
    if pos == 0 or pos == n:
        print(f"{name[:33]:<34}{n:>5}{pos:>5}{'N/A(无正负样本)':>18}")
        continue
    delta = (oau - sau) if (oau is not None and sau is not None) else None
    print(f"{name[:33]:<34}{n:>5}{pos:>5}{sau:>10.3f}{oau:>10.3f}{delta:>+8.3f}")
    pts, cv = curve
    print(f"      在线曲线(feedback数→AUROC): " + "  ".join(f"{p}:{c:.2f}" for p,c in zip(pts,cv) if c is not None))
print("\n说明:'静态'=冻结门控直接预测;'在线'=从固定门控起点、逐条 prequential 更新头。")
print("在线>静态 说明门控在该模型的反馈上自适应变好(用得越久越准);增量更新,非重训。")
