"""持久累积在线学习流水线(核心实验)。

一个持久门控实例,从 Sonnet 训练的 models/gate.pt 出发,按【固定顺序】依次流过每个
执行模型的 rollout。门控**不重置**——跨模型累积在线更新(编码器冻结,头逐条 1 步 SGD)。
见的反馈越多,门控越准。**每跑完一个模型存一个 checkpoint** → 完全可复现。

这不是"现训现用",而是"一个模型、边用边更新、用得越久越好"。是对已采集 rollout 的
确定性离线重放,不需要重跑模型。

顺序默认:先国外强模型(GPT/Gemini/Claude/Llama/Mistral),后国产(DeepSeek/Qwen/GLM/Kimi)。
用 --order 覆盖(逗号分隔的模型名子串)。

用法: python scripts/run_online_stream.py [--lr 0.02] [--order gpt-4o,gemini-2.5-pro,...]
输出: models/stream/gate_after_<model>.pt (每阶段 checkpoint) + 各阶段在线 AUROC 进展表。
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
ap.add_argument("--lr", type=float, default=0.02, help="在线头 SGD 学习率")
ap.add_argument("--order", default="", help="逗号分隔的模型名子串顺序(默认国外先国产后)")
ap.add_argument("--seed", type=int, default=0, help="每个模型内部 rollout 打乱的种子")
a = ap.parse_args()
order = [s.strip() for s in a.order.split(",") if s.strip()] or DEFAULT_ORDER

gate, mu, sd = load_gate(MODELS_DIR)
enc = gate.enc
for p in enc.parameters(): p.requires_grad = False          # 编码器永久冻结(固定表示)
head = nn.Linear(64, 1); head.load_state_dict(gate.head.state_dict())  # 持久头(从 Sonnet 头出发)
opt = torch.optim.SGD(head.parameters(), lr=a.lr); crit = nn.BCEWithLogitsLoss()

qemb, semb = load_embeddings(DATA_DIR)
feat = make_feat(qemb, semb)
_, models = load_rollouts(DATA_DIR, BASE_MODEL)
STREAM_DIR = Path(MODELS_DIR) / "stream"; STREAM_DIR.mkdir(parents=True, exist_ok=True)

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s); P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0: return None
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))

def match(sub):
    for name, dd in models.items():
        if sub in name and dd: return name, dd
    return None, None

# 阶段0:初始 checkpoint
torch.save(head.state_dict(), STREAM_DIR / "gate_after_stage0_initial.pt")
print(f"持久累积在线学习 | 编码器冻结, 持久头 lr={a.lr}")
print(f"顺序: {order}\n")
print(f"{'阶段':<4}{'模型':<28}{'n':>5}{'正类':>5}{'该模型在线AUROC':>16}{'累计已见反馈':>12}")
seen = 0; results = []
for stage, sub in enumerate(order, 1):
    name, dd = match(sub)
    if not dd:
        print(f"{stage:<4}{sub:<28}{'(无数据,跳过)':>10}"); continue
    keys = list(dd)
    rng = np.random.default_rng(a.seed); idx = rng.permutation(len(keys))
    X = np.array([feat(dd[keys[i]]) for i in idx], np.float32)
    Xs = ((X - mu) / sd).astype(np.float32)
    Y = np.array([1 if dd[keys[i]]["reward"] > 0 else 0 for i in idx], np.int64)
    with torch.no_grad(): H = enc(torch.tensor(Xs)).numpy()
    probs = np.zeros(len(keys))
    for t in range(len(keys)):
        h = torch.tensor(H[t:t+1]); head.eval()
        with torch.no_grad(): probs[t] = 1/(1+np.exp(-head(h).item()))   # 先预测(持久门控当前状态)
        head.train(); opt.zero_grad()                                    # 揭晓真值→1步SGD更新持久头
        loss = crit(head(h), torch.tensor([[float(Y[t])]])); loss.backward(); opt.step()
        seen += 1
    au = auroc(Y, probs); aus = f"{au:.3f}" if au is not None else "N/A(无正负)"
    torch.save(head.state_dict(), STREAM_DIR / f"gate_after_stage{stage}_{name.split('/')[-1]}.pt")  # 阶段 checkpoint
    print(f"{stage:<4}{name.split('/')[-1][:27]:<28}{len(keys):>5}{int(Y.sum()):>5}{aus:>16}{seen:>12}")
    results.append({"stage": stage, "model": name, "n": len(keys), "pos": int(Y.sum()), "online_auroc": au, "seen": seen})

(STREAM_DIR / "stream_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\n每阶段 checkpoint 存于 {STREAM_DIR}/ (可复现);进展存 stream_results.json")
print("门控跨模型累积:后面阶段带着前面所有模型的在线更新 → 反馈越多越准。")
