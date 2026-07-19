"""跨模型泛化三表:门控AUROC / 时间 / 金额。
【加载】已训练保存的固定门控(models/gate.pt),绝不现训现用——保证所有模型用同一个门控评估。
先跑 scripts/train_gate.py 生成 checkpoint。

用法: python scripts/build_gen_tables.py
"""
import os, sys, json, statistics
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gate import load_embeddings, make_feat, load_rollouts, load_gate, predict

DATA_DIR = os.environ.get("DATA_DIR", "data")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
BASE_MODEL = os.environ.get("BASE_MODEL", "claude-sonnet-4.6")
PRICE = {
    "deepseek": (0.27, 1.10), "gpt-4o": (2.50, 10.0),
    "gemini-2.5-pro": (1.25, 10.0), "gemini-2.5-flash": (0.30, 2.50),
    "qwen3-max": (1.20, 6.0), "glm-4.6": (0.60, 2.0), "kimi-k2": (0.60, 2.5),
    "llama-4-maverick": (0.20, 0.60), "claude-sonnet-4": (3.0, 15.0), "mistral-large": (2.0, 6.0),
}
def price_of(m):
    for k, v in PRICE.items():
        if k in m: return v
    return None

if not (Path(MODELS_DIR) / "gate.pt").exists():
    sys.exit(f"未找到 {MODELS_DIR}/gate.pt — 请先运行: python scripts/train_gate.py")

model, mu, sd = load_gate(MODELS_DIR)           # 加载固定门控
qemb, semb = load_embeddings(DATA_DIR)
feat = make_feat(qemb, semb)
_, models = load_rollouts(DATA_DIR, BASE_MODEL)

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s); P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0: return None
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))

meta = json.loads((Path(MODELS_DIR) / "gate_meta.json").read_text())
print(f"加载固定门控 models/gate.pt (基座 {meta['base_model']}, {meta['n_train']} 训练样本)\n")
print("=" * 78); print("表1 · 泛化模型 (固定门控 → 预测各模型 reward>0)"); print("=" * 78)
print(f"{'模型':<34}{'n':>5}{'正类':>5}{'AUROC':>8}{'省调用':>8}{'保留R':>8}{'R/调用':>8}")
for name, dd in sorted(models.items()):
    keys = list(dd)
    if not keys: continue
    X = np.array([feat(dd[k]) for k in keys], np.float32)
    Y = np.array([1 if dd[k]["reward"] > 0 else 0 for k in keys])
    R = np.array([dd[k]["reward"] for k in keys], np.float32)
    prob = predict(model, mu, sd, X)
    pred = (prob >= 0.5).astype(int); call = pred.astype(bool); tot = R.sum() or 1.0
    au = auroc(Y, prob); aus = f"{au:.3f}" if au is not None else "N/A"
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
    it = sum(r.get("input_tokens") or 0 for r in dd.values()); ot = sum(r.get("output_tokens") or 0 for r in dd.values())
    p = price_of(name); cost = f"{it/1e6*p[0]+ot/1e6*p[1]:.2f}" if p else "?"
    print(f"{name[:33]:<34}{it:>14,}{ot:>12,}{cost:>10}")
