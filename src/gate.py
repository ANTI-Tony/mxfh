"""执行门控(surrogate)共享库:MLP 结构 + 特征构造 + 数据加载 + 保存/加载。

门控输入 776 维 = query_emb[384] + bundle_emb[384] + bundle_type_onehot[4] + ppr_feats[4]，
不含执行模型身份，故可零样本套到任意执行模型上预测 P(reward>0)。
"""
import os, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn

BUNDLE_TYPES = ("gos_original", "delete_top", "add_irrelevant", "replace_similar")
EMB = 384
FEAT_DIM = 776


class GateMLP(nn.Module):
    """776 -> 256 -> 128 -> 64 -> 1 (logits)。"""
    def __init__(self, d=FEAT_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(64, 1)
    def forward(self, x): return self.head(self.enc(x))


def load_embeddings(data_dir):
    data_dir = Path(data_dir)
    def _load(p):
        d = np.load(p, allow_pickle=True); return {k: d["E"][i] for i, k in enumerate(d["ids"])}
    return _load(data_dir / "surrogate" / "dyn_query_emb.npz"), _load(data_dir / "surrogate" / "dyn_skill_emb.npz")


def make_feat(qemb, semb):
    def feat(r):
        qe = qemb.get(r["query_id"], np.zeros(EMB, np.float32))
        es = [semb[s] for s in r["skill_ids"] if s in semb]
        be = np.mean(es, axis=0) if es else np.zeros(EMB, np.float32)
        oh = np.zeros(4, np.float32); oh[BUNDLE_TYPES.index(r["bundle_type"])] = 1.0
        p = np.array(r.get("ppr_scores") or [0.0], dtype=np.float32)
        return np.concatenate([qe, be, oh, np.array([p.sum(), p.mean(), p.max(), p.std()], np.float32)]).astype(np.float32)
    return feat


def _mdl(r): return r.get("model_name") or r.get("model") or ""
def _err(r): v = r.get("error_type"); return v if v is not None else r.get("error")
def _rw(r): v = r.get("reward"); return v if isinstance(v, (int, float)) else None


def load_rollouts(data_dir, base_model="claude-sonnet-4.6"):
    """返回 (base_dict, {model: dict})，dict 键=(query_id,bundle_type)，仅干净行。"""
    rows = [json.loads(l) for l in (Path(data_dir) / "runs.jsonl").read_text().splitlines() if l.strip()]
    base = {}; models = {}
    for r in rows:
        if _err(r) is not None or _rw(r) is None: continue
        k = (r["query_id"], r["bundle_type"]); m = _mdl(r)
        if m == base_model: base[k] = r
        else: models.setdefault(m, {})[k] = r
    return base, models


def save_gate(model, mu, sd, meta, models_dir):
    models_dir = Path(models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / "gate.pt")
    np.savez(models_dir / "scaler.npz", mu=mu, sd=sd)
    (models_dir / "gate_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def load_gate(models_dir):
    models_dir = Path(models_dir)
    model = GateMLP(); model.load_state_dict(torch.load(models_dir / "gate.pt", map_location="cpu"))
    model.eval()
    d = np.load(models_dir / "scaler.npz"); mu, sd = d["mu"], d["sd"]
    return model, mu, sd


def predict(model, mu, sd, X):
    Xs = ((np.asarray(X, np.float32) - mu) / sd).astype(np.float32)
    with torch.no_grad():
        logit = model(torch.tensor(Xs)).view(-1).numpy()
    return 1.0 / (1.0 + np.exp(-logit))
