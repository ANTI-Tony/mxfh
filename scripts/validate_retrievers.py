"""检索器无关(retrieval-agnostic)验证:用固定门控预测【不同检索器】产出的 bundle 的 reward>0。
先分别验证每种检索器(bm25/embedding/gold_distractor/hybrid/...),再混合验证。

- 非 GoS 检索器无 PPR → 用去-PPR 门控 gate_noppr(推理时 ppr 特征块置零)。
- GoS 数据(有 PPR)可用全门控 gate.pt 作对照参考。
- skill 库 200 个全部已缓存嵌入,任何检索器选的 skill 都能建特征。

用法:
  # 数据到了后:
  python -m scripts.validate_retrievers --data retriever_runs.jsonl
  # 先测通管线 + 出 GoS 参考基准(用现有 Sonnet 数据):
  python -m scripts.validate_retrievers --gos-baseline

数据格式(JSONL,一行一个 (query, 检索器, 扰动) rollout),字段名可在 load_rows 里改:
  query_id, skill_ids(list), reward(float), error_type(None=干净),
  retriever(检索器名,或用 --retriever-field 指定), bundle_type(4 种扰动之一)
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gate import GateMLP, load_embeddings, make_feat, load_rollouts, FEAT_DIM, BUNDLE_TYPES

DATA_DIR = os.environ.get("DATA_DIR", "data")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
PPR = slice(772, 776)
# 检索器数据里字段名的候选(适配未知格式,按需增改)
RETRIEVER_KEYS = ["retriever", "retriever_type", "retriever_name", "method", "retriever_kind"]
BUNDLE_KEYS = ["bundle_type", "perturbation", "perturbation_type", "variant"]
# 若检索器数据的扰动标签与我们的 4 种不同,在此映射到 BUNDLE_TYPES
BUNDLE_MAP = {"original": "gos_original", "base": "gos_original", "gos": "gos_original",
              "delete": "delete_top", "delete_top": "delete_top",
              "add": "add_irrelevant", "add_irrelevant": "add_irrelevant",
              "replace": "replace_similar", "replace_similar": "replace_similar"}


def load_gate_noppr(md):
    m = GateMLP(FEAT_DIM); m.load_state_dict(torch.load(Path(md) / "gate_noppr.pt", weights_only=True)); m.eval()
    z = np.load(Path(md) / "scaler_noppr.npz"); return m, z["mu"], z["sd"]


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s); P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0: return None
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


def validate(rows, feat, semb, gate, mu, sd, son_pos, label):
    """rows: list of rollout dict(含 query_id/skill_ids/bundle_type/reward)。返回指标 dict。"""
    miss_skl = sum(1 for r in rows for s in r["skill_ids"] if s not in semb)
    X = np.array([feat(r) for r in rows], np.float32); X[:, PPR] = 0.0     # 去 PPR
    Xs = ((X - mu) / sd).astype(np.float32)
    with torch.no_grad(): prob = 1 / (1 + np.exp(-gate(torch.tensor(Xs)).view(-1).numpy()))
    Y = np.array([1 if r["reward"] > 0 else 0 for r in rows]); R = np.array([r["reward"] for r in rows], np.float32)
    au = auroc(Y, prob); pred = prob >= 0.5; tot = R.sum() or 1.0
    # 导师选样口径:仅 Sonnet 正类对
    pm = np.array([son_pos.get((r["query_id"], r["bundle_type"])) == True for r in rows])
    pau = auroc(Y[pm], prob[pm]) if pm.sum() > 0 else None
    return {"检索器": label, "n": len(rows), "正类": int(Y.sum()), "AUROC": None if au is None else round(au, 3),
            "省调用%": round(100 * (~pred).mean()), "保留R%": round(100 * R[pred].sum() / tot),
            "正类对n": int(pm.sum()), "正类对AUROC": None if pau is None else round(pau, 3),
            "缺失skill嵌入": miss_skl}


def norm_bundle(v):
    if v in BUNDLE_TYPES: return v
    return BUNDLE_MAP.get(str(v).lower(), "gos_original")


def load_rows(path, ret_field=None):
    rows = []
    for l in Path(path).read_text().splitlines():
        l = l.strip()
        if not l or l[0] != "{": continue
        r = json.loads(l)
        if r.get("error_type") is not None or not isinstance(r.get("reward"), (int, float)): continue
        # 检索器标签
        ret = None
        for k in ([ret_field] if ret_field else []) + RETRIEVER_KEYS:
            if k and r.get(k): ret = r[k]; break
        # 扰动/bundle
        bt = None
        for k in BUNDLE_KEYS:
            if r.get(k): bt = norm_bundle(r[k]); break
        rows.append({"query_id": r["query_id"], "skill_ids": r.get("skill_ids") or [],
                     "bundle_type": bt or "gos_original", "reward": float(r["reward"]),
                     "retriever": ret or "unknown", "ppr_scores": r.get("ppr_scores")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="检索器 rollout 的 JSONL")
    ap.add_argument("--retriever-field", default=None, help="检索器名所在字段(默认自动探测)")
    ap.add_argument("--gos-baseline", action="store_true", help="用现有 Sonnet GoS 数据测通管线+出参考")
    a = ap.parse_args()

    qemb, semb = load_embeddings(DATA_DIR); feat = make_feat(qemb, semb)
    gate, mu, sd = load_gate_noppr(MODELS_DIR)
    base, _ = load_rollouts(DATA_DIR)
    son_pos = {k: (r["reward"] > 0) for k, r in base.items()}

    def show(rowset):
        cols = ["检索器", "n", "正类", "AUROC", "省调用%", "保留R%", "正类对n", "正类对AUROC", "缺失skill嵌入"]
        print("  " + "".join(f"{c:>12}" for c in cols))
        for m in rowset:
            print("  " + "".join(f"{str(m[c]):>12}" for c in cols))

    print("去-PPR 门控 gate_noppr | 检索器无关验证\n")
    print("参考基准:GoS 内 w/o-ppr 门控 held-out AUROC ≈ 0.742(消融结果)\n")

    if a.gos_baseline or not a.data:
        rows = [{"query_id": q, "skill_ids": r["skill_ids"], "bundle_type": b, "reward": r["reward"]}
                for (q, b), r in base.items()]
        print("=== 管线自测:GoS(Sonnet)数据(注:in-sample,仅验管线可跑)===")
        show([validate(rows, feat, semb, gate, mu, sd, son_pos, "GoS(self-test)")])
        if not a.data:
            print("\n(数据文件未给。retriever 数据到了后加 --data <file>,自动分检索器+混合验证。)")
            return

    rows = load_rows(a.data, a.retriever_field)
    rets = sorted({r["retriever"] for r in rows})
    print(f"\n=== 分检索器验证({len(rows)} 行, 检索器: {rets}) ===")
    results = [validate([r for r in rows if r["retriever"] == rt], feat, semb, gate, mu, sd, son_pos, rt) for rt in rets]
    show(results)
    print("\n=== 混合验证(所有检索器汇总) ===")
    show([validate(rows, feat, semb, gate, mu, sd, son_pos, "MIXED")])
    out = {"per_retriever": results, "mixed": validate(rows, feat, semb, gate, mu, sd, son_pos, "MIXED"),
           "gos_reference_wo_ppr": 0.742}
    Path(DATA_DIR, "results").mkdir(exist_ok=True)
    Path(DATA_DIR, "results", "retriever_validation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n✅ 已存 {DATA_DIR}/results/retriever_validation.json")


if __name__ == "__main__":
    main()
