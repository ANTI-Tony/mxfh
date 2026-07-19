"""跨模型泛化实验：用 Sonnet 跑过的【完全相同的 skill bundle】，换成 GPT-5.5(codex) 重跑。

为什么 replay 而非重新检索：
  - 公平：同一任务、同一 bundle(skill_ids 完全一致), 只换 agent 模型 → 干净对比模型差异
  - 绕开 vertexai 卡死：不调 gos 检索/嵌入, 不 import gos.core.engine(其 fast_graphrag→vertexai
    import 在本机会卡死)

从 runs.jsonl 读 Sonnet(claude-sonnet-4.6) 每个 (query,bundle) 的 skill_ids + ppr_scores,
原样喂给 run_once(backend=openai, model=gpt-5.5)。结果 append 回 runs.jsonl(model_name=gpt-5.5)。

用法:
  # 先验证 1 任务 1 bundle:
  python -m scripts.run_gpt_replay --tasks data-to-d3 --bundle-types gos_original
  # 全量 10 任务 4 bundle:
  python -m scripts.run_gpt_replay --tasks <逗号分隔> --bundle-types gos_original,delete_top,add_irrelevant,replace_similar
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from src.agent_runner import run_once

B_ALL = ("gos_original", "delete_top", "add_irrelevant", "replace_similar")

def _err(r): v=r.get("error_type"); return v if v is not None else r.get("error")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_gpt.yaml")
    ap.add_argument("--tasks", required=True, help="逗号分隔的 query_id; 'auto:N' 自动选N个有信号的; 'all-positive' 选所有含 Sonnet reward>0 bundle 的 query")
    ap.add_argument("--bundle-types", default=",".join(B_ALL))
    ap.add_argument("--src-model", default="claude-sonnet-4.6", help="复用哪个模型跑过的bundle")
    ap.add_argument("--only-positive", action="store_true",
                    help="只跑 Sonnet reward>0 的(query,bundle)对。老师建议:先在GoS里选reward>0的样本再跑,"
                         "别随便选(随便选大多reward=0,别的模型也失败,白烧钱、无信号)。")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    rd = Path(cfg["paths"]["results_dir"]).expanduser().resolve()
    jsonl = rd / "runs.jsonl"
    runs = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    src = [r for r in runs if (r.get("model_name") or r.get("model")) == a.src_model]
    latest = {}
    for r in src: latest[(r["query_id"], r["bundle_type"])] = r
    clean = lambda q,b: (q,b) in latest and _err(latest[(q,b)]) is None and latest[(q,b)].get("reward") is not None

    bundle_types = [b.strip() for b in a.bundle_types.split(",") if b.strip()]

    # 选任务
    if a.tasks.startswith("auto:"):
        n = int(a.tasks.split(":")[1])
        comp = [q for q in sorted({q for q,_ in latest}) if all(clean(q,b) for b in B_ALL)]
        # 优先选 Sonnet 下 4bundle reward 有差异(有信号)的
        sig = [q for q in comp if len({latest[(q,b)]["reward"] for b in B_ALL}) > 1]
        tasks = (sig + [q for q in comp if q not in sig])[:n]
    elif a.tasks == "all-positive":
        # 老师建议:所有【含 Sonnet reward>0 bundle】的 query(配合 --only-positive 只跑那些正类对)
        tasks = sorted({q for (q, b), r in latest.items()
                        if _err(r) is None and (r.get("reward") or 0) > 0})
    else:
        tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]

    print(f"[gpt-replay] model={cfg['agent']['model']} backend={cfg['agent']['backend']}")
    print(f"[gpt-replay] 任务({len(tasks)}): {tasks}")
    print(f"[gpt-replay] bundle_types: {bundle_types}")

    done_gpt = {(r["query_id"], r["bundle_type"]) for r in runs
                if (r.get("model_name") or r.get("model")) == cfg["agent"]["model"]
                and _err(r) is None and r.get("reward") is not None}

    for q in tasks:
        for b in bundle_types:
            if (q, b) in done_gpt:
                print(f"[skip] {q} {b} 已有 {cfg['agent']['model']} 结果"); continue
            if not clean(q, b):
                print(f"[skip] {q} {b}: Sonnet 无干净 bundle 可复用"); continue
            if a.only_positive and not ((latest[(q, b)].get("reward") or 0) > 0):
                print(f"[skip] {q} {b}: Sonnet reward=0(--only-positive 只跑正类对)"); continue
            src_r = latest[(q, b)]
            skill_ids = src_r["skill_ids"]
            ppr_list = src_r.get("ppr_scores") or []
            ppr_scores = {s: (ppr_list[i] if i < len(ppr_list) else 0.0)
                          for i, s in enumerate(skill_ids)}
            print(f"[run] {q} [{b}] {len(skill_ids)} skills via {cfg['agent']['model']}")
            run_once(
                query_id=q,
                query=src_r["query"],
                bundle_type=b,
                bundle=skill_ids,
                library={},                       # lite: skill_names 退化为 skill_ids, 不 import gos
                ppr_scores=ppr_scores,
                agent_cfg=cfg["agent"],
                paths_cfg=cfg["paths"],
                results_path=jsonl,
                perturbation_meta=src_r.get("perturbation_meta") or {},
            )
    print("\nDone.")

if __name__ == "__main__":
    main()
