#!/bin/bash
# 全量跨模型评估:在 22 个 bundle-sensitive query × 4 bundle = 88 对上,
# 换模型重跑 Sonnet 检索出的同一 bundle(公平对比)。可断点续跑(自动跳过已完成对)。
# 按成本从低到高:glm -> qwen -> kimi(若余额耗尽,便宜的先跑完)。带余额守卫。
set -u
cd /Users/tonygpt/Desktop/gos-sanity-release
source .env.local 2>/dev/null; export OPENROUTER_KEY
OR="$OPENROUTER_KEY"
# 22 个 bundle-sensitive query(Sonnet 下 bundle 扰动会翻转成功/失败或改变 reward)
SENS="adaptive-cruise-control,civ6-adjacency-optimizer,court-form-filling,data-to-d3,debug-trl-grpo,drone-planning-control,energy-ac-optimal-power-flow,exceltable-in-ppt,exoplanet-detection-period,fix-erlang-ssh-cve,gravitational-wave-detection,invoice-fraud-detection,jpg-ocr-stat,lab-unit-harmonization,lean4-proof,mars-clouds-clustering,organize-messy-files,pddl-tpp-planning,powerlifting-coef-calc,r2r-mpc-control,seismic-phase-picking,tictoc-unnecessary-abort-detection"
BUNDLES="gos_original,delete_top,add_irrelevant,replace_similar"
MIN_BAL=2.0   # 余额低于此(美元)就停,保护用户

bal() { curl -s -m 15 "https://openrouter.ai/api/v1/credits" -H "Authorization: Bearer $OR" \
  | python3 -c "import sys,json;d=json.load(sys.stdin)['data'];print(f\"{d['total_credits']-d['total_usage']:.2f}\")" 2>/dev/null; }

# 成本从低到高
declare -a RUN=("z-ai/glm-4.6" "qwen/qwen3-max" "moonshotai/kimi-k2")

for m in "${RUN[@]}"; do
  b=$(bal); echo "===== 余额 \$$b | 准备跑 $m $(date +%H:%M) ====="
  if python3 -c "import sys;sys.exit(0 if float('$b')>= $MIN_BAL else 1)"; then :; else
    echo "!! 余额 \$$b < \$$MIN_BAL,停止以保护预算"; break
  fi
  short=$(echo "$m" | sed 's#.*/##')
  sed "s|model: openrouter/google/gemini-2.5-flash|model: openrouter/$m|" configs/experiment_or.yaml > configs/experiment_$short.yaml
  .venv/bin/python -m scripts.run_gpt_replay --config configs/experiment_$short.yaml \
      --tasks "$SENS" --bundle-types "$BUNDLES" 2>&1 | tail -4
  echo "===== $m 完成 $(date +%H:%M) 余额 \$$(bal) ====="
done
echo "FULL_DONE 余额 \$$(bal)"
