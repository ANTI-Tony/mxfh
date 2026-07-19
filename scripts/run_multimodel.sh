#!/bin/bash
# 多模型跨模型泛化跑批:固定集 22query×4bundle=88,每模型断点续跑,并发3,自动清停止容器
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS=$(echo "adaptive-cruise-control,civ6-adjacency-optimizer,court-form-filling,data-to-d3,debug-trl-grpo,drone-planning-control,energy-ac-optimal-power-flow,exceltable-in-ppt,exoplanet-detection-period,fix-erlang-ssh-cve,gravitational-wave-detection,invoice-fraud-detection,jpg-ocr-stat,lab-unit-harmonization,lean4-proof,mars-clouds-clustering,organize-messy-files,pddl-tpp-planning,powerlifting-coef-calc,r2r-mpc-control,seismic-phase-picking,tictoc-unnecessary-abort-detection")
BUNDLES="gos_original,delete_top,add_irrelevant,replace_similar"
MODELS=(
  "google/gemini-2.5-pro"
  "qwen/qwen3-max"
  "z-ai/glm-4.6"
  "moonshotai/kimi-k2"
  "anthropic/claude-sonnet-4"
  "meta-llama/llama-4-maverick"
  "google/gemini-2.5-flash"
)
MAXJOBS=1   # 串行:并发会让多个 harbor 抢 Docker 导致 RuntimeError 秒失败
LOGDIR=logs/mm_logs; mkdir -p $LOGDIR

run_one() {
  local M="$1"; local SAFE=$(echo "$M" | tr '/' '_')
  local CFG="configs/experiment_${SAFE}.yaml"
  sed "s|model: openrouter/google/gemini-2.5-flash|model: openrouter/${M}|" configs/experiment_or.yaml > "$CFG"
  echo "[$(date '+%H:%M')] START $M"
  ${PYTHON:-python3} -m scripts.run_gpt_replay --config "$CFG" --tasks "$TASKS" --bundle-types "$BUNDLES" > "$LOGDIR/${SAFE}.log" 2>&1
  docker container prune -f >/dev/null 2>&1
  echo "[$(date '+%H:%M')] DONE  $M | 磁盘空闲 $(df -h / | tail -1 | awk '{print $4}')"
}

# 后台磁盘守卫:只清停止容器(绝不清构建缓存/镜像——否则每条重build会联网超时)
# 仅当磁盘<5G 危急时才清 dangling 镜像
( while true; do
    docker container prune -f >/dev/null 2>&1
    FREE=$(df -g / | tail -1 | awk '{print $4}')
    if [ "${FREE:-99}" -lt 5 ]; then docker image prune -f >/dev/null 2>&1; fi
    sleep 180
done ) &
GUARD=$!

for M in "${MODELS[@]}"; do
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXJOBS" ]; do sleep 20; done
  run_one "$M" &
  sleep 8
done
wait $(jobs -rp | grep -v "$GUARD" 2>/dev/null)
kill $GUARD 2>/dev/null
echo "ALL_MODELS_DONE $(date)"
