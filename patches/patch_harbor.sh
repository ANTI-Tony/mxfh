#!/bin/bash
# 修复 mini-swe-agent 容器内 LiteLLM 缺 proxy 依赖:OpenRouter 带斜杠模型名(如 google/gemini-2.5-pro)
# 会让 LiteLLM 走 proxy 代码路径,import fastapi/orjson。默认镜像没装,导致 agent_setup_failed。
F=$(find ~/.local/share/uv/tools/harbor -name mini_swe_agent.py -path "*agents/installed*" 2>/dev/null | head -1)
[ -z "$F" ] && { echo "找不到 harbor mini_swe_agent.py — 请先: uv tool install harbor"; exit 1; }
if grep -q "with orjson" "$F"; then echo "已打过补丁: $F"; exit 0; fi
perl -i -pe "s/--with 'litellm\[proxy\]'/--with 'litellm[proxy]' --with orjson --with fastapi --with uvicorn --with backoff --with pyyaml/" "$F"
grep -q "with orjson" "$F" && echo "✅ 补丁已应用: $F" || echo "❌ 失败 — 请手动在 install 命令追加: --with orjson --with fastapi --with uvicorn"
