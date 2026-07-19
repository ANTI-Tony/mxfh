#!/bin/bash
# 可选:给 SkillsBench 任务 Dockerfile 的 pip install 加国内镜像 + 长超时 + 重试。
# 何时用:pip 下载重包(pycbc/lalsuite 等)超时(ReadTimeoutError)导致镜像 build 失败、
#         进而 harbor_exception:RuntimeError。网络好(海外/公司网)可跳过本步。
# 幂等:重复运行不会重复注入。撤销见末尾。
# 用法: bash patches/patch_pip_mirror.sh [graph-of-skills路径]  (默认 ../graph-of-skills)
set -u
GOS="${1:-../graph-of-skills}"
TASKS="$GOS/evaluation/skillsbench/tasks"
MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"   # 也可设 PIP_MIRROR 换阿里/中科大
[ -d "$TASKS" ] || { echo "找不到 $TASKS ,请传入 graph-of-skills 路径"; exit 1; }
n=0
for D in $(find "$TASKS" -name Dockerfile); do
  grep -q "$MIRROR" "$D" 2>/dev/null && continue
  sed -i.bak -E "s#pip install #pip install -i $MIRROR --timeout 600 --retries 20 #g" "$D" && rm -f "$D.bak" && n=$((n+1))
done
echo "已给 $n 个 Dockerfile 注入镜像($MIRROR)。"
echo "撤销: cd $TASKS && git checkout ."
