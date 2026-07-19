# 跨模型泛化实验 (Cross-Model Generalization of the Execution Gate)

验证一个在 **Claude Sonnet** 数据上训练的「执行门控」(cheap MLP surrogate),能否**零样本迁移**到其它执行模型(GPT / Gemini / Qwen / DeepSeek / GLM / Kimi / Llama …)——即换个 agent 模型,门控还能不能准确预测"这次调用值不值得(reward>0)"。

## 核心思想
门控输入只有 `query_emb + bundle_emb + bundle_type + ppr`(**不含执行模型身份**),所以同一个门控天然可套到任何模型上。若它在没训练过的模型上仍能把"会成功"的调用排到前面(AUROC 高),就证明门控学到的是**模型无关的可迁移信号**,可"训一次、任何执行模型前都能用"。

## 实验设计(重要)
- **固定评估集** = 22 个 bundle-sensitive query × 4 个 bundle = **88 个 (query,bundle)**,所有模型跑同一套,直接可比。
- 这 88 条**含 Sonnet 成功 46 + 失败 42**(有正有负),这样每个模型才有正负样本 → **能算门控 AUROC**。
  ⚠️ 若只跑"有信号"(全 Sonnet 成功)的子集,门控 AUROC 会退化(没有负样本可跳过)。
- 门控在**全部 Sonnet rollout**(`data/runs.jsonl`,~290 条)上训练;每个模型的 rollout 作为测试集,预测其 `reward>0`。

## 依赖
1. **Docker**(跑 SkillsBench 任务环境)。
2. **harbor**(agent runner):`uv tool install harbor`,然后打补丁(见下)。
3. **graph-of-skills** 仓库:需放在本仓库的**同级目录** `../graph-of-skills/`(内含 SkillsBench 任务 `evaluation/skillsbench/tasks` 和 skill 库 `data/skillsets/skills_200`)。向作者索取或从对应 repo 克隆。
4. **Python**:`pip install -r requirements.txt`(numpy / torch / pyyaml)。
5. **OpenRouter API key**(一个 key 覆盖所有家族):https://openrouter.ai → 充值 → 建 key。

## 一键跑

```bash
# 0. 目录结构应为:
#    parent/
#      mxfh/            <- 本仓库
#      graph-of-skills/ <- SkillsBench 任务+skill库(同级)

cd mxfh

# 1. 装依赖 + harbor + 补丁
pip install -r requirements.txt
uv tool install harbor
bash patches/patch_harbor.sh          # 修复容器内 LiteLLM 缺 proxy 依赖(带斜杠模型名必需)

# 2. 填 key
cp .env.example .env.local            # 编辑填入 OPENROUTER_KEY

# 3. 跑所有模型(串行,断点续跑,自动清停止容器)
#    模型列表在 scripts/run_multimodel.sh 顶部,可增删
bash scripts/run_multimodel.sh

# 4a. 【核心】在线持续更新:就一个门控模型,按固定顺序流过所有模型的 rollout,
#     每验证一条(或 --update-every 10/20 条)就原地更新它的权重,跨模型不重置、
#     一直累积。见的反馈越多越准。国外强模型先、国产后。全程只有一个模型。
python scripts/run_online_stream.py            # -> models/gate_online.pt(那一个持续更新后的模型)
# python scripts/run_online_stream.py --update-every 10   # 攒10条更新一次

# 4b. 三张表(门控AUROC / 时间 / 金额)—— 加载固定门控 models/gate.pt(静态基线)
python scripts/build_gen_tables.py
```

## ⚠️ 关键:全程就一个模型,一直原地更新它,不是每个模型独立、也不存分阶段 checkpoint
`run_online_stream.py` 用**一个门控模型**,从 `models/gate.pt`(Sonnet 训练)出发,
按顺序流过每个模型的 rollout,**编码器冻结、头每验证一条(或攒 `--update-every` 条)就
原地更新一次、不重置** —— 它在模型 A 上更新完的权重**带着**去跑模型 B……跨模型累积。
**从头到尾只有这一个模型**,跑完存成 `models/gate_online.pt`(那一个持续更新后的当前
状态),**不分阶段另存 checkpoint**。这是"一个模型、边用边更新、用得越久越好",不是
"现训现用"、也不是"每个模型各存一个"。

> 复现:初始 `gate.pt`(`train_gate.py` 确定性生成) + 本脚本确定性重放这条流 → 同一个在线模型。
> 这是对**已采集 rollout** 的确定性离线重放,几秒完成,不需要重跑昂贵的模型调用。
> 前提是各模型的 rollout 里有**足够的成功样本**(正负都有)——若某模型近乎全失败
> (可能是 agent 脚手架不适配该模型),它的反馈信息量低,累积效果会打折。

## 门控是一个固定的 checkpoint(不现训现用)
门控**已训练好并存在仓库里**:`models/gate.pt`(权重) + `models/scaler.npz`(标准化) + `models/gate_meta.json`(元信息)。
所有模型的评估都**加载这同一个固定门控**,保证可比、可复现。无需重训。

如需**重新训练**(例如换了基座数据):
```bash
python scripts/train_gate.py     # 在 data/runs.jsonl 的 Sonnet 数据上训练,seed=0 确定性,覆盖 models/gate.pt
```

单独跑某个模型:
```bash
sed 's|model: openrouter/google/gemini-2.5-flash|model: openrouter/qwen/qwen3-max|' \
    configs/experiment_or.yaml > configs/experiment_qwen.yaml
python -m scripts.run_gpt_replay --config configs/experiment_qwen.yaml \
    --tasks adaptive-cruise-control,data-to-d3,fix-erlang-ssh-cve \
    --bundle-types gos_original,delete_top,add_irrelevant,replace_similar
```

## 输出
`scripts/build_gen_tables.py` 打印三张表:
1. **泛化模型**:每模型的门控 AUROC / 省调用% / 保留reward% / reward每调用。
2. **花费时间**:每模型 rollout 数、均值/中位/总时长。
3. **花费金额**:每模型输入/输出 token、估算 USD。

所有 rollout 结果追加进 `data/runs.jsonl`(model_name = `openrouter/<家族/型号>`)。**可断点续跑**:重跑脚本会自动跳过已完成的 (model, query, bundle)。

## 文件
```
src/agent_runner.py           harbor + mini-swe-agent 运行器(含 OpenRouter/vLLM 后端)
scripts/run_gpt_replay.py     复用 Sonnet 检索出的同一 bundle,换模型重跑(公平对比)
scripts/run_multimodel.sh     多模型批量(串行,一键)
scripts/build_gen_tables.py   门控AUROC + 时间 + 金额三表
scripts/eval_mlp_generalization.py   单模型门控评估(细节版)
configs/experiment_or.yaml    OpenRouter 配置模板(改 model 一行即可)
data/runs.jsonl               Sonnet 基座 rollouts(训门控+提供bundle)+ 新采集的
data/surrogate/*.npz          缓存的 query/skill embedding
patches/patch_harbor.sh       harbor mini-swe-agent 依赖补丁
```

## 已知坑
- **Docker build 网络超时**(`ReadTimeoutError`):任务镜像首次 build 要 pip 下载科学计算包,网络慢会超时。**别清 Docker 构建缓存/镜像**(否则每条重 build、重复下载)。网络好的机器一次 build 成功后会缓存复用。
- **脚手架混淆**:Sonnet 原始数据用的是 claude-code;本实验其它模型用 mini-swe-agent(claude-code 不支持它们)。模型间 reward 差异含 agent 框架因素,论文需注明。观察到部分模型经 mini-swe-agent 成功率很低,需甄别是模型能力还是脚手架。
- **门控 AUROC 需正负样本**:见上"实验设计"。
