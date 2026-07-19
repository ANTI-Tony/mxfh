"""Run one (query, bundle) trial via Harbor and persist a JSONL line.

The bundle is materialized by symlinking each chosen skill package into a
temp directory, then setting SKILLSBENCH_SKILLS_HOST_DIR so the SkillsBench
docker-compose mounts it at /opt/skillsbench/skills. The agent therefore sees
exactly the bundle we picked -- no GoS runtime retrieval, no full library.

Output schema (Tony 2026-06-01 spec)：
  query_id, query, bundle_type, skill_ids, skill_names, ppr_scores,
  reward, success, error_type,
  input_tokens, output_tokens,
  cache_creation_input_tokens, cache_read_input_tokens,
  total_api_tokens, total_cost_usd,
  execution_time, model_name, provider, run_id,
  agent_output, skip_reason, perturbation_meta
One JSON object per line, appended to results/runs.jsonl.

Back-compat：旧 jsonl 用 token_count / model / api_provider，读取处自己 fallback。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# 按模型查价表（单位：USD per 1M tokens）。来源：Anthropic 公开定价 2026-06。
# cache_write 用 5min ephemeral 价格；claude-code 默认就是 5min。1h ephemeral
# 价格更高（1.5x base），如果你的 agent 配置改用 1h，下面表里加一个 *_1h 字段
# 并改 _compute_cost 即可。
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5": {
        "input_per_m": 3.0,
        "output_per_m": 15.0,
        "cache_write_5m_per_m": 3.75,
        "cache_read_per_m": 0.30,
    },
    # alias 不带版本号的常见叫法，便于以后切型号
    "claude-sonnet-4-5-20250929": {
        "input_per_m": 3.0,
        "output_per_m": 15.0,
        "cache_write_5m_per_m": 3.75,
        "cache_read_per_m": 0.30,
    },
    # 第三方中转 apicursor 的模型名（claude-sonnet-4.6 点号写法）。
    # 注意：用的是 Anthropic 官方 sonnet 价位做估算；中转实际计费 + token
    # 计数都不可信（直接 curl 测过 "hi" 报 84860 input），total_cost_usd 仅作
    # 参考，真实花销以 apicursor 网站面板为准。
    "claude-sonnet-4.6": {
        "input_per_m": 3.0,
        "output_per_m": 15.0,
        "cache_write_5m_per_m": 3.75,
        "cache_read_per_m": 0.30,
    },
}


def _compute_cost_usd(
    model_name: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_creation_input_tokens: int | None,
    cache_read_input_tokens: int | None,
) -> float | None:
    """按模型定价表算单次 rollout 实际花了多少美元。

    Anthropic API 的 input/output/cache_* 是 4 个独立 bucket（每个都按各自 rate 计费），
    *不要*把 cache_read 重复加到 input 里。token 字段缺失时默认 0。
    """
    p = MODEL_PRICING.get(model_name or "")
    if p is None:
        return None
    return (
        (input_tokens or 0) * p["input_per_m"] / 1e6
        + (output_tokens or 0) * p["output_per_m"] / 1e6
        + (cache_creation_input_tokens or 0) * p["cache_write_5m_per_m"] / 1e6
        + (cache_read_input_tokens or 0) * p["cache_read_per_m"] / 1e6
    )


@dataclass
class RunRecord:
    query_id: str
    query: str
    bundle_type: str                 # "gos_original" | "delete_top" | "add_irrelevant" | "replace_similar"
    skill_ids: list[str]
    skill_names: list[str]
    ppr_scores: list[float]
    # ---- reward / 状态 ----
    agent_output: str
    reward: float | None
    success: bool
    execution_time: float
    error_type: str | None
    # ---- 6.1 spec: 成本/token 细分 ----
    # 4 个独立 bucket，对应 Anthropic API usage 字段。任一缺失保留 None
    # （便于 reader 区分"没数据"和"真 0"）。
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    total_api_tokens: int | None = None        # 4 个 bucket 求和
    total_cost_usd: float | None = None        # 按 MODEL_PRICING 算的 USD
    # ---- perturbation / 调试元数据 ----
    # 该 trial 因 perturbation 不适用而未实际跑 agent 时的原因（如
    # "no_ppr_neighbor_within_epsilon"）；正常 trial 为 None。
    # 与 error_type 的语义区别：error_type 是 agent/verifier/基建出错；
    # skip_reason 是设计上 perturbation 找不到合法候选——不算"错误"，
    # 但需写入 jsonl 让结果完整可分析。
    skip_reason: str | None = None
    # Perturbation-specific metadata (delete_top / add_irrelevant / replace_similar)。
    # gos_original 时为空 dict。schema 由 src/perturbations.py 决定。
    perturbation_meta: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # 跑这条 trial 时实际用的 agent 模型 + provider。surrogate 训练时不同
    # model/provider 的 reward 必须分桶处理，所以每行都要带。
    # 6.1 spec 把字段名固定为 model_name / provider（旧 jsonl 是 model / api_provider）。
    model_name: str | None = None
    provider: str | None = None


def _stage_bundle(bundle: list[str], full_library_dir: Path) -> Path:
    """Build a temp dir containing only the bundle's skill packages (symlinks)."""
    stage = Path(tempfile.mkdtemp(prefix="gos_sanity_"))
    for skill_id in bundle:
        src = full_library_dir / skill_id
        if not src.exists():
            shutil.rmtree(stage, ignore_errors=True)
            raise FileNotFoundError(
                f"Skill {skill_id!r} not found in {full_library_dir}"
            )
        (stage / skill_id).symlink_to(src.resolve())
    return stage


def _build_extra_mounts() -> list[dict]:
    """构造容器额外 bind 挂载列表。

    目前只做一件事：把 host 上的 playwright chromium 缓存挂到容器，
    避免 react-performance-debugging / fix-visual-stability 两个 task 的
    test.sh 在容器内重新从 playwright.azureedge.net 下载 chromium 失败。
    要求 host 上先 `python3 -m playwright install chromium`（>= 1.49.1）。
    """
    mounts: list[dict] = []
    pw_cache = Path.home() / ".cache" / "ms-playwright"
    if pw_cache.exists():
        # 不能 read_only：playwright install 会写 __dirlock，即便 chromium 已存在。
        # gos-sanity trial 串行执行，不会有并发写冲突。
        mounts.append({
            "type": "bind",
            "source": str(pw_cache),
            "target": "/root/.cache/ms-playwright",
        })
    return mounts


def _harbor_run(
    task_dir: Path,
    out_dir: Path,
    agent_cfg: dict,
    bundle_skills_dir: Path,
    timeout_s: int,
) -> subprocess.CompletedProcess:
    backend_to_agent = {"openai": "codex", "anthropic": "claude-code", "gemini": "gemini-cli",
                        "vllm": "mini-swe-agent"}
    cmd = [
        "harbor", "run",
        "--agent", backend_to_agent.get(agent_cfg["backend"], agent_cfg["backend"]),
        "--model", agent_cfg["model"],
        # 不再 --force-build：skill bundle 是运行时挂载（SKILLSBENCH_SKILLS_HOST_DIR），
        # 不进镜像，所以每个 rollout 重建镜像纯属浪费——会让 docker build cache 爆盘
        # （226 rollout 攒了 45G 撑爆磁盘）。复用已 build 的镜像，省盘省时间。
        "--timeout-multiplier", str(agent_cfg.get("harbor_timeout_multiplier", 5)),
        "-p", str(task_dir),
        "-o", str(out_dir),
    ]
    extra_mounts = _build_extra_mounts()
    if extra_mounts:
        cmd += ["--mounts-json", json.dumps(extra_mounts)]
    env = {**os.environ, "SKILLSBENCH_SKILLS_HOST_DIR": str(bundle_skills_dir)}

    # 从 .env.local 读 key + 可选 base_url
    _env_local = Path(__file__).parent.parent / ".env.local"
    _vals: dict[str, str] = {}
    if _env_local.exists():
        for _line in _env_local.read_text().splitlines():
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                k, v = _line.split("=", 1)
                _vals[k.strip()] = v.strip()
    _key = _vals.get("ANTHROPIC_API_KEY", "")
    _base_url = _vals.get("ANTHROPIC_BASE_URL", "")

    # 先清掉主机继承来的所有 Anthropic 路由变量（~/.claude/settings.json 里可能有
    # gpugeek 代理），再按模式重设，避免污染。
    for k in (
        "ANTHROPIC_BASE_URL", "ANTHROPIC_BASE_URLS", "ANTHROPIC_API_BASE",
        "ANTHROPIC_BEDROCK_BASE_URL", "ANTHROPIC_VERTEX_PROJECT_ID",
        "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ):
        env.pop(k, None)

    _backend = agent_cfg.get("backend", "anthropic")
    if _backend == "vllm":
        # 通用 OpenAI 兼容 chat 端点（本地 vLLM / DeepSeek / DashScope-Qwen / OpenAI），
        # 走 harbor 的 mini-swe-agent。model 用 provider/model 格式(如 openai/deepseek-chat)。
        # base_url 从 config.agent.base_url 取；key 从 .env.local 的 key_var 指定变量取。
        _oai_base = agent_cfg.get("base_url") or _vals.get("VLLM_BASE_URL", "")
        _key_var = agent_cfg.get("key_var", "")
        _oai_key = (_vals.get(_key_var, "") if _key_var else "") or "EMPTY"
        _model = agent_cfg.get("model", "")
        for k in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_ORG_ID",
                  "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            env.pop(k, None)
        if _model.startswith("openrouter/"):
            # OpenRouter: 用 LiteLLM 原生 openrouter provider。model=openrouter/<家族/型号>，
            # key 走 OPENROUTER_API_KEY（harbor 的 get_api_key_var_names 会为 openrouter/ 转发它），
            # 不设 OPENAI_API_BASE（避免 openai/ 双斜杠把 LiteLLM 带进 proxy 代码路径报 fastapi 缺失）。
            env["OPENROUTER_API_KEY"] = _oai_key
        else:
            # 通用 OpenAI 兼容 chat（本地 vLLM / DeepSeek / OpenAI）：openai/ + OPENAI_API_BASE。
            # 只设 OPENAI_API_KEY：harbor 若见 MSWEA_API_KEY 会只转发它、不转发 OPENAI_API_KEY。
            if _oai_base:
                env["OPENAI_API_BASE"] = _oai_base
            env["OPENAI_API_KEY"] = _oai_key
        env.pop("MSWEA_API_KEY", None)
    elif _backend == "openai":
        # codex agent (GPT) 走 OpenAI 兼容接口。中转的 sk-lkapi key 对 GPT 模型同样通用。
        # codex 读 OPENAI_API_KEY(合成 auth.json) + OPENAI_BASE_URL(写进 config.toml)。
        # 注意: 容器内这个 OPENAI_API_KEY 是中转 key, 与主机上 GoS embedding 用的真
        # OpenAI key 互不影响(embedding 在 _harbor_run 之前、父进程里跑)。
        _proxy_key = _vals.get("OPENAI_PROXY_KEY") or _key   # 默认复用中转 key
        _proxy_base = _vals.get("OPENAI_PROXY_BASE") or (
            (_base_url.rstrip("/") + "/v1") if _base_url else ""
        )
        for k in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_ORG_ID",
                  "CODEX_AUTH_JSON_PATH", "CODEX_FORCE_AUTH_JSON"):
            env.pop(k, None)
        env["OPENAI_API_KEY"] = _proxy_key
        if _proxy_base:
            env["OPENAI_BASE_URL"] = _proxy_base
    elif _base_url:
        # 第三方中转模式（如 apicursor.com）：claude-code 走 ANTHROPIC_BASE_URL +
        # ANTHROPIC_AUTH_TOKEN（Bearer）。这些中转不认 x-api-key，且 token 计数
        # 通常不可信——这是临时打通用，正式 cost 数据需用官方 key。
        env["ANTHROPIC_BASE_URL"] = _base_url
        env["ANTHROPIC_AUTH_TOKEN"] = _key
    elif _key.startswith("sk-ant-"):
        # 官方 API 模式：x-api-key，不设 base_url。
        env["ANTHROPIC_API_KEY"] = _key

    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout_s, check=False
    )


def _read_harbor_result(out_dir: Path) -> dict:
    """读取 harbor 写出的 trial result.json。

    harbor 在 out_dir 下会同时写两个 result.json：
      out_dir/<ts>/result.json                  ← batch summary（含 stats，无 reward）
      out_dir/<ts>/<trial_name>/result.json     ← 单个 trial 结果（含 verifier_result/agent_result）
    我们要的是后者。优先选含 ``verifier_result`` 字段的那个，避免任何路径深度
    断言耦合到 harbor 内部目录约定。
    """
    candidates = list(out_dir.glob("**/result.json"))
    if not candidates:
        return {}
    for c in candidates:
        try:
            data = json.loads(c.read_text())
        except json.JSONDecodeError:
            continue
        if "verifier_result" in data or "agent_result" in data:
            return data
    return json.loads(candidates[0].read_text())


def _classify_error(payload: dict, harbor_stderr: str) -> str | None:
    """Coarse error bucket. None when the run completed and verifier returned a reward.

    优先级：exception_info > agent_tokens==0 > reward 存在性。
    verifier 在 agent setup 失败时仍会写 reward=0.0（看到空 solution），
    必须先排除 exception 才能信任 reward 值。
    """
    # 1. exception_info 优先——agent/harbor 内部异常
    exc_type = (payload.get("exception_info") or {}).get("exception_type") or ""
    if exc_type:
        if "Timeout" in exc_type or "timeout" in exc_type.lower():
            return "agent_timeout"
        if "NonZeroAgentExitCodeError" in exc_type:
            # agent 进程非零退出：可能是 install 失败或 claude-code 本身 crash
            ar = payload.get("agent_result") or {}
            tok = (ar.get("n_input_tokens") or 0) + (ar.get("n_output_tokens") or 0)
            return "agent_setup_failed" if tok == 0 else "agent_failed"
        return f"harbor_exception:{exc_type}"

    # 2. harbor_stderr 兜底（exception_info 缺失时）
    if "Timeout" in harbor_stderr or "timeout" in harbor_stderr.lower():
        return "agent_timeout"
    if "NonZeroAgentExitCodeError" in harbor_stderr:
        return "agent_setup_failed"

    # 3. 无 exception：看 reward
    if not payload:
        return "harbor_no_result"
    vr = payload.get("verifier_result") or {}
    rewards = vr.get("rewards") or {}
    if rewards.get("reward") is not None:
        return None  # 正常完成
    if payload.get("agent_result") is None:
        return "agent_failed"
    if payload.get("verifier_result") is None:
        return "verifier_failed"
    return "unknown"


def run_once(
    *,
    query_id: str,
    query: str,
    bundle_type: str,
    bundle: list[str],
    library: dict,                       # skill_id -> SkillRecord (for skill_names)
    ppr_scores: dict[str, float],
    agent_cfg: dict,
    paths_cfg: dict,
    results_path: Path,
    perturbation_meta: dict[str, Any] | None = None,
) -> RunRecord:
    skill_names = [getattr(library.get(s), "name", s) for s in bundle]
    bundle_ppr = [float(ppr_scores.get(s, 0.0)) for s in bundle]

    record = RunRecord(
        query_id=query_id,
        query=query,
        bundle_type=bundle_type,
        skill_ids=list(bundle),
        skill_names=skill_names,
        ppr_scores=bundle_ppr,
        agent_output="",
        reward=None,
        success=False,
        execution_time=0.0,
        error_type=None,
        perturbation_meta=dict(perturbation_meta or {}),
        model_name=agent_cfg.get("model"),
        provider=agent_cfg.get("backend"),
    )

    t0 = time.time()
    stage_dir: Path | None = None
    try:
        stage_dir = _stage_bundle(
            bundle, Path(paths_cfg["skills_library"]).expanduser().resolve()
        )
        task_dir = (
            Path(paths_cfg["skillsbench_tasks"]).expanduser().resolve() / query_id
        )
        if not task_dir.exists():
            raise FileNotFoundError(f"SkillsBench task missing: {task_dir}")

        out_dir = Path(paths_cfg["results_dir"]) / "harbor_jobs" / f"{query_id}__{bundle_type}__{record.run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        proc = _harbor_run(
            task_dir, out_dir, agent_cfg, stage_dir,
            timeout_s=int(agent_cfg.get("timeout_s", 1800)),
        )
        payload = _read_harbor_result(out_dir) or {}
        # 防御：harbor 写出 "verifier_result": null / "agent_result": null（agent 容器 setup 失败常见）
        rewards = (payload.get("verifier_result") or {}).get("rewards") or {}
        agent_result = payload.get("agent_result") or {}

        record.reward = (
            float(rewards["reward"]) if rewards.get("reward") is not None else None
        )
        record.success = bool(record.reward and record.reward > 0)

        # 6.1 spec：4 个 token bucket 各自记录。harbor 把 Anthropic API usage
        # 字段保留 n_ 前缀传过来；同时兜底 cache_* 没前缀的写法。
        def _pick(*names: str) -> int | None:
            for n in names:
                v = agent_result.get(n)
                if v is not None:
                    return int(v)
            return None

        record.input_tokens = _pick("n_input_tokens", "input_tokens")
        record.output_tokens = _pick("n_output_tokens", "output_tokens")
        record.cache_creation_input_tokens = _pick(
            "n_cache_creation_input_tokens", "cache_creation_input_tokens"
        )
        record.cache_read_input_tokens = _pick(
            "n_cache_read_input_tokens", "cache_read_input_tokens"
        )
        # total_api_tokens = 4 个 bucket 求和（任一为 None 当 0）
        if any(v is not None for v in (
            record.input_tokens, record.output_tokens,
            record.cache_creation_input_tokens, record.cache_read_input_tokens,
        )):
            record.total_api_tokens = (
                (record.input_tokens or 0)
                + (record.output_tokens or 0)
                + (record.cache_creation_input_tokens or 0)
                + (record.cache_read_input_tokens or 0)
            )
        record.total_cost_usd = _compute_cost_usd(
            record.model_name,
            record.input_tokens,
            record.output_tokens,
            record.cache_creation_input_tokens,
            record.cache_read_input_tokens,
        )

        record.agent_output = (agent_result.get("final_output") or "")[:8000]
        record.error_type = _classify_error(payload, proc.stderr)

    except subprocess.TimeoutExpired:
        record.error_type = "agent_timeout"
    except FileNotFoundError as exc:
        record.error_type = f"missing:{exc}"
    except Exception as exc:                           # noqa: BLE001
        record.error_type = f"exception:{type(exc).__name__}:{exc}"
    finally:
        record.execution_time = time.time() - t0
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)
        _append_jsonl(record, results_path)

    return record


def _append_jsonl(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
