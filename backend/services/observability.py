"""
Langfuse LLM 可观测性集成（参考掘金文章：https://juejin.cn/post/7633462423300407336）

设计原则（重要程度从高到低）：
  1. **优雅降级**（最关键）：未安装 langfuse 或未配 KEY 时，所有函数降级为 no-op，
     主流程零影响、零异常、零日志噪音。这保证已有部署不会被新依赖破坏。
  2. **OpenAI drop-in 替换**：用 `from langfuse.openai import OpenAI` 替换 `from openai import OpenAI`，
     让单次 LLM 调用自动产生 generation 记录（model、prompt、completion、token、cost）。
  3. **`@observe` 装饰器**：在管线入口（Creation / Interaction / Growth / Director / Actor）
     标记 span，让 Langfuse UI 上看到"trace 树"而非扁平列表。
  4. **graceful shutdown flush**：FastAPI 进程结束时主动 flush，避免最后几条 trace 丢失
     （文章踩坑 #3）。
  5. **采样率可配**：`LANGFUSE_SAMPLE_RATE=0.1` 表示只记录 10% 的请求。

关键环境变量（全部可选）：
  - LANGFUSE_ENABLED        (bool, default false)  显式开关；未设 KEY 时即使=true 也会降级
  - LANGFUSE_PUBLIC_KEY     (str)                  Langfuse 项目 Public Key（pk-...）
  - LANGFUSE_SECRET_KEY     (str)                  Langfuse 项目 Secret Key（sk-...）
  - LANGFUSE_HOST           (str, default cloud)   自部署填 http://localhost:3000
  - LANGFUSE_SAMPLE_RATE    (float, default 1.0)   0.0~1.0，1.0=全量记录
  - LANGFUSE_RELEASE        (str, default "")      标识发布版本，方便按版本切片

踩坑提醒（来自文章）：
  - 异步函数不要手动 `langfuse.trace()` 串联子 span——contextvars 会丢，必须用装饰器。
  - Web 服务问题不大（进程常驻），但 startup_event / shutdown_event 必须配套使用。
  - `langfuse>=2.40` 之后 API 才稳定，老版本 trace 树会显示为空。
"""

from __future__ import annotations

import logging
import os
import functools
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部状态
# ---------------------------------------------------------------------------
# 用模块级变量缓存"langfuse 是否可用"和"装饰器原函数"，避免每次 import 重新探测
_ENABLED: Optional[bool] = None  # None=未初始化；True/False=已探测
_LANGFUSE_OBSERVE = None         # langfuse.decorators.observe 函数；不可用时为 None
_LANGFUSE_CONTEXT = None         # langfuse.decorators.langfuse_context；不可用时为 None
_OPENAI_FACTORY = None           # langfuse.openai.OpenAI 类；不可用时为 None
_RAW_LANGFUSE = None             # 原始 langfuse 模块（用于 Langfuse() 客户端创建）


def _read_env_bool(key: str, default: bool = False) -> bool:
    """容忍多种真值写法：1/true/yes/on（大小写不敏感）"""
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _read_env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _detect() -> bool:
    """
    探测 Langfuse 是否真的可用（同时满足：开关 + 装好包 + 配好 key）。

    只跑一次，结果缓存在 _ENABLED。后续 is_enabled() 直接读缓存。
    """
    global _ENABLED, _LANGFUSE_OBSERVE, _LANGFUSE_CONTEXT, _OPENAI_FACTORY, _RAW_LANGFUSE

    if _ENABLED is not None:
        return _ENABLED

    # 1. 开关检查
    if not _read_env_bool("LANGFUSE_ENABLED", False):
        logger.info("[langfuse] 未启用（LANGFUSE_ENABLED=false）")
        _ENABLED = False
        return False

    # 2. 依赖检查
    try:
        from langfuse.decorators import observe, langfuse_context
        from langfuse.openai import OpenAI as LangfuseOpenAI
        import langfuse as langfuse_pkg
    except ImportError:
        logger.warning(
            "[langfuse] LANGFUSE_ENABLED=true 但未安装 langfuse 包；"
            "请 `pip install langfuse>=2.40,<3.0`；自动降级为 no-op"
        )
        _ENABLED = False
        return False

    # 3. Key 检查
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    sec = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    if not pub or not sec:
        logger.warning(
            "[langfuse] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 未配置；"
            "自动降级为 no-op（请到 Langfuse Settings → API Keys 复制粘贴）"
        )
        _ENABLED = False
        return False

    # 4. 全部通过 → 装入缓存
    _LANGFUSE_OBSERVE = observe
    _LANGFUSE_CONTEXT = langfuse_context
    _OPENAI_FACTORY = LangfuseOpenAI
    _RAW_LANGFUSE = langfuse_pkg

    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    sample = _read_env_float("LANGFUSE_SAMPLE_RATE", 1.0)
    release = os.environ.get("LANGFUSE_RELEASE", "")
    logger.info(
        "[langfuse] 启用成功：host=%s, sample_rate=%.2f, release=%s",
        host, sample, release or "(unset)",
    )
    _ENABLED = True

    # 5. Health check：验证 key 是否能通过 Langfuse 服务认证
    #    必要性：langfuse.openai.OpenAI 2.60 在 chat 调用时会**同步**触发 trace 上报，
    #    如果 key 无效 / 服务不可达，错误会**冒泡**到主 LLM 调用并触发重试
    #    （实测日志：每个 chat 报 3 次 ConnectionError，全部用降级输出）
    #    健康检查失败时，自动回退到原生 openai.OpenAI + identity 装饰器，
    #    @observe_safe 仍可作为装饰器（不真上报，但不报错）
    if not _health_check():
        logger.warning(
            "[langfuse] health check 失败（key 无效或服务不可达），"
            "自动回退到 no-op 模式：OpenAI 改用原生版（避免上报失败污染 LLM 调用），"
            "@observe_safe 装饰器保留为 no-op（不影响主流程）"
        )
        from openai import OpenAI as NativeOpenAI
        _OPENAI_FACTORY = NativeOpenAI
        _LANGFUSE_OBSERVE = None  # 让 observe_safe 退化为 identity
        _LANGFUSE_CONTEXT = None
        _RAW_LANGFUSE = None
        _ENABLED = False  # 视为"未启用"，让 flush() 等函数 no-op

    return _ENABLED


def _health_check() -> bool:
    """
    探测 Langfuse 服务是否真的能连通 + key 是否有效。

    实现：调用 `langfuse.Langfuse().auth_check()`，捕获所有异常。
    返回 True 表示服务可达且 key 有效；False 表示需要回退。

    注意：只在 _detect() 内调用一次（结果由 _ENABLED 缓存），
          不会对每次 LLM 调用都做 health check。

    关键：必须**显式**把 env vars 传给 Langfuse()，否则 SDK 默认从
          `os.environ` 读，但 uvicorn 启动时 `load_dotenv()` 的顺序
          可能导致 SDK 拿到空值 → 永远 health check 失败。
    """
    try:
        client = _RAW_LANGFUSE.Langfuse(  # type: ignore
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        client.auth_check()
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "[langfuse] health check 失败：%s: %s",
            type(e).__name__, str(e)[:150],
        )
        return False


def is_enabled() -> bool:
    """外部查询：当前是否真的启用了 Langfuse tracing"""
    return _detect()


# ---------------------------------------------------------------------------
# 公开 API 1：OpenAI drop-in 工厂
# ---------------------------------------------------------------------------

def get_openai_class():
    """
    返回 OpenAI 客户端类。
    - Langfuse 启用时：返回 langfuse.openai.OpenAI（自动 wrap，零侵入）
    - 未启用时：返回 openai.OpenAI（行为完全不变）

    用法（在 llm_service.py 中）：
        from backend.services.observability import get_openai_class
        OpenAI = get_openai_class()
        client = OpenAI(api_key=..., base_url=..., timeout=...)
    """
    if _detect() and _OPENAI_FACTORY is not None:
        return _OPENAI_FACTORY
    # 降级：返回原生 OpenAI
    from openai import OpenAI
    return OpenAI


# ---------------------------------------------------------------------------
# 公开 API 2：@observe_safe 装饰器（核心）
# ---------------------------------------------------------------------------

def observe_safe(
    name: Optional[str] = None,
    *,
    as_type: str = "span",
    sample_rate: Optional[float] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    "安全版" @observe 装饰器。
    - Langfuse 启用时 → 等价于 `@observe(name=name, as_type=as_type)`
    - Langfuse 关闭时 → 退化为一个 ~10ns 的 identity 装饰器，零开销
    - **同步/异步都安全**（关键！文章踩坑 #2：异步函数不要手动 langfuse.trace()）

    Args:
        name:  装饰后的 span/trace 名，会出现在 Langfuse UI
        as_type: "span"（默认，子调用） 或 "generation"（一次 LLM 调用） 或 "tool"
        sample_rate: 覆盖全局 LANGFUSE_SAMPLE_RATE（不传则用全局）

    用法：
        @observe_safe("creation.run", as_type="span")
        def run(self, user_input):
            ...
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not _detect() or _LANGFUSE_OBSERVE is None:
            # 降级：原样返回，不做任何包装
            return func

        # 实际启用：透传给 langfuse.observe
        kwargs: dict = {"as_type": as_type}
        if name:
            kwargs["name"] = name
        if sample_rate is not None:
            kwargs["sample_rate"] = sample_rate
        return _LANGFUSE_OBSERVE(**kwargs)(func)

    return decorator


# ---------------------------------------------------------------------------
# 公开 API 3：给当前 trace 打元数据（user_id / session_id / tags）
# ---------------------------------------------------------------------------

def update_current_trace(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    在 `@observe` 装饰的函数内部调用，给"当前 trace"附加元数据。
    Langfuse 关闭时 → no-op。

    用法（放在被 @observe 装饰的函数体内）：
        @observe_safe("chat.run")
        def run(...):
            update_current_trace(
                user_id="anonymous",         # 实际项目里用登录用户 ID
                session_id=str(session_id),  # 关联到 ChatSession.id
                tags=["director+actor"],
                metadata={"character_id": character_id},
            )
    """
    if not _detect() or _LANGFUSE_CONTEXT is None:
        return
    try:
        payload: dict = {}
        if user_id is not None:
            payload["user_id"] = user_id
        if session_id is not None:
            payload["session_id"] = session_id
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata
        if payload:
            _LANGFUSE_CONTEXT.update_current_trace(**payload)
    except Exception as e:  # noqa: BLE001
        # 元数据写入失败绝不能影响主流程
        logger.debug("[langfuse] update_current_trace 失败（已忽略）: %s", e)


def score_current_trace(
    name: str,
    value: Any,
    comment: Optional[str] = None,
) -> None:
    """
    给当前 trace 打分（Langfuse "Scoring" 能力）。
    Langfuse 关闭时 → no-op。

    Args:
        name: 评分名（如 "answer_length_ok", "contains_hallucination"）
        value: 分数（数字 0-1，或字符串分类如 "good"/"bad"）
        comment: 可选的人类可读说明
    """
    if not _detect() or _LANGFUSE_CONTEXT is None:
        return
    try:
        kwargs: dict = {"name": name, "value": value}
        if comment:
            kwargs["comment"] = comment
        _LANGFUSE_CONTEXT.score_current_trace(**kwargs)
    except Exception as e:  # noqa: BLE001
        logger.debug("[langfuse] score_current_trace 失败（已忽略）: %s", e)


# ---------------------------------------------------------------------------
# 公开 API 4：flush（应用退出时调用，避免丢最后几条 trace）
# ---------------------------------------------------------------------------

def flush() -> None:
    """
    同步刷新所有待上报的 trace。在 FastAPI shutdown_event 中调用。

    文章踩坑 #3：脚本 / Lambda 等短生命周期进程必须显式 flush，
    否则最后几条 trace 会随进程一起死掉。Web 服务影响小但仍建议加。
    """
    if not _detect() or _LANGFUSE_CONTEXT is None:
        return
    try:
        t0 = time.time()
        _LANGFUSE_CONTEXT.flush()
        logger.info("[langfuse] flush 完成（%.1fms）", (time.time() - t0) * 1000)
    except Exception as e:  # noqa: BLE001
        logger.warning("[langfuse] flush 失败（已忽略）: %s", e)


# ---------------------------------------------------------------------------
# 公开 API 5：创建 Langfuse 客户端（用于模型价格配置等高级能力）
# ---------------------------------------------------------------------------

def get_langfuse_client():
    """
    返回 langfuse.Langfuse() 客户端，用于：
      - langfuse.create_model(...) 自定义模型价格表
      - 手动 score / flush / 获取 trace 等

    Langfuse 关闭时返回 None。
    """
    if not _detect() or _RAW_LANGFUSE is None:
        return None
    try:
        return _RAW_LANGFUSE.Langfuse()
    except Exception as e:  # noqa: BLE001
        logger.warning("[langfuse] 创建 Langfuse 客户端失败: %s", e)
        return None
