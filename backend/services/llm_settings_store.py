"""
LLM 设置存储（File-based JSON Store）

设计目标：
  - 把"LLM 厂商 / API Key / 模型"等配置从 .env 文件迁出
    改为可视化设置页可改、可持久化（参考 NextChat 的做法）。
  - 存储位置遵循项目规则：usercontext/llm_settings.json
    （usercontext 目录是用户级别数据的存放点）。
  - 原子写：先写 .tmp，再 os.replace，避免写入中途崩溃导致配置损坏。

文件结构：
  {
    "active_provider": "agnes",
    "providers": {
      "deepseek": {"api_key": "...", "base_url": "...", "model": "..."},
      "qwen":     {"api_key": "...", "base_url": "...", "model": "..."},
      "zhipu":    {"api_key": "...", "base_url": "...", "model": "..."},
      "ollama":   {"api_key": "",   "base_url": "...", "model": "..."},
      "openai":   {"api_key": "...", "base_url": "...", "model": "..."},
      "agnes":    {"api_key": "...", "base_url": "...", "model": "..."}
    },
    "default_temperature": 0.7,
    "default_max_tokens": 1000
  }

线程安全：单进程文件读写不加锁（FastAPI 单进程下请求串行 + 短临界区）。
        如未来要部署多 worker，可换 threading.Lock 或外部 Redis。
"""
import json
import logging
import os
import threading
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 路径常量（文件仅在用户通过设置页 PUT 保存时创建）
# ---------------------------------------------------------------------------
# 首次启动时不自动创建，所有配置优先从 .env 读取。
# 用户用过设置页 PUT 后文件才会被 _write() 创建，
# 此后 _read() 以文件内容为准（不再读 .env）。
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SETTINGS_DIR = os.path.join(_PROJECT_ROOT, "usercontext")
_SETTINGS_FILE = os.path.join(_SETTINGS_DIR, "llm_settings.json")

# 同一进程内多次调用 SettingsStore 共享文件锁
_file_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Provider 默认配置（首次启动时写入文件）
# ---------------------------------------------------------------------------
PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "qwen": {
        "api_key": "",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-turbo",
    },
    "zhipu": {
        "api_key": "",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
    },
    "ollama": {
        # Ollama 是本地服务，不需要 API Key
        "api_key": "",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b",
    },
    "openai": {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "agnes": {
        "api_key": "",
        "base_url": "https://apihub.agnes-ai.com/v1",
        "model": "agnes-1.5-flash",
    },
}

# 用于前端下拉选择时显示的"厂商信息表"
PROVIDER_META: List[Dict[str, str]] = [
    {"id": "deepseek", "name": "DeepSeek",      "needs_key": "true"},
    {"id": "qwen",     "name": "通义千问 (Qwen)", "needs_key": "true"},
    {"id": "zhipu",    "name": "智谱 GLM",       "needs_key": "true"},
    {"id": "ollama",   "name": "Ollama (本地)",  "needs_key": "false"},
    {"id": "openai",   "name": "OpenAI",         "needs_key": "true"},
    {"id": "agnes",    "name": "Agnes AI",       "needs_key": "true"},
]

DEFAULT_ACTIVE = "agnes"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _default_settings() -> Dict[str, Any]:
    """首次启动时使用的默认配置（含每个 provider 的默认值）"""
    return {
        "active_provider": DEFAULT_ACTIVE,
        "providers": {pid: dict(cfg) for pid, cfg in PROVIDER_DEFAULTS.items()},
        "default_temperature": DEFAULT_TEMPERATURE,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
    }


def _merge_defaults(stored: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并"硬盘上的配置"与"代码中的 PROVIDER_DEFAULTS"。

    目的：当代码新增了一个 provider / 改了默认值时，
    已存在的用户配置文件应自动获得新字段（不覆盖用户已有的 api_key）。

    Returns:
        合并后的完整配置（已保证包含所有 provider 字段）。
    """
    base = _default_settings()
    base["active_provider"] = stored.get("active_provider", DEFAULT_ACTIVE)
    base["default_temperature"] = float(
        stored.get("default_temperature", DEFAULT_TEMPERATURE)
    )
    base["default_max_tokens"] = int(
        stored.get("default_max_tokens", DEFAULT_MAX_TOKENS)
    )
    stored_providers = stored.get("providers") or {}
    for pid, default_cfg in PROVIDER_DEFAULTS.items():
        existing = stored_providers.get(pid) or {}
        # 已有值 → 用 existing；缺失字段 → 用 default
        merged = dict(default_cfg)
        merged.update({k: v for k, v in existing.items() if v not in (None,)})
        base["providers"][pid] = merged
    return base


def _atomic_write(path: str, data: str) -> None:
    """原子写：先写 .tmp 再 os.replace，避免半写状态污染主文件"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())  # 强刷盘，防止 OS 缓存丢数据
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------
class LLMSettingsStore:
    """
    LLM 设置文件存储。

    用法：
        store = LLMSettingsStore()           # 自动加载 / 初始化
        active = store.get_active_provider() # 取得当前激活 provider 的完整配置
        store.set_active_provider("openai")  # 切换
        store.update_provider("openai", api_key="sk-...")  # 修改字段
    """

    def __init__(self) -> None:
        self._ensure_loaded()

    # -------------------- 文件 IO --------------------
    def _ensure_loaded(self) -> None:
        """
        确保配置可读。

        不再自动创建 usercontext/ 目录和 llm_settings.json。
        文件仅在用户通过设置页 PUT 主动保存时由 _write() 创建。
        文件不存在时 _read() 自动回退到 .env 构建配置。
        """
        pass

    def _read(self) -> Dict[str, Any]:
        """读取配置。文件不存在时从 .env 环境变量构建。"""
        if not os.path.exists(_SETTINGS_FILE):
            return self._build_from_env()
        with _file_lock:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                raw = f.read()
        try:
            stored = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            logger.warning("LLM 设置文件损坏，回退到 .env 配置")
            return self._build_from_env()
        return _merge_defaults(stored)

    @staticmethod
    def _build_from_env() -> Dict[str, Any]:
        """
        从 .env 环境变量构建完整配置（文件不存在时使用）。

        读取规则：
          - LLM_PROVIDER → 激活的 provider（不设则用默认值 "agnes"）
          - {PID}_API_KEY / {PID}_BASE_URL / {PID}_MODEL → 各 provider 配置
            （不设则用 PROVIDER_DEFAULTS 中的默认值）
        """
        from dotenv import load_dotenv
        load_dotenv()

        data = _default_settings()

        # 从 LLM_PROVIDER 读取激活 provider（兼容大小写）
        env_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if env_provider in PROVIDER_DEFAULTS:
            data["active_provider"] = env_provider

        # 为每个 provider 从环境变量读取配置
        for pid in PROVIDER_DEFAULTS:
            cfg = data["providers"][pid]
            prefix = pid.upper()
            env_key = os.environ.get(f"{prefix}_API_KEY")
            env_url = os.environ.get(f"{prefix}_BASE_URL")
            env_model = os.environ.get(f"{prefix}_MODEL")
            if env_key:
                cfg["api_key"] = env_key
            if env_url:
                cfg["base_url"] = env_url
            if env_model:
                cfg["model"] = env_model

        logger.debug("从 .env 构建 LLM 配置: active=%s", data["active_provider"])
        return data

    def _write(self, data: Dict[str, Any]) -> None:
        with _file_lock:
            os.makedirs(_SETTINGS_DIR, exist_ok=True)
            _atomic_write(
                _SETTINGS_FILE,
                json.dumps(data, ensure_ascii=False, indent=2),
            )

    # -------------------- 对外 API --------------------
    def get_all(self) -> Dict[str, Any]:
        """读取完整配置（包含所有 provider）。"""
        return self._read()

    def get_active_provider_id(self) -> str:
        return self._read()["active_provider"]

    def get_active_provider(self) -> Dict[str, str]:
        """取得当前激活 provider 的 {api_key, base_url, model} 字典。"""
        data = self._read()
        pid = data["active_provider"]
        return dict(data["providers"][pid])

    def get_provider(self, provider_id: str) -> Dict[str, str]:
        data = self._read()
        if provider_id not in data["providers"]:
            raise KeyError(f"未知 provider: {provider_id}")
        return dict(data["providers"][provider_id])

    def get_default_params(self) -> Dict[str, float]:
        data = self._read()
        return {
            "temperature": float(data["default_temperature"]),
            "max_tokens": int(data["default_max_tokens"]),
        }

    def set_active_provider(self, provider_id: str) -> None:
        data = self._read()
        if provider_id not in data["providers"]:
            raise KeyError(f"未知 provider: {provider_id}")
        data["active_provider"] = provider_id
        self._write(data)
        logger.info("切换激活 provider: %s", provider_id)

    def update_provider(
        self,
        provider_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        更新指定 provider 的字段（None 表示不修改）。
        返回更新后的完整配置。
        """
        data = self._read()
        if provider_id not in data["providers"]:
            raise KeyError(f"未知 provider: {provider_id}")
        cfg = data["providers"][provider_id]
        if api_key is not None:
            cfg["api_key"] = api_key
        if base_url is not None:
            cfg["base_url"] = base_url
        if model is not None:
            cfg["model"] = model
        data["providers"][provider_id] = cfg
        self._write(data)
        return dict(cfg)

    def update_default_params(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        data = self._read()
        if temperature is not None:
            data["default_temperature"] = float(temperature)
        if max_tokens is not None:
            data["default_max_tokens"] = int(max_tokens)
        self._write(data)

    @staticmethod
    def list_providers_meta() -> List[Dict[str, str]]:
        """返回 provider 元信息（id / name / needs_key），给前端渲染下拉。"""
        return [dict(m) for m in PROVIDER_META]

    def get_provider_with_env_fallback(self, provider_id: str) -> Dict[str, str]:
        """
        取得 provider 配置，缺失字段自动从环境变量补齐。

        设计动机：用户初次启动时 JSON 文件里 api_key 为空（设置页未配置），
        但 .env 中可能有 AGNES_API_KEY 等。此时应让 .env 作为兜底，
        避免出现"配置已存在但系统认为没配置"的诡异状态。

        Returns:
            完整 provider 配置（api_key / base_url / model 一定非空，
            除非连环境变量也没有）。
        """
        import os
        cfg = self.get_provider(provider_id)
        if not cfg.get("api_key"):
            env_val = os.environ.get(f"{provider_id.upper()}_API_KEY")
            if env_val:
                cfg["api_key"] = env_val
        if not cfg.get("base_url"):
            env_val = os.environ.get(f"{provider_id.upper()}_BASE_URL")
            if env_val:
                cfg["base_url"] = env_val
        if not cfg.get("model"):
            env_val = os.environ.get(f"{provider_id.upper()}_MODEL")
            if env_val:
                cfg["model"] = env_val
        return cfg

    # -------------------- 辅助 --------------------
    @staticmethod
    def settings_file_path() -> str:
        """暴露文件路径，方便前端展示与运维排错。"""
        return _SETTINGS_FILE

    @staticmethod
    def mask_api_key(api_key: str) -> str:
        """API Key 脱敏：保留首尾各 4 字符，中间用 **** 代替。"""
        if not api_key:
            return ""
        if len(api_key) <= 8:
            return "****"
        return f"{api_key[:4]}{'*' * max(4, len(api_key) - 8)}{api_key[-4:]}"
