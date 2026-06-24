"""
CharacterSeed Frontend — API Client 模块
==========================================

设计理念（Why this module exists）：
  将全部 HTTP 调用逻辑集中在单一模块中，UI 层（app.py）不涉及 URL 拼接、
  requests 参数构造、异常捕获细节。

核心设计决策：
  1. 同步 requests 而非异步 aiohttp
     —— Streamlit 内部以同步方式运行脚本（每次 rerun 从头执行），
        使用同步 requests 在语义上完全匹配，且无需 asyncio 事件循环管理。
  2. 每个 API 函数独立封装
     —— 8 个端点 = 8 个函数，一一对应。
        避免用一个"通用调用函数"导致类型安全丢失和参数歧义。
  3. 异常统一转换为 dict 结果
     —— 不向上抛异常，而是返回 {"error": True, "detail": "..."} 字典。
        Streamlit 页面代码只需检查 "error" key，无需 try/except 样板。
  4. 文件上传使用 requests.post(files=...)
     —— 后端端点 require UploadFile via Form，对应 requests 的 files 参数。
        文件名推断从 file_uploader 返回的 BytesIO 对象获取。
"""

import requests
import json
from typing import Optional, List, Dict, Any


# ============================================================
# 配置常量
# ============================================================
# 为何硬编码而非从配置文件读取：
#   本地 Demo 场景，前端和后端在 localhost 上固定端口运行。
#   生产环境可通过环境变量覆盖，但 Demo 阶段的确定性比灵活性更重要。
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api"
TIMEOUT_SECONDS = 120  # LLM 调用可能耗时较长（Creation 可达 60s+）


# ============================================================
# 内部辅助函数
# ============================================================

def _build_url(path: str) -> str:
    """拼接完整 URL。不暴露给外部，所有 URL 逻辑集中在此。"""
    return f"{BASE_URL}{path}"


def _handle_response(response: requests.Response) -> Dict[str, Any]:
    """
    统一处理 HTTP 响应。

    设计考量：
      - 200/201 → 返回 JSON 数据（dict 或 list）
      - 4xx/5xx → 返回 {"error": True, "detail": "..."}
      - 连接错误 → 在调用函数处通过 except 捕获并转换

    为何不在此函数中捕获 ConnectionError：
      此函数只处理"已得到响应"的情况。连接的建立/失败在调用方捕获，
      因为不同 API 的 ConnectionError 可能有不同的用户提示。
    """
    if response.ok:
        return response.json()
    else:
        return {
            "error": True,
            "detail": f"HTTP {response.status_code}: {response.text[:500]}"
        }


# ============================================================
# 角色 API (4 个端点)
# ============================================================

def create_character_text(description: str) -> Dict[str, Any]:
    """
    通过文本描述创建角色。
    端点: POST /api/characters/create (Form data)

    Args:
        description: 用户输入的角色描述文本

    Returns:
        成功: CharacterResponse dict (含 id/name/personality 等)
        失败: {"error": True, "detail": "..."}

    为何用 Form data 而非 JSON body：
      后端端点 design 将 description 定义为 Form(...)，与 story_file
      的 UploadFile 共存于同一端点。使用 data= 而非 json= 匹配 Form 语义。
    """
    try:
        response = requests.post(
            _build_url(f"{API_PREFIX}/characters/create"),
            data={"description": description},
            timeout=TIMEOUT_SECONDS,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端，请确认 uvicorn 已启动在 :8000"}
    except requests.Timeout:
        return {"error": True, "detail": "请求超时，LLM 生成可能耗时过长，请重试"}


def create_character_file(file_bytes: bytes, filename: str, description: str = "") -> Dict[str, Any]:
    """
    通过 TXT 文件上传创建角色，可选追加用户寄语。
    端点: POST /api/characters/create (multipart/form-data)

    Args:
        file_bytes:   文件内容的字节数据
        filename:     原始文件名（用于推断 content-type）
        description:  创建寄语（可选），后端会拼接到 TXT 文件内容末尾

    Returns:
        成功: CharacterResponse dict
        失败: {"error": True, "detail": "..."}

    关于 data + files 同时发送：
      requests 允许 data dict 和 files dict 共存，data 中的字段以 Form 字段
      发送，files 中的字段以文件上传发送。FastAPI 端同时接收 Form 和 UploadFile。
    """
    try:
        data = {}
        if description:
            data["description"] = description
        response = requests.post(
            _build_url(f"{API_PREFIX}/characters/create"),
            data=data,
            files={"story_file": (filename, file_bytes, "text/plain")},
            timeout=TIMEOUT_SECONDS,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端，请确认 uvicorn 已启动在 :8000"}
    except requests.Timeout:
        return {"error": True, "detail": "请求超时，LLM 生成可能耗时过长，请重试"}


def get_characters(skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
    """
    获取角色列表。
    端点: GET /api/characters

    Returns:
        成功: List[CharacterResponse]
        失败: 空列表 []（并 st.error 显示错误）
    """
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/characters"),
            params={"skip": skip, "limit": limit},
            timeout=10,
        )
        result = _handle_response(response)
        if isinstance(result, list):
            return result
        return []
    except (requests.ConnectionError, requests.Timeout):
        return []


def get_character(character_id: int) -> Dict[str, Any]:
    """
    获取单个角色详情。
    端点: GET /api/characters/{character_id}

    Returns:
        成功: CharacterResponse dict（personality/current_state 为 JSON 字符串）
        失败: {"error": True, "detail": "..."}
    """
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/characters/{character_id}"),
            timeout=10,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "请求超时"}


def delete_character(character_id: int) -> Dict[str, Any]:
    """
    删除角色及其全部关联数据（级联删除）。
    端点: DELETE /api/characters/{character_id}

    Returns:
        成功: {"detail": "角色「xxx」及 N 条记忆、N 条对话、N 条成长记录已永久删除"}
        失败: {"error": True, "detail": "..."}
    """
    try:
        response = requests.delete(
            _build_url(f"{API_PREFIX}/characters/{character_id}"),
            timeout=10,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端，请确认 uvicorn 已启动"}
    except requests.Timeout:
        return {"error": True, "detail": "删除请求超时"}


# ============================================================
# 对话 API (1 个端点)
# ============================================================

def send_message(
    character_id: int,
    message: str,
    session_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    发送玩家消息，获取 NPC 回复（Director + Actor 双 LLM 管线）。
    端点: POST /api/chat (JSON body)

    Args:
        character_id: 目标角色 ID
        message:      玩家输入的消息文本
        session_id:   可选，会话 ID。
                       - None → 后端自动创建新会话（首条消息自动作为标题）
                       - 有效 ID → 复用该会话累积多轮消息

    Returns:
        成功: ChatResponse dict
              {id, character_id, user_input, npc_response,
               emotion, action, expression, director_raw, actor_raw, timestamp,
               session_id, session_title}   ← 后两个为新增字段
        失败: {"error": True, "detail": "..."}

    为何使用 JSON body 而非 Form：
      ChatRequest 是一个 Pydantic BaseModel（character_id + message + session_id），
      后端通过请求体解析。requests.post(json=...) 自动设置
      Content-Type: application/json。
    """
    payload: Dict[str, Any] = {
        "character_id": character_id, "message": message,
    }
    if session_id is not None:
        payload["session_id"] = session_id
    try:
        response = requests.post(
            _build_url(f"{API_PREFIX}/chat"),
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端，请确认 uvicorn 已启动"}
    except requests.Timeout:
        return {"error": True, "detail": "对话请求超时，请重试"}


# ============================================================
# 成长 API (1 个端点)
# ============================================================

def trigger_growth(character_id: int) -> Dict[str, Any]:
    """
    触发角色成长（Growth LLM 管线）。
    端点: POST /api/growth/trigger (JSON body)

    管线流程（后端执行）：
      1. 读取角色当前人格 + 最近 10 条对话
      2. Growth LLM 分析 → personality_delta / new_memories / event_summary
      3. 计算新人格 = 旧人格 + delta（钳位 [0, 100]）
      4. 持久化 growth_log + memories + 更新 character.personality

    Returns:
        成功: GrowthResponse dict
              {id, character_id, personality_delta, event_summary,
               new_memories, growth_raw, created_at}
        失败: {"error": True, "detail": "..."}

    注意：Growth 是异步触发接口，不存在降级策略。
          LLM 失败时直接返回错误，调用方可自行决定是否重试。
    """
    try:
        response = requests.post(
            _build_url(f"{API_PREFIX}/growth/trigger"),
            json={"character_id": character_id},
            timeout=TIMEOUT_SECONDS,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "成长分析请求超时，请重试"}


# ============================================================
# 数据查询 API (3 个端点)
# ============================================================

def get_memories(
    character_id: int,
    memory_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    获取角色记忆列表。
    端点: GET /api/characters/{character_id}/memories?memory_type=...

    Args:
        character_id: 角色 ID
        memory_type:  可选筛选（"conversation" / "event" / "growth" / None=all）
        skip/limit:   分页参数

    Returns:
        成功: List[MemoryResponse]
             每条: {id, character_id, content, importance, memory_type, created_at}
        失败: 空列表 []

    为何返回空列表而非抛异常：
      记忆查询是展示型操作，失败时不影响核心流程。
      空列表在 UI 上自然表现为"暂无数据"，不需要额外的错误态处理。
    """
    try:
        params = {"skip": skip, "limit": limit}
        if memory_type:
            params["memory_type"] = memory_type
        response = requests.get(
            _build_url(f"{API_PREFIX}/characters/{character_id}/memories"),
            params=params,
            timeout=10,
        )
        result = _handle_response(response)
        if isinstance(result, list):
            return result
        return []
    except (requests.ConnectionError, requests.Timeout):
        return []


def get_conversations(
    character_id: int,
    skip: int = 0,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    获取角色对话历史。
    端点: GET /api/characters/{character_id}/conversations

    Returns:
        成功: List[ChatResponse 风格 dict]
        失败: 空列表 []
    """
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/characters/{character_id}/conversations"),
            params={"skip": skip, "limit": limit},
            timeout=10,
        )
        result = _handle_response(response)
        if isinstance(result, list):
            return result
        return []
    except (requests.ConnectionError, requests.Timeout):
        return []


def get_growth_logs(
    character_id: int,
    skip: int = 0,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    获取角色成长记录。
    端点: GET /api/characters/{character_id}/growth-logs

    Returns:
        成功: List[GrowthResponse 风格 dict]
        失败: 空列表 []
    """
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/characters/{character_id}/growth-logs"),
            params={"skip": skip, "limit": limit},
            timeout=10,
        )
        result = _handle_response(response)
        if isinstance(result, list):
            return result
        return []
    except (requests.ConnectionError, requests.Timeout):
        return []


# ============================================================
# 健康检查
# ============================================================

def check_backend_health() -> bool:
    """
    检查后端是否可达。
    终点: GET /

    在 Streamlit 首次渲染时调用，确定是否展示"后端未启动"警告。
    """
    try:
        response = requests.get(
            _build_url("/"),
            timeout=3,
        )
        return response.ok
    except (requests.ConnectionError, requests.Timeout):
        return False


# ============================================================
# LLM 设置 API (4 个端点)
# ============================================================

def get_llm_settings() -> Dict[str, Any]:
    """
    获取当前 LLM 设置。
    端点: GET /api/settings/llm

    Returns:
        成功: {
          active_provider, active_provider_name, config, providers,
          default_temperature, default_max_tokens, settings_file_path
        }
        失败: {"error": True, "detail": "..."}
    """
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/settings/llm"),
            timeout=10,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端，请确认 uvicorn 已启动在 :8000"}
    except requests.Timeout:
        return {"error": True, "detail": "请求设置超时"}


def list_llm_providers() -> Dict[str, Any]:
    """
    列出所有支持的 LLM 厂商及其元信息。
    端点: GET /api/settings/llm/providers

    Returns:
        成功: {"providers": [{id, name, needs_key}], "defaults": {provider_id: {base_url, model}}}
        失败: {"error": True, "detail": "..."}
    """
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/settings/llm/providers"),
            timeout=10,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "请求超时"}


def update_llm_settings(
    active_provider: Optional[str] = None,
    active_config: Optional[Dict[str, Optional[str]]] = None,
    default_temperature: Optional[float] = None,
    default_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    更新 LLM 设置（部分更新友好）。
    端点: PUT /api/settings/llm

    Args:
        active_provider:       要切换到的 provider id（None=不切换）
        active_config:         当前激活 provider 的配置覆盖（None=不修改）
                               形如 {"api_key": "sk-...", "base_url": "...", "model": "..."}
        default_temperature:   新的默认温度
        default_max_tokens:    新的默认 max_tokens

    Returns:
        成功: 更新后的完整设置（同 get_llm_settings 的响应）
        失败: {"error": True, "detail": "..."}
    """
    payload: Dict[str, Any] = {}
    if active_provider is not None:
        payload["active_provider"] = active_provider
    if active_config is not None:
        # 过滤掉 None 字段，避免后端误以为是 "显式置空"
        cleaned = {k: v for k, v in active_config.items() if v is not None}
        if cleaned:
            payload["active_config"] = cleaned
    if default_temperature is not None:
        payload["default_temperature"] = default_temperature
    if default_max_tokens is not None:
        payload["default_max_tokens"] = default_max_tokens

    try:
        response = requests.put(
            _build_url(f"{API_PREFIX}/settings/llm"),
            json=payload,
            timeout=15,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "保存设置超时"}


def test_llm_connection(
    provider_id: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    test_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    测试 LLM 连接是否可用。
    端点: POST /api/settings/llm/test

    Args:
        provider_id: 可选，覆盖当前激活 provider
        api_key:     可选，本次测试用的 api_key（不写盘）
        base_url:    可选，本次测试用的 base_url
        model:       可选，本次测试用的 model
        test_prompt: 可选，测试用的提示词

    Returns:
        成功/失败字典: {
          success, message, provider_id, model,
          response_text (成功时), latency_ms
        }
    """
    payload: Dict[str, Any] = {}
    if provider_id is not None:
        payload["provider_id"] = provider_id
    if api_key is not None:
        payload["api_key"] = api_key
    if base_url is not None:
        payload["base_url"] = base_url
    if model is not None:
        payload["model"] = model
    if test_prompt is not None:
        payload["test_prompt"] = test_prompt

    try:
        response = requests.post(
            _build_url(f"{API_PREFIX}/settings/llm/test"),
            json=payload,
            timeout=30,  # 测试调用应比正常聊天略宽裕
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "测试请求超时（30s）"}


# ============================================================
# API 测试 API (3 个端点，移植自 web-tools/react-vite)
# ============================================================

def list_models(
    provider_id: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    拉取 provider 的 /v1/models 列表（含耗时）。
    端点: GET /api/test/models?provider_id=&base_url=&api_key=

    Args:
        provider_id: 可选，指定 provider；默认用当前激活
        base_url:    可选，一次性覆盖 base_url
        api_key:     可选，一次性覆盖 api_key

    Returns:
        成功: {
          provider_id, base_url, models: [{id, owned_by, object}],
          duration_ms, raw_count
        }
        失败: {"error": True, "detail": "..."}
    """
    params: Dict[str, Any] = {}
    if provider_id:
        params["provider_id"] = provider_id
    if base_url:
        params["base_url"] = base_url
    if api_key:
        params["api_key"] = api_key
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/test/models"),
            params=params,
            timeout=30,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "拉取 models 超时（30s）"}


def test_latency(
    provider_id: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    test_message: Optional[str] = "Hi",
    max_tokens: int = 16,
) -> Dict[str, Any]:
    """
    流式延迟测试（TTFT + 总延迟）。
    端点: POST /api/test/latency (JSON body)

    Returns:
        成功: {
          provider_id, model, status, ttft_ms, total_ms,
          content, chunks, error
        }
        失败: {"error": True, "detail": "..."}

    注意：test_message 与 max_tokens 仅在本测试中使用，不会写盘。
    """
    payload: Dict[str, Any] = {
        "test_message": test_message,
        "max_tokens": max_tokens,
    }
    if provider_id is not None:
        payload["provider_id"] = provider_id
    if api_key is not None:
        payload["api_key"] = api_key
    if base_url is not None:
        payload["base_url"] = base_url
    if model is not None:
        payload["model"] = model
    try:
        response = requests.post(
            _build_url(f"{API_PREFIX}/test/latency"),
            json=payload,
            timeout=90,  # 流式测试可能略长
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "延迟测试超时（90s）"}


def probe_llm(
    provider_id: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    test_message: Optional[str] = "Hi",
    max_tokens: int = 16,
) -> Dict[str, Any]:
    """
    原始请求探针：返回完整 request/response 头/体（密钥已脱敏）。
    端点: POST /api/test/probe (JSON body)
    """
    payload: Dict[str, Any] = {
        "test_message": test_message,
        "max_tokens": max_tokens,
    }
    if provider_id is not None:
        payload["provider_id"] = provider_id
    if api_key is not None:
        payload["api_key"] = api_key
    if base_url is not None:
        payload["base_url"] = base_url
    try:
        response = requests.post(
            _build_url(f"{API_PREFIX}/test/probe"),
            json=payload,
            timeout=30,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "探针请求超时（30s）"}


# ============================================================
# ChatSession API（5 个端点，移植自 NextChat 的会话管理）
# ============================================================

def list_sessions(
    character_id: int,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    列出某角色的全部会话（按 updated_at 倒序）。
    端点: GET /api/sessions?character_id=&search=&limit=&offset=

    Returns:
        成功: [{id, character_id, title, created_at, updated_at, message_count}, ...]
        失败: []
    """
    params: Dict[str, Any] = {
        "character_id": character_id, "limit": limit, "offset": offset,
    }
    if search:
        params["search"] = search
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/sessions"),
            params=params,
            timeout=10,
        )
        result = _handle_response(response)
        if isinstance(result, list):
            return result
        return []
    except (requests.ConnectionError, requests.Timeout):
        return []


def get_session_detail(session_id: int) -> Dict[str, Any]:
    """
    获取会话详情（含全部消息）。
    端点: GET /api/sessions/{session_id}

    Returns:
        成功: {id, character_id, title, ..., message_count, messages: [...]}
        失败: {"error": True, "detail": "..."}
    """
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/sessions/{session_id}"),
            timeout=10,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "加载会话超时"}


def create_session(
    character_id: int,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    主动创建空会话（前端"新对话"按钮调用）。
    端点: POST /api/sessions (JSON body)
    """
    payload: Dict[str, Any] = {"character_id": character_id}
    if title:
        payload["title"] = title
    try:
        response = requests.post(
            _build_url(f"{API_PREFIX}/sessions"),
            json=payload,
            timeout=10,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "创建会话超时"}


def rename_session(session_id: int, new_title: str) -> Dict[str, Any]:
    """
    重命名会话。
    端点: PATCH /api/sessions/{session_id} (JSON body)
    """
    try:
        response = requests.patch(
            _build_url(f"{API_PREFIX}/sessions/{session_id}"),
            json={"title": new_title},
            timeout=10,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "重命名超时"}


def delete_session(session_id: int) -> Dict[str, Any]:
    """
    删除会话（级联删除其下全部 conversation）。
    端点: DELETE /api/sessions/{session_id}
    """
    try:
        response = requests.delete(
            _build_url(f"{API_PREFIX}/sessions/{session_id}"),
            timeout=10,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "删除超时"}


# ============================================================
# Event / Time API（Day4 新增：事件推进系统）
# ============================================================

def advance_event(character_id: int) -> Dict[str, Any]:
    """
    推进下一个待处理事件。
    端点: POST /api/event/advance (JSON body)

    Args:
        character_id: 角色 ID

    Returns:
        成功: EventResponse dict
              {id, character_id, day_number, order_index, event_type,
               content, metadata_json, result_json, status, session_id,
               time_period, created_at}
        失败: {"error": True, "detail": "..."}
        无待办事件时返回 404（前端据此判断是否可触发 iterate）
    """
    try:
        response = requests.post(
            _build_url(f"{API_PREFIX}/event/advance"),
            json={"character_id": character_id},
            timeout=TIMEOUT_SECONDS,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端，请确认 uvicorn 已启动"}
    except requests.Timeout:
        return {"error": True, "detail": "事件推进请求超时，请重试"}


def iterate_day(character_id: int) -> Dict[str, Any]:
    """
    触发 Growth 迭代一天（控制台模式）。
    端点: POST /api/time/iterate (JSON body)
    """
    try:
        response = requests.post(
            _build_url(f"{API_PREFIX}/time/iterate"),
            json={"character_id": character_id},
            timeout=TIMEOUT_SECONDS,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "迭代请求超时，请重试"}


def auto_advance(character_id: int) -> Dict[str, Any]:
    """
    自动推进全部待处理事件 + 触发迭代（自动模式）。
    端点: POST /api/time/auto (JSON body)
    """
    try:
        response = requests.post(
            _build_url(f"{API_PREFIX}/time/auto"),
            json={"character_id": character_id},
            timeout=TIMEOUT_SECONDS * 3,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "自动推进请求超时，请重试"}


def get_events(character_id: int, day_number: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    获取角色的事件列表。
    端点: GET /api/characters/{character_id}/events?day_number=
    """
    params: Dict[str, Any] = {}
    if day_number is not None:
        params["day_number"] = day_number
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/characters/{character_id}/events"),
            params=params,
            timeout=10,
        )
        result = _handle_response(response)
        if isinstance(result, list):
            return result
        return []
    except (requests.ConnectionError, requests.Timeout):
        return []


# ============================================================
# World / Scene API（v1.6 Phase 1 新增：世界系统）
# ============================================================

def get_world(world_id: int) -> Dict[str, Any]:
    """
    获取世界详情。
    端点: GET /api/worlds/{world_id}
    """
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/worlds/{world_id}"),
            timeout=10,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "请求超时"}


def get_character_world(character_id: int) -> Dict[str, Any]:
    """
    获取角色所属世界。
    端点: GET /api/characters/{character_id}/world
    """
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/characters/{character_id}/world"),
            timeout=10,
        )
        return _handle_response(response)
    except requests.ConnectionError:
        return {"error": True, "detail": "无法连接后端"}
    except requests.Timeout:
        return {"error": True, "detail": "请求超时"}


def get_world_scenes(world_id: int, scene_layer: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    获取世界下的场景列表。
    端点: GET /api/worlds/{world_id}/scenes?scene_layer=
    """
    params: Dict[str, Any] = {}
    if scene_layer:
        params["scene_layer"] = scene_layer
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/worlds/{world_id}/scenes"),
            params=params,
            timeout=10,
        )
        result = _handle_response(response)
        if isinstance(result, list):
            return result
        return []
    except (requests.ConnectionError, requests.Timeout):
        return []


def get_character_scenes(character_id: int) -> List[Dict[str, Any]]:
    """
    获取角色所属世界的全部场景。
    端点: GET /api/characters/{character_id}/scenes
    """
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/characters/{character_id}/scenes"),
            timeout=10,
        )
        result = _handle_response(response)
        if isinstance(result, list):
            return result
        return []
    except (requests.ConnectionError, requests.Timeout):
        return []


def get_scene_path(scene_id: int) -> List[Dict[str, Any]]:
    """
    获取场景完整路径（面包屑导航）。
    端点: GET /api/scenes/{scene_id}/path
    """
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/scenes/{scene_id}/path"),
            timeout=10,
        )
        result = _handle_response(response)
        if isinstance(result, list):
            return result
        return []
    except (requests.ConnectionError, requests.Timeout):
        return []


def get_scene_changes(scene_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """
    获取场景变化历史。
    端点: GET /api/scenes/{scene_id}/changes?limit=
    """
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/scenes/{scene_id}/changes"),
            params={"limit": limit},
            timeout=10,
        )
        result = _handle_response(response)
        if isinstance(result, list):
            return result
        return []
    except (requests.ConnectionError, requests.Timeout):
        return []


def get_character_world_changes(character_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """
    获取角色世界的场景变化时间轴。
    端点: GET /api/characters/{character_id}/world-changes?limit=
    """
    try:
        response = requests.get(
            _build_url(f"{API_PREFIX}/characters/{character_id}/world-changes"),
            params={"limit": limit},
            timeout=10,
        )
        result = _handle_response(response)
        if isinstance(result, list):
            return result
        return []
    except (requests.ConnectionError, requests.Timeout):
        return []
