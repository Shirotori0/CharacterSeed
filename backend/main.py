from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional, List
import logging
import time
import os

# 关键：在所有其他导入前加载 .env
# 这样 os.environ.get("AGNES_API_KEY") 就能拿到 .env 中的值
# 作为 LLM settings store 的兜底
try:
    from dotenv import load_dotenv
    load_dotenv()  # 默认加载当前目录 .env
    print("[startup] 已加载 .env 文件")
except ImportError:
    # python-dotenv 未安装时静默降级
    pass
except Exception as e:
    print(f"[startup] 加载 .env 失败: {e}")

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional, List

from backend.database import engine, get_db, Base
from backend.models import Character, Conversation, Memory, GrowthLog, ChatSession
from backend.schemas import (
    CharacterCreate, CharacterResponse,
    ChatRequest, ChatResponse,
    MemoryResponse,
    GrowthTriggerRequest, GrowthResponse,
    LLMSettingsResponse, LLMUpdateRequest, LLMTestRequest, LLMTestResponse,
    ProviderConfigMasked,
    ModelsListResponse, LatencyTestRequest, LatencyTestResponse,
    ProbeRequest, ProbeResponse,
    ChatSessionCreate, ChatSessionUpdate, ChatSessionInfo, ChatSessionWithMessages,
    ConversationRow,
)
from backend.crud import character as character_crud
from backend.crud import memory as memory_crud
from backend.crud import conversation as conversation_crud
from backend.crud import growth as growth_crud
from backend.services import chat_session_crud
from backend.services import db_migration
from backend.modules.creation import CreationModule
from backend.modules.interaction import InteractionPipeline
from backend.modules.growth import GrowthModule
from backend.services.llm_settings_store import (
    LLMSettingsStore, PROVIDER_META, PROVIDER_DEFAULTS
)
from backend.services import llm_api_tester

logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="CharacterSeed API",
    description="AI NPC生命模拟系统",
    version="0.1.0"
)

# ==================== 启动事件 ====================
@app.on_event("startup")
def startup_event():
    """应用启动时执行"""
    # 1) 确保所有表存在
    Base.metadata.create_all(bind=engine)
    # 2) 执行 schema 迁移（幂等，可重复运行）
    try:
        history = db_migration.run_all_migrations(engine)
        for h in history:
            if h.get("backfilled", 0) > 0 or h.get("added_column"):
                logger.info("[migration] %s", h)
    except Exception as e:
        logger.exception("迁移失败: %s", e)
        # 不阻塞启动，但打印错误便于排查
    print("=" * 50)
    print("CharacterSeed API 启动成功！")
    print("访问 http://localhost:8000/docs 查看API文档")
    print("=" * 50)

# ==================== Character Endpoints ====================

@app.post("/api/characters/create", response_model=CharacterResponse)
async def create_character(
    description: Optional[str] = Form(None),
    story_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """
    创建角色（支持一句话描述或TXT文件上传）
    
    - description: 一句话描述（可选）
    - story_file: TXT故事文件（可选，与description二选一）
    """
    # 确定输入类型和内容
    if story_file:
        # 读取文件内容
        content = await story_file.read()
        user_input = content.decode("utf-8")
        # 如果同时提供了 description（用户寄语），拼接到文件内容末尾
        if description:
            user_input = user_input + "\n\n[额外的角色期望]\n" + description
        input_type = "file"
    elif description:
        user_input = description
        input_type = "text"
    else:
        raise HTTPException(status_code=400, detail="必须提供description或story_file")
    
    # 调用Creation Module
    try:
        creation_module = CreationModule()
        parsed_data, raw_response = creation_module.run(user_input, input_type)
        
        # 提取数据（personality/current_state 以 dict 形式传入 CRUD，
        # 由 CRUD 层统一完成 JSON 序列化，消除调用方重复的 json.dumps）
        name = parsed_data.get("name", "未命名角色")
        world_setting = parsed_data.get("world_setting")
        personality = parsed_data.get("personality", {})
        current_state = parsed_data.get("current_state", {})
        initial_memories = parsed_data.get("initial_memories", [])
        
        # 保存到数据库（CRUD 层自动将 dict 序列化为 JSON 字符串）
        db_character = character_crud.create_character(
            db=db,
            name=name,
            description=user_input[:500],  # 限制长度
            world_setting=world_setting,
            personality=personality,
            current_state=current_state,
            creation_raw=raw_response
        )
        
        # Day 3：保存初始记忆到 memories 表
        # 初始记忆来自 Creation LLM 的输出，每条含 content + importance 字段
        for mem in initial_memories:
            if isinstance(mem, dict):
                memory_crud.create_memory(
                    db=db,
                    character_id=db_character.id,
                    content=mem.get("content", ""),
                    importance=mem.get("importance", 5),
                    memory_type="event"  # 初始记忆标记为 event 类型
                )
        
        return db_character
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"角色创建失败: {str(e)}")

@app.get("/api/characters", response_model=List[CharacterResponse])
def get_characters(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """获取角色列表"""
    characters = character_crud.get_characters(db, skip=skip, limit=limit)
    return characters

@app.get("/api/characters/{character_id}", response_model=CharacterResponse)
def get_character(character_id: int, db: Session = Depends(get_db)):
    """获取单个角色详情"""
    character = character_crud.get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    return character


@app.delete("/api/characters/{character_id}")
def delete_character(character_id: int, db: Session = Depends(get_db)):
    """
    删除角色及其所有关联数据（级联删除）。
    
    级联清理顺序：memories → conversations → growth_logs → characters，
    确保数据库无孤儿记录残留。
    """
    result = character_crud.cascade_delete_character(db, character_id)
    if not result["deleted"]:
        raise HTTPException(status_code=404, detail="角色不存在")
    
    return {
        "detail": (
            f"角色「{result['name']}」及 {result['memories_deleted']} 条记忆、"
            f"{result['conversations_deleted']} 条对话、"
            f"{result['growth_logs_deleted']} 条成长记录已永久删除"
        )
    }

# ==================== Chat Endpoints（Day 2实现） ====================

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    与角色对话（Day 2 实现：Director + Actor 双 LLM 管线）

    管线流程:
      1. 从数据库获取角色、最近记忆
      2. Director.analyze()  → emotion / focus_memories / goal / style
      3. Actor.generate()    → action / expression / speech
      4. 持久化对话记录到 conversations 表
      5. 返回 ChatResponse

    新增（NextChat 会话管理）:
      - request.session_id 缺省/None：自动创建新 session，标题 = user_message 前 30 字
      - request.session_id 有效：复用该 session 累积多轮消息
      - 响应额外返回 session_id / session_title，前端可立即更新侧栏
    """
    try:
        pipeline = InteractionPipeline()
        result = pipeline.run(
            character_id=request.character_id,
            user_message=request.message,
            db=db,
            session_id=request.session_id,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对话处理失败: {str(e)}")


# ==================== ChatSession Endpoints（NextChat 会话管理） ====================

def _serialize_session_row(row: dict) -> dict:
    """ORM 出来的 ChatSession 含 datetime，统一转 iso 字符串方便前端"""
    out = dict(row)
    for k in ("created_at", "updated_at"):
        v = out.get(k)
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
    return out


@app.get("/api/sessions", response_model=List[ChatSessionInfo])
def list_sessions(
    character_id: int,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """
    列出某角色的所有 session，支持按 title 模糊搜索。

    性能：用单条 SQL 带 LEFT JOIN 计算 message_count，避免 N+1。
    """
    # 校验角色存在
    char = character_crud.get_character(db, character_id)
    if not char:
        raise HTTPException(status_code=404, detail=f"角色不存在: id={character_id}")
    rows = chat_session_crud.list_sessions_with_message_count(
        db, character_id, search=search, limit=limit, offset=offset,
    )
    return [_serialize_session_row(r) for r in rows]


@app.post("/api/sessions", response_model=ChatSessionInfo)
def create_session(request: ChatSessionCreate, db: Session = Depends(get_db)):
    """
    主动创建新会话（不立刻发消息时使用，比如想预先起个标题）。
    """
    char = character_crud.get_character(db, request.character_id)
    if not char:
        raise HTTPException(status_code=404, detail=f"角色不存在: id={request.character_id}")
    sess = chat_session_crud.create_session(
        db, request.character_id, title=request.title,
    )
    return _serialize_session_row({
        "id": sess.id,
        "character_id": sess.character_id,
        "title": sess.title,
        "created_at": sess.created_at,
        "updated_at": sess.updated_at,
        "message_count": 0,
    })


@app.get("/api/sessions/{session_id}", response_model=ChatSessionWithMessages)
def get_session_detail(
    session_id: int,
    db: Session = Depends(get_db),
):
    """
    获取会话详情 + 全部消息（按时间升序）。
    """
    sess = chat_session_crud.get_session(db, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"会话不存在: id={session_id}")
    conversations = conversation_crud.get_session_conversations(db, session_id, limit=200)
    # 转成 dict + 序列化 datetime → iso string，兼容 Pydantic
    messages = []
    for c in conversations:
        row = {
            "id": c.id,
            "session_id": c.session_id,
            "character_id": c.character_id,
            "user_input": c.user_input,
            "npc_response": c.npc_response,
            "emotion": c.emotion,
            "action": c.action,
            "expression": c.expression,
            "director_raw": c.director_raw,
            "actor_raw": c.actor_raw,
            "timestamp": c.timestamp.isoformat() if c.timestamp else None,
        }
        # 用 model_validate 仍走一次 Pydantic 校验，保证响应体字段齐全
        messages.append(ConversationRow.model_validate(row).model_dump(mode="json"))
    info = {
        "id": sess.id,
        "character_id": sess.character_id,
        "title": sess.title,
        "created_at": sess.created_at,
        "updated_at": sess.updated_at,
        "message_count": len(messages),
    }
    out = _serialize_session_row(info)
    out["messages"] = messages
    return out


@app.patch("/api/sessions/{session_id}", response_model=ChatSessionInfo)
def update_session(
    session_id: int,
    request: ChatSessionUpdate,
    db: Session = Depends(get_db),
):
    """重命名会话"""
    sess = chat_session_crud.rename_session(db, session_id, request.title)
    if not sess:
        raise HTTPException(status_code=404, detail=f"会话不存在: id={session_id}")
    msg_count = len(conversation_crud.get_session_conversations(
        db, session_id, limit=1000,
    ))
    return _serialize_session_row({
        "id": sess.id,
        "character_id": sess.character_id,
        "title": sess.title,
        "created_at": sess.created_at,
        "updated_at": sess.updated_at,
        "message_count": msg_count,
    })


@app.delete("/api/sessions/{session_id}")
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
):
    """
    删除会话（级联删除其下全部 conversation，依赖外键 ON DELETE CASCADE）。
    """
    ok = chat_session_crud.delete_session(db, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"会话不存在: id={session_id}")
    return {"deleted": True, "session_id": session_id}


# ==================== Growth Endpoints（Day 3实现） ====================

@app.post("/api/growth/trigger", response_model=GrowthResponse)
def trigger_growth(request: GrowthTriggerRequest, db: Session = Depends(get_db)):
    """
    触发角色成长（Day 3 实现：Growth LLM 管线）

    管线流程:
      1. 从数据库获取角色、昨日最近对话
      2. GrowthModule.run() → personality_delta / new_memories / event_summary
      3. 计算新人格 = 旧人格 + delta
      4. 持久化 growth_log + memories + 更新 character.personality
      5. 返回 GrowthResponse

    注意：Growth 是异步触发接口，不设降级策略——LLM 失败时直接抛异常，
          调用方可自行决定何时重试。
    """
    try:
        growth_module = GrowthModule()
        result = growth_module.run(
            character_id=request.character_id,
            db=db,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"成长处理失败: {str(e)}")

# ==================== Memory Endpoints（Day 3实现） ====================

@app.get("/api/characters/{character_id}/memories", response_model=List[MemoryResponse])
def get_character_memories(
    character_id: int,
    memory_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取角色记忆列表"""
    memories = memory_crud.get_character_memories(
        db, character_id, memory_type=memory_type, skip=skip, limit=limit
    )
    return memories


@app.get("/api/characters/{character_id}/conversations")
def get_character_conversations(
    character_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取角色对话历史"""
    conversations = conversation_crud.get_character_conversations(
        db, character_id, skip=skip, limit=limit
    )
    return conversations


@app.get("/api/characters/{character_id}/growth-logs")
def get_character_growth_logs(
    character_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取角色成长记录列表"""
    logs = growth_crud.get_character_growth_logs(
        db, character_id, skip=skip, limit=limit
    )
    return logs


# ==================== LLM Settings Endpoints ====================

def _build_settings_response(store: LLMSettingsStore) -> LLMSettingsResponse:
    """组装 LLM 设置的对外响应（api_key 全部脱敏）"""
    all_data = store.get_all()
    active = all_data["active_provider"]
    providers_masked: dict = {}
    for pid, cfg in all_data["providers"].items():
        providers_masked[pid] = ProviderConfigMasked(
            api_key=LLMSettingsStore.mask_api_key(cfg.get("api_key", "")),
            base_url=cfg.get("base_url", ""),
            model=cfg.get("model", ""),
        )
    active_meta = next(
        (m for m in PROVIDER_META if m["id"] == active), {"name": active}
    )
    return LLMSettingsResponse(
        active_provider=active,
        active_provider_name=active_meta["name"],
        config=providers_masked[active],
        default_temperature=float(all_data["default_temperature"]),
        default_max_tokens=int(all_data["default_max_tokens"]),
        providers=providers_masked,
        settings_file_path=LLMSettingsStore.settings_file_path(),
    )


@app.get("/api/settings/llm", response_model=LLMSettingsResponse)
def get_llm_settings():
    """获取当前 LLM 设置（含所有 provider 的脱敏配置）"""
    return _build_settings_response(LLMSettingsStore())


@app.get("/api/settings/llm/providers")
def list_llm_providers():
    """列出所有支持的 provider（含展示用元信息）"""
    return {
        "providers": LLMSettingsStore.list_providers_meta(),
        "defaults": PROVIDER_DEFAULTS,
    }


@app.put("/api/settings/llm", response_model=LLMSettingsResponse)
def update_llm_settings(request: LLMUpdateRequest):
    """
    更新 LLM 设置。

    支持的更新动作（任意组合）：
      - 切换激活 provider:        request.active_provider
      - 修改当前激活 provider 配置: request.active_config
      - 修改默认温度:             request.default_temperature
      - 修改默认 max_tokens:      request.default_max_tokens

    设计考量：保持接口幂等 + 部分更新友好。
    前端只需把表单中"用户实际改过的字段"带上即可。
    """
    store = LLMSettingsStore()

    # 1. 先切换激活 provider（如果指定了）
    if request.active_provider:
        if request.active_provider not in PROVIDER_DEFAULTS:
            raise HTTPException(
                status_code=400,
                detail=f"未知 provider: {request.active_provider}",
            )
        store.set_active_provider(request.active_provider)

    # 2. 修改当前激活 provider 的配置（如果指定了）
    if request.active_config:
        # 注意：用最新激活的 provider id（可能被上一步切换过）
        target = store.get_active_provider_id()
        cfg = request.active_config
        # 兼容 pydantic 可能将 None 字段丢弃的情况：用 exclude_unset 才是用户真实意图
        # 但 BaseModel 默认是字段为 None 即保留 None；这里我们用默认值填充
        try:
            store.update_provider(
                target,
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                model=cfg.model,
            )
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 3. 修改默认参数
    if request.default_temperature is not None or request.default_max_tokens is not None:
        store.update_default_params(
            temperature=request.default_temperature,
            max_tokens=request.default_max_tokens,
        )

    logger.info("LLM 设置已更新: active=%s", store.get_active_provider_id())
    return _build_settings_response(store)


@app.post("/api/settings/llm/test", response_model=LLMTestResponse)
def test_llm_connection(request: LLMTestRequest):
    """
    测试 LLM 连接是否可用。

    支持两种用法：
      1. 不传任何 provider 字段 → 用当前激活 provider 的设置做测试
      2. 传 provider 字段（部分或全部）→ 用传入的字段覆盖当前设置后测试
         （覆盖仅在本次测试中生效，**不写盘**；如需保存请用 PUT 接口）

    设计动机：用户改了 API Key 后想"先试一下能不能用"，再点保存。
    """
    from openai import OpenAI

    store = LLMSettingsStore()

    # 1. 决定用哪份配置做测试
    pid = request.provider_id or store.get_active_provider_id()
    if pid not in PROVIDER_DEFAULTS:
        raise HTTPException(status_code=400, detail=f"未知 provider: {pid}")

    if request.provider_id or request.api_key or request.base_url or request.model:
        # 用户传了覆盖字段 → 临时构造（不写盘）
        base_cfg = store.get_provider_with_env_fallback(pid)
        api_key = request.api_key if request.api_key is not None else base_cfg["api_key"]
        base_url = request.base_url if request.base_url is not None else base_cfg["base_url"]
        model = request.model if request.model is not None else base_cfg["model"]
    else:
        cfg = store.get_provider_with_env_fallback(pid)
        api_key, base_url, model = cfg["api_key"], cfg["base_url"], cfg["model"]

    # 2. Ollama 不需要 api_key
    if not api_key and pid != "ollama":
        return LLMTestResponse(
            success=False,
            message=f"API Key 为空，请先在设置中填写 {pid} 的 API Key",
            provider_id=pid,
            model=model,
        )

    # 3. 真正发一次请求
    test_prompt = request.test_prompt or "你好"
    t0 = time.time()
    try:
        client = OpenAI(api_key=api_key or "ollama", base_url=base_url, timeout=20)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=80,
        )
        latency_ms = int((time.time() - t0) * 1000)
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return LLMTestResponse(
                success=False,
                message="LLM 返回了空内容（可能模型名错误或权限不足）",
                provider_id=pid,
                model=model,
                latency_ms=latency_ms,
            )
        return LLMTestResponse(
            success=True,
            message=f"连接成功（{latency_ms}ms）",
            provider_id=pid,
            model=model,
            response_text=text[:200],
            latency_ms=latency_ms,
        )
    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        logger.warning("LLM 连接测试失败: provider=%s, err=%s", pid, str(e)[:300])
        return LLMTestResponse(
            success=False,
            message=f"连接失败: {str(e)[:200]}",
            provider_id=pid,
            model=model,
            latency_ms=latency_ms,
        )


# ==================== API Test Endpoints ====================
# 参考 https://github.com/joker1point/web-tools 的 API 联通测试 Dashboard
# 三大能力：拉取 /v1/models、流式延迟测试（TTFT）、原始请求探针

@app.get("/api/test/models", response_model=ModelsListResponse)
def list_models(
    provider_id: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    拉取 provider 的 /v1/models 列表（含耗时）。

    复用 LLMSettingsStore；query 参数用于一次性覆盖（不写盘）。
    """
    try:
        result = llm_api_tester.fetch_models(
            provider_id=provider_id,
            override_api_key=api_key,
            override_base_url=base_url,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception("拉取 models 失败")
        raise HTTPException(status_code=500, detail=f"拉取失败: {str(e)[:200]}")


@app.post("/api/test/latency", response_model=LatencyTestResponse)
def test_latency(request: LatencyTestRequest):
    """
    流式延迟测试：发送 stream=true 请求，测量 TTFT + 总延迟 + 响应内容。

    设计要点（与 web-tools 的 testLatency 一致）：
      - 增量解析 SSE，遇到 finish_reason / message_stop 即结束
      - 失败时也返回 total_duration_ms，便于排错
    """
    try:
        return llm_api_tester.test_stream_latency(
            provider_id=request.provider_id,
            override_api_key=request.api_key,
            override_base_url=request.base_url,
            override_model=request.model,
            test_message=request.test_message,
            max_tokens=request.max_tokens or 16,
        )
    except Exception as e:
        logger.exception("延迟测试异常")
        return {
            "provider_id": request.provider_id or "",
            "model": request.model or "",
            "status": 0,
            "ttft_ms": None,
            "total_ms": None,
            "content": "",
            "chunks": 0,
            "error": f"测试异常: {str(e)[:200]}",
        }


@app.post("/api/test/probe", response_model=ProbeResponse)
def probe_llm(request: ProbeRequest):
    """
    原始请求探针：发送一次非流式请求，返回完整 request/response 头/体。
    用于排查 provider 鉴权/路由/协议差异。请求头中密钥字段已脱敏。
    """
    try:
        return llm_api_tester.probe_request(
            provider_id=request.provider_id,
            override_api_key=request.api_key,
            override_base_url=request.base_url,
            test_message=request.test_message,
            max_tokens=request.max_tokens or 16,
        )
    except Exception as e:
        logger.exception("探针异常")
        return {
            "provider_id": request.provider_id or "",
            "model": "",
            "base_url": "",
            "request": {},
            "response": {},
            "error": f"探针异常: {str(e)[:200]}",
        }


# ==================== 根路径 ====================

@app.get("/")
def root():
    return {
        "message": "CharacterSeed API is running!",
        "docs": "http://localhost:8000/docs",
        "version": "0.1.0"
    }
