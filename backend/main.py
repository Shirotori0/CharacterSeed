from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional, List
import json
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
from backend.models import Character, Conversation, Memory, GrowthLog, ChatSession, Event
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
    # Day4 新增：事件推进系统
    AdvanceRequest, EventResponse,
    IterateRequest, IterateResponse, AutoResponse,
    # v1.6 Phase 1：世界系统
    WorldResponse, SceneResponse, SceneChangeResponse,
    WorldUpdateRequest,
)
from backend.crud import character as character_crud
from backend.crud import memory as memory_crud
from backend.crud import conversation as conversation_crud
from backend.crud import growth as growth_crud
from backend.crud import event as event_crud  # Day4 新增
from backend.crud import world as world_crud      # v1.6 Phase 1
from backend.crud import scene as scene_crud      # v1.6 Phase 1
from backend.crud import scene_change as scene_change_crud  # v1.6 Phase 1
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

        # Day4 新增：说话风格/信念/习惯/长期目标
        speaking_style = parsed_data.get("speaking_style", [])
        values = parsed_data.get("values", [])
        habits = parsed_data.get("habits", [])
        long_term_goal = parsed_data.get("long_term_goal", "")

        # ---- v1.6 Phase 1：创建 World + Scene 并关联角色 ----
        world_name = parsed_data.get("world_name", f"{name}的世界")
        core_worldview = parsed_data.get("core_worldview", world_setting[:100] if world_setting else "")
        scenes_data = parsed_data.get("scenes", [])

        # 1) 创建 World 记录
        db_world = world_crud.create_world(
            db=db,
            name=world_name,
            core_worldview=core_worldview,
        )
        world_id = db_world.id

        # 2) 逐条创建 Scene 记录（先概念后实际）
        index_to_db_id: dict = {}  # scenes 数组索引 → 数据库 scene ID
        first_actual_scene_id: Optional[int] = None

        for idx, scene_item in enumerate(scenes_data):
            # 解析 parent_scene_id：通过 parent_index 映射到已创建的场景 ID
            parent_scene_id: Optional[int] = None
            parent_index = scene_item.get("parent_index", -1)
            if parent_index >= 0 and parent_index in index_to_db_id:
                parent_scene_id = index_to_db_id[parent_index]

            try:
                db_scene = scene_crud.create_scene(
                    db=db,
                    world_id=world_id,
                    name=scene_item["name"],
                    scene_layer=scene_item["scene_layer"],
                    scene_type=scene_item.get("scene_type"),
                    parent_scene_id=parent_scene_id,
                    description=scene_item.get("description"),
                    created_day=1,
                )
                index_to_db_id[idx] = db_scene.id
                # 记录第一个 actual 场景作为角色的初始位置
                if scene_item["scene_layer"] == "actual" and first_actual_scene_id is None:
                    first_actual_scene_id = db_scene.id
            except Exception as scene_err:
                logger.warning("场景创建失败 (idx=%d, name=%s): %s",
                               idx, scene_item.get("name", "?"), scene_err)

        # 3) 如果 LLM 没有生成 actual 场景，创建一个保底场景
        if first_actual_scene_id is None:
            location_name = current_state.get("location", "未知地点") if isinstance(current_state, dict) else "初始地点"
            fallback_scene = scene_crud.create_scene(
                db=db,
                world_id=world_id,
                name=location_name,
                scene_layer="actual",
                scene_type="location",
                parent_scene_id=list(index_to_db_id.values())[0] if index_to_db_id else None,
                description=f"角色{name}所在之处",
                created_day=1,
            )
            first_actual_scene_id = fallback_scene.id

        # ---- 保存角色到数据库 ----
        db_character = character_crud.create_character(
            db=db,
            name=name,
            description=user_input[:500],  # 限制长度
            world_setting=world_setting,
            personality=personality,
            current_state=current_state,
            creation_raw=raw_response,
            # Day4 新增：持久化人格扩充字段
            speaking_style=json.dumps(speaking_style, ensure_ascii=False) if isinstance(speaking_style, list) else speaking_style,
            values=json.dumps(values, ensure_ascii=False) if isinstance(values, list) else values,
            habits=json.dumps(habits, ensure_ascii=False) if isinstance(habits, list) else habits,
            long_term_goal=long_term_goal,
        )

        # ---- 关联 World + Scene 到角色 ----
        character_crud.update_character(
            db=db,
            character_id=db_character.id,
            world_id=world_id,
            current_scene_id=first_actual_scene_id,
        )
        # 刷新后重新读取（因为 update_character 在另一个事务中）
        db.refresh(db_character)

        # Day 3：保存初始记忆到 memories 表
        for mem in initial_memories:
            if isinstance(mem, dict):
                memory_crud.create_memory(
                    db=db,
                    character_id=db_character.id,
                    content=mem.get("content", ""),
                    importance=mem.get("importance", 5),
                    memory_type="event"  # 初始记忆标记为 event 类型
                )

        # Day 5：将 Creation LLM 输出的 day1_schedule 持久化为 events 表记录
        day1_schedule = parsed_data.get("day1_schedule", [])
        for item in day1_schedule:
            if isinstance(item, dict) and item.get("content", "").strip():
                event_crud.create_event(
                    db=db,
                    character_id=db_character.id,
                    day_number=1,
                    order_index=item.get("order_index", 1),
                    event_type=item.get("event_type", "schedule_action"),
                    content=item["content"].strip(),
                    status="pending",
                    time_period=item.get("time_period"),
                )

        # Step 13 新增（v1.6 Phase 3）：持久化 short_term_goals
        # 短期目标是 long_term_goal 与日程事件之间的桥梁，
        # 以 JSON 数组格式存储在 Character 表中。
        short_term_goals = parsed_data.get("short_term_goals", [])
        if short_term_goals:
            character_crud.update_character(
                db=db,
                character_id=db_character.id,
                short_term_goals=json.dumps(short_term_goals, ensure_ascii=False),
            )
            db.refresh(db_character)

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

# ==================== Event Advance Endpoints（Day4 新增） ====================

def _package_session_dialogue_for_event(db: Session, character_id: int) -> Optional[int]:
    """
    在 advance_event 前，检查当前是否有 session 的未打包对话。
    若有，将该 session 的本日全部 Conversation 打包为一个 player_dialogue 事件。

    设计考量（为什么在 advance 前做这件事）：
      用户可能已经聊了几轮但从未触发过 event advance。
      如果直接跳到"推进下一个事件"，这些对话就会从 Growth 的"观察材料"中消失。
      因此本函数作为 advance 的前置步骤，确保对话总是被观测到。

    打包逻辑：
      - 找到该角色最新的活跃 session（最近 updated）
      - 检查该 session 的对话是否已被打包（通过查询已有的 player_dialogue 事件）
      - 若未打包 → 创建新的 Event(event_type=player_dialogue, status=completed)
      - content = LLM 生成的对话摘要（调用 LLM 或简单拼接摘要）
      - metadata_json = 完整对话记录 JSON
      - result_json = 对话摘要（前端展示用）

    Returns:
        创建的 Event ID，无对话需打包时返回 None
    """
    from backend.services import chat_session_crud
    from backend.crud import conversation as conversation_crud

    # 找最新的活跃 session
    sessions = chat_session_crud.list_sessions_with_message_count(
        db, character_id, limit=1,
    )
    if not sessions:
        return None

    latest_session = sessions[0]
    sid = latest_session["id"]

    # 检查这个 session 是否已有打包事件（避免重复打包）
    existing = (
        db.query(Event)
        .filter(
            Event.character_id == character_id,
            Event.session_id == sid,
            Event.event_type == "player_dialogue",
        )
        .first()
    )
    if existing:
        return None  # 已打包过，无需重复

    # 读取该 session 的对话
    conversations = conversation_crud.get_session_conversations(
        db, sid, limit=200,
    )
    if not conversations:
        return None

    # 组装对话文本用于摘要
    dialogue_lines = []
    for conv in conversations:
        player = (conv.user_input or "").strip()
        npc = (conv.npc_response or "").strip()
        if player:
            dialogue_lines.append(f"[玩家]: {player}")
        if npc:
            dialogue_lines.append(f"[NPC]: {npc}")

    full_dialogue = "\n".join(dialogue_lines)
    summary = full_dialogue[:200] if len(full_dialogue) > 200 else full_dialogue
    result_summary = f"与角色进行了 {len(conversations)} 轮对话"

    # 获取角色当前 day_number
    character = character_crud.get_character(db, character_id)
    current_day = character.day_number if character else 1

    ev = event_crud.create_event(
        db=db,
        character_id=character_id,
        day_number=current_day,
        order_index=0,  # 对话事件 order_index=0（排在日程事件之前）
        event_type="player_dialogue",
        content=summary,
        metadata_json=json.dumps(
            {"dialogue": full_dialogue, "total_turns": len(conversations)},
            ensure_ascii=False,
        ),
        status="completed",
        session_id=sid,
        time_period=None,
    )
    # 对话事件打包后直接标记为 completed（因为对话已发生）
    event_crud.complete_event(db, ev.id, result_summary)
    logger.info("对话事件已打包: event_id=%d, session_id=%d, turns=%d",
                ev.id, sid, len(conversations))
    return ev.id


def _write_scene_changes_from_growth(
    db: Session,
    character,
    growth_log_id: int,
    world_changes_json: Optional[str],
    day_number: int,
) -> int:
    """
    v1.6 Phase 1：将 Growth 产出的 world_changes 写入 scene_changes 表。

    设计考量：
      - 如果角色已关联世界和当前场景，则在该场景下创建变化记录
      - 如果 world_changes 为空或无效，跳过（不创建空记录）
      - change_type 默认为 "external"（Growth 描述的是外部世界变化）

    Args:
        db: 数据库会话
        character: Character ORM 对象
        growth_log_id: 关联的 GrowthLog ID
        world_changes_json: Growth 输出的世界变化 JSON
        day_number: 当前天数

    Returns:
        创建的 SceneChange 记录数
    """
    if not world_changes_json:
        return 0

    try:
        changes_data = json.loads(world_changes_json) if isinstance(world_changes_json, str) else world_changes_json
    except (json.JSONDecodeError, TypeError):
        return 0

    # 提取变化描述
    if isinstance(changes_data, dict):
        description = changes_data.get("description", "")
        change_type = changes_data.get("change_type", "external")
        details = changes_data
    elif isinstance(changes_data, str):
        description = changes_data.strip()
        change_type = "external"
        details = None
    else:
        return 0

    if not description:
        return 0

    # 获取角色当前场景
    scene_id = character.current_scene_id if hasattr(character, "current_scene_id") and character.current_scene_id else None
    if not scene_id:
        # 角色未关联场景，跳过
        return 0

    # 验证场景存在
    scene = scene_crud.get_scene(db, scene_id)
    if not scene:
        return 0

    # 创建 SceneChange 记录
    try:
        scene_change_crud.create_scene_change(
            db=db,
            scene_id=scene_id,
            change_type=change_type if change_type in ("character_driven", "external") else "external",
            description=description,
            day_number=day_number,
            growth_log_id=growth_log_id,
            change_details_json=json.dumps(details, ensure_ascii=False) if details and isinstance(details, dict) else None,
        )
        # 同步更新场景描述（保留 initial_description 不变）
        scene_crud.update_scene(db, scene_id, description=description)
        logger.info("场景变化已记录: scene=%d, day=%d, type=%s", scene_id, day_number, change_type)
        return 1
    except Exception as e:
        logger.warning("场景变化写入失败: %s", e)
        return 0


@app.post("/api/event/advance", response_model=EventResponse)
def advance_event(request: AdvanceRequest, db: Session = Depends(get_db)):
    """
    推进下一个待处理事件（控制台模式）。

    操作顺序：
      1. 检查是否有未打包的对话 session → 打包为 player_dialogue 事件
      2. 取下一个 status=pending 且 order_index 最小的 Event
      3. 根据事件类型写入 result_json
      4. 标记 status → completed
      5. 返回 EventResponse

    设计考量（Why this order）：
      对话打包必须先于"取下一个事件"，因为打包本身会创建一个 completed 事件，
      如果先取再打包，打包的事件不会被计入本次 advance 的结果。
      但**对话打包与日程推进是独立的**——用户推进日程事件时，
      对话作为"已完成的观测材料"被提前存好。
    """
    character_id = request.character_id

    # 校验角色存在
    character = character_crud.get_character(db, character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"角色不存在: id={character_id}")

    current_day = character.day_number or 1

    # 步骤1：打包未处理对话（如有）
    try:
        _package_session_dialogue_for_event(db, character_id)
    except Exception as e:
        logger.warning("对话打包失败（不阻断推进流程）: %s", e)

    # 步骤2：取下一个 pending 事件
    next_event = event_crud.get_next_pending_event(db, character_id, current_day)
    if not next_event:
        # 还有事件但已经全部 completed → 告诉用户去迭代
        pending_count = event_crud.has_pending_events(db, character_id, current_day)
        if not pending_count:
            raise HTTPException(
                status_code=404,
                detail=f"角色 {character.name} 在 Day {current_day} 的所有事件已推进完成。"
                       "请调用 /api/time/iterate 触发角色成长并生成次日事件。",
            )
        raise HTTPException(
            status_code=404,
            detail=f"角色 {character.name} 在 Day {current_day} 暂无待推进事件。",
        )

    # 步骤3：调用交互管线处理事件（v1.6 B8：双 LLM 管线替代硬编码）
    # ------------------------------------------------------------------
    # 核心设计：将日程事件纳入 Director+Actor 双 LLM 管线
    #
    # 数据流：
    #   Event ORM ──→ run_event() ──→ {action, speech, expression,
    #                                    emotion, goal, capabilities,
    #                                    event_attitude, plan_modifications,
    #                                    dialogue_pending, director_raw, actor_raw}
    #
    # 降级策略：
    #   LLM 调用失败时 run_event() 内部使用 FALLBACK_*_EVENT_OUTPUT
    #   保证管线不崩溃，始终有合法返回值
    #
    # personality 从 character 反序列化后传入
    # pipeline.run_event() 内部自行处理场景上下文/日程/人格加权
    # ------------------------------------------------------------------
    pipeline = InteractionPipeline()

    try:
        pipeline_result = pipeline.run_event(next_event, character, db)
    except Exception as e:
        logger.warning("事件管线处理失败，回退到降级文本: %s", e)
        # 完全不可恢复的情况下使用最简回退
        pipeline_result = {
            "action": f"角色完成了日程安排：{next_event.content}",
            "speech": None,
            "expression": "表情平静",
            "emotion": "平静",
            "goal": "完成当前日程",
            "capabilities": ["respond_normally", "complete_event(succeed)"],
            "event_attitude": "平常心对待",
            "plan_modifications": [],
            "dialogue_pending": None,
            "director_raw": None,
            "actor_raw": None,
        }

    # 步骤3.5：处理 capabilities（能力执行）
    # ----------------------------------------------------------------
    # 能力执行在 main.py 中（而非 run_event() 中），保持管线纯粹性。
    # 管线只负责"产生决策"，main.py 负责"执行决策的副作用"。
    # ----------------------------------------------------------------

    capabilities = pipeline_result.get("capabilities", [])

    # 3.5a: initiate_dialogue → 创建 character_initiative 事件
    if "initiate_dialogue" in capabilities:
        dialogue_pending = pipeline_result.get("dialogue_pending")
        if dialogue_pending and isinstance(dialogue_pending, dict):
            dialogue_content = dialogue_pending.get("content", "角色有话想说")
        else:
            dialogue_content = pipeline_result.get("speech") or "角色想主动和你聊聊"

        try:
            # 在当日日程末尾插入一个 character_initiative 事件
            # order_index 设为当前最大 order_index + 1
            existing_max_order = 0
            all_today = event_crud.get_events_by_day(db, character_id, current_day)
            if all_today:
                existing_max_order = max(
                    getattr(ev, "order_index", 0) for ev in all_today
                )
            event_crud.create_event(
                db=db,
                character_id=character_id,
                day_number=current_day,
                order_index=existing_max_order + 1,
                event_type="character_initiative",
                content=dialogue_content,
                status="pending",
            )
            logger.info(
                "角色发起主动对话: char=%d, content=%s",
                character_id, dialogue_content[:60],
            )
        except Exception as e:
            logger.warning("创建角色主动事件失败: %s", e)

    # 3.5b: modify_plan → 修改日程/目标
    if "modify_plan" in capabilities:
        plan_modifications = pipeline_result.get("plan_modifications", [])
        if isinstance(plan_modifications, list):
            for mod in plan_modifications:
                if not isinstance(mod, dict):
                    continue
                target_event_id = mod.get("event_id")
                action = mod.get("action", "")

                if action == "update_content" and target_event_id:
                    event_crud.update_event_content(
                        db, target_event_id,
                        new_content=mod.get("new_content", ""),
                        new_event_type=mod.get("new_event_type"),
                        new_time_period=mod.get("new_time_period"),
                    )
                elif action == "reorder" and target_event_id:
                    new_order = mod.get("new_order_index", 1)
                    event_crud.reorder_event(db, target_event_id, new_order)

                # 目标侧修改
                if mod.get("goal_progress") is not None or mod.get("new_goal"):
                    # 读取并更新 character.short_term_goals
                    goals_raw = character.short_term_goals
                    goals = []
                    if goals_raw:
                        try:
                            goals = json.loads(goals_raw)
                        except (json.JSONDecodeError, TypeError):
                            goals = []
                    if not isinstance(goals, list):
                        goals = []

                    # 进度更新
                    goal_idx = mod.get("goal_index")
                    if goal_idx is not None and isinstance(goal_idx, int):
                        try:
                            new_progress = float(mod.get("goal_progress", 0.5))
                            new_progress = max(0.0, min(1.0, new_progress))
                            if 0 <= goal_idx < len(goals):
                                goals[goal_idx]["progress"] = new_progress
                        except (ValueError, TypeError, IndexError):
                            pass

                    # 添加新目标
                    new_goal_text = mod.get("new_goal", "")
                    if new_goal_text and isinstance(new_goal_text, str) and new_goal_text.strip():
                        existing_texts = {
                            g.get("goal", "") for g in goals
                            if isinstance(g, dict)
                        }
                        if new_goal_text.strip() not in existing_texts:
                            goals.append({
                                "goal": new_goal_text.strip(),
                                "progress": 0.0,
                                "created_day": current_day,
                                "source": "character",
                            })

                    # 持久化
                    if goals:
                        character_crud.update_character(
                            db, character_id,
                            short_term_goals=json.dumps(goals, ensure_ascii=False),
                        )

    # 步骤4：持久化结果（含叙事元数据）
    # ----------------------------------------------------------------
    # result_json = action（Actor 行为叙事文本），让 Growth 迭代时能读到叙事化回执
    # 新增字段（director_raw/actor_raw/capabilities_applied/emotion/expression）
    # 让前端可以直接展示完整的事件推进叙事
    # ----------------------------------------------------------------
    result_action = pipeline_result.get("action", f"角色完成了日程安排：{next_event.content}")
    result_speech = pipeline_result.get("speech")
    # 如果有 speech，拼接到 result_json 中让 Growth 看到完整输出
    result_text = result_action
    if result_speech:
        result_text = f"{result_action}\n角色说：「{result_speech}」"

    completed = event_crud.complete_event(
        db, next_event.id, result_text,
        director_raw=pipeline_result.get("director_raw"),
        actor_raw=pipeline_result.get("actor_raw"),
        capabilities_applied=json.dumps(capabilities, ensure_ascii=False),
        emotion=pipeline_result.get("emotion"),
        expression=pipeline_result.get("expression"),
    )
    if not completed:
        raise HTTPException(status_code=500, detail="事件推进失败")

    return completed


@app.post("/api/time/iterate", response_model=IterateResponse)
def iterate_day(request: IterateRequest, db: Session = Depends(get_db)):
    """
    触发 Growth 迭代一天（控制台模式）。

    操作顺序：
      1. 收集角色当天所有 status=completed 的 Event
      2. 调用 GrowthModule.run() 输出人格变化/新记忆/次日日程
      3. 将 schedule 数组逐项写入 events 表 (status=pending, day_number+1)
      4. 将 world_changes 写入 scene_changes 表
      5. 角色 day_number += 1
      6. 返回 IterateResponse

    设计考量（Why not put this logic in GrowthModule）：
      GrowthModule 的职责是"LLM 推理 + 人格计算"，不负责
      数据库写入安排。将"schedule → events 表"的写操作放在
      main.py 中，保持 GrowthModule 的可测试性。
    """
    character_id = request.character_id

    character = character_crud.get_character(db, character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"角色不存在: id={character_id}")

    current_day = character.day_number or 1

    # 步骤1：运行 Growth 模块
    growth_module = GrowthModule()
    try:
        result = growth_module.run(
            character_id=character_id,
            db=db,
            events=None,  # 自动从 DB 读取当日 completed 事件
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"成长分析失败: {str(e)}")

    # 步骤2：将 schedule 写入 events 表（次日事件）
    schedule_json = result.get("schedule_json", "[]")
    try:
        schedule_items = json.loads(schedule_json) if isinstance(schedule_json, str) else schedule_json
    except (json.JSONDecodeError, TypeError):
        schedule_items = []

    new_day_number = current_day + 1
    events_created = 0
    for item in schedule_items:
        event_crud.create_event(
            db=db,
            character_id=character_id,
            day_number=new_day_number,
            order_index=item.get("order_index", 1),
            event_type=item.get("event_type", "schedule_action"),
            content=item.get("content", ""),
            status="pending",
            time_period=item.get("time_period"),
        )
        events_created += 1

    # ---- v1.6 Phase 1：写入 SceneChange 记录 ----
    _write_scene_changes_from_growth(
        db=db,
        character=character,
        growth_log_id=result.get("growth_log_id", 0),
        world_changes_json=result.get("world_changes_json"),
        day_number=current_day,
    )

    # 步骤3：获取更新后的角色 day_number
    updated_char = character_crud.get_character(db, character_id)
    final_day = updated_char.day_number if updated_char else new_day_number

    # 步骤4：组装响应
    return IterateResponse(
        growth_log_id=result.get("growth_log_id", 0),
        character_id=character_id,
        day_number=final_day,
        personality_delta=result.get("personality_delta"),
        event_summary=result.get("event_summary"),
        new_memories=result.get("new_memories"),
        world_changes_json=result.get("world_changes_json"),
        schedule_json=schedule_json,
        events_created=events_created,
        growth_raw=result.get("growth_raw"),
        created_at=str(result.get("created_at", "")),
    )


@app.post("/api/time/auto", response_model=AutoResponse)
def auto_advance(request: AdvanceRequest, db: Session = Depends(get_db)):
    """
    自动推进全部待处理事件 + 触发迭代（自动模式）。

    操作顺序：
      1. 循环调用 advance_event 直至无更多 pending 事件
      2. 所有事件完成后自动调用 iterate_day
      3. 返回 AutoResponse（含全部完成的事件列表 + 迭代结果）

    设计考量（Why concatenate two endpoints instead of separate calls）：
      控制台模式的 advance + iterate 需要用户手动点击两次。
      自动模式提供"一键推演一天"的语法糖，减少用户操作步骤。
      若 iterate 失败，已 advance 的 events 仍可在返回值中查看。
    """
    character_id = request.character_id

    character = character_crud.get_character(db, character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"角色不存在: id={character_id}")

    completed_events = []
    current_day = character.day_number or 1

    # 步骤1：循环推进所有 pending 事件
    while True:
        try:
            # 先打包对话
            _package_session_dialogue_for_event(db, character_id)
            # 取下一个 pending
            ev = event_crud.get_next_pending_event(db, character_id, current_day)
            if not ev:
                break
            # 执行推进
            rejson = f"自动推进：{ev.content}"
            event_crud.complete_event(db, ev.id, rejson)
            # 刷新后再读
            db.refresh(ev)
            completed_events.append(EventResponse.model_validate(ev))
        except Exception as e:
            logger.warning("自动推进中断于事件 #%d: %s",
                           len(completed_events) + 1, e)
            break

    # 步骤2：自动触发迭代
    iterate_result = None
    try:
        growth_module = GrowthModule()
        growth_result = growth_module.run(
            character_id=character_id,
            db=db,
            events=None,
        )
        # 写入次日事件
        schedule_json = growth_result.get("schedule_json", "[]")
        try:
            schedule_items = json.loads(schedule_json) if isinstance(schedule_json, str) else schedule_json
        except (json.JSONDecodeError, TypeError):
            schedule_items = []

        new_day = (character.day_number or 1) + 1
        ec = 0
        for item in schedule_items:
            event_crud.create_event(
                db=db, character_id=character_id,
                day_number=new_day,
                order_index=item.get("order_index", 1),
                event_type=item.get("event_type", "schedule_action"),
                content=item.get("content", ""),
                status="pending",
                time_period=item.get("time_period"),
            )
            ec += 1

        updated_char = character_crud.get_character(db, character_id)
        iterate_result = IterateResponse(
            growth_log_id=growth_result.get("growth_log_id", 0),
            character_id=character_id,
            day_number=updated_char.day_number if updated_char else new_day,
            personality_delta=growth_result.get("personality_delta"),
            event_summary=growth_result.get("event_summary"),
            new_memories=growth_result.get("new_memories"),
            world_changes_json=growth_result.get("world_changes_json"),
            schedule_json=schedule_json,
            events_created=ec,
            growth_raw=growth_result.get("growth_raw"),
            created_at=str(growth_result.get("created_at", "")),
        )

        # v1.6 Phase 1：写入 SceneChange 记录
        _write_scene_changes_from_growth(
            db=db,
            character=character,
            growth_log_id=growth_result.get("growth_log_id", 0),
            world_changes_json=growth_result.get("world_changes_json"),
            day_number=current_day,
        )
    except Exception as e:
        logger.warning("自动迭代失败（事件已推进，迭代可稍后重试）: %s", e)
        return AutoResponse(
            character_id=character_id,
            completed_events=completed_events,
            iterate_result=None,
            error=f"迭代失败: {str(e)}",
        )

    return AutoResponse(
        character_id=character_id,
        completed_events=completed_events,
        iterate_result=iterate_result,
    )


@app.get("/api/characters/{character_id}/events", response_model=List[EventResponse])
def get_character_events(
    character_id: int,
    day_number: Optional[int] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    获取角色的事件列表（按 order_index 升序）。

    参数：
      - day_number: 可选，指定天数
      - status: 可选，筛选状态（pending/completed）

    设计考量：
      - 无 day_number 时返回所有天的事件（按 day_number, order_index 排序）
      - 前端用此接口展示"事件时间轴"
    """
    character = character_crud.get_character(db, character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"角色不存在: id={character_id}")

    if day_number is not None:
        events = event_crud.get_events_by_day(
            db, character_id, day_number, status_filter=status,
        )
    else:
        query = db.query(Event)
        if status:
            query = query.filter(Event.status == status)
        events = (
            query
            .filter(Event.character_id == character_id)
            .order_by(Event.day_number.asc(), Event.order_index.asc())
            .all()
        )

    return events


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


# ============================================================
# v1.6 Phase 1：世界系统 API 端点
# ============================================================

@app.get("/api/worlds", response_model=List[WorldResponse])
def list_worlds(db: Session = Depends(get_db)):
    """
    获取所有世界列表。

    设计考量：前端"世界选择"页面用此接口展示所有已创建的世界。
    """
    worlds = world_crud.get_all_worlds(db)
    return worlds


@app.get("/api/worlds/{world_id}", response_model=WorldResponse)
def get_world(world_id: int, db: Session = Depends(get_db)):
    """获取单个世界详情。"""
    world = world_crud.get_world(db, world_id)
    if not world:
        raise HTTPException(status_code=404, detail=f"世界不存在: id={world_id}")
    return world


@app.patch("/api/worlds/{world_id}", response_model=WorldResponse)
def update_world(world_id: int, request: WorldUpdateRequest, db: Session = Depends(get_db)):
    """
    更新世界信息（部分更新）。

    设计考量：Phase 1 支持修改世界观名称和核心世界观文本。
    """
    updated = world_crud.update_world(
        db, world_id,
        name=request.name,
        core_worldview=request.core_worldview,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"世界不存在: id={world_id}")
    return updated


@app.get("/api/worlds/{world_id}/scenes", response_model=List[SceneResponse])
def get_world_scenes(
    world_id: int,
    scene_layer: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    获取指定世界下的场景列表。

    Args:
        world_id: 世界 ID
        scene_layer: 可选筛选 "conceptual" / "actual"

    设计考量：前端"世界详情页"展示完整的场景地图（概念层 + 实际层）。
    """
    world = world_crud.get_world(db, world_id)
    if not world:
        raise HTTPException(status_code=404, detail=f"世界不存在: id={world_id}")
    scenes = scene_crud.get_scenes_by_world(db, world_id, scene_layer=scene_layer)
    return scenes


@app.get("/api/scenes/{scene_id}", response_model=SceneResponse)
def get_scene(scene_id: int, db: Session = Depends(get_db)):
    """获取单个场景详情。"""
    scene = scene_crud.get_scene(db, scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"场景不存在: id={scene_id}")
    return scene


@app.get("/api/scenes/{scene_id}/path", response_model=List[SceneResponse])
def get_scene_path(scene_id: int, db: Session = Depends(get_db)):
    """
    获取从根场景到当前场景的完整路径（面包屑导航）。

    返回从最顶层概念场景到当前场景的有序列表。
    用于前端展示"角色身处何地"的上下文。
    """
    path = scene_crud.get_scene_path(db, scene_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"场景不存在: id={scene_id}")
    return path


@app.get("/api/scenes/{scene_id}/adjacent", response_model=List[SceneResponse])
def get_adjacent_scenes(scene_id: int, db: Session = Depends(get_db)):
    """
    获取同一父场景下的兄弟场景（不含自身）。

    用于向玩家展示"附近还有哪些地方可以去"。
    """
    scenes = scene_crud.get_adjacent_scenes(db, scene_id)
    return scenes


@app.get("/api/scenes/{scene_id}/changes", response_model=List[SceneChangeResponse])
def get_scene_changes(
    scene_id: int,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """
    获取指定场景的最近 N 条变化记录。

    用于前端展示"场景历史时间轴"，让玩家了解场景如何演变。
    """
    scene = scene_crud.get_scene(db, scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"场景不存在: id={scene_id}")
    changes = scene_change_crud.get_recent_changes(db, scene_id, limit=limit)
    return changes


@app.get("/api/characters/{character_id}/world", response_model=WorldResponse)
def get_character_world(character_id: int, db: Session = Depends(get_db)):
    """
    获取角色所属的世界。

    返回 None 时表示角色尚未关联世界（旧数据通过 migration 回填）。
    """
    character = character_crud.get_character(db, character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"角色不存在: id={character_id}")
    world = world_crud.get_world_by_character(db, character_id)
    if not world:
        raise HTTPException(status_code=404, detail="角色尚未关联世界")
    return world


@app.get("/api/characters/{character_id}/scenes", response_model=List[SceneResponse])
def get_character_scenes(character_id: int, db: Session = Depends(get_db)):
    """
    获取角色所属世界的所有场景。

    用于前端"世界探索"面板，展示角色身处的整个场景地图。
    """
    character = character_crud.get_character(db, character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"角色不存在: id={character_id}")
    scenes = scene_crud.get_scenes_by_character(db, character_id)
    return scenes


@app.get("/api/characters/{character_id}/world-changes", response_model=List[SceneChangeResponse])
def get_character_world_changes(
    character_id: int,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """
    获取角色所属世界的场景变化时间轴。

    用于前端在角色详情页展示"世界正在发生什么"。
    """
    character = character_crud.get_character(db, character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"角色不存在: id={character_id}")
    changes = scene_change_crud.get_scene_changes_by_character(
        db, character_id, limit=limit,
    )
    return changes


# ==================== 根路径 ====================

@app.get("/")
def root():
    return {
        "message": "CharacterSeed API is running!",
        "docs": "http://localhost:8000/docs",
        "version": "0.1.0"
    }
