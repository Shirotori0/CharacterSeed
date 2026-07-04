from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Index
from sqlalchemy.sql import func
from backend.database import Base

class Character(Base):
    __tablename__ = "characters"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)  # 用户原始输入
    world_setting = Column(Text, nullable=True)  # 世界设定（LLM生成）
    personality = Column(Text, nullable=True)  # 人格属性（JSON格式）
    current_state = Column(Text, nullable=True)  # 当前状态（JSON格式）
    creation_raw = Column(Text, nullable=True)  # Creation LLM原始响应
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Day4 新增：事件推进系统所需字段
    day_number = Column(Integer, default=1)  # 当前天数（Growth迭代时自增）
    speaking_style = Column(Text, nullable=True)  # 说话风格描述（JSON数组字符串）
    values = Column(Text, nullable=True)  # 核心信念（JSON数组字符串）
    habits = Column(Text, nullable=True)  # 日常习惯（JSON数组字符串）
    long_term_goal = Column(Text, nullable=True)  # 长期目标（纯文本）

    # v1.6 Phase 1：世界系统字段
    world_id = Column(
        Integer, ForeignKey("worlds.id", ondelete="SET NULL"), nullable=True, index=True,
    )  # 所属世界（N:1 共享世界）
    current_scene_id = Column(
        Integer, ForeignKey("scenes.id", ondelete="SET NULL"), nullable=True,
    )  # 角色当前所在的实际场景
    short_term_goals = Column(Text, nullable=True)  # 短期目标（JSON 数组字符串）

    """
    设计考量：
      day_number: 事件推进轴的核心计数器。每个 Growth 迭代将 day_number+1，
                  所有 Event 按 (character_id, day_number, order_index) 排序。
                  使用 Integer 而非 DateTime 表示"天数"，避免跨天边界判断逻辑。
      speaking_style/values/habits: JSON 数组字符串，由 Creation LLM 生成，
                  也可在 Growth 迭代中更新。数组格式让前端可逐条展示。
      long_term_goal: 纯文本字段，Growth LLM 可基于事件观察修改。
      world_id: 指向独立的 worlds 表，多角色可共享同一世界（N:1）。
                nullable=True 兼容存量角色（迁移回填后设为非空）。
      current_scene_id: 结构化位置标识，取代 current_state.location 纯文本。
                        指向 scenes 表中 scene_layer='actual' 的行。
      short_term_goals: v1.6 Phase 3 引入，JSON 数组格式：
          [{"goal":"...", "progress":0.0, "created_day":1, "source":"creation"}]
    """


class ChatSession(Base):
    """
    对话会话（多轮消息的容器，参考 NextChat 的 session 概念）

    与 Conversation 的关系：
      - ChatSession 1 → N Conversation
      - 每个 session 有一个 title（自动生成首条消息前缀 or 用户手动改）
      - 删除 session 会级联删除其下所有 conversation
    """
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    character_id = Column(
        Integer, ForeignKey("characters.id"), nullable=False, index=True,
    )
    title = Column(String(200), nullable=False, default="新对话")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        index=True,  # 列表页按更新时间倒序，常查
    )


class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    character_id = Column(Integer, ForeignKey("characters.id"), nullable=False, index=True)
    # 会话归属（可空以兼容旧数据；migrate 时会回填到默认 session）
    session_id = Column(
        Integer,
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    user_input = Column(Text, nullable=False)
    npc_response = Column(Text, nullable=True)
    emotion = Column(String(50), nullable=True)
    action = Column(Text, nullable=True)
    expression = Column(String(100), nullable=True)
    director_raw = Column(Text, nullable=True)  # Director LLM原始响应
    actor_raw = Column(Text, nullable=True)  # Actor LLM原始响应
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class Event(Base):
    """
    事件表（Day4 新增）

    职责：作为 Growth 迭代与用户推进之间的桥梁。
      - Growth 迭代产出次日待办事件列表（status=pending, day_number+1）
      - 用户通过"推进事件"逐个完成（status=completed）
      - 所有 completed 事件作为 Growth 下一次迭代的"观察材料"

    状态机：pending → active → completed
      pending:   Growth 刚产出，等待用户推进
      active:    正在进行中（对话事件打包时临时标记）
      completed: 用户已推进（result_json 已写入）

    对话打包设计：
      一组玩家对话不逐条成 Event，而是整个 session 打包为一个 Event。
      打包时机：advance_event 前，检查本日是否有未打包对话。
      content = 对话摘要, metadata_json = 完整聊天记录, event_type = player_dialogue
    """
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    character_id = Column(Integer, ForeignKey("characters.id"), nullable=False, index=True)
    day_number = Column(Integer, nullable=False, default=1)
    order_index = Column(Integer, nullable=False, default=1)
    event_type = Column(
        String(30), nullable=False, default="schedule_action",
        index=True,
    )
    # event_type 枚举值：player_dialogue / schedule_action / scene_event / character_initiative
    content = Column(Text, nullable=False)  # 事件描述文本
    metadata_json = Column(Text, nullable=True)  # 附加数据（如对话记录JSON）
    result_json = Column(Text, nullable=True)  # 执行回执（completed后由 advance 写入）
    status = Column(String(20), nullable=False, default="pending", index=True)
    # status 枚举值：pending / active / completed
    session_id = Column(
        Integer, ForeignKey("chat_sessions.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )  # 对话事件关联的 session
    time_period = Column(String(20), nullable=True)  # 时段标签（如 morning/afternoon，仅元数据）
    # v1.6 B6 新增：事件叙事与决策元数据字段
    director_raw = Column(Text, nullable=True)       # Director 决策原始 JSON
    actor_raw = Column(Text, nullable=True)          # Actor 叙事原始 JSON
    capabilities_applied = Column(Text, nullable=True)  # 角色选择的能力列表（JSON 数组字符串）
    emotion = Column(String(50), nullable=True)      # 角色处理事件时的情绪
    expression = Column(String(100), nullable=True)  # 角色处理事件时的表情
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    """
    复合索引设计：
      (character_id, day_number, order_index) — 按角色/天排序的完整事件列表
      (character_id, day_number, status)      — 高效查询"某天待推进"事件
    """


Index(
    "ix_events_char_day_order",
    Event.character_id, Event.day_number, Event.order_index,
)
Index(
    "ix_events_char_day_status",
    Event.character_id, Event.day_number, Event.status,
)


class Memory(Base):
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True, index=True)
    character_id = Column(Integer, ForeignKey("characters.id"), nullable=False)
    content = Column(Text, nullable=False)
    importance = Column(Integer, default=5)  # 1-10，默认5
    memory_type = Column(String(50), default="conversation")  # conversation, event, growth
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GrowthLog(Base):
    __tablename__ = "growth_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    character_id = Column(Integer, ForeignKey("characters.id"), nullable=False)
    personality_delta = Column(Text, nullable=True)  # 人格变化（JSON格式）
    event_summary = Column(Text, nullable=True)
    new_memories = Column(Text, nullable=True)  # 新增记忆（JSON数组）
    growth_raw = Column(Text, nullable=True)  # Growth LLM原始响应

    # Day4 新增：事件维度输出
    schedule_json = Column(Text, nullable=True)  # 次日事件实体列表（JSON数组）
    world_changes_json = Column(Text, nullable=True)  # 世界变化描述（JSON对象）
    """
    设计考量：
      schedule_json: Growth LLM 输出的事件实体数组，
                     写入后由 main.py 逐项插入 events 表（status=pending）。
                     包含 {content, event_type, time_period, order_index}。
      world_changes_json: 记录角色所处世界的宏观变化（如"酒馆来了新客人"），
                          供后续 Growth 迭代参考上下文。
    """

    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ============================================================
# v1.6 Phase 1：世界系统三表
# ============================================================

class World(Base):
    """
    世界表 — 独立持久化，多角色可共享

    设计考量：
      - 不含 character_id FK（由 characters.world_id 反向引用 N:1）
      - core_worldview 是 Creation LLM 产出的核心世界观文本
      - 独立持久化使得世界数据在角色删除后仍可保留（除非最后一个角色）
    """
    __tablename__ = "worlds"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    core_worldview = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Scene(Base):
    """
    场景表 — 仅两层：概念场景 (conceptual) + 实际场景 (actual)

    层级约束：
      - conceptual 可嵌套 conceptual（世界观框架：大陆→王国→区域）
      - actual 的 parent_scene_id 必须指向 conceptual 节点（校验在 schema 层完成）
      - actual 不能有子实际场景，也不能嵌套 actual

    attributes_json 扩展预留：
      - heatmap_score / visit_count / mood / danger_level 等热力图字段
      - 当前 Phase 1 仅保留结构，Phase 4 前端按需使用
    """
    __tablename__ = "scenes"

    id = Column(Integer, primary_key=True, index=True)
    world_id = Column(
        Integer, ForeignKey("worlds.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    name = Column(String(100), nullable=False)
    scene_layer = Column(String(20), nullable=False)   # "conceptual" | "actual"
    scene_type = Column(String(30), nullable=True)      # continent/kingdom/town/tavern/cave/...
    parent_scene_id = Column(
        Integer, ForeignKey("scenes.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    description = Column(Text, nullable=True)            # 当前场景描述（反映最新状态）
    initial_description = Column(Text, nullable=True)    # Creation 时的原始描述（不可变锚点）
    attributes_json = Column(Text, nullable=True)        # 扩展属性（heatmap_score/mood...）
    created_day = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SceneChange(Base):
    """
    场景迭代记录表 — 以叙事描述为核心，桥接硬编码与自由描述

    change_type 两分法：
      - character_driven: 角色行动导致的变化（"打翻油灯引发火灾"）
      - external: 外界/环境/他人导致的变化（"暴风雪封路"）

    与 Scene 的关系：
      - SceneChange 创建时同步更新 Scene.description（钩子在 Growth 模块处理）
      - Scene.initial_description 保留创建时的原始描述不变
      - SceneChange 保留完整叙事因果链——谁/什么导致了变化
    """
    __tablename__ = "scene_changes"

    id = Column(Integer, primary_key=True, index=True)
    scene_id = Column(
        Integer, ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    growth_log_id = Column(
        Integer, ForeignKey("growth_logs.id", ondelete="SET NULL"), nullable=True,
    )
    change_type = Column(String(20), nullable=False)     # "character_driven" | "external"
    description = Column(Text, nullable=False)            # 核心：叙事化的变化描述
    change_details_json = Column(Text, nullable=True)     # 可选结构化详情
    day_number = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ============================================================
# 复合索引：高频查询模式
# ============================================================
# 1) 会话列表：按 character_id + updated_at desc
Index(
    "ix_chat_sessions_char_updated",
    ChatSession.character_id, ChatSession.updated_at.desc(),
)
# 2) 单会话的消息列表：按 session_id + timestamp
Index(
    "ix_conversations_session_timestamp",
    Conversation.session_id, Conversation.timestamp,
)
# 3) 场景变更：按场景+天数排序（前端展示场景历史时高频查询）
Index(
    "ix_scene_changes_scene_day",
    SceneChange.scene_id, SceneChange.day_number,
)
