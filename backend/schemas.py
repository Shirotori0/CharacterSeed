from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# ==================== Character Schemas ====================

class CharacterCreate(BaseModel):
    description: str  # 用户描述（一句话或故事）
    # 注意：文件上传通过FastAPI的UploadFile处理，不在这里定义

class CharacterResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    world_setting: Optional[str] = None
    personality: Optional[str] = None  # JSON字符串
    current_state: Optional[str] = None  # JSON字符串
    creation_raw: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    # Day4 新增：事件推进系统所需字段
    day_number: int = 1
    speaking_style: Optional[str] = None  # JSON数组字符串
    values: Optional[str] = None  # JSON数组字符串
    habits: Optional[str] = None  # JSON数组字符串
    long_term_goal: Optional[str] = None

    # v1.6 Phase 1：世界系统字段
    world_id: Optional[int] = None          # 所属世界 ID
    current_scene_id: Optional[int] = None  # 当前所在实际场景 ID
    short_term_goals: Optional[str] = None  # 短期目标（JSON 数组字符串）
    """
    设计考量：5个字段均从数据库 ORM 属性自动映射，Pydantic from_attributes=True
    确保无论 ORM 直接返回值还是人工构造 dict，都能正确序列化。
    day_number 默认 1 兼容旧角色（迁移后自动获得初始值）。
    """
    
    class Config:
        from_attributes = True

# ==================== Conversation Schemas ====================
# ChatRequest 统一在底部 "ChatSession Schemas" 之后定义（带可选 session_id）

class ChatResponse(BaseModel):
    id: int
    character_id: int
    user_input: str
    npc_response: str
    emotion: Optional[str] = None
    action: Optional[str] = None
    expression: Optional[str] = None
    director_raw: Optional[str] = None
    actor_raw: Optional[str] = None
    timestamp: datetime
    session_id: Optional[int] = None  # ← 新增：返回消息所属 session
    session_title: Optional[str] = None  # ← 新增：方便前端立即更新侧栏

    class Config:
        from_attributes = True

# ==================== Memory Schemas ====================

class MemoryResponse(BaseModel):
    id: int
    character_id: int
    content: str
    importance: int = 5
    memory_type: str = "conversation"
    created_at: datetime
    
    class Config:
        from_attributes = True

# ==================== Growth Schemas ====================

class GrowthTriggerRequest(BaseModel):
    character_id: int

class GrowthResponse(BaseModel):
    id: int
    character_id: int
    personality_delta: Optional[str] = None
    event_summary: Optional[str] = None
    new_memories: Optional[str] = None
    growth_raw: Optional[str] = None
    created_at: datetime

    # Day4 新增：事件维度输出
    schedule_json: Optional[str] = None  # 次日事件实体列表（JSON数组字符串）
    world_changes_json: Optional[str] = None  # 世界变化（JSON对象字符串）
    
    class Config:
        from_attributes = True

# ==================== Event Iterate Schemas（Day4 新增） ====================

class EventResponse(BaseModel):
    """
    事件响应模型。

    v1.6 B6 新增字段：
      - director_raw: Director 决策原始 JSON（前端可展示"角色思考过程"）
      - actor_raw: Actor 叙事原始 JSON（前端可展示完整行为描述）
      - capabilities_applied: 角色选择的能力列表（前端可展示"角色选择了什么行动方针"）
      - emotion: 角色当前情绪（移出 director_raw 到顶层，方便前端直接使用）
      - expression: 角色表情（移出 actor_raw 到顶层）

    设计考量：
      - content / result_json 为纯文本而非嵌套对象，降低前端解析负担
      - metadata_json 保留为原始 JSON 字符串，前端需要时自行解析
      - 时间字段 datetime 由 Pydantic 自动序列化为 ISO 字符串
    """
    id: int
    character_id: int
    day_number: int
    order_index: int
    event_type: str
    content: str
    metadata_json: Optional[str] = None
    result_json: Optional[str] = None
    status: str = "pending"
    session_id: Optional[int] = None
    time_period: Optional[str] = None
    created_at: datetime
    # v1.6 B6 新增：事件叙事字段
    director_raw: Optional[str] = None       # Director LLM 原始响应 JSON
    actor_raw: Optional[str] = None          # Actor LLM 原始响应 JSON
    capabilities_applied: Optional[str] = None  # 角色选择的能力列表（JSON 数组字符串）
    emotion: Optional[str] = None            # 角色推进事件时的情绪
    expression: Optional[str] = None         # 角色推进事件时的表情

    class Config:
        from_attributes = True


class AdvanceRequest(BaseModel):
    """
    推进事件请求。

    字段最小化设计：当前仅需 character_id，
    推进逻辑在服务端自动取 order_index 最小的 pending 事件。
    字段精简避免前端拼参错误，也为后续扩展保留弹性。
    """
    character_id: int


class IterateRequest(BaseModel):
    """迭代一天请求（触发 Growth 生成次日事件）。"""
    character_id: int


class IterateResponse(BaseModel):
    """
    迭代一天响应。

    包含 Growth 的全部产出 + 新生成的事件列表。
    前端可展示：人格变化、新记忆、世界变化、次日日程安排。
    """
    growth_log_id: int
    character_id: int
    day_number: int  # 迭代后角色的新 day_number
    personality_delta: Optional[str] = None
    event_summary: Optional[str] = None
    new_memories: Optional[str] = None
    world_changes_json: Optional[str] = None
    schedule_json: Optional[str] = None  # 生成的事件实体JSON数组
    events_created: int = 0  # 实际插入 events 表的记录数
    growth_raw: Optional[str] = None
    created_at: Optional[str] = None


class AutoResponse(BaseModel):
    """
    自动迭代响应。

    completed_events: 本次推进完成的事件列表（按完成顺序）
    iterate_result:   最终迭代一天的结果（完成后自动调 iterate）
                      若 iterate 失败则为 None
    error:            整体操作中的错误信息
    """
    character_id: int
    completed_events: List[EventResponse] = []
    iterate_result: Optional[IterateResponse] = None
    error: Optional[str] = None
    """
    设计考量：AutoResponse 将"推进全部pending事件+迭代"串联为一个原子操作，
    减少前端调用链（不需要先 advance 再等响应再 iterate）。
    如果中间某步失败，已完成的 events 仍可在 completed_events 中查看。
    """


# ==================== Creation Response (Special) ====================

class CreationResponse(BaseModel):
    """Creation Module的完整响应"""
    id: int
    name: str
    world_setting: Optional[str] = None
    personality: Optional[str] = None
    initial_memories: Optional[List[str]] = None
    current_state: Optional[str] = None
    creation_raw: Optional[str] = None

# ==================== LLM Settings Schemas ====================

class ProviderMeta(BaseModel):
    """前端下拉选项用的厂商元信息"""
    id: str
    name: str
    needs_key: str  # "true" / "false"（用字符串是因前端 JS 解析方便）


class ProviderConfig(BaseModel):
    """单个 provider 的配置（写入侧：明文 api_key）"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


class ProviderConfigMasked(BaseModel):
    """单个 provider 的配置（读取侧：api_key 已脱敏）"""
    api_key: str        # 已脱敏：保留首尾 4 字符
    base_url: str
    model: str


class LLMSettingsResponse(BaseModel):
    """GET /api/settings/llm 的响应体"""
    active_provider: str
    active_provider_name: str
    config: ProviderConfigMasked
    default_temperature: float
    default_max_tokens: int
    providers: dict  # {provider_id: ProviderConfigMasked}
    settings_file_path: str  # 给前端展示用，便于排错


class LLMUpdateRequest(BaseModel):
    """PUT /api/settings/llm 的请求体（部分字段可选）"""
    active_provider: Optional[str] = None       # 切换激活 provider
    active_config: Optional[ProviderConfig] = None  # 修改当前激活 provider 的配置
    default_temperature: Optional[float] = None
    default_max_tokens: Optional[int] = None


class LLMTestRequest(BaseModel):
    """POST /api/settings/llm/test 的请求体（可选覆盖当前配置）"""
    # 不传则用当前激活 provider 的配置测试
    provider_id: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    test_prompt: Optional[str] = "你好，请用一句话自我介绍。"


class LLMTestResponse(BaseModel):
    """POST /api/settings/llm/test 的响应体"""
    success: bool
    message: str
    provider_id: str
    model: str
    response_text: Optional[str] = None
    latency_ms: Optional[int] = None


# ==================== API Test Schemas ====================
# 参考 https://github.com/joker1point/web-tools 的 API 联通测试 Dashboard
# 三大能力：models 列表 / 流式延迟 / 原始请求探针

class TestModelItem(BaseModel):
    """provider /v1/models 返回的单个模型条目"""
    id: str
    owned_by: str = ""
    object: str = "model"


class ModelsListResponse(BaseModel):
    """GET /api/test/models 响应体"""
    provider_id: str
    base_url: str
    models: List[TestModelItem]
    duration_ms: int
    raw_count: int


class LatencyTestRequest(BaseModel):
    """POST /api/test/latency 请求体"""
    provider_id: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    test_message: Optional[str] = "Hi"
    max_tokens: Optional[int] = 16


class LatencyTestResponse(BaseModel):
    """POST /api/test/latency 响应体"""
    provider_id: str
    model: str
    status: int
    ttft_ms: Optional[int] = None      # Time To First Token
    total_ms: Optional[int] = None     # 完整响应耗时
    content: str = ""
    chunks: int = 0
    error: Optional[str] = None


class ProbeRequest(BaseModel):
    """POST /api/test/probe 请求体（debug 模式）"""
    provider_id: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    test_message: Optional[str] = "Hi"
    max_tokens: Optional[int] = 16


class ProbeResponse(BaseModel):
    """POST /api/test/probe 响应体（含完整 request/response，密钥脱敏）"""
    provider_id: str
    model: str
    base_url: str
    request: dict
    response: dict
    error: Optional[str] = None


# ==================== ChatSession Schemas ====================
# 参考 https://github.com/ChatGPTNextWeb/NextChat 的会话管理
# 提供：list / create / rename / delete / get-detail（带 messages）/ search

class ChatSessionCreate(BaseModel):
    """POST /api/sessions 请求体"""
    character_id: int
    title: Optional[str] = None  # 缺省时用"新对话"占位


class ChatSessionUpdate(BaseModel):
    """PATCH /api/sessions/{id} 请求体（目前只支持改 title）"""
    title: str


class ChatSessionInfo(BaseModel):
    """会话概要（列表用）"""
    id: int
    character_id: int
    title: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    message_count: int = 0

    class Config:
        from_attributes = True


class ChatSessionWithMessages(ChatSessionInfo):
    """会话详情（含所有消息）"""
    messages: List["ConversationRow"] = []


class ConversationRow(BaseModel):
    """单条对话（与数据库行 1:1）"""
    id: int
    session_id: Optional[int] = None
    character_id: int
    user_input: str
    npc_response: Optional[str] = None
    emotion: Optional[str] = None
    action: Optional[str] = None
    expression: Optional[str] = None
    director_raw: Optional[str] = None
    actor_raw: Optional[str] = None
    timestamp: Optional[str] = None

    class Config:
        from_attributes = True


# ChatRequest 增加可选的 session_id（向后兼容：None 时自动创建新 session）
class ChatRequest(BaseModel):
    character_id: int
    message: str
    session_id: Optional[int] = None  # ← 新增


# 解决 ChatSessionWithMessages 中 ConversationRow 的前向引用
ChatSessionWithMessages.model_rebuild()


# ============================================================
# v1.6 Phase 1：世界系统 Schemas
# ============================================================

class WorldResponse(BaseModel):
    """世界查询响应"""
    id: int
    name: str
    core_worldview: str = ""
    created_at: datetime

    class Config:
        from_attributes = True


class SceneResponse(BaseModel):
    """场景查询响应"""
    id: int
    world_id: int
    name: str
    scene_layer: str           # "conceptual" | "actual"
    scene_type: Optional[str] = None
    parent_scene_id: Optional[int] = None
    description: Optional[str] = None
    initial_description: Optional[str] = None
    attributes_json: Optional[str] = None
    created_day: int = 1
    created_at: datetime

    class Config:
        from_attributes = True


class SceneChangeResponse(BaseModel):
    """场景迭代记录响应"""
    id: int
    scene_id: int
    growth_log_id: Optional[int] = None
    change_type: str            # "character_driven" | "external"
    description: str
    change_details_json: Optional[str] = None
    day_number: int
    created_at: datetime

    class Config:
        from_attributes = True


class WorldUpdateRequest(BaseModel):
    """世界更新请求（Phase 4 前端 PATCH 预留）"""
    name: Optional[str] = None
    core_worldview: Optional[str] = None
