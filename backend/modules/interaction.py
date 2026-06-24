"""
Day 2 — 交互运行时（Interaction Runtime）

设计理念：采用"注意力聚焦 + 行为生成"双 LLM 管路。
    Director.analyze() → 角色"感知"世界，决定该关注什么
    Actor.generate()   → 角色"表达"自我，生成动作/表情/语言

该架构的核心优势：
  - 可解释性：Director 的中间输出（emotion / focus_memories / goal）
              可独立可视化，让观察者看到"角色的思考过程"
  - 可调试性：两个 LLM 独立调试 prompt 与温度参数，互不干扰
  - 鲁棒性：  每个 LLM 调用点都有独立降级策略，任一失败不影响管线完整性

温度参数选择依据：
  - Director temperature=0.5：注意力聚焦需要逻辑一致性，偏低减少随机性
  - Actor temperature=0.8：  行为/语言生成需要创造性，偏高避免千篇一律
"""

import json
import logging
import collections
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from sqlalchemy.orm import Session

from backend.services.llm_service import LLMService
from backend.crud import character as character_crud
from backend.crud import memory as memory_crud
from backend.crud import conversation as conversation_crud
from backend.crud import scene as scene_crud           # v1.6: Scene 上下文
from backend.crud import scene_change as scene_change_crud  # v1.6: 场景变化
from backend.crud import event as event_crud           # v1.6: 事件上下文

logger = logging.getLogger(__name__)


# ============================================================================
# 降级常量：当 LLM 调用失败时，保证管线不崩溃
# 设计考量：降级值采用"中立/保守"策略——
#   宁可返回一个 bland but correct 的回复，也不返回空值或报错
# ============================================================================

FALLBACK_DIRECTOR_OUTPUT: Dict[str, Any] = {
    "emotion": "平静",
    "focus_memories": [],
    "goal": "与玩家进行友好交谈",
    "style": "温和有礼的",
}

FALLBACK_ACTOR_OUTPUT: Dict[str, Any] = {
    "action": "站在原地，注视着玩家",
    "expression": "表情平静",
    "speech": "（角色暂时无法回应）",
}

# ============================================================================
# v1.6 新增：事件模式降级常量
# 当事件模式 LLM 调用失败时使用，确保管线不崩溃
# ============================================================================

FALLBACK_DIRECTOR_EVENT_OUTPUT: Dict[str, Any] = {
    "emotion": "平静",
    "goal": "完成当前日常安排",
    "capabilities": ["respond_normally", "complete_event(succeed)"],
    "event_attitude": "平常心对待这个日常事件",
    "plan_modifications": [],
}

FALLBACK_ACTOR_EVENT_OUTPUT: Dict[str, Any] = {
    "action": "按照日程计划完成了该做的事情",
    "speech": None,
    "expression": "表情平静",
    "dialogue_pending": None,
}


# ============================================================================
# Director：注意力聚焦模块
# ============================================================================

class DirectorModule:
    """
    注意力聚焦模块（Director）

    职责：给定角色状态 + 玩家输入，决定角色"该关注什么"。
    ——这是双 LLM 管路的第一阶段，模拟人类的"感知→关注"认知过程。

    输入 → 输出链路：
        character_name + personality + current_state
        + recent_memories + user_input
            ↓  一次 LLM 调用 (temperature=0.5, response_format=json_object)
        emotion + focus_memories + goal + style
    """

    def __init__(self):
        self.llm_service = LLMService()
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """加载 Director prompt 模板文件"""
        with open("backend/prompts/director.txt", "r", encoding="utf-8") as f:
            return f.read()

    def analyze(
        self,
        character_name: str,
        personality: Dict[str, Any],
        current_state: Dict[str, Any],
        recent_memories: List[str],
        user_input: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
        scene_context: Optional[str] = None,  # v1.6 新增：场景结构化上下文
        event_mode: bool = False,              # v1.6 新增：事件模式标志
        event_context: Optional[Dict[str, str]] = None,  # v1.6 新增：事件模式附加占位符
    ) -> Tuple[Dict[str, Any], str]:
        """
        执行注意力聚焦分析。

        Args:
            character_name:  角色名称
            personality:     人格属性字典（如 {"optimism": 70, ...}）
            current_state:   当前状态字典（如 {"location": "酒馆", ...}）
            recent_memories: 最近记忆内容列表（字符串，最多5条）
            user_input:      玩家输入文本
            history_messages: 可选的历史对话消息列表。
                传入时启用多轮模式 ——
                messages 数组会按 [system, ...history, current_user(prompt)] 顺序组装，
                LLM 能感知完整对话上下文。
                传 None 或空列表则回退到单轮（system + user）模式。
            scene_context:   v1.6 新增，结构化场景上下文文本块。
                为空时 Director 工作在"无坐标"模式（向后兼容）。
            event_mode:      v1.6 新增，True 时注入事件模式 prompt 区块。
            event_context:   v1.6 新增，事件模式的附加占位符 dict。
                包含 {today_full_schedule, event_type, event_content, personality_influence}。

        Returns:
            (parsed_data, raw_response) 元组
            - parsed_data: 校验通过后的字典 {emotion, focus_memories, goal, style}
            - raw_response: LLM 原始 JSON 字符串

        降级策略：
            LLM 调用异常 → 返回 FALLBACK_DIRECTOR_OUTPUT + 错误日志
        """
        # --- 步骤 1：组装 prompt ---
        personality_str = json.dumps(personality, ensure_ascii=False)
        current_state_str = json.dumps(current_state, ensure_ascii=False)
        memories_str = "\n".join(
            f"  - {mem}" for mem in (recent_memories or [])
        ) or "  （无最近记忆）"

        prompt = self.prompt_template.format(
            character_name=character_name,
            personality=personality_str,
            current_state=current_state_str,
            recent_memories=memories_str,
            user_input=user_input,
        )

        # v1.6 新增：注入场景上下文到 Director prompt
        if scene_context:
            prompt += "\n\n[场景上下文 - 角色所处的世界环境]\n" + scene_context

        # v1.6 新增：事件模式 prompt 区块注入
        if event_mode and event_context:
            ec = event_context
            event_block = (
                "\n\n[事件模式 - 角色正在处理一个日程事件]\n"
                f"今日全部日程（含本事件）：\n{ec.get('today_full_schedule', '(无)')}\n"
                f"当前正在处理：[{ec.get('event_type', 'schedule_action')}] {ec.get('event_content', '')}\n"
                f"\n{ec.get('personality_influence', '')}\n"
                "\n可用能力集合（必须在此集合内选择，可组合多个）：\n"
                "- respond_normally: 正常完成事件，生成叙事化行为描述\n"
                "- initiate_dialogue: 主动向玩家发起对话\n"
                "- modify_plan: 修改当前或今日后续事件\n"
                "- complete_event: 必须选择以下之一：succeed / exceed / linger / fail / skip\n"
                "\n请输出严格的 JSON（不要包含 ```json``` 标记）：\n"
                "{\n"
                '  "emotion": "角色当前情绪标签",\n'
                '  "goal": "角色处理此事件时的目标",\n'
                '  "capabilities": ["respond_normally", "complete_event(succeed)"],\n'
                '  "event_attitude": "角色对当前事件的态度（一句话）",\n'
                '  "plan_modifications": []\n'
                "}"
            )
            prompt += event_block

        # --- 步骤 2：调用 LLM ---
        # temperature=0.5 的设计考量：
        #   注意力聚焦是"决策型"任务，需要偏确定的逻辑推导。
        #   过高的温度会导致情绪标签与实际情况不匹配。
        system_prompt = (
            "你是一个专业的角色行为分析师，"
            "擅长根据上下文推导角色的心理状态和注意力焦点。"
        )

        if history_messages:
            # 多轮模式：system + 历史 user/assistant 交替 + 当前 user(prompt)
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt}
            ]
            messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            raw_response = self.llm_service.call_with_messages(
                messages=messages,
                temperature=0.5,
                response_format={"type": "json_object"},
            )
        else:
            # 单轮模式（向后兼容）
            raw_response = self.llm_service.call(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                response_format={"type": "json_object"},
            )

        # --- 步骤 3：解析并校验 ---
        parsed = self.llm_service.parse_json_response(raw_response)
        parsed = LLMService.validate_director_schema(parsed)

        return parsed, raw_response

    def analyze_with_fallback(
        self,
        character_name: str,
        personality: Dict[str, Any],
        current_state: Dict[str, Any],
        recent_memories: List[str],
        user_input: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
        scene_context: Optional[str] = None,
        event_mode: bool = False,
        event_context: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        带降级的注意力分析。

        与 analyze() 的区别：捕获异常后不向上抛，而是返回降级值。
        这是管线中的"安全网"节点，确保 Director 的失败不会阻塞 Actor。

        v1.6 新增参数与 analyze() 一致，透传即可。

        Returns:
            (parsed_data, raw_response_or_None)
            成功时 raw_response 为 LLM 原始 JSON 字符串
            降级时 raw_response 为 None（事件模式下降级到 FALLBACK_DIRECTOR_EVENT_OUTPUT）
        """
        try:
            return self.analyze(
                character_name, personality, current_state,
                recent_memories, user_input,
                history_messages=history_messages,
                scene_context=scene_context,
                event_mode=event_mode,
                event_context=event_context,
            )
        except Exception as e:
            logger.warning(
                "Director LLM 调用失败，使用降级输出: %s", e
            )
            if event_mode:
                return dict(FALLBACK_DIRECTOR_EVENT_OUTPUT), None
            return dict(FALLBACK_DIRECTOR_OUTPUT), None


# ============================================================================
# Actor：行为生成模块
# ============================================================================

class ActorModule:
    """
    行为生成模块（Actor）

    职责：根据 Director 聚焦结果，生成角色的具体动作/表情/语言。
    ——这是双 LLM 管路的第二阶段，模拟人类的"关注→表达"行为过程。

    输入 → 输出链路：
        character_name + personality
        + emotion + focus_memories + goal + style  ← 来自 Director
        + user_input
            ↓  一次 LLM 调用 (temperature=0.8, response_format=json_object)
        action + expression + speech
    """

    def __init__(self):
        self.llm_service = LLMService()
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """加载 Actor prompt 模板文件"""
        with open("backend/prompts/actor.txt", "r", encoding="utf-8") as f:
            return f.read()

    def generate(
        self,
        character_name: str,
        personality: Dict[str, Any],
        emotion: str,
        focus_memories: List[str],
        goal: str,
        style: str,
        user_input: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
        event_mode: bool = False,  # v1.6 新增：事件模式标志
        scene_context: Optional[str] = None,  # v1.6.fix 新增：场景上下文
    ) -> Tuple[Dict[str, Any], str]:
        """
        生成角色行为（动作 + 表情 + 语言）。

        Args:
            character_name:  角色名称
            personality:     人格属性字典
            emotion:         Director 输出的情绪标签
            focus_memories:  Director 筛选的关键记忆
            goal:            Director 设定的对话目标
            style:           Director 确定的回复风格
            user_input:      玩家输入文本
            history_messages: 可选的历史对话消息列表。
            event_mode:      v1.6 新增，True 时注入事件模式约束区块。
            scene_context:   v1.6.fix 新增，场景上下文文本。
                让 Actor 生成行为时感知角色所处环境，
                产出的 action 能自然地融入场景元素。

        Returns:
            (parsed_data, raw_response) 元组

        降级策略：
            LLM 调用异常 → 返回 FALLBACK_ACTOR_OUTPUT + 错误日志
        """
        # --- 步骤 1：组装 prompt ---
        personality_str = json.dumps(personality, ensure_ascii=False)
        memories_str = "\n".join(
            f"  - {mem}" for mem in (focus_memories or [])
        ) or "  （无特殊关注的记忆）"

        safe_dict = collections.defaultdict(str, {
            "character_name": character_name,
            "personality": personality_str,
            "emotion": emotion,
            "focus_memories": memories_str,
            "goal": goal,
            "style": style,
            "user_input": user_input,
            "scene_context": scene_context or "  （暂无场景信息）",  # v1.6.fix
        })

        # 使用 string.Template 风格安全性建 prompt
        prompt = self.prompt_template
        for key, val in safe_dict.items():
            prompt = prompt.replace("{" + key + "}", val)

        # v1.6 新增：事件模式约束区块注入
        if event_mode:
            prompt += (
                "\n\n[事件模式约束]\n"
                "- speech 可为 null（当无对话对象时）\n"
                "- action 承担主要叙事输出，描述事件执行的具体过程\n"
                "- dialogue_pending 仅当 Director 选择 initiate_dialogue 时存在\n"
                "- expression 描述角色执行事件时的表情\n"
                "\n请输出严格的 JSON（不要包含 ```json``` 标记）：\n"
                "{\n"
                '  "action": "第三人称叙事描述，角色如何处理该事件",\n'
                '  "speech": null,\n'
                '  "expression": "角色表情描述",\n'
                '  "dialogue_pending": null\n'
                "}"
            )

        # --- 步骤 2：调用 LLM ---
        # temperature=0.8 的设计考量：
        #   行为生成是"创意型"任务，需要一定的随机性来产生多样的回复。
        #   但也不宜超过 0.9，否则可能产生不符合角色设定的内容。
        system_prompt = (
            "你是一个沉浸式角色扮演系统，"
            "你能精准地根据角色的情绪、记忆和目标生成自然的动作和对话。"
        )

        if history_messages:
            # 多轮模式：system + 历史 user/assistant 交替 + 当前 user(prompt)
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt}
            ]
            messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            raw_response = self.llm_service.call_with_messages(
                messages=messages,
                temperature=0.8,
                response_format={"type": "json_object"},
            )
        else:
            # 单轮模式（向后兼容）
            raw_response = self.llm_service.call(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.8,
                response_format={"type": "json_object"},
            )

        # --- 步骤 3：解析并校验 ---
        parsed = self.llm_service.parse_json_response(raw_response)
        parsed = LLMService.validate_actor_schema(parsed)

        return parsed, raw_response

    def generate_with_fallback(
        self,
        character_name: str,
        personality: Dict[str, Any],
        emotion: str,
        focus_memories: List[str],
        goal: str,
        style: str,
        user_input: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
        event_mode: bool = False,
        scene_context: Optional[str] = None,  # v1.6.fix 新增
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        带降级的行为生成。所有参数透传给 generate()。
        """
        try:
            return self.generate(
                character_name, personality, emotion,
                focus_memories, goal, style, user_input,
                history_messages=history_messages,
                event_mode=event_mode,
                scene_context=scene_context,
            )
        except Exception as e:
            logger.warning(
                "Actor LLM 调用失败，使用降级输出: %s", e
            )
            if event_mode:
                return dict(FALLBACK_ACTOR_EVENT_OUTPUT), None
            return dict(FALLBACK_ACTOR_OUTPUT), None


# ============================================================================
# InteractionPipeline：对话管线编排层
# ============================================================================

# ============================================================================
# v1.6 B4：人格加权函数（模块级纯函数，无外部依赖）
# ============================================================================

# 人格维度常量（与 Growth 模块保持一致）
_PERSONALITY_DIMS = [
    "courage", "intelligence", "sociability",
    "empathy", "loyalty", "optimism",
]

def compute_personality_influence(
    personality: Dict[str, int],
    event_type: str = "schedule_action",
) -> str:
    """
    基于角色 6 维人格计算事件处理倾向的自然语言引导。

    核心数学逻辑：
      1. 6 维人格归一化到 [0, 1]
      2. 计算 5 种 complete_event 子类型权重：
         - succeed:  勇气 + 智力（主动解决问题的能力）
         - exceed:   智力 + 乐观（超额完成的创造力与积极心态）
         - linger:   (1-勇气) + (1-社交)（犹豫不决/回避倾向）
         - fail:     (1-智力) + (1-勇气)（认知与行动能力不足）
         - skip:     (1-忠诚) + (1-同理心)（不在意/不关心）
      3. 计算 2 个能力偏置：
         - dialogue_bias:  社交 + 乐观（主动交流倾向）
         - modify_bias:    智力 − 忠诚（计划修改倾向——聪明但不一定守规矩）
      4. 归一化后格式化为自然语言引导字符串

    设计考量（为什么是"引导"而非"决策"）：
      函数不替代 Director——它只提供"倾向建议"。
      Director 保有最终决策权，可以在 prompt 中覆盖此建议。
      此设计与双 LLM 管路的核心哲学一致：
      "概率引导在管道上游提供压缩上下文，LLM 在下游做最终判断。"

    Args:
        personality: 6 维人格字典，值域 [0, 100]
        event_type:  事件类型字符串（保留以备未来扩展，当前未使用）

    Returns:
        自然语言引导文本，可直接注入 Director prompt
    """
    if not personality or not isinstance(personality, dict):
        return "（无足够人格数据用于计算倾向）"

    # --- 步骤 1：提取 6 维人格，缺省值 50 ---
    dims = {}
    for dim in _PERSONALITY_DIMS:
        try:
            val = float(personality.get(dim, 50))
        except (ValueError, TypeError):
            val = 50.0
        dims[dim] = max(0.0, min(100.0, val))

    # --- 步骤 2：归一化到 [0, 1] ---
    c = dims["courage"] / 100.0       # 勇气
    i = dims["intelligence"] / 100.0  # 智力
    s = dims["sociability"] / 100.0   # 社交
    e = dims["empathy"] / 100.0       # 同理心
    l = dims["loyalty"] / 100.0       # 忠诚
    o = dims["optimism"] / 100.0      # 乐观

    # --- 步骤 3：5 种 complete_event 子类型原始权重 ---
    raw_weights = {
        "succeed": (c + i) / 2.0,
        "exceed":  (i + o) / 2.0,
        "linger":  ((1.0 - c) + (1.0 - s)) / 2.0,
        "fail":    ((1.0 - i) + (1.0 - c)) / 2.0,
        "skip":    ((1.0 - l) + (1.0 - e)) / 2.0,
    }

    # Softmax 归一化为百分比
    import math
    exp_weights = {k: math.exp(v * 3.0) for k, v in raw_weights.items()}
    total_exp = sum(exp_weights.values())
    pct_weights = {k: v / total_exp for k, v in exp_weights.items()}

    # --- 步骤 4：2 个能力偏置 ---
    dialogue_bias_raw = (s + o) / 2.0
    modify_bias_raw = max(0.0, (i - l))  # 智力高+忠诚低 → 更愿意改计划

    # 偏置映射到自然语言等级
    def _bias_to_label(val: float) -> str:
        if val > 0.65:
            return "高"
        elif val > 0.35:
            return "中"
        else:
            return "低"

    # --- 步骤 5：格式化为自然语言引导 ---
    # 找出 top 2 子类型
    sorted_pcts = sorted(pct_weights.items(), key=lambda x: x[1], reverse=True)
    top_lines = ", ".join(
        f"{k}={v*100:.0f}%" for k, v in sorted_pcts
    )

    lines = [
        "人格倾向分析（由系统计算，仅作引导参考）：",
        f"你的性格特征：勇气{dims['courage']:.0f} 智力{dims['intelligence']:.0f} "
        f"社交{dims['sociability']:.0f} 同理心{dims['empathy']:.0f} "
        f"忠诚{dims['loyalty']:.0f} 乐观{dims['optimism']:.0f}",
        f"事件完成倾向：{top_lines}",
        f"主动对话倾向：{_bias_to_label(dialogue_bias_raw)}",
        f"修改计划倾向：{_bias_to_label(modify_bias_raw)}",
        "",
        "注意：以上仅为统计分析，你仍需根据当前事件的具体内容做出最终判断。",
    ]

    return "\n".join(lines)


# ============================================================================
# InteractionPipeline：对话管线编排层
# ============================================================================

class InteractionPipeline:
    """
    对话管线编排层

    职责：
      1. 数据库读取（角色 → 记忆 → 历史对话）
      2. 数据组装（字典反序列化、列表提取）
      3. 串联 Director → Actor 两步 LLM 调用
      4. 持久化对话记录到数据库
      5. 返回完整 ChatResponse

    管线节点依赖图（→ 表示数据流方向）：

        character_crud.get_character ────┐
        memory_crud.get_character_memories ─┤──□ Director.analyze()
        conversation_crud.get_character_conversations ─┘     │
                                                             ▼
                                                      Actor.generate()
                                                             │
                                                             ▼
                                            conversation_crud.create()

    设计考量：
      - Pipeline 本身不调用 LLM，LLM 调用封装在 Director/Actor 中
      - Pipeline 仅负责"读数据 → 协调调用 → 写数据"的编排逻辑
      - 这样保证了单一职责：模块内聚 LLM 调用，管线负责流程
    """

    def __init__(self):
        """初始化 Director 和 Actor 实例（两个 LLM 子模块）"""
        self.director = DirectorModule()
        self.actor = ActorModule()

    @staticmethod
    def _safe_load_json(raw: Optional[str]) -> dict:
        """安全地将数据库中的 JSON 字符串转为 dict"""
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _build_scene_context(character: Any, db: Session) -> str:
        """
        v1.6 A1：组装场景结构化上下文文本。

        从数据库读取角色当前所在场景的完整层级路径、相邻场景、最近变化，
        并格式化为紧凑文本块，供 Director prompt 注入。

        Token 预算控制：
          - 场景路径最多 6 层（约 200 字）
          - 相邻场景限 5 条（约 150 字）
          - 最近变化限 3 条（约 200 字）
          - 总计约 550 字，在 Director 的 token 预算内安全

        Args:
            character: 已加载的 Character ORM 对象（避免重复查 DB）
            db: 数据库会话

        Returns:
            结构化场景文本块；角色无场景信息时返回空字符串
        """
        scene_id = getattr(character, "current_scene_id", None)
        if not scene_id:
            return ""

        lines = []

        # 1) 场景完整路径（面包屑导航）
        try:
            path = scene_crud.get_scene_path(db, scene_id)
            if path:
                path_str = " > ".join(
                    f"{s.name}({s.scene_type or s.scene_layer})"
                    for s in path
                )
                lines.append(f"当前位置（完整路径）：{path_str}")
                # 获取最深层实际场景的描述
                current_scene = path[-1]
                if current_scene.description:
                    lines.append(f"当前场景描述：{current_scene.description}")
        except Exception as e:
            logger.debug("构建场景路径失败: %s", e)

        # 2) 相邻已知场景（兄弟节点）
        try:
            adjacent = scene_crud.get_adjacent_scenes(db, scene_id)
            if adjacent:
                adj_names = ", ".join(
                    f"{s.name}({s.scene_type or 'location'})"
                    for s in adjacent[:5]
                )
                lines.append(f"相邻已知场所：{adj_names}")
        except Exception as e:
            logger.debug("获取相邻场景失败: %s", e)

        # 3) 最近场景变化
        try:
            changes = scene_change_crud.get_recent_changes(db, scene_id, limit=3)
            if changes:
                change_lines = [
                    f"  · Day {ch.day_number}: {ch.description}"
                    for ch in changes
                ]
                lines.append("最近场景变化：\n" + "\n".join(change_lines))
        except Exception as e:
            logger.debug("获取场景变化失败: %s", e)

        return "\n".join(lines) if lines else ""

    @staticmethod
    def _build_history_messages(
        conversations: List[Any],
        max_turns: int = 10,
    ) -> List[Dict[str, str]]:
        """
        把数据库中最近 N 条对话记录组装为 OpenAI 风格的 messages 数组。

        数据结构（OpenAI 格式）：
            [
              {"role": "user",      "content": <user_input>},
              {"role": "assistant", "content": <npc_response>},
              ... 交替 ...
            ]

        Args:
            conversations: Conversation ORM 对象列表（按时间升序）。
                          调用方需自行做"取最近 N 条"的截断。
            max_turns: 最多保留多少轮（每轮 = 1 user + 1 assistant）。
                       截断采用"保留最近 N 轮"策略：取列表尾部而非头部，
                       避免最早的对话覆盖最近的语义。

        Returns:
            OpenAI 风格 messages 数组（不含 system，由调用方追加在最前）。
            空列表表示无历史。

        健壮性设计：
          - 一轮对话必须 user_input *和* npc_response 都非空才保留。
            原因：OpenAI messages 必须 user/assistant 严格交替，
            若只保留一侧会导致连续同角色消息，触发 API 报错或语义混乱。
          - 跳过"半轮"（任一字段为空）—— 在脏数据或部分写入失败时保护 LLM 调用。
        """
        if not conversations or max_turns <= 0:
            return []

        # 截断到最近 N 轮
        recent = conversations[-max_turns:]

        history: List[Dict[str, str]] = []
        for conv in recent:
            user_text = (conv.user_input or "").strip()
            npc_text = (conv.npc_response or "").strip()
            # 严格成对：两端都有非空内容才纳入 messages
            if user_text and npc_text:
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": npc_text})
        return history

    def run(
        self,
        character_id: int,
        user_message: str,
        db: Session,
        history_turns: int = 10,
        session_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        运行完整的对话管线。

        Args:
            character_id:  角色 ID
            user_message:  玩家输入文本
            db:            SQLAlchemy 数据库会话
            history_turns: 注入到 LLM messages 的最近对话轮数。
                           默认 10 轮 = 20 条消息；
                           设为 0 即可禁用多轮模式（回退到单轮）。
            session_id:    会话 ID（None → 自动创建新 session 并用首条消息做标题）

        Returns:
            字典，包含以下字段，可直接用于 ChatResponse schema：
            {
                "id": int,               # 对话记录 ID
                "character_id": int,
                "user_input": str,
                "npc_response": str,     # 角色的语言回复
                "emotion": str,          # Director 输出的情绪
                "action": str,           # Actor 输出的动作
                "expression": str,       # Actor 输出的表情
                "director_raw": str|None,# Director LLM 原始响应
                "actor_raw": str|None,   # Actor LLM 原始响应
                "timestamp": datetime,
                "session_id": int,       # 新增：所属会话
                "session_title": str,    # 新增：会话标题（前端可立即更新侧栏）
            }

        Raises:
            ValueError: 角色不存在时抛出
        """
        # ---- 节点 1：获取角色基础数据 ----
        character = character_crud.get_character(db, character_id)
        if not character:
            raise ValueError(f"角色不存在: id={character_id}")

        personality = self._safe_load_json(character.personality)
        current_state = self._safe_load_json(character.current_state)

        # ---- 节点 2：获取最近记忆（最多 5 条） ----
        # 设计考量：限制 5 条是 prompt token 预算与上下文丰富度之间的平衡点。
        # 5 条记忆 + 其他变量 ≈ 总 token < 2000，确保在模型上下文限制内安全。
        recent_memories = memory_crud.get_character_memories(
            db, character_id, limit=5
        )
        memory_texts = [mem.content for mem in recent_memories]

        # ---- 节点 2.4：获取/创建会话（多轮消息的容器） ----
        #   - session_id 传了就复用（角色不匹配时降级为创建新 session）
        #   - 没传就创建一个新 session，标题取首条消息前 30 字
        #   - 必须在"取历史消息"之前完成，否则会把"上一会话"的内容串味到新会话
        from backend.services import chat_session_crud
        session = chat_session_crud.get_or_create_session(
            db, session_id=session_id, character_id=character_id,
            first_user_message=user_message,
        )
        session_id = session.id
        session_title = session.title

        # ---- 节点 2.5：组装多轮历史消息 ----
        #   从当前 session 取最近 N 轮对话，按时间升序拼接为 OpenAI 风格 messages。
        #   重要：必须在持久化新对话 *之前* 取历史，否则会把"当前轮"也塞回去造成重复。
        history_messages: List[Dict[str, str]] = []
        if history_turns and history_turns > 0:
            # 优先用 session 级历史（更聚焦），但若 session 为空且没有显式 session_id
            # 则退回到角色级历史，避免首次进入"默认会话"时空白
            recent_conversations = conversation_crud.get_session_conversations(
                db, session_id=session_id, limit=history_turns,
            )
            if not recent_conversations and session_id is None:
                recent_conversations = conversation_crud.get_character_conversations(
                    db, character_id, skip=0, limit=history_turns,
                )
            history_messages = self._build_history_messages(
                recent_conversations, max_turns=history_turns,
            )
            if history_messages:
                logger.info(
                    "InteractionPipeline: 注入 %d 条历史消息（%d 轮）",
                    len(history_messages), len(history_messages) // 2,
                )

        # ---- 节点 3：执行 Director 注意力聚焦 ----
        # v1.6 A1：构建场景上下文并注入
        scene_context = self._build_scene_context(character, db)

        # 使用带降级的版本，确保 LLM 失败时管线不崩溃
        director_data, director_raw = self.director.analyze_with_fallback(
            character_name=character.name,
            personality=personality,
            current_state=current_state,
            recent_memories=memory_texts,
            user_input=user_message,
            history_messages=history_messages or None,
            scene_context=scene_context or None,
        )

        # ---- 节点 4：执行 Actor 行为生成 ----
        # Actor 接收 Director 的完整输出 + 场景上下文
        actor_data, actor_raw = self.actor.generate_with_fallback(
            character_name=character.name,
            personality=personality,
            emotion=director_data["emotion"],
            focus_memories=director_data["focus_memories"],
            goal=director_data["goal"],
            style=director_data["style"],
            user_input=user_message,
            history_messages=history_messages or None,
            scene_context=scene_context or None,  # v1.6.fix
        )

        # ---- 节点 5：持久化对话记录（带 session_id） ----
        conversation = conversation_crud.create_conversation(
            db=db,
            character_id=character_id,
            user_input=user_message,
            npc_response=actor_data["speech"],
            emotion=director_data["emotion"],
            action=actor_data["action"],
            expression=actor_data["expression"],
            director_raw=director_raw,
            actor_raw=actor_raw,
            session_id=session_id,
        )

        # 刷新 session.updated_at，让活跃会话在侧栏里排前面
        chat_session_crud.touch_session(db, session_id)

        # ---- 节点 6：返回结果 ----
        return {
            "id": conversation.id,
            "character_id": character_id,
            "user_input": user_message,
            "npc_response": actor_data["speech"],
            "emotion": director_data["emotion"],
            "action": actor_data["action"],
            "expression": actor_data["expression"],
            "director_raw": director_raw,
            "actor_raw": actor_raw,
            "timestamp": conversation.timestamp,
            "session_id": session_id,
            "session_title": session_title,
        }

    # =========================================================================
    # v1.6 B1：事件处理入口 — run_event()
    # =========================================================================

    @staticmethod
    def _format_today_schedule(character_id: int, day_number: int, db: Session) -> str:
        """
        格式化今日全部日程为人类可读文本（含状态标记）。

        用于事件模式 Director prompt 注入，让 LLM 了解当前事件在整个日程中的位置。

        设计考量：
          - 标记当前事件为 [← 当前处理中] 帮助 Director 定位
          - 显示已完成/待处理状态，让 Director 感知当天进度
          - 同一事件可能有不同 agent（schedule_action 标记为"日程"等）
        """
        events = event_crud.get_events_by_day(db, character_id, day_number)
        if not events:
            return "  （今日无安排）"

        lines = []
        for ev in events:
            period = f" [{ev.time_period}]" if getattr(ev, "time_period", None) else ""
            status_mark = ""
            if getattr(ev, "status", "") == "completed":
                status_mark = " ✅"
            elif getattr(ev, "status", "") == "pending":
                status_mark = " ⏳"
            lines.append(
                f"  #{ev.order_index}{period} {ev.event_type}{status_mark}: {ev.content}"
            )
        return "\n".join(lines)

    def run_event(
        self,
        event: Any,           # Event ORM 对象
        character: Any,       # Character ORM 对象（避免重复查 DB）
        db: Session,
    ) -> Dict[str, Any]:
        """
        v1.6 B1：事件处理入口 — 将日程事件纳入 Director+Actor 双 LLM 管线。

        核心流程（8 步）：
          1. 计算人格加权引导 → compute_personality_influence()
          2. 读取 Scene 上下文 → _build_scene_context()
          3. 读取今日全部日程 → _format_today_schedule()
          4. 组装 Director 事件模式 prompt
          5. Director.analyze(event_mode=True) → emotion/goal/capabilities/attitude
          6. Actor.generate(event_mode=True) → action/speech/expression/dialogue_pending
          7. 校验事件模式输出
          8. 返回 pipeline_result（含 capabilities_applied）

        设计考量：
          - 复用现有 DirectorModule/ActorModule 实例，仅 prompt 参数不同
          - 能力集白名单在 Director prompt 中定义，校验在 llm_service 中
          - plan_modifications 不在此处持久化，返回给调用方（main.py）处理

        Args:
            event:     待处理的 Event ORM 对象（必须 status=pending）
            character: 已加载的 Character ORM 对象
            db:        SQLAlchemy 数据库会话

        Returns:
            {
                "action": str,              # Actor 行为叙事
                "speech": Optional[str],    # Actor 语言（可能为 None）
                "expression": str,          # Actor 表情
                "emotion": str,             # Director 情绪
                "goal": str,                # Director 当前目标
                "capabilities": list,       # Director 选择的能力列表
                "event_attitude": str,      # Director 对事件的态度
                "plan_modifications": list, # Director 的日程/目标修改计划
                "dialogue_pending": dict|None,  # Actor 待处理对话
                "director_raw": str|None,   # Director LLM 原始响应
                "actor_raw": str|None,      # Actor LLM 原始响应
            }
        """
        # ---- 步骤 1：计算人格加权引导（B4） ----
        personality = self._safe_load_json(character.personality)
        event_type = getattr(event, "event_type", "schedule_action") or "schedule_action"
        personality_influence = compute_personality_influence(
            personality, event_type
        )

        # ---- 步骤 2：读取 Scene 上下文（A1） ----
        scene_context = self._build_scene_context(character, db)

        # ---- 步骤 3：读取今日全部日程 ----
        day_number = getattr(event, "day_number", getattr(character, "day_number", 1)) or 1
        today_full_schedule = self._format_today_schedule(
            getattr(character, "id", 0), day_number, db
        )

        # ---- 步骤 4：组装 Director 事件模式 context ----
        event_context = {
            "today_full_schedule": today_full_schedule,
            "event_type": event_type,
            "event_content": getattr(event, "content", "") or "",
            "personality_influence": personality_influence,
        }

        # ---- 步骤 5：Director 事件模式分析 ----
        # 使用"事件内容"作为 user_input，让 Director 聚焦于这个具体事件
        event_description = (
            f"[{event_type}] {getattr(event, 'content', '未知事件')}"
        )

        director_data, director_raw = self.director.analyze_with_fallback(
            character_name=getattr(character, "name", "角色"),
            personality=personality,
            current_state=self._safe_load_json(
                getattr(character, "current_state", None)
            ),
            recent_memories=[],  # 事件模式暂不注入记忆（避免 token 超预算）
            user_input=event_description,
            scene_context=scene_context or None,
            event_mode=True,
            event_context=event_context,
        )

        # 事件模式 style 字段被 capabilities+event_attitude 替代，按降级值处理
        style_value = director_data.get("style", "自然的")
        capabilities = director_data.get("capabilities", ["respond_normally", "complete_event(succeed)"])
        event_attitude = director_data.get("event_attitude", "平常心对待此事件")
        plan_modifications = director_data.get("plan_modifications", [])

        # 确保 capabilities 和 plan_modifications 是列表
        if not isinstance(capabilities, list):
            capabilities = ["respond_normally", "complete_event(succeed)"]
        if not isinstance(plan_modifications, list):
            plan_modifications = []

        # ---- 步骤 6：Actor 事件模式行为生成 ----
        actor_data, actor_raw = self.actor.generate_with_fallback(
            character_name=getattr(character, "name", "角色"),
            personality=personality,
            emotion=director_data.get("emotion", "平静"),
            focus_memories=director_data.get("focus_memories", []),
            goal=director_data.get("goal", "完成当前事件"),
            style=style_value,
            user_input=event_description,
            event_mode=True,
            scene_context=scene_context or None,  # v1.6.fix
        )

        # ---- 步骤 7：校验事件模式输出 ----
        # 使用 LLMService 的专用校验函数
        capabilities = LLMService.validate_event_capabilities(capabilities)
        action = actor_data.get("action", "处理了当前事件")
        speech = actor_data.get("speech")
        expression = actor_data.get("expression", "表情平静")
        dialogue_pending = actor_data.get("dialogue_pending")

        # ---- 步骤 8：返回 pipeline_result ----
        return {
            "action": action,
            "speech": speech,  # 可能为 None（无对话对象时）
            "expression": expression,
            "emotion": director_data.get("emotion", "平静"),
            "goal": director_data.get("goal", "完成当前事件"),
            "capabilities": capabilities,
            "event_attitude": event_attitude,
            "plan_modifications": plan_modifications,
            "dialogue_pending": dialogue_pending,
            "director_raw": director_raw,
            "actor_raw": actor_raw,
        }
