"""
Day4 — 事件驱动角色成长模块（Growth Module，重写）

设计变更（相对于 Day3）：
  旧：从对话记录（Conversation 表）读取角色经历
  新：从事件记录（Event 表，status=completed）读取角色经历
      + 输出事件日程（schedule 数组）供下一日使用

核心数据流：
  Growth.observe(今日 completed 事件列表)
    → LLM 分析：人格变化 + 新记忆 + 次日事件日程 + 世界变化
    → 持久化：growth_log / memories / character 更新
    → 返回：schedule 数组（由 main.py 写入 events 表）

输入 → 输出链路：
    character_name + personality + events_today(content+result_json)
        ↓  一次 LLM 调用 (temperature=0.5, response_format=json_object)
    {
      personality_delta: {...},
      new_memories: [...],
      event_summary: "...",
      schedule: [{content, event_type, time_period, order_index}, ...],
      world_changes: "..."
    }

设计考量（为什么改为事件驱动）：
  1. 明确观测粒度：事件是"有意义的行为单元"，对话记录包含大量
     日常闲聊（"你好""再见"），直接喂给 LLM 浪费 token 且引入噪声。
  2. 执行回执（result_json）：让 LLM 知道事件实际如何收场，
     而非仅看到事件描述。"事件完成了 vs 事件被跳过了"对人格影响不同。
  3. 日程生成：Growth 同时兼任"编剧"角色——基于今天的变化，
     推演明天角色可能做什么。这是"事件推进轴"的核心机制。
"""
import json
import logging
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session

from backend.services.llm_service import LLMService
from backend.crud import character as character_crud
from backend.crud import memory as memory_crud
from backend.crud import growth as growth_crud
from backend.crud import event as event_crud
from backend.crud import scene as scene_crud             # v1.6 Phase 1: Scene 上下文
from backend.crud import scene_change as scene_change_crud  # v1.6 Phase 1: 场景变化

logger = logging.getLogger(__name__)


# ============================================================================
# 人格维度常量
# ============================================================================

PERSONALITY_DIMENSIONS = [
    "optimism", "courage", "empathy",
    "loyalty", "intelligence", "sociability"
]


class GrowthModule:
    """
    事件驱动角色成长模块（Pipeline 模式，Day4 重写）

    职责：分析角色今日经历（事件），
          推导人格变化、提炼新记忆、生成次日日程。

    与 Day3 的关键区别：
      - run() 改为接受 events 列表（已完成的事件）
      - 改用 validate_growth_schema_v2（含 schedule/world_changes 校验）
      - 返回 schedule 数组供调用方写入 events 表
    """

    def __init__(self):
        self.llm_service = LLMService()
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """加载 Growth prompt 模板文件"""
        with open("backend/prompts/growth.txt", "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _safe_load_json(raw: Optional[str]) -> dict:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _build_scene_context(character: Any, db: Session) -> str:
        """
        v1.6 A2：组装场景上下文文本注入 Growth prompt。

        与 InteractionPipeline._build_scene_context() 逻辑一致，
        但独立定义以保持 GrowthModule 的自包含性。

        Args:
            character: Character ORM 对象
            db: 数据库会话

        Returns:
            格式化的场景上下文字符串，无场景信息时返回空字符串
        """
        scene_id = getattr(character, "current_scene_id", None)
        if not scene_id:
            return "  （暂无场景信息）"

        lines = []

        # 场景完整路径
        try:
            path = scene_crud.get_scene_path(db, scene_id)
            if path:
                path_str = " > ".join(
                    f"{s.name}({s.scene_type or s.scene_layer})"
                    for s in path
                )
                lines.append(f"当前位置：{path_str}")
                current_scene = path[-1]
                if current_scene.description:
                    lines.append(f"场景描述：{current_scene.description}")
        except Exception:
            pass

        # 相邻场景
        try:
            adjacent = scene_crud.get_adjacent_scenes(db, scene_id)
            if adjacent:
                adj_names = ", ".join(s.name for s in adjacent[:5])
                lines.append(f"可前往的场所：{adj_names}")
        except Exception:
            pass

        # 最近场景变化
        try:
            changes = scene_change_crud.get_recent_changes(db, scene_id, limit=3)
            if changes:
                change_lines = [
                    f"  · Day {ch.day_number}: {ch.description}"
                    for ch in changes
                ]
                lines.append("最近场景变化：\n" + "\n".join(change_lines))
        except Exception:
            pass

        return "\n".join(lines) if lines else "  （暂无场景信息）"

    def _format_events_today(
        self, events: List[Any]
    ) -> str:
        """
        将事件列表格式化为 LLM 可读的文本。

        格式：
          [日程#1 上午] 事件描述
            执行结果：result_json（已完成事件的回执）
          [日程#2 下午] 事件描述
            执行结果：...

        设计考量：
          - 每个事件包含 content + result_json，让 LLM 看出"发生了什么"和"结果如何"
          - 标记 time_period 帮助 LLM 理解时间线
          - 事件按 order_index 排列，保持时间序
        """
        if not events:
            return "  （今日无已完成事件）"

        lines = []
        for ev in events:
            period = f" [{ev.time_period}]" if ev.time_period else ""
            prefix = f"[{ev.event_type}{period} #{ev.order_index}]"
            lines.append(f"  {prefix} {ev.content}")
            if ev.result_json:
                lines.append(f"    执行结果: {ev.result_json}")

        return "\n".join(lines) if lines else "  （今日无已完成事件）"

    def _calculate_new_personality(
        self,
        old_personality: Dict[str, int],
        delta: Dict[str, int]
    ) -> Dict[str, int]:
        """
        计算新人格 = 旧人格 + delta（与 Day3 相同）。

        钳位逻辑：确保每个属性值在 [0, 100] 范围内，
        防止累积误差导致属性值越界。
        """
        new_personality = {}
        for dim in PERSONALITY_DIMENSIONS:
            old_val = old_personality.get(dim, 50)
            delta_val = delta.get(dim, 0)
            new_val = max(0, min(100, old_val + delta_val))
            new_personality[dim] = new_val
        return new_personality

    def run(
        self,
        character_id: int,
        db: Session,
        events: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        运行事件驱动成长管线。

        Args:
            character_id: 角色 ID
            db: SQLAlchemy 数据库会话
            events: 可选，当日已完成的 Event 对象列表。
                    不传时自动从数据库读取当天所有 completed 事件。

        Returns:
            {
                "growth_log_id": int,
                "character_id": int,
                "personality_delta": str (JSON),
                "event_summary": str,
                "new_memories": str (JSON 数组),
                "schedule_json": str (JSON 数组 — 次日事件实体),
                "world_changes_json": str (JSON 对象),
                "growth_raw": str,
                "created_at": datetime,
            }

        Raises:
            ValueError: 角色不存在时抛出
        """
        # ---- 节点 1：读取角色当前状态 ----
        character = character_crud.get_character(db, character_id)
        if not character:
            raise ValueError(f"角色不存在: id={character_id}")

        old_personality = self._safe_load_json(character.personality)
        for dim in PERSONALITY_DIMENSIONS:
            if dim not in old_personality:
                old_personality[dim] = 50

        current_day = character.day_number or 1

        # ---- 节点 2：获取当日事件 ----
        if events is None:
            events = event_crud.get_events_by_day(
                db, character_id, current_day, status_filter="completed",
            )
        events_text = self._format_events_today(events)

        # 也收集角色的当前世界观信息（speaking_style 等可选的上下文信息）
        speaking_style_str = character.speaking_style or "[]"
        values_str = character.values or "[]"
        habits_str = character.habits or "[]"
        long_term_goal_str = character.long_term_goal or ""

        # ---- Step 14 新增（v1.6 Phase 3）：注入短期目标到 Growth prompt ----
        # 仅注入未完成的目标（progress < 1.0），让 Growth LLM 知道角色当前在追求什么。
        all_goals = self._safe_load_json(character.short_term_goals) if character.short_term_goals else []
        if not isinstance(all_goals, list):
            all_goals = []
        active_goals = [
            g for g in all_goals
            if isinstance(g, dict) and float(g.get("progress", 0.0)) < 1.0
        ]
        short_term_goals_str = json.dumps(active_goals, ensure_ascii=False) if active_goals else "[]"

        # ---- v1.6 A2：构建场景上下文注入到 Growth prompt ----
        # 让 Growth LLM 在生成次日日程时感知角色当前身处何地，
        # 避免生成与场景状态不匹配的日程（如在"矿洞"中生成"去酒馆喝酒"）
        scene_context_str = self._build_scene_context(character, db)

        # ---- 节点 3：组装 prompt → 调用 Growth LLM ----
        personality_str = json.dumps(old_personality, ensure_ascii=False)

        prompt = self.prompt_template.format(
            character_name=character.name,
            personality=personality_str,
            events_today=events_text,
            speaking_style=speaking_style_str,
            values=values_str,
            habits=habits_str,
            long_term_goal=long_term_goal_str,
            short_term_goals_active=short_term_goals_str,  # Step 14 新增：注入活跃短期目标
            scene_context=scene_context_str,  # v1.6 A2 新增：注入场景上下文
        )

        system_prompt = (
            "你是一个专业的角色成长分析师与编剧，"
            "擅长根据角色今日的事件经历推导其人格变化，"
            "并为角色规划明日的日程事件。"
        )

        raw_response = self.llm_service.call(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            response_format={"type": "json_object"},
        )

        # ---- 节点 4：解析并校验（v2 版本，含 schedule/world_changes）----
        parsed = self.llm_service.parse_json_response(raw_response)
        parsed = LLMService.validate_growth_schema_v2(parsed)

        personality_delta = parsed["personality_delta"]
        new_memories = parsed["new_memories"]
        event_summary = parsed["event_summary"]
        schedule = parsed.get("schedule", [])
        world_changes = parsed.get("world_changes", "")

        # ---- Step 14 新增（v1.6 Phase 3）：处理短期目标更新 ----
        # Growth LLM 产出 goal_updates（进度调整）+ new_goals（新目标生成），
        # 应用层合并到 character.short_term_goals JSON 数组中持久化。
        goal_updates = parsed.get("goal_updates", [])
        if not isinstance(goal_updates, list):
            goal_updates = []
        new_goals = parsed.get("new_goals", [])
        if not isinstance(new_goals, list):
            new_goals = []

        # 应用 goal_updates 到活跃目标
        updated_all_goals = list(all_goals)  # 复制全部目标（含已完成的）
        for update in goal_updates:
            if not isinstance(update, dict):
                continue
            try:
                idx = int(update.get("index", -1))
                new_progress = float(update.get("new_progress", 0.0))
                new_progress = max(0.0, min(1.0, new_progress))
            except (ValueError, TypeError):
                continue
            # 索引映射：goal_updates 中的 index 对应 active_goals 的位置，
            # 需要在 all_goals 中找到对应条目并更新 progress
            if 0 <= idx < len(active_goals):
                target_goal = active_goals[idx]
                for g in updated_all_goals:
                    if (isinstance(g, dict)
                            and g.get("goal") == target_goal.get("goal")
                            and g.get("created_day") == target_goal.get("created_day")):
                        g["progress"] = new_progress
                        break

        # 追加 new_goals（Growth 生成的新目标）
        for ng in new_goals:
            if not isinstance(ng, dict):
                continue
            goal_text = ng.get("goal", "")
            if not isinstance(goal_text, str) or not goal_text.strip():
                continue
            # 不重复添加：检查是否已存在相同内容的目标
            existing_texts = {
                g["goal"] for g in updated_all_goals
                if isinstance(g, dict) and "goal" in g
            }
            if goal_text.strip() not in existing_texts:
                updated_all_goals.append({
                    "goal": goal_text.strip(),
                    "progress": 0.0,
                    "created_day": current_day + 1,  # 新目标创建于次日
                    "source": "growth",
                })

        # ---- 节点 5：计算新人格 ----
        new_personality = self._calculate_new_personality(
            old_personality, personality_delta
        )

        # ---- 节点 6：持久化更新 ----
        # 6a. 创建 growth_log（含 schedule_json / world_changes_json）
        growth_log = growth_crud.create_growth_log(
            db=db,
            character_id=character_id,
            personality_delta=json.dumps(personality_delta, ensure_ascii=False),
            event_summary=event_summary,
            new_memories=json.dumps(new_memories, ensure_ascii=False),
            growth_raw=raw_response,
            schedule_json=json.dumps(schedule, ensure_ascii=False),
            world_changes_json=json.dumps(
                {"description": world_changes}, ensure_ascii=False
            ) if world_changes else None,
        )

        # 6b. 将新记忆写入 memories 表
        for mem in new_memories:
            memory_crud.create_memory(
                db=db,
                character_id=character_id,
                content=mem["content"],
                importance=mem["importance"],
                memory_type="growth",
            )

        # 6c. 更新角色人格 + 天数 + 可选的世界观字段 + 短期目标
        update_kwargs = {
            "personality": new_personality,
            "day_number": current_day + 1,
        }
        # 如果 growth 输出了 long_term_goal 更新，也一并持久化
        if parsed.get("long_term_goal_update"):
            update_kwargs["long_term_goal"] = parsed["long_term_goal_update"]
        # Step 14 新增：持久化更新后的短期目标
        if updated_all_goals:
            update_kwargs["short_term_goals"] = json.dumps(updated_all_goals, ensure_ascii=False)
        character_crud.update_character(
            db=db,
            character_id=character_id,
            **update_kwargs,
        )

        # ---- 节点 7：返回 ----
        return {
            "growth_log_id": growth_log.id,
            "character_id": character_id,
            "personality_delta": growth_log.personality_delta,
            "event_summary": growth_log.event_summary,
            "new_memories": growth_log.new_memories,
            "schedule_json": growth_log.schedule_json,
            "world_changes_json": growth_log.world_changes_json,
            "growth_raw": growth_log.growth_raw,
            "created_at": growth_log.created_at,
        }
