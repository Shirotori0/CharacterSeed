import json
import logging
import time
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError

from backend.config import settings
from backend.services.llm_settings_store import LLMSettingsStore

logger = logging.getLogger(__name__)


def _extract_first_balanced_json(text: str) -> str:
    """从文本中提取第一个平衡的 { ... } JSON 字符串（支持嵌套结构）"""
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""  # 未闭合


class LLMService:
    """LLM服务封装类 - 支持多模型切换 + 运行时热更新

    配置来源（优先级从高到低）：
      1. usercontext/llm_settings.json （由设置页写入）
      2. 环境变量（向后兼容老配置，API_KEY / *_BASE_URL / *_MODEL）

    行为：
      - 每次 __init__ 都会从 LLMSettingsStore 重新读取配置
        ——保证设置页改动后，下一次对话/角色创建即可生效，**无需重启后端**。
      - 内部维护 self._loaded_at 时间戳；可在外部调用 reload_config() 强制重读。
    """

    _MAX_RETRIES = 3
    _RETRY_DELAY = 1.0
    _TIMEOUT = 60

    def __init__(self):
        self.reload_config()

    def reload_config(self) -> None:
        """
        从 LLMSettingsStore 重新加载配置，并重建 OpenAI client。

        调用场景：
          - __init__ 内部（默认）
          - 设置页 PUT 成功后由 main.py 显式调用
            （不显式调用也行——下次 chat 请求时会自动重读）

        Fallback 行为（v2 新增）：
          当当前激活 provider 的 API Key / base_url / model 任一缺失时，
          自动遍历全部 6 个 provider，取第一个配置完整的做降级替代。
          仅所有 provider 都不可用时才抛出 ValueError。

        特别处理：
          pydantic-settings 读取 .env 但不会将变量导出到 os.environ，
          而 get_provider_with_env_fallback() 依赖 os.environ 做 env 兜底。
          因此在此显式 load_dotenv() 确保 .env 变量可被 env fallback 找到。
        """
        from dotenv import load_dotenv
        load_dotenv()  # 注入 .env → os.environ，供 get_provider_with_env_fallback 使用

        from backend.services.llm_settings_store import PROVIDER_DEFAULTS

        store = LLMSettingsStore()
        original_id = store.get_active_provider_id()

        # 优先顺序：激活 provider 排第一，其余按默认顺序排列
        candidates = [original_id] + [
            pid for pid in PROVIDER_DEFAULTS if pid != original_id
        ]

        last_reason = None
        for pid in candidates:
            cfg = store.get_provider_with_env_fallback(pid)
            api_key = cfg.get("api_key", "") or ""
            base_url = cfg.get("base_url", "")
            model = cfg.get("model", "")

            # --- 校验配置完整性 ---
            if pid != "ollama" and not api_key:
                last_reason = f"provider={pid} 的 API Key 为空"
                continue
            if not base_url:
                last_reason = f"provider={pid} 的 base_url 为空"
                continue
            if not model:
                last_reason = f"provider={pid} 的 model 为空"
                continue
            try:
                self._validate_base_url(base_url)
            except ValueError as e:
                last_reason = str(e)
                continue

            # --- 找到可用 provider ---
            if pid != original_id:
                logger.warning(
                    "LLM provider 降级: %s → %s (model=%s, base_url=%s). "
                    "原因: %s",
                    original_id, pid, model, base_url, last_reason,
                )

            self.provider = pid
            self.model = model
            self.base_url = base_url
            self._api_key = api_key
            self.client = OpenAI(
                api_key=api_key if pid != "ollama" else "ollama",
                base_url=base_url,
                timeout=self._TIMEOUT,
            )
            self._loaded_at = time.time()
            logger.info(
                "LLMService 重新加载: provider=%s, model=%s, base_url=%s",
                self.provider, self.model, self.base_url,
            )
            return

        # 全部 provider 均不可用
        raise ValueError(
            f"所有 LLM provider 均未配置完整（最近尝试: {last_reason}）。"
            "请前往设置页至少配置一个可用的 provider。"
        )

    @staticmethod
    def _try_env_fallback(provider_id: str, suffix: str) -> Optional[str]:
        """
        从环境变量回退读取（仅当 JSON 文件里没值时使用）。
        兼容 .env 中形如 AGNES_API_KEY / DEEPSEEK_API_KEY / QWEN_BASE_URL 等命名。
        保留为 @staticmethod 以便其他场景使用；reload_config 主路径已统一走 store。
        """
        import os
        env_name = f"{provider_id.upper()}_{suffix}"
        return os.environ.get(env_name) or None

    def _validate_base_url(self, base_url: str) -> None:
        """校验 base_url 格式合法性"""
        if not base_url:
            raise ValueError("base_url 不能为空")

        try:
            parsed = urlparse(base_url)
            if not parsed.scheme or parsed.scheme not in ("http", "https"):
                raise ValueError("base_url 必须以 http:// 或 https:// 开头")
            if not parsed.netloc:
                raise ValueError("base_url 缺少有效的域名或IP地址")
        except ValueError as e:
            raise ValueError(f"base_url 格式错误: {e}")

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[dict] = None
    ) -> str:
        """
        调用LLM（单轮：system + user）

        Args:
            prompt: 用户prompt
            system_prompt: 系统prompt（可选）
            temperature: 温度参数（0-1）
            max_tokens: 最大token数
            response_format: 响应格式约束（可选，例如 {"type": "json_object"}）。
                           默认 None 即不约束格式，由调用方按需传入。

        Returns:
            LLM的响应文本
        """
        self._validate_call_params(prompt, system_prompt, temperature, max_tokens)

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        return self._call_with_retry(kwargs)

    def call_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[dict] = None
    ) -> str:
        """
        使用已组装好的多轮 messages 数组调用 LLM。

        与 call() 的区别：
          - call()          只能传单条 prompt，自动拼成 [system?, user]
          - call_with_messages() 接受调用方已组装好的完整消息列表，
                                  支持多轮对话上下文（system + 历史 user/assistant + 当前 user）

        Args:
            messages: 已组装的消息数组，每个元素必须是 {"role": ..., "content": ...}
                      至少包含 1 条消息；role 必须是 system/user/assistant 之一
            temperature: 温度参数（0-2）
            max_tokens: 最大token数（1-32000）
            response_format: 响应格式约束（可选）

        Returns:
            LLM的响应文本

        Raises:
            ValueError: 参数非法时
        """
        # --- 校验 messages ---
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages 必须是非空列表")

        valid_roles = {"system", "user", "assistant"}
        validated: List[Dict[str, str]] = []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"messages[{idx}] 必须是字典")
            role = msg.get("role")
            content = msg.get("content")
            if role not in valid_roles:
                raise ValueError(
                    f"messages[{idx}].role 必须是 {valid_roles} 之一，得到 {role!r}"
                )
            if not isinstance(content, str):
                raise ValueError(f"messages[{idx}].content 必须是字符串")
            validated.append({"role": role, "content": content})

        # --- 校验其他参数 ---
        if not isinstance(temperature, (int, float)):
            raise ValueError("temperature 必须是数值")
        if temperature < 0 or temperature > 2:
            raise ValueError("temperature 必须在 [0, 2] 范围内")
        if not isinstance(max_tokens, int):
            raise ValueError("max_tokens 必须是整数")
        if max_tokens < 1 or max_tokens > 32000:
            raise ValueError("max_tokens 必须在 [1, 32000] 范围内")

        kwargs = dict(
            model=self.model,
            messages=validated,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        logger.debug(
            "call_with_messages: total=%d, history_turns=%d",
            len(validated),
            sum(1 for m in validated if m["role"] in ("user", "assistant")) // 2,
        )
        return self._call_with_retry(kwargs)

    def _call_with_retry(self, kwargs: Dict[str, Any]) -> str:
        """
        执行带重试的 LLM 调用（被 call / call_with_messages 共用）。

        抽离此方法的动机：
          - call 与 call_with_messages 的重试/异常处理逻辑完全一致
          - 集中一处便于未来统一调整重试策略（如指数退避、熔断等）
        """
        last_exception = None
        for attempt in range(self._MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return self._extract_content(response)

            except AuthenticationError as e:
                logger.error(f"LLM认证失败: {str(e)[:200]}")
                raise

            except RateLimitError as e:
                logger.warning(f"LLM限流: attempt={attempt+1}/{self._MAX_RETRIES}, {str(e)[:200]}")
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(self._RETRY_DELAY * (attempt + 1))
                    continue
                last_exception = e

            except APIConnectionError as e:
                logger.warning(f"LLM连接失败: attempt={attempt+1}/{self._MAX_RETRIES}, {str(e)[:200]}")
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(self._RETRY_DELAY * (attempt + 1))
                    continue
                last_exception = e

            except APIError as e:
                logger.error(f"LLM API错误: attempt={attempt+1}/{self._MAX_RETRIES}, {str(e)[:200]}")
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(self._RETRY_DELAY * (attempt + 1))
                    continue
                last_exception = e

            except Exception as e:
                logger.error(f"LLM调用未知错误: {str(e)[:200]}")
                raise

        if last_exception:
            raise last_exception

        # 理论上不会到达这里（每次循环要么 return 要么 continue 要么 raise）
        raise RuntimeError("LLM调用异常结束：未返回结果也未抛出异常")

    def _validate_call_params(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
        ) -> None:
        """校验调用参数合法性"""
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空字符串")

        if system_prompt is not None:
            if not isinstance(system_prompt, str):
                raise ValueError("system_prompt 必须是字符串")
            if not system_prompt.strip():
                system_prompt = None

        if not isinstance(temperature, (int, float)):
            raise ValueError("temperature 必须是数值")
        if temperature < 0 or temperature > 2:
            raise ValueError("temperature 必须在 [0, 2] 范围内")

        if not isinstance(max_tokens, int):
            raise ValueError("max_tokens 必须是整数")
        if max_tokens < 1 or max_tokens > 32000:
            raise ValueError("max_tokens 必须在 [1, 32000] 范围内")

    def _extract_content(self, response: Any) -> str:
        """安全提取响应内容"""
        if not response:
            logger.warning("LLM返回空响应")
            return ""

        if not hasattr(response, "choices") or not response.choices:
            logger.warning("LLM返回空choices")
            return ""

        first_choice = response.choices[0]
        if not first_choice:
            logger.warning("LLM第一个choice为空")
            return ""

        message = getattr(first_choice, "message", None)
        if not message:
            logger.warning("LLM响应中message为空")
            return ""

        content = getattr(message, "content", None)
        if content is None:
            logger.warning("LLM响应content为None")
            return ""

        if not isinstance(content, str):
            logger.warning(f"LLM响应content类型异常: {type(content)}")
            try:
                return str(content)
            except Exception:
                return ""

        return content.strip()

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        """剥除 ```json ... ``` 或 ``` ... ``` 包裹的 markdown 代码块"""
        t = text.strip()
        for fence in ("```json", "```"):
            if t.startswith(fence):
                t = t[len(fence):].lstrip("\n\r")
                if t.endswith("```"):
                    t = t[:-3].rstrip()
                break
        return t

    def parse_json_response(self, response: str) -> dict:
        """
        解析LLM的JSON响应

        容错策略（按顺序）：
          1. 直接 json.loads(清理后的文本)
          2. 剥除 markdown fence 后重试
          3. 括号配对提取第一个 { ... } 后解析
          4. 仍失败则抛出含尾部上下文的异常便于诊断

        Args:
            response: LLM返回的字符串

        Returns:
            解析后的字典
        """
        if not response or not isinstance(response, str) or not response.strip():
            raise ValueError("响应为空，无法解析JSON")

        cleaned = self._strip_markdown_fence(response)
        if not cleaned:
            raise ValueError("响应经清理后为空，无法解析JSON")

        # --- 尝试 1：直接解析 ---
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # --- 尝试 2：括号配对提取 ---
        candidate = _extract_first_balanced_json(cleaned)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # --- 所有尝试失败：报错时同时显示首部和尾部，便于判断是截断还是格式错误 ---
        head = response[:120]
        tail = response[-120:] if len(response) > 240 else ""
        detail = (
            f"无法解析LLM响应为JSON（总长度={len(response)}字符）。\n"
            f"  首部: {head}...\n"
            f"  尾部: ...{tail}"
        )
        raise ValueError(detail)

    @staticmethod
    def validate_creation_schema(data: dict) -> dict:
        """
        轻量级 schema 校验：验证 Creation LLM 输出的必要字段与类型。

        校验内容：
        1. 顶层必填字段：name, world_setting, personality, current_state
        2. personality 子字段：optimism, courage, empathy, loyalty,
           intelligence, sociability（要求为 0-100 整数值）
        3. current_state 子字段：location, activity, mood
        4. （Day4 新增）可选字段：speaking_style, values, habits, long_term_goal
        5. （Day5 新增）day1_schedule：Day 1 初始日程数组
        6. （v1.6 Phase 1 新增）world_name / core_worldview / scenes：世界+场景结构化数据

        Args:
            data: 解析后的字典

        Returns:
            校验通过后的字典（personality 数值已转为 int）

        Raises:
            ValueError: 缺少必要字段或类型错误时
        """
        if not isinstance(data, dict):
            raise ValueError("数据必须是字典")

        required_top = ["name", "world_setting", "personality", "current_state"]
        for field in required_top:
            if field not in data or data[field] is None:
                raise ValueError(f"LLM响应缺少必填字段: '{field}'")

        personality = data["personality"]
        if not isinstance(personality, dict):
            raise ValueError("'personality' 必须是 JSON 对象")

        personality_fields = [
            "optimism", "courage", "empathy",
            "loyalty", "intelligence", "sociability"
        ]
        for field in personality_fields:
            if field not in personality:
                raise ValueError(f"personality 缺少字段: '{field}'")
            try:
                val = int(personality[field])
                if val < 0 or val > 100:
                    val = max(0, min(100, val))
                personality[field] = val
            except (ValueError, TypeError):
                personality[field] = 50

        current_state = data["current_state"]
        if not isinstance(current_state, dict):
            raise ValueError("'current_state' 必须是 JSON 对象")

        for field in ["location", "activity", "mood"]:
            if field not in current_state:
                current_state[field] = ""
            elif not isinstance(current_state[field], str):
                current_state[field] = str(current_state[field])

        # Day4 新增：speaking_style / values / habits 字段校验（可选，缺省用默认值）
        for field, default in [
            ("speaking_style", ["说话自然"]),
            ("values", ["追求真实"]),
            ("habits", ["保持日常作息"]),
        ]:
            val = data.get(field)
            if val is None:
                data[field] = default
            elif isinstance(val, list):
                data[field] = [str(v).strip() for v in val if v and str(v).strip()]
                if not data[field]:
                    data[field] = default
            else:
                data[field] = default

        long_term_goal = data.get("long_term_goal")
        if not long_term_goal or not isinstance(long_term_goal, str) or not long_term_goal.strip():
            data["long_term_goal"] = ""
        else:
            data["long_term_goal"] = long_term_goal.strip()

        # -- v1.6 Phase 1 新增：world_name / core_worldview / scenes 校验 --
        world_name = data.get("world_name")
        if not world_name or not isinstance(world_name, str) or not world_name.strip():
            data["world_name"] = data.get("name", "未命名世界") + "的世界"
        else:
            data["world_name"] = world_name.strip()

        core_worldview = data.get("core_worldview")
        if not core_worldview or not isinstance(core_worldview, str) or not core_worldview.strip():
            data["core_worldview"] = data.get("world_setting", "")[:100]
        else:
            data["core_worldview"] = core_worldview.strip()

        # scenes 数组校验
        scenes_raw = data.get("scenes")
        if not scenes_raw or not isinstance(scenes_raw, list) or len(scenes_raw) == 0:
            # 保底：从 world_setting 生成一个最小场景树
            scenes_raw = [
                {
                    "name": data.get("world_name", "世界"),
                    "scene_layer": "conceptual",
                    "scene_type": "world",
                    "parent_index": -1,
                    "description": data.get("world_setting", ""),
                },
                {
                    "name": current_state.get("location", "未知地点"),
                    "scene_layer": "actual",
                    "scene_type": "location",
                    "parent_index": 0,
                    "description": f"角色{data.get('name', '')}所在之处",
                },
            ]

        validated_scenes = []
        index_to_id: dict = {}  # 数组索引 → 数据库 ID (占位符，由 main.py 填入)

        for idx, item in enumerate(scenes_raw):
            if not isinstance(item, dict):
                continue
            name = item.get("name", f"场景{idx+1}")
            if not isinstance(name, str) or not name.strip():
                continue
            scene_layer = item.get("scene_layer", "conceptual")
            if scene_layer not in ("conceptual", "actual"):
                scene_layer = "conceptual"
            scene_type = item.get("scene_type", "")
            if not isinstance(scene_type, str):
                scene_type = str(scene_type) if scene_type else ""
            parent_index = item.get("parent_index", -1)
            try:
                parent_index = int(parent_index)
            except (ValueError, TypeError):
                parent_index = -1
            # 根场景和概念场景都可以有 parent_index=-1
            if parent_index < 0:
                parent_index = -1
            description = item.get("description", "")
            if not isinstance(description, str):
                description = str(description) if description else ""

            validated_scenes.append({
                "name": name.strip(),
                "scene_layer": scene_layer,
                "scene_type": scene_type.strip() or None,
                "parent_index": parent_index,  # 保存为索引，在 main.py 中解析为 parent_scene_id
                "description": description.strip(),
            })

        # 保底：至少 1 概念 + 1 实际
        has_conceptual = any(s["scene_layer"] == "conceptual" for s in validated_scenes)
        has_actual = any(s["scene_layer"] == "actual" for s in validated_scenes)
        if not has_conceptual:
            validated_scenes.insert(0, {
                "name": data.get("world_name", "世界"),
                "scene_layer": "conceptual",
                "scene_type": "world",
                "parent_index": -1,
                "description": data.get("world_setting", ""),
            })
        if not has_actual:
            validated_scenes.append({
                "name": current_state.get("location", "未知地点"),
                "scene_layer": "actual",
                "scene_type": "location",
                "parent_index": 0,
                "description": f"角色{data.get('name', '')}所在之处",
            })
        data["scenes"] = validated_scenes

        # -- day1_schedule 校验（Day5 新增）：复用 Growth schedule 校验模式 --
        day1_raw = data.get("day1_schedule")
        if not day1_raw or not isinstance(day1_raw, list):
            day1_raw = []
        validated_day1 = []
        for idx, item in enumerate(day1_raw):
            if not isinstance(item, dict):
                continue
            content = item.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            event_type = item.get("event_type", "schedule_action")
            if event_type not in (
                "schedule_action", "player_dialogue",
                "scene_event", "character_initiative",
            ):
                event_type = "schedule_action"
            time_period = item.get("time_period", "")
            if time_period and time_period not in (
                "morning", "afternoon", "evening", "night",
            ):
                time_period = "morning"
            order_index = item.get("order_index", idx + 1)
            try:
                order_index = int(order_index)
            except (ValueError, TypeError):
                order_index = idx + 1
            validated_day1.append({
                "content": content.strip(),
                "event_type": event_type,
                "time_period": time_period or None,
                "order_index": order_index,
            })

        # 保底：至少 1 条初始事件
        if not validated_day1:
            validated_day1.append({
                "content": "新的一天开始了",
                "event_type": "schedule_action",
                "time_period": None,
                "order_index": 1,
            })
        data["day1_schedule"] = validated_day1

        # -- Step 13 新增：short_term_goals 校验（v1.6 Phase 3） --
        # 短期目标是 long_term_goal 与 schedule 之间的桥梁，
        # 每条目标的生命依附于日程事件。
        goals_raw = data.get("short_term_goals")
        if not goals_raw or not isinstance(goals_raw, list):
            goals_raw = []

        validated_goals = []
        for item in goals_raw:
            if not isinstance(item, dict):
                continue
            goal_text = item.get("goal", "")
            if not isinstance(goal_text, str) or not goal_text.strip():
                continue
            # progress 值域 [0.0, 1.0]，非法值钳位
            try:
                progress = float(item.get("progress", 0.0))
                progress = max(0.0, min(1.0, progress))
            except (ValueError, TypeError):
                progress = 0.0
            # created_day 至少为 1
            try:
                created_day = int(item.get("created_day", 1))
                if created_day < 1:
                    created_day = 1
            except (ValueError, TypeError):
                created_day = 1
            # source 白名单：creation / growth / character
            source = item.get("source", "creation")
            if source not in ("creation", "growth", "character"):
                source = "creation"

            validated_goals.append({
                "goal": goal_text.strip(),
                "progress": progress,
                "created_day": created_day,
                "source": source,
            })

        # 保底：至少 1 条初始短期目标（与长期目标对齐）
        if not validated_goals:
            long_term = data.get("long_term_goal", "")
            fallback_goal = long_term.strip() if long_term.strip() else "探索周围世界，了解自己所处的环境"
            validated_goals.append({
                "goal": fallback_goal,
                "progress": 0.0,
                "created_day": 1,
                "source": "creation",
            })
        data["short_term_goals"] = validated_goals

        return data

    @staticmethod
    def validate_director_schema(data: dict) -> dict:
        """
        轻量级 schema 校验：验证 Director LLM 输出的必要字段与类型。

        校验内容：
        1. 顶层必填字段：emotion, focus_memories, goal, style（均为 string 或 list）
        2. focus_memories 必须是 list[str] 类型，最多 3 条
        3. 所有字符串字段不能为空

        设计考量：
          - 不做 emotion 枚举约束，给 LLM 自由发挥空间（"悲喜交加"、"怅然若失"等复合情绪）
          - focus_memories 截断到 3 条，作为 prompt 工程之外的兜底保护

        Args:
            data: 解析后的字典

        Returns:
            校验通过后的字典

        Raises:
            ValueError: 缺少必要字段或类型错误时
        """
        if not isinstance(data, dict):
            raise ValueError("数据必须是字典")

        defaults = {
            "emotion": "neutral",
            "focus_memories": [],
            "goal": "继续对话",
            "style": "natural"
        }

        for field, default in defaults.items():
            if field not in data or data[field] is None:
                data[field] = default

        emotion = data["emotion"]
        if not isinstance(emotion, str) or not emotion.strip():
            data["emotion"] = "neutral"
        else:
            data["emotion"] = emotion.strip()

        focus_memories = data["focus_memories"]
        if not isinstance(focus_memories, list):
            data["focus_memories"] = []
        else:
            data["focus_memories"] = [
                str(m).strip() for m in focus_memories if m and str(m).strip()
            ][:3]

        goal = data["goal"]
        if not isinstance(goal, str) or not goal.strip():
            data["goal"] = "继续对话"
        else:
            data["goal"] = goal.strip()

        style = data["style"]
        if not isinstance(style, str) or not style.strip():
            data["style"] = "natural"
        else:
            data["style"] = style.strip()

        return data

    @staticmethod
    def validate_actor_schema(data: dict) -> dict:
        """
        轻量级 schema 校验：验证 Actor LLM 输出的必要字段与类型。

        校验内容：
        1. 顶层必填字段：action, expression, speech（均为非空字符串）
        2. speech 做最小长度校验（>= 1 字符）以防止空回复

        设计考量：
          - Actor 输出结构简单（3 个字符串），校验逻辑轻薄
          - speech 不做最大长度限制，给 LLM 充分的表达空间
          - 不做 OOC 检测（超出角色设定的回复），这是 prompt 层面的责任

        Args:
            data: 解析后的字典

        Returns:
            校验通过后的字典

        Raises:
            ValueError: 缺少必要字段或类型错误时
        """
        if not isinstance(data, dict):
            raise ValueError("数据必须是字典")

        defaults = {
            "action": "stand",
            "expression": "neutral",
            "speech": "..."
        }

        for field, default in defaults.items():
            if field not in data or data[field] is None:
                data[field] = default

        for field in ["action", "expression", "speech"]:
            value = data[field]
            if not isinstance(value, str):
                data[field] = str(value) if value else defaults[field]
            if not data[field].strip():
                data[field] = defaults[field]

        return data

    @staticmethod
    def validate_growth_schema(data: dict) -> dict:
        """
        轻量级 schema 校验：验证 Growth LLM 输出的必要字段与类型。

        校验内容：
        1. 顶层必填字段：personality_delta (dict), new_memories (list), event_summary (str)
        2. personality_delta 子字段：6 个人格维度，值域 [-30, 30]
        3. new_memories 数组元素：每条含 content(str) + importance(int 1-10)，最多 3 条
        4. event_summary 为非空字符串

        设计考量：
          - delta 范围限制在 [-30, 30]：防止 LLM 一次输出极端变化（如 optimism 直接 -90）
          - new_memories 截断到 3 条：prompt 已要求 ≤3 条，但 schema 层二次兜底
          - 不对事件摘要做最大长度限制：给 LLM 充分的叙事空间

        Args:
            data: 解析后的字典

        Returns:
            校验通过后的字典（personality_delta 数值已转为 int）

        Raises:
            ValueError: 缺少必要字段或类型/范围错误时
        """
        if not isinstance(data, dict):
            raise ValueError("数据必须是字典")

        if "personality_delta" not in data or data["personality_delta"] is None:
            data["personality_delta"] = {}
        personality_delta = data["personality_delta"]
        if not isinstance(personality_delta, dict):
            data["personality_delta"] = {}
            personality_delta = {}

        personality_fields = [
            "optimism", "courage", "empathy",
            "loyalty", "intelligence", "sociability"
        ]
        for field in personality_fields:
            if field not in personality_delta:
                personality_delta[field] = 0
            else:
                try:
                    val = int(personality_delta[field])
                    val = max(-30, min(30, val))
                    personality_delta[field] = val
                except (ValueError, TypeError):
                    personality_delta[field] = 0

        if "new_memories" not in data or data["new_memories"] is None:
            data["new_memories"] = []
        new_memories = data["new_memories"]
        if not isinstance(new_memories, list):
            data["new_memories"] = []
            new_memories = []

        validated_memories = []
        for mem in new_memories:
            if not isinstance(mem, dict):
                continue
            content = mem.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            try:
                importance = int(mem.get("importance", 5))
                importance = max(1, min(10, importance))
            except (ValueError, TypeError):
                importance = 5
            validated_memories.append({
                "content": content.strip(),
                "importance": importance
            })

        data["new_memories"] = validated_memories[:3]

        if "event_summary" not in data or data["event_summary"] is None:
            data["event_summary"] = "角色经历了一次成长"
        else:
            event_summary = data["event_summary"]
            if not isinstance(event_summary, str) or not event_summary.strip():
                data["event_summary"] = "角色经历了一次成长"

        return data

    @staticmethod
    def validate_growth_schema_v2(data: dict) -> dict:
        """
        v2 schema 校验：验证 Growth+编剧 LLM 输出的字段与类型（Day4 新增）。

        新增字段（相对于 v1）：
          - schedule: 次日事件实体数组（必填），每条含 content/event_type/time_period/order_index
          - world_changes: 世界变化描述（可选，缺省空字符串）

        设计考量（为什么分 v1 和 v2）：
          保留 v1（validate_growth_schema）保持对旧 Growth 调用的向后兼容；
          v2 仅由事件推进轴的新 Growth.run() 使用。
          如果 v2 调用失败，fallback 会尝试用 v1 解析后补默认 schedule。
        """
        if not isinstance(data, dict):
            raise ValueError("数据必须是字典")

        # -- 先走 v1 校验（确保人格 delta/记忆/摘要合法） --
        try:
            data = LLMService.validate_growth_schema(data)
        except ValueError:
            # v1 校验失败时，仍尝试部分恢复
            if "personality_delta" not in data or data["personality_delta"] is None:
                data["personality_delta"] = {}
            if "new_memories" not in data or data["new_memories"] is None:
                data["new_memories"] = []
            if "event_summary" not in data or data["event_summary"] is None:
                data["event_summary"] = "角色经历了一次成长"

        # -- schedule 校验（v2 新增） --
        schedule_raw = data.get("schedule")
        if not schedule_raw or not isinstance(schedule_raw, list):
            schedule_raw = []
        validated_schedule = []
        for idx, item in enumerate(schedule_raw):
            if not isinstance(item, dict):
                continue
            content = item.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            event_type = item.get("event_type", "schedule_action")
            if event_type not in (
                "schedule_action", "player_dialogue",
                "scene_event", "character_initiative",
            ):
                event_type = "schedule_action"
            time_period = item.get("time_period", "")
            if time_period and time_period not in (
                "morning", "afternoon", "evening", "night",
            ):
                time_period = "morning"
            order_index = item.get("order_index", idx + 1)
            try:
                order_index = int(order_index)
            except (ValueError, TypeError):
                order_index = idx + 1
            validated_schedule.append({
                "content": content.strip(),
                "event_type": event_type,
                "time_period": time_period or None,
                "order_index": order_index,
            })

        # schedule 至少保底 1 条
        if not validated_schedule:
            validated_schedule.append({
                "content": "新的一天开始了",
                "event_type": "schedule_action",
                "time_period": None,
                "order_index": 1,
            })
        data["schedule"] = validated_schedule

        # -- world_changes 校验 --
        world_changes = data.get("world_changes", "")
        if not isinstance(world_changes, str):
            world_changes = str(world_changes) if world_changes else ""
        data["world_changes"] = world_changes.strip()

        return data

    # ========================================================================
    # v1.6 B5：事件模式输出校验函数
    # ========================================================================

    # Director 事件模式可用能力白名单
    _VALID_EVENT_CAPABILITIES = {
        "respond_normally", "initiate_dialogue", "modify_plan",
    }

    # complete_event 合法子类型（必须带括号前缀）
    _VALID_COMPLETE_EVENT_SUBTYPES = {
        "succeed", "exceed", "linger", "fail", "skip",
    }

    @staticmethod
    def validate_event_capabilities(capabilities: list) -> list:
        """
        校验并清理 Director 事件模式输出的 capabilities 列表。

        校验规则：
          1. 必须是列表
          2. 列表中的元素只保留合法的"基础能力"和合法格式的 "complete_event(X)"
          3. 不含法元素被静默丢弃
          4. 至少保留 respond_normally + complete_event(succeed) 兜底

        设计考量：
          - 白名单校验而非黑名单，防止 LLM 幻觉输出非法能力名称
          - complete_event 子类型枚举限定，防止 LLM 生成 undefined 行为
          - 静默丢弃而非报错，因为能力选择是"增强"而非"必须"语义

        Args:
            capabilities: Director 输出的 capabilities 列表

        Returns:
            清理后的合法能力列表
        """
        if not isinstance(capabilities, list):
            return ["respond_normally", "complete_event(succeed)"]

        cleaned = []
        for cap in capabilities:
            if not isinstance(cap, str) or not cap.strip():
                continue
            cap = cap.strip()

            # 检查基础能力
            if cap in LLMService._VALID_EVENT_CAPABILITIES:
                cleaned.append(cap)
                continue

            # 检查 complete_event 格式: "complete_event(subtype)"
            if cap.startswith("complete_event(") and cap.endswith(")"):
                subtype = cap[len("complete_event("):-1].strip()
                if subtype in LLMService._VALID_COMPLETE_EVENT_SUBTYPES:
                    cleaned.append(cap)
                    continue

            logger.debug("事件能力白名单过滤: 丢弃非法值 '%s'", cap[:60])

        # 兜底：至少保证基础能力存在
        if not cleaned:
            cleaned = ["respond_normally", "complete_event(succeed)"]

        # 确保至少有一个 complete_event 子类型
        has_complete = any(
            c.startswith("complete_event(") for c in cleaned
        )
        if not has_complete:
            cleaned.append("complete_event(succeed)")

        return cleaned

    @staticmethod
    def validate_event_actor_output(data: dict) -> dict:
        """
        校验 Actor 事件模式输出的必要字段与类型。

        与 validate_actor_schema 的关键区别：
          - speech 可为 None/空（无对话对象时合法）
          - action 为必填（主叙事输出）
          - dialogue_pending 为可选

        Args:
            data: Actor 输出字典

        Returns:
            校验通过后的字典
        """
        if not isinstance(data, dict):
            raise ValueError("数据必须是字典")

        # action 必填（事件模式下是主输出）
        action = data.get("action", "")
        if not isinstance(action, str) or not action.strip():
            data["action"] = "按照计划处理了当前事件"

        # expression 必填
        expression = data.get("expression", "")
        if not isinstance(expression, str) or not expression.strip():
            data["expression"] = "表情平静"

        # speech 可选（无对话对象时可为 None 或空字符串）
        speech = data.get("speech")
        if speech is not None and (not isinstance(speech, str) or not speech.strip()):
            speech = None
        data["speech"] = speech

        # dialogue_pending 可选
        dialogue_pending = data.get("dialogue_pending")
        if dialogue_pending is not None and not isinstance(dialogue_pending, dict):
            dialogue_pending = None
        data["dialogue_pending"] = dialogue_pending

        return data
