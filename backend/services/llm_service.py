import json
import logging
import re
import time
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError

from backend.config import settings
from backend.services.llm_settings_store import LLMSettingsStore

logger = logging.getLogger(__name__)


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
        """
        store = LLMSettingsStore()
        provider_id = store.get_active_provider_id()
        # get_provider_with_env_fallback 自动从环境变量补齐缺失字段
        # （向后兼容老的 .env 配置）
        cfg = store.get_provider_with_env_fallback(provider_id)

        api_key = cfg.get("api_key", "") or ""
        base_url = cfg.get("base_url", "")
        model = cfg.get("model", "")

        # 校验
        if not api_key and provider_id != "ollama":
            raise ValueError(
                f"provider={provider_id} 的 API Key 为空。"
                f"请在设置页填写，或在 .env 中设置 {provider_id.upper()}_API_KEY"
            )
        if not base_url:
            raise ValueError(f"provider={provider_id} 的 base_url 为空")
        if not model:
            raise ValueError(f"provider={provider_id} 的 model 为空")

        self._validate_base_url(base_url)

        self.provider = provider_id
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self.client = OpenAI(
            api_key=api_key if provider_id != "ollama" else "ollama",
            base_url=base_url,
            timeout=self._TIMEOUT,
        )
        self._loaded_at = time.time()
        logger.info(
            "LLMService 重新加载: provider=%s, model=%s, base_url=%s",
            self.provider, self.model, self.base_url,
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

    def parse_json_response(self, response: str) -> dict:
        """
        解析LLM的JSON响应

        Args:
            response: LLM返回的字符串

        Returns:
            解析后的字典
        """
        if not response or not isinstance(response, str) or not response.strip():
            raise ValueError("响应为空，无法解析JSON")

        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        try:
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, re.error):
            pass

        raise ValueError(f"无法解析LLM响应为JSON: {response[:200]}...")

    @staticmethod
    def validate_creation_schema(data: dict) -> dict:
        """
        轻量级 schema 校验：验证 Creation LLM 输出的必要字段与类型。

        校验内容：
        1. 顶层必填字段：name, world_setting, personality, current_state
        2. personality 子字段：optimism, courage, empathy, loyalty,
           intelligence, sociability（要求为 0-100 整数值）
        3. current_state 子字段：location, activity, mood

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
