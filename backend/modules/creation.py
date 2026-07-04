from typing import Dict, Any, Optional
from backend.services.llm_service import LLMService
from backend.config import settings
from backend.services.observability import observe_safe, update_current_trace


class CreationModule:
    """角色创建模块（Pipeline模式）"""

    def __init__(self):
        self.llm_service = LLMService()
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """加载prompt模板"""
        with open("backend/prompts/creation.txt", "r", encoding="utf-8") as f:
            return f.read()

    def validate_input(self, user_input: str, input_type: str = "text") -> str:
        """
        步骤1：验证输入

        Args:
            user_input: 用户输入
            input_type: 输入类型（"text"或"file"）

        Returns:
            验证后的输入字符串
        """
        if not user_input or len(user_input.strip()) == 0:
            raise ValueError("输入不能为空")

        # 如果是文件输入，可能已经读取为字符串
        return user_input.strip()

    def build_prompt(self, validated_input: str) -> str:
        """
        步骤2：组装Prompt

        Args:
            validated_input: 验证后的输入

        Returns:
            组装好的prompt
        """
        prompt = self.prompt_template.replace(
            "{user_description}",
            validated_input
        )
        return prompt

    @observe_safe("creation.call_llm", as_type="generation")
    def call_llm(self, prompt: str) -> str:
        """
        步骤3：调用LLM

        Args:
            prompt: 组装好的prompt

        Returns:
            LLM的原始响应（JSON字符串）

        Langfuse: 用 as_type="generation" 标记这次调用为一次 LLM generation，
                  Langfuse UI 会自动渲染 model、token、cost 等元数据。
        """
        system_prompt = "你是一个专业的角色创建助手，擅长从描述中提取角色特征。"
        raw_response = self.llm_service.call(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=8000,                          # 14字段嵌套JSON约3000~5000tokens（含scenes/day1_schedule数组），2500不足导致截断→json.loads失败
            response_format={"type": "json_object"}  # Creation 需要 JSON 输出
        )
        return raw_response

    def parse_response(self, raw_response: str) -> Dict[str, Any]:
        """
        步骤4：解析并校验LLM响应

        Args:
            raw_response: LLM的原始响应

        Returns:
            校验通过后的字典

        Raises:
            ValueError: 解析失败或 schema 校验失败时
        """
        # 先做 robust JSON 解析（含 regex fallback）
        parsed = self.llm_service.parse_json_response(raw_response)
        # 再做轻量级 schema 校验
        parsed = LLMService.validate_creation_schema(parsed)
        return parsed

    @observe_safe("creation.run", as_type="span")
    def run(self, user_input: str, input_type: str = "text") -> tuple[Dict[str, Any], str]:
        """
        运行完整的Creation Pipeline

        Args:
            user_input: 用户输入（一句话或故事文本）
            input_type: 输入类型（"text"或"file"）

        Returns:
            (parsed_data, raw_response) 元组
            parsed_data 包含以下字段（Day4 新增 speaking_style/values/habits/long_term_goal；Day5 新增 day1_schedule）：
              - name, world_setting, personality, current_state, initial_memories
              - speaking_style (list[str]), values (list[str]),
                habits (list[str]), long_term_goal (str)
              - day1_schedule (list[dict]): Day 1 初始日程，每条含
                {content, event_type, time_period, order_index}
        """
        # 步骤1：验证输入
        validated_input = self.validate_input(user_input, input_type)

        # 步骤2：组装Prompt
        prompt = self.build_prompt(validated_input)

        # 附加 trace 元数据（用于在 Langfuse UI 按输入类型筛选）
        update_current_trace(
            tags=["creation", f"input_type={input_type}"],
            metadata={
                "input_type": input_type,
                "input_length": len(validated_input),
            },
        )

        # 步骤3：调用LLM
        raw_response = self.call_llm(prompt)

        # 步骤4：解析响应
        parsed_data = self.parse_response(raw_response)
        
        # Day4 新增：确保说话风格/信念/习惯/长期目标有值
        # 这些字段由 Creation LLM 的 prompt 要求生成，
        # 但 schema 校验层已提供默认值兜底，此处不再额外处理。
        # parse_response() 内部的 validate_creation_schema() 已做完整校验。
        
        return parsed_data, raw_response
