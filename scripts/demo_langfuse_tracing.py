# -*- coding: utf-8 -*-
"""
Langfuse Tracing Demo
=====================
跑 10 轮真实 chat pipeline (Director -> Actor -> DB persist),
把每轮产生的 trace 树以 ASCII 树状图打印到终端,
让用户直观看到 Langfuse 启用时 UI 上会看到什么.

设计要点:
  1. 不依赖真 Langfuse 服务: TraceCollector 在 observability.observe_safe
     / update_current_trace 这一层拦截,把所有 span/generation 收集到内存.
  2. Mock LLM 调用: LLMService.call() 被替换为返回 canned JSON,
     10 轮对话只需 1-2 秒.
  3. 真实 InteractionPipeline.run: 除了 LLM 是 mock,其它
     (数据库读写、会话管理、Director 分析、Actor 演绎、trace 装饰)
     都是真实代码路径.

跑法:
  cd CharacterSeed
  python -m scripts.demo_langfuse_tracing
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# 把项目根目录加入 sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# =====================================================================
# 1. TraceCollector: 拦截 observability.* 把数据收集到内存
# =====================================================================
class _SpanNode:
    """模拟 Langfuse 的 Span / Generation 数据结构"""
    __slots__ = (
        "name", "as_type", "start_time", "end_time", "duration_ms",
        "metadata", "tags", "user_id", "session_id", "parent", "children",
        "return_summary",
    )

    def __init__(self, name: str, as_type: str, parent: Optional["_SpanNode"]):
        self.name = name
        self.as_type = as_type
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0.0
        self.metadata: Dict[str, Any] = {}
        self.tags: List[str] = []
        self.user_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.parent = parent
        self.children: List["_SpanNode"] = []
        self.return_summary: str = ""

    def close(self, return_value: Any = None) -> None:
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        if isinstance(return_value, dict):
            keys = list(return_value.keys())[:6]
            self.return_summary = "{" + ", ".join(f"{k}=..." for k in keys) + "}"
        elif isinstance(return_value, str):
            s = return_value[:40] + ("..." if len(return_value) > 40 else "")
            self.return_summary = repr(s)
        elif return_value is not None:
            self.return_summary = repr(return_value)[:50]
        else:
            self.return_summary = "None"


class TraceCollector:
    def __init__(self) -> None:
        self.all_roots: List[_SpanNode] = []
        self._stack: List[_SpanNode] = []

    def begin(self, name: str, as_type: str) -> _SpanNode:
        parent = self._stack[-1] if self._stack else None
        node = _SpanNode(name=name, as_type=as_type, parent=parent)
        if parent:
            parent.children.append(node)
        else:
            self.all_roots.append(node)
        self._stack.append(node)
        return node

    def end(self, node: _SpanNode, return_value: Any = None) -> None:
        node.close(return_value)
        if self._stack and self._stack[-1] is node:
            self._stack.pop()
        else:
            while self._stack and self._stack[-1] is not node:
                self._stack.pop()
            if self._stack:
                self._stack.pop()

    def attach_metadata(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._stack:
            return
        cur = self._stack[-1]
        if user_id is not None:
            cur.user_id = user_id
        if session_id is not None:
            cur.session_id = session_id
        if tags:
            cur.tags.extend(tags)
        if metadata:
            cur.metadata.update(metadata)


COLLECTOR = TraceCollector()


# =====================================================================
# 2. Monkey-patch observability 模块
# =====================================================================
from backend.services import observability  # noqa: E402

observability.is_enabled = lambda: True


def _patched_observe_safe(name=None, *, as_type="span", sample_rate=None):
    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            node = COLLECTOR.begin(span_name, as_type)
            try:
                rv = func(*args, **kwargs)
                COLLECTOR.end(node, rv)
                return rv
            except Exception as e:
                node.close(None)
                node.return_summary = f"[EXC] {type(e).__name__}: {e}"
                if COLLECTOR._stack and COLLECTOR._stack[-1] is node:
                    COLLECTOR._stack.pop()
                raise

        return wrapper

    return decorator


observability.observe_safe = _patched_observe_safe
observability.update_current_trace = lambda **kw: COLLECTOR.attach_metadata(**kw)
observability.score_current_trace = lambda *a, **k: None
observability.flush = lambda: None


# =====================================================================
# 3. Mock LLMService.call: 返回 canned JSON
# =====================================================================
from backend.services import llm_service  # noqa: E402

_DEMO_TURNS = [
    {"emotion": "温柔", "goal": "让用户感到被欢迎", "style": "亲切", "mood": "暖意"},
    {"emotion": "好奇", "goal": "关心用户近况", "style": "俏皮", "mood": "活泼"},
    {"emotion": "心疼", "goal": "安抚用户的疲惫", "style": "温柔", "mood": "柔软"},
    {"emotion": "欣喜", "goal": "分享自己最拿手的茶", "style": "推荐式", "mood": "温暖"},
    {"emotion": "自信", "goal": "展现自己泡茶的手艺", "style": "自豪", "mood": "得意"},
    {"emotion": "神秘", "goal": "讲一个古老的故事", "style": "叙述式", "mood": "悠远"},
    {"emotion": "惊讶", "goal": "倾听用户的奇遇", "style": "共鸣", "mood": "共情"},
    {"emotion": "认真", "goal": "帮用户出主意", "style": "分析式", "mood": "沉稳"},
    {"emotion": "害羞", "goal": "回应用户的特别", "style": "含蓄", "mood": "微微红脸"},
    {"emotion": "不舍", "goal": "温暖地道别", "style": "温柔", "mood": "期盼再见"},
]

_SAMPLE_REPLIES = [
    "哎呀客官来啦！今儿个天气好,坐下喝杯龙井吧~",
    "店里生意还行,您瞧这满堂的茶客就知道啦~",
    "累了就歇歇,茶凉了我再给您添热水。",
    "我最拿手的是桂花龙井,秋天的时候桂花刚摘下来,泡出来满屋都是香。",
    "（眯眼笑）那我献丑啦——水温、投茶、注水、出汤,每一步都有讲究呢。",
    "（端起茶壶）听我爷爷讲,这茶馆啊是民国那年开的……",
    "哟,您这奇遇可真是有意思！后来呢？",
    "您这事儿我觉得可以这样……容我先帮您理一理。",
    "（低头笑）您……您过奖啦,我只是个泡茶的丫头。",
    "下次再来啊,我给您留一罐新茶~",
]

_turn_counter = [0]


def _mock_llm_call(self, prompt, system_prompt=None, temperature=0.7, response_format=None):
    # 真实 system_prompt 是中文:
    #   Director: "你是一个专业的角色行为分析师,..."
    #   Actor:    "你是一个沉浸式角色扮演系统,..."
    is_director = system_prompt and "角色行为分析师" in system_prompt
    is_actor = system_prompt and "角色扮演系统" in system_prompt
    return _mock_payload(is_director, is_actor)


def _mock_llm_call_with_messages(self, messages, temperature=0.7, response_format=None, **kw):
    """call_with_messages 的 mock: 从 messages 数组里反查 system prompt"""
    system_prompt = ""
    for m in (messages or []):
        if m.get("role") == "system":
            system_prompt = m.get("content", "")
            break
    is_director = "角色行为分析师" in system_prompt
    is_actor = "角色扮演系统" in system_prompt
    return _mock_payload(is_director, is_actor)


def _mock_payload(is_director: bool, is_actor: bool) -> str:
    if is_director:
        idx = _turn_counter[0]
        t = _DEMO_TURNS[idx % len(_DEMO_TURNS)]
        return json.dumps({
            "emotion": t["emotion"],
            "focus_memories": [
                "童年在茶馆帮爷爷擦茶具的回忆",
                "第一次给客人泡出好茶时爷爷的赞许",
            ],
            "goal": t["goal"],
            "style": t["style"],
        }, ensure_ascii=False)

    if is_actor:
        idx = _turn_counter[0]
        t = _DEMO_TURNS[idx % len(_DEMO_TURNS)]
        _turn_counter[0] += 1
        return json.dumps({
            "npc_response": f"[{t['mood']}] {_SAMPLE_REPLIES[idx % len(_SAMPLE_REPLIES)]}",
            "expression": t["emotion"],
            "action": "把茶碗轻轻推向用户",
        }, ensure_ascii=False)

    return "{}"


llm_service.LLMService.call = _mock_llm_call
llm_service.LLMService.call_with_messages = _mock_llm_call_with_messages


# =====================================================================
# 4. 跑 10 轮真实 InteractionPipeline.run
# =====================================================================
def main(character_id: int = 4, n_turns: int = 10):
    from backend.database import SessionLocal
    from backend.modules.interaction import InteractionPipeline

    # 强制 UTF-8 输出 (Windows 终端默认 GBK)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    print("=" * 80)
    print("[Langfuse Tracing Demo] 10 轮对话追踪演示")
    print("=" * 80)
    print(f"角色: id={character_id}")
    print(f"轮次: {n_turns}")
    print(f"LLM 模式: MOCK (canned JSON, 不调真模型)")
    print(f"Langfuse 模式: MOCK (数据收集到内存, 不发 HTTP)")
    print()

    db = SessionLocal()
    try:
        from backend.crud import character as char_crud
        char = char_crud.get_character(db, character_id)
        if not char:
            print(f"[FAIL] 找不到角色 id={character_id}")
            return
        print(f"角色名称: {char.name}")
        print(f"角色描述: {char.description}")
        print()

        pipeline = InteractionPipeline()
        user_inputs = [
            "你好呀!",
            "最近店里怎么样?",
            "我今天有点累。",
            "你最拿手的是什么茶?",
            "能教教我怎么泡茶吗?",
            "讲个故事给我听吧。",
            "我今天遇到一件奇怪的事……",
            "你能帮我出出主意吗?",
            "我觉得你很特别。",
            "下次再来找你,再见~",
        ][:n_turns]

        t_start = time.time()
        for i, msg in enumerate(user_inputs, 1):
            print(f"[轮 {i:>2}/{n_turns}] user: {msg}")
            t0 = time.time()
            try:
                result = pipeline.run(
                    character_id=character_id,
                    user_message=msg,
                    db=db,
                    history_turns=10,
                    session_id=None,
                )
                npc_resp = (result.get("npc_response") or "")[:60]
                dt = (time.time() - t0) * 1000
                print(f"          npc : {npc_resp}  ({dt:.0f}ms)")
            except Exception as e:
                print(f"          [FAIL] {e}")
        total_dt = (time.time() - t_start) * 1000
        print()
        print(f"[OK] 10 轮跑完, 总耗时 {total_dt:.0f}ms")
        print()
    finally:
        db.close()

    print_trace_trees()

    # 额外写一份 UTF-8 文件，方便 IDE 打开查看（绕过 Windows 终端 GBK 编码问题）
    try:
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print("=" * 80)
            print("[Langfuse Tracing Demo] 10 轮对话追踪演示")
            print("=" * 80)
            print(f"角色: id={character_id}")
            print(f"轮次: {n_turns}")
            print(f"LLM 模式: MOCK (canned JSON, 不调真模型)")
            print(f"Langfuse 模式: MOCK (数据收集到内存, 不发 HTTP)")
            print()
            db2 = SessionLocal()
            try:
                from backend.crud import character as char_crud
                char = char_crud.get_character(db2, character_id)
                if char:
                    print(f"角色名称: {char.name}")
                    print(f"角色描述: {char.description}")
                    print()
            finally:
                db2.close()
            print_trace_trees()
        finally:
            sys.stdout = old_stdout
        out_path = os.path.join(_ROOT, "scripts", "demo_output.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print(f"[INFO] 完整输出已写入: {out_path}")
    except Exception as e:
        print(f"[WARN] 写文件失败: {e}")


# =====================================================================
# 5. ASCII 树状图打印
# =====================================================================
def print_trace_trees() -> None:
    print("=" * 80)
    print("[Langfuse Trace 树] 如果接了 Langfuse, UI 上会看到这些:")
    print("=" * 80)

    for idx, root in enumerate(COLLECTOR.all_roots, 1):
        _print_node(root, prefix="", is_root=True, idx=idx, total=len(COLLECTOR.all_roots))
        print()

    all_nodes = _collect_all(COLLECTOR.all_roots)
    n_generations = sum(1 for n in all_nodes if n.as_type == "generation")
    n_spans = sum(1 for n in all_nodes if n.as_type == "span")
    total_ms = sum(n.duration_ms for n in all_nodes)
    print("-" * 80)
    print(f"[汇总] {len(COLLECTOR.all_roots)} 个 trace, "
          f"{n_spans} 个 span, {n_generations} 个 generation, "
          f"总 wall-time {total_ms:.0f}ms")
    print()


def _collect_all(roots):
    out = []
    stack = list(roots)
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children)
    return out


def _print_node(node, prefix, is_root, idx, total):
    icon = "[LLM]" if node.as_type == "generation" else "[SPAN]"

    if is_root:
        print("-" * 80)
        print(f"Trace #{idx}/{total}  |  {icon} {node.name}  |  {node.duration_ms:.0f}ms")
    else:
        print(f"{prefix}|-- {icon} {node.name}  |  {node.duration_ms:.0f}ms")

    meta_prefix = prefix + ("   " if not is_root else "  ")
    meta_lines = []
    if node.user_id:
        meta_lines.append(f"user_id={node.user_id}")
    if node.session_id:
        meta_lines.append(f"session_id={node.session_id}")
    if node.tags:
        meta_lines.append("tags=[" + ", ".join(node.tags) + "]")
    if node.metadata:
        for k, v in node.metadata.items():
            v_str = str(v)
            if len(v_str) > 40:
                v_str = v_str[:37] + "..."
            meta_lines.append(f"{k}={v_str}")
    if node.return_summary:
        meta_lines.append(f"return={node.return_summary}")
    for line in meta_lines:
        print(f"{meta_prefix}|  {line}")

    for child in node.children:
        _print_node(child, meta_prefix + "|  ", is_root=False, idx=idx, total=total)


# =====================================================================
# 入口
# =====================================================================
if __name__ == "__main__":
    main(character_id=4, n_turns=10)
