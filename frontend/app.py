"""
CharacterSeed Frontend — Streamlit 三页面应用
==============================================

架构设计（Why single-file multi-page）：
  Streamlit 支持两种多页面方案：
    A. pages/ 目录（每个文件一个页面）—— 需要共享 session_state 的 hack
    B. st.sidebar.radio 手动切换 —— 所有页面共享同一个 session_state

  选择方案 B 的原因：
    - 三个页面需要共享 selected_character_id 状态
    - 跨文件 session_state 在 st 1.29 中仍然不完美
    - 单个 app.py 更适合 Demo 场景：一键启动、零配置、代码量可控

页面切换设计：
  st.sidebar.radio 返回当前选中的页面名（str），
  主函数根据返回值 dispatch 到对应的 render 函数。
  每次 rerun 都会重新执行整个脚本，所以 st.cache_data 用于缓存
  不变数据（如角色列表）。

状态管理设计：
  st.session_state 中持久化的 key：
    - selected_character_id: int | None  ← 跨页面共享的角色选择
    - chat_history: list[dict]           ← 当前会话的对话气泡缓存
    - creation_result: dict | None       ← 最近一次创建结果（用于页面1展示）

运行方式：
  cd CharacterSeed
  streamlit run frontend/app.py
"""

import sys
import os
import json
import time
import html
from typing import Any, Dict, Optional

# 确保项目根目录在 path 中（本地开发时 streamlit run 的 cwd 是项目根）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from frontend.api_client import (
    check_backend_health,
    create_character_text,
    create_character_file,
    get_characters,
    get_character,
    delete_character,
    send_message,
    trigger_growth,
    get_memories,
    get_conversations,
    get_growth_logs,
    get_llm_settings,
    list_llm_providers,
    update_llm_settings,
    test_llm_connection,
    list_models,
    test_latency,
    probe_llm,
    # NextChat 会话管理
    list_sessions,
    get_session_detail,
    create_session,
    rename_session,
    delete_session,
)

# ============================================================
# 页面配置（必须是第一个 st 调用）
# ============================================================
st.set_page_config(
    page_title="CharacterSeed — AI NPC 生命模拟",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 人格维度常量
# ============================================================
# 为何在此处重新定义而非从 backend 导入：
#   frontend 包与 backend 包逻辑隔离。重新定义使前端零后端依赖，
#   即使后端重构/移动文件也不影响前端。6 个维度名是不变的业务常量。
PERSONALITY_DIMS = [
    ("optimism", "乐观"),
    ("courage", "勇气"),
    ("empathy", "同理心"),
    ("loyalty", "忠诚"),
    ("intelligence", "智慧"),
    ("sociability", "社交"),
]
PERSONALITY_KEYS = [k for k, _ in PERSONALITY_DIMS]


# ============================================================
# 工具函数
# ============================================================

def safe_parse_json(raw: str | None) -> dict:
    """
    安全解析 JSON 字符串为 dict。

    为何需要此函数：
      CharacterResponse.personality 和 current_state 在 API 返回中
      是 JSON 字符串。Streamlit UI 渲染需要 dict 形式。
      如果 JSON 格式异常（极少情况），返回空 dict 而非崩溃。
    """
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def format_personality_progress(personality_dict: dict) -> list:
    """
    将人格 dict {optimism: 75, ...} 转为有序的 (英文key, 中文名, 值) 三元组。

    为何需要排序：
      进度条展示时，固定顺序给用户一致的视觉参照。
      随机顺序会降低可读性。
    """
    result = []
    for key, cn_name in PERSONALITY_DIMS:
        val = personality_dict.get(key, 50)  # 默认 50 分
        result.append((key, cn_name, val))
    return result


def display_personality_bars(personality_dict: dict, prefix: str = ""):
    """
    渲染 6 维人格进度条。

    设计考量：
      - st.progress 接受 0.0-1.0 的浮点数，而人格值是 0-100 的整数
      - 直接除以 100 转换
      - 每行显示：中文名 | 进度条 | 数值
      - 为何分两列：三列各 2 条，减少纵向滚动，提升信息密度
    """
    items = format_personality_progress(personality_dict)
    mid = 3
    col1, col2 = st.columns(2)

    with col1:
        for key, cn_name, val in items[:mid]:
            pct = val / 100.0
            st.caption(f"{cn_name} ({key})")
            st.progress(pct, text=f"{val}/100")
    with col2:
        for key, cn_name, val in items[mid:]:
            pct = val / 100.0
            st.caption(f"{cn_name} ({key})")
            st.progress(pct, text=f"{val}/100")


# ============================================================
# Director 摘要解析与 CSS 动画注入
# ============================================================

def parse_director_summary(director_raw: str | None) -> dict | None:
    """
    解析 Director 原始 JSON，提取面向用户的摘要信息。

    返回：
      {"goal": str, "style": str, "focus_count": int}
      解析失败时返回 None（兼容低版本数据中 director_raw 为 None 的场景）
    """
    if not director_raw:
        return None
    try:
        data = json.loads(director_raw)
        return {
            "goal": data.get("goal", "继续对话"),
            "style": data.get("style", "自然"),
            "focus_count": len(data.get("focus_memories", [])),
        }
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


def _inject_chat_styles():
    """
    注入 Director/Actor 分时渐进呈现所需的 CSS 动画。

    设计意图：
      - .director-card: 蓝色左边框 + 浅蓝背景的摘要卡片，fadeIn 0.35s
      - .actor-tags.latest: 仅最新消息的 Actor 标签带 0.35s 延迟滑入
      - 历史消息的标签使用普通淡入（无延迟），保持加载时的一致性
    """
    st.markdown(
        '<style>\n'
        '.director-card {\n'
        '    animation: directorFadeIn 0.35s ease-out both;\n'
        '    border-left: 3px solid #4A90D9;\n'
        '    background: #f0f7ff;\n'
        '    padding: 8px 12px;\n'
        '    border-radius: 4px;\n'
        '    margin: 8px 0;\n'
        '    font-size: 0.9em;\n'
        '    line-height: 1.6;\n'
        '}\n'
        '.director-card .d-label {\n'
        '    color: #888;\n'
        '}\n'
        '.actor-tags {\n'
        '    animation: directorFadeIn 0.3s ease-out both;\n'
        '}\n'
        '.actor-tags.latest {\n'
        '    animation: actorSlideUp 0.45s ease-out 0.35s both;\n'
        '}\n'
        '@keyframes directorFadeIn {\n'
        '    from { opacity: 0; transform: translateY(-6px); }\n'
        '    to { opacity: 1; transform: translateY(0); }\n'
        '}\n'
        '@keyframes actorSlideUp {\n'
        '    from { opacity: 0; transform: translateY(10px); }\n'
        '    to { opacity: 1; transform: translateY(0); }\n'
        '}\n'
        '</style>',
        unsafe_allow_html=True,
    )


def render_character_selector(key_prefix: str = "") -> int | None:
    """
    渲染角色下拉选择器并返回选中的 character_id。

    为何作为独立函数：
      页面2和页面3都需要角色选择器，代码复用避免重复。
      key_prefix 参数确保不同页面的 st.selectbox 使用不同的 widget key，
      避免 Streamlit 报 "DuplicateWidgetID" 错误。
    """
    characters = get_characters()
    if not characters:
        st.info("📭 暂无角色，请先去「角色创建」页面创建")
        return None

    options = {f"[{c['id']}] {c['name']}": c["id"] for c in characters}
    labels = list(options.keys())

    # 尝试恢复之前的选中状态
    default_idx = 0
    if "selected_character_id" in st.session_state and st.session_state.selected_character_id:
        for i, c in enumerate(characters):
            if c["id"] == st.session_state.selected_character_id:
                default_idx = i
                break

    selected_label = st.selectbox(
        "👤 选择角色",
        labels,
        index=min(default_idx, len(labels) - 1),
        key=f"{key_prefix}_char_selector",
    )
    char_id = options[selected_label]
    st.session_state.selected_character_id = char_id
    return char_id


# ============================================================
# 页面 1：角色创建
# ============================================================

def render_page_create():
    """
    角色创建页面。

    UI 布局（自上而下）：
      ┌────────────────────────────────────┐
      │  标题：🌱 创建新角色               │
      │  模式切换：文本描述 | 文件上传      │
      │  ┌──────────────────────────┐      │
      │  │ 输入区（text_area 或      │      │
      │  │ file_uploader）           │      │
      │  └──────────────────────────┘      │
      │  [ 🔮 生成角色 ]                   │
      │  ┌──────────────────────────┐      │
      │  │ 结果区（角色卡片）        │      │
      │  │ - 名称 + 世界设定         │      │
      │  │ - 6维人格进度条           │      │
      │  │ - 初始状态                │      │
      │  │ - 初始记忆表              │      │
      │  │ - LLM 原始响应展开        │      │
      │  └──────────────────────────┘      │
      └────────────────────────────────────┘

    交互流程：
      1. 用户选择输入模式（文本 / 文件）
      2. 输入描述或上传 TXT 文件
      3. 点击「生成角色」→ 调用 POST /api/characters/create
      4. 展示生成结果卡片
      5. 自动将新角色 ID 写入 session_state 供其他页面使用
    """
    st.title("🌱 创建新角色")
    st.markdown("输入一段描述或上传 TXT 故事文件，让 AI 为你创造一个鲜活的 NPC 角色。")

    # ---- 模式选择 ----
    # 为何用 radio 而非 tabs：
    #   radio 的状态切换更直观，且 Streamlit 1.29 中 radio 的 horizontal 模式
    #   提供了清晰的视觉分组
    input_mode = st.radio(
        "📥 输入模式",
        ["📝 文本描述", "📄 文件上传 (.txt)"],
        horizontal=True,
        key="create_input_mode",
    )

    user_input_text = ""
    file_content_bytes = None
    file_name = ""
    file_wish = ""

    # ---- 输入区 ----
    if input_mode == "📝 文本描述":
        user_input_text = st.text_area(
            "角色描述",
            placeholder=(
                "用一段话描述你想要的角色...\n\n"
                "示例：「一位在雪山中修炼十年的剑客，性格孤傲但内心善良，\n"
                "师父临终前将一把古剑托付给他...」"
            ),
            height=200,
            key="create_text_input",
        )
    else:
        uploaded_file = st.file_uploader(
            "上传 TXT 故事文件",
            type=["txt"],
            accept_multiple_files=False,
            key="create_file_uploader",
            help="选择一个 .txt 文件，文件内容将作为角色故事描述",
        )
        if uploaded_file is not None:
            file_content_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name
            # 预览文件前 500 字符
            preview = file_content_bytes.decode("utf-8", errors="replace")[:500]
            st.text_area("📋 文件预览", preview, height=100, disabled=True)
            # 创建寄语（可选）
            file_wish = st.text_input(
                "💬 创建寄语（可选）",
                placeholder="添加对角色的额外期望... 如「希望她是一个有忧郁气质但内心坚强的角色」",
                key="create_file_wish",
            )

    # ---- 生成按钮 ----
    # 为何用 button 而非 form_submit_button：
    #   button 更灵活，不需要包装在 st.form 中。
    #   生成操作是幂等的（每次点击创建新角色），不需要 form 的"确认"语义。
    col_btn, col_spacer = st.columns([1, 3])
    with col_btn:
        generate_clicked = st.button(
            "🔮 生成角色",
            type="primary",
            use_container_width=True,
            key="create_generate_btn",
            disabled=(input_mode == "📝 文本描述" and not user_input_text.strip())
            or (input_mode == "📄 文件上传 (.txt)" and file_content_bytes is None),
        )

    if generate_clicked:
        with st.spinner("⏳ 正在调用 Creation LLM 生成角色..."):
            if input_mode == "📝 文本描述":
                result = create_character_text(user_input_text.strip())
            else:
                result = create_character_file(file_content_bytes, file_name, description=file_wish.strip())

        if result.get("error"):
            st.error(f"❌ 创建失败：{result['detail']}")
        else:
            # 创建成功 → 写入 session_state + 展示结果
            st.session_state.creation_result = result
            st.session_state.selected_character_id = result["id"]
            st.success(f"✅ 角色「{result['name']}」创建成功！(ID: {result['id']})")
            st.balloons()

    # ---- 结果展示区 ----
    # 为何同时检查 creation_result 和 result：
    #   如果用户刚点击生成 → generation_result 已写入
    #   如果用户刷新页面（翻到别的 tab 再回来）→ 从 session_state 恢复
    if "creation_result" in st.session_state and st.session_state.creation_result:
        result = st.session_state.creation_result
        personality = safe_parse_json(result.get("personality"))
        current_state = safe_parse_json(result.get("current_state"))

        st.divider()
        st.subheader(f"🎭 {result['name']}")

        # 世界设定
        if result.get("world_setting"):
            with st.expander("🌍 世界设定", expanded=True):
                st.markdown(result["world_setting"])

        # 人格属性
        st.subheader("📊 人格属性")
        if personality:
            display_personality_bars(personality)
        else:
            st.info("暂无初始人格数据")

        # 当前状态
        if current_state:
            st.subheader("📍 当前状态")
            cols = st.columns(3)
            with cols[0]:
                st.markdown("**位置**")
                st.caption(current_state.get("location", "未知"))
            with cols[1]:
                st.markdown("**活动**")
                st.caption(current_state.get("activity", "空闲"))
            with cols[2]:
                st.markdown("**心情**")
                st.caption(current_state.get("mood", "平静"))

        # 初始记忆
        st.subheader("🧠 初始记忆")
        # 从后端获取该角色的记忆列表
        memories = get_memories(result["id"], memory_type="event", limit=50)
        if memories:
            mem_data = []
            for m in memories:
                mem_data.append({
                    "内容": m.get("content", ""),
                    "重要性": m.get("importance", 5),
                    "类型": m.get("memory_type", "event"),
                })
            st.dataframe(mem_data, use_container_width=True, hide_index=True)
        else:
            st.info("暂无初始记忆数据")

        # LLM 原始响应（折叠，供调试）
        if result.get("creation_raw"):
            with st.expander("🔍 Creation LLM 原始响应 (Raw JSON)", expanded=False):
                try:
                    raw_parsed = json.loads(result["creation_raw"])
                    st.json(raw_parsed)
                except json.JSONDecodeError:
                    st.text(result["creation_raw"])

        # 清除按钮
        if st.button("🔄 清除结果，重新创建", key="create_clear_btn"):
            st.session_state.creation_result = None
            st.rerun()


# ============================================================
# 页面 2：对话交互
# ============================================================

def _format_session_time(iso_str: str) -> str:
    """把 ISO 时间格式化成"刚刚 / X 分钟前 / YYYY-MM-DD HH:MM"风格"""
    if not iso_str:
        return ""
    try:
        from datetime import datetime, timezone
        # 兼容带/不带时区的 ISO 字符串
        s = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - dt
        if delta.days >= 1:
            return dt.strftime("%m-%d %H:%M")
        secs = int(delta.total_seconds())
        if secs < 60:
            return "刚刚"
        if secs < 3600:
            return f"{secs // 60} 分钟前"
        return f"{secs // 3600} 小时前"
    except Exception:
        return iso_str[:16]


def _load_session_messages_into_history(char_id: int, session_id: int) -> None:
    """从后端拉取指定 session 的全部消息，写入 st.session_state.chat_history"""
    detail = get_session_detail(session_id)
    if "error" in detail:
        st.session_state.chat_history = []
        return
    msgs_raw = detail.get("messages", [])
    history = []
    for m in msgs_raw:
        # 用户消息
        history.append({
            "role": "user",
            "content": m.get("user_input", ""),
            "timestamp": m.get("timestamp", ""),
        })
        # NPC 回复
        history.append({
            "role": "assistant",
            "content": m.get("npc_response", ""),
            "emotion": m.get("emotion", ""),
            "action": m.get("action", ""),
            "expression": m.get("expression", ""),
            "director_raw": m.get("director_raw", ""),
            "actor_raw": m.get("actor_raw", ""),
            "timestamp": m.get("timestamp", ""),
        })
    st.session_state.chat_history = history
    st.session_state.chat_loaded_char_id = char_id
    st.session_state.chat_loaded_session_id = session_id
    st.session_state.current_session_title = detail.get("title", "")


def _refresh_sessions_cache(char_id: int) -> list:
    """从后端拉取某角色的全部 session 列表，写入 cache"""
    sessions = list_sessions(char_id)
    st.session_state.sessions_cache = sessions
    st.session_state.sessions_cache_char_id = char_id
    return sessions


def _render_session_sidebar(char_id: int) -> int:
    """
    渲染左侧"会话管理"侧栏。
    返回：当前激活的 session_id（可能因用户操作而变化）

    UI 元素：
      - [+ 新对话] 按钮
      - 搜索框
      - 会话列表（每行：标题、消息数、更新时间、操作按钮）
    """
    # ---- 顶部操作 ----
    st.markdown("##### 💬 会话管理")
    if st.button(
        "➕ 新对话", use_container_width=True, key="sidebar_new_session",
        type="primary",
    ):
        result = create_session(char_id)
        if "error" in result:
            st.error(f"创建失败：{result['detail']}")
        else:
            new_sid = result.get("id")
            st.session_state.current_session_id = new_sid
            st.session_state.current_session_title = result.get("title", "新对话")
            st.session_state.chat_history = []
            st.session_state.chat_loaded_char_id = char_id
            st.session_state.chat_loaded_session_id = new_sid
            _refresh_sessions_cache(char_id)
            st.rerun()

    # ---- 搜索框 ----
    search = st.text_input(
        "🔍 搜索会话标题",
        value=st.session_state.get("session_search", ""),
        key="session_search_box",
        placeholder="输入关键字过滤…",
    )
    # session_state 同步：保持 list_sessions 调用能拿到最新值
    st.session_state.session_search = search

    # ---- 加载会话列表 ----
    sessions = _refresh_sessions_cache(char_id)
    if search.strip():
        kw = search.strip().lower()
        sessions = [s for s in sessions if kw in s.get("title", "").lower()]

    # ---- 会话列表 ----
    if not sessions:
        st.caption("暂无会话，点 ➕ 新对话 开始")
        return st.session_state.get("current_session_id")

    # 如果当前没有激活 session，自动选第一个（最活跃的）
    if not st.session_state.get("current_session_id"):
        st.session_state.current_session_id = sessions[0]["id"]
        st.session_state.current_session_title = sessions[0].get("title", "")

    current_sid = st.session_state.current_session_id
    for s in sessions:
        sid = s["id"]
        is_active = (sid == current_sid)
        title = s.get("title", "未命名") or "未命名"
        msg_count = s.get("message_count", 0)
        updated = _format_session_time(s.get("updated_at", ""))

        # 容器：每个 session 一行；激活态用 border 高亮
        with st.container(border=is_active):
            # 第一行：标题 + 消息数
            cols = st.columns([5, 1])
            with cols[0]:
                # 标题按钮（点击切换）
                display = f"{'📍' if is_active else '💬'} **{title}**"
                if st.button(
                    display,
                    key=f"sidebar_session_btn_{sid}",
                    use_container_width=True,
                    help=f"{msg_count} 条消息 · {updated}",
                ):
                    if not is_active:
                        st.session_state.current_session_id = sid
                        st.session_state.current_session_title = title
                        # 加载该 session 的消息
                        _load_session_messages_into_history(char_id, sid)
                        st.rerun()
            with cols[1]:
                st.caption(f"`{msg_count}`")

            # 第二行：更新时间
            st.caption(f"🕐 {updated}")

            # 第三行：操作按钮（重命名 / 删除）— 仅激活态展开，避免误操作
            if is_active:
                op_cols = st.columns(2)
                with op_cols[0]:
                    with st.popover("✏️ 重命名", use_container_width=True):
                        new_title = st.text_input(
                            "新标题", value=title, key=f"rename_input_{sid}",
                            max_chars=200,
                        )
                        if st.button(
                            "保存", key=f"rename_save_{sid}",
                            type="primary", use_container_width=True,
                        ):
                            r = rename_session(sid, new_title)
                            if "error" in r:
                                st.error(f"失败：{r['detail']}")
                            else:
                                st.session_state.current_session_title = new_title
                                _refresh_sessions_cache(char_id)
                                st.rerun()
                with op_cols[1]:
                    with st.popover("🗑️ 删除", use_container_width=True):
                        st.warning(f"删除 **{title}** 及其全部 {msg_count} 条消息？")
                        if st.button(
                            "确认删除", key=f"delete_confirm_{sid}",
                            type="primary", use_container_width=True,
                        ):
                            r = delete_session(sid)
                            if "error" in r:
                                st.error(f"失败：{r['detail']}")
                            else:
                                st.session_state.current_session_id = None
                                st.session_state.current_session_title = ""
                                st.session_state.chat_history = []
                                st.session_state.chat_loaded_session_id = None
                                _refresh_sessions_cache(char_id)
                                st.rerun()

    return st.session_state.current_session_id


def render_page_chat():
    """
    对话交互页面（NextChat 风格：左侧会话管理 + 右侧对话气泡）。

    UI 布局：
      ┌──────────────────┬──────────────────────────────────────┐
      │  标题：💬 角色对话                                          │
      │  [角色选择器]                                                │
      │  角色信息栏                                                  │
      ├──────────────────┼──────────────────────────────────────┤
      │  会话侧栏          │  对话气泡区                              │
      │  + 新对话          │  用户消息 (右)                            │
      │  🔍 搜索           │  NPC 回复 (左) + 标签                    │
      │  📜 会话列表       │                                          │
      │   💬 标题 3         │  [消息输入框 chat_input]                │
      │      3 条 · 5分钟前 │  ──── 侧栏 ────                          │
      │      ✏️  🗑️         │  [🌱 触发成长]                            │
      └──────────────────┴──────────────────────────────────────┘

    交互流程（Director + Actor 双 LLM 管线）：
      1. 用户输入消息 → POST /api/chat（带 session_id）
      2. 后端执行 Director.analyze() → emotion/focus_memories/goal/style
      3. 后端执行 Actor.generate()  → action/expression/speech
      4. 后端把消息写入对应 session，并 touch updated_at
      5. 响应返回 session_id / session_title，前端刷新侧栏
    """
    st.title("💬 角色对话")
    st.markdown("选择一名角色，开始对话。AI 将通过 Director+Actor 双管线生成角色的反应。")
    _inject_chat_styles()

    # ---- 角色选择 ----
    char_id = render_character_selector(key_prefix="chat")
    if char_id is None:
        return

    # ---- 角色信息栏 ----
    char_detail = get_character(char_id)
    if "error" in char_detail:
        st.error(f"无法加载角色信息：{char_detail['detail']}")
        return

    personality = safe_parse_json(char_detail.get("personality"))
    current_state = safe_parse_json(char_detail.get("current_state"))

    if personality:
        top_trait = max(personality.items(), key=lambda x: x[1]) if personality else (None, 0)
        bottom_trait = min(personality.items(), key=lambda x: x[1]) if personality else (None, 0)
        trait_summary = (
            f"**{char_detail['name']}** · "
            f"最突出: {dict(PERSONALITY_DIMS).get(top_trait[0], top_trait[0])} ({top_trait[1]}) · "
            f"最薄弱: {dict(PERSONALITY_DIMS).get(bottom_trait[0], bottom_trait[0])} ({bottom_trait[1]})"
        )
    else:
        trait_summary = char_detail.get("name", "未知角色")

    st.markdown(trait_summary)
    if current_state:
        st.caption(
            f"📍 {current_state.get('location', '未知')} · "
            f"🔄 {current_state.get('activity', '空闲')} · "
            f"😊 {current_state.get('mood', '平静')}"
        )

        # ---- 场景上下文卡片（提升沉浸感） ----
        # 将角色当前的位置、活动、心情与世界设定拼合为一段场景描述，
        # 使用 st.info 呈现，并附加基于情绪的轻度引导提示。
        world_setting = char_detail.get("world_setting", "")
        scene_lines = []
        if current_state:
            loc = current_state.get("location", "未知地点")
            act = current_state.get("activity", "空闲")
            mood = current_state.get("mood", "平静")
            scene_lines.append(f"📍 **现在你在：** {loc}")
            scene_lines.append(f"🕐 **角色状态：** {act} · 心情 {mood}")

        # 世界氛围（截取前 80 字避免卡片过于冗长）
        if world_setting:
            brief_world = world_setting[:80].rstrip()
            if len(world_setting) > 80:
                brief_world += "…"
            scene_lines.insert(0, f"🌍 **世界氛围：** {brief_world}")

        # 基于情绪的轻度引导提示
        mood_guide = {
            "开心": "她看起来心情不错，聊聊开心事吧 🌟",
            "悲伤": "她似乎有些低落，温柔地关心一下吧 💙",
            "愤怒": "她现在有点生气，小心说话哦 🔥",
            "焦虑": "她有些不安，试着让她放松下来 🤗",
            "平静": "她很平静，聊聊日常也是好选择 ☕",
            "激动": "她好像很兴奋，问问发生了什么好事吧 🎉",
        }
        mood_val = current_state.get("mood", "") if current_state else ""
        hint_text = mood_guide.get(mood_val, "试试从她当前的状态切入对话吧 ✨")

        scene_md = "\n\n".join(scene_lines) + f"\n\n💡 **提示：** {hint_text}"
        st.info(scene_md)

    st.divider()

    # ---- 角色切换检测：清空 session 缓存 + 重置激活 session ----
    if st.session_state.get("chat_loaded_char_id") != char_id:
        st.session_state.sessions_cache = []
        st.session_state.sessions_cache_char_id = None
        st.session_state.current_session_id = None
        st.session_state.current_session_title = ""
        st.session_state.chat_loaded_session_id = None

    # ---- 左右分栏：会话侧栏 + 对话区 ----
    sidebar_col, chat_col = st.columns([1, 2.5], gap="medium")

    with sidebar_col:
        active_sid = _render_session_sidebar(char_id)
    with chat_col:
        # 渲染当前 session 的标题 + 消息 + 输入
        _render_chat_area(char_id, active_sid)


def _render_chat_area(char_id: int, session_id: Optional[int]) -> None:
    """右侧对话区：标题 / 气泡列表 / 输入框 / 成长按钮"""
    # ---- 当前会话标题 ----
    title = st.session_state.get("current_session_title") or "（未选择会话）"
    st.markdown(f"#### 📍 {title}")
    if session_id is None:
        st.info("👈 请在左侧选择或创建一个会话")
        return

    # ---- 加载历史（如果 session 变了） ----
    if (
        st.session_state.get("chat_loaded_char_id") != char_id
        or st.session_state.get("chat_loaded_session_id") != session_id
    ):
        _load_session_messages_into_history(char_id, session_id)

    # 确保 chat_history 存在
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---- 渲染对话气泡 ----
    total_msgs = len(st.session_state.chat_history)
    for idx, msg in enumerate(st.session_state.chat_history):
        is_latest_fresh = (idx == total_msgs - 1) and msg.get("is_fresh", False)
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                # --- 1. Director 摘要卡片（分时渐进：卡片先淡入） ---
                director_info = parse_director_summary(msg.get("director_raw"))
                if director_info:
                    cls = "director-card" + (" latest" if is_latest_fresh else "")
                    st.markdown(
                        f'<div class="{cls}">'
                        f'  <div style="font-weight:600;color:#4A90D9;margin-bottom:4px;">🧠 角色思考</div>'
                        f'  <div><span class="d-label">目标：</span>{html.escape(director_info["goal"])}</div>'
                        f'  <div><span class="d-label">风格：</span>{html.escape(director_info["style"])}</div>'
                        f'  <div><span class="d-label">关联记忆：</span>{director_info["focus_count"]} 条</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # --- 2. Actor 标签（渐进式延迟淡入） ---
                tags = []
                if msg.get("emotion"):
                    tags.append(f"😶 {html.escape(msg['emotion'])}")
                if msg.get("expression"):
                    tags.append(f"🎭 {html.escape(msg['expression'])}")
                if msg.get("action"):
                    tags.append(f"🏃 {html.escape(msg['action'])}")
                if tags:
                    tags_str = " · ".join(tags)
                    cls = "actor-tags" + (" latest" if is_latest_fresh else "")
                    st.markdown(f'<div class="{cls}">{tags_str}</div>', unsafe_allow_html=True)

    # ---- 消息输入 ----
    user_msg = st.chat_input("输入消息...", key="chat_input_box")

    if user_msg:
        # 追加用户消息到历史
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_msg,
        })

        # 调用后端对话 API（带 session_id）
        with st.spinner("💭 角色正在思考..."):
            chat_result = send_message(char_id, user_msg, session_id=session_id)

        if chat_result.get("error"):
            st.error(f"对话失败：{chat_result['detail']}")
            st.session_state.chat_history.pop()
        else:
            # 同步后端返回的 session_id / session_title
            new_sid = chat_result.get("session_id") or session_id
            new_title = chat_result.get("session_title") or title
            st.session_state.current_session_id = new_sid
            st.session_state.current_session_title = new_title
            st.session_state.chat_loaded_session_id = new_sid

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": chat_result.get("npc_response", ""),
                "emotion": chat_result.get("emotion", ""),
                "action": chat_result.get("action", ""),
                "expression": chat_result.get("expression", ""),
                "director_raw": chat_result.get("director_raw", ""),
                "actor_raw": chat_result.get("actor_raw", ""),
                "timestamp": str(chat_result.get("timestamp", "")),
                "is_fresh": True,
            })

            # 展示 LLM 原始响应（折叠）
            if chat_result.get("director_raw") or chat_result.get("actor_raw"):
                with st.expander("🔍 LLM 管线内部响应 (调试)", expanded=False):
                    if chat_result.get("director_raw"):
                        st.markdown("**Director（注意力聚焦）原始输出：**")
                        try:
                            st.json(json.loads(chat_result["director_raw"]))
                        except json.JSONDecodeError:
                            st.text(chat_result["director_raw"])
                    if chat_result.get("actor_raw"):
                        st.markdown("**Actor（行为生成）原始输出：**")
                        try:
                            st.json(json.loads(chat_result["actor_raw"]))
                        except json.JSONDecodeError:
                            st.text(chat_result["actor_raw"])

            # 刷新侧栏 session 列表（让 updated_at / message_count 立即更新）
            _refresh_sessions_cache(char_id)

        st.rerun()

    # ---- 侧边操作 ----
    st.divider()
    if st.button("🌱 触发角色成长", use_container_width=True, key="chat_growth_btn"):
        with st.spinner("⏳ Growth LLM 正在分析角色成长..."):
            growth_result = trigger_growth(char_id)

        if growth_result.get("error"):
            st.error(f"成长触发失败：{growth_result['detail']}")
        else:
            st.success(f"✅ 成长分析完成！")
            st.markdown(f"**事件摘要：** {growth_result.get('event_summary', '无')}")

            delta_raw = growth_result.get("personality_delta", "{}")
            delta = safe_parse_json(delta_raw)
            if delta:
                st.markdown("**人格变化 (Δ)：**")
                for key, cn_name in PERSONALITY_DIMS:
                    dv = delta.get(key, 0)
                    if dv != 0:
                        arrow = "↑" if dv > 0 else "↓"
                        color = "green" if dv > 0 else "red"
                        st.markdown(f"  {cn_name}: :{color}[{arrow}{abs(dv)}]")

            new_memories_raw = growth_result.get("new_memories", "[]")
            new_memories = safe_parse_json(new_memories_raw) if isinstance(new_memories_raw, str) else new_memories_raw
            if new_memories:
                with st.expander("🧠 新增记忆", expanded=True):
                    for nm in new_memories:
                        if isinstance(nm, dict):
                            st.markdown(f"- [{nm.get('importance', 5)}/10] {nm.get('content', '')}")

            if growth_result.get("growth_raw"):
                with st.expander("🔍 Growth LLM 原始响应", expanded=False):
                    try:
                        st.json(json.loads(growth_result["growth_raw"]))
                    except json.JSONDecodeError:
                        st.text(growth_result["growth_raw"])

    # ---- 角色删除（带二次确认） ----
    st.markdown("---")
    st.markdown("**⚠️ 危险操作**")

    if st.session_state.get("chat_confirm_delete"):
        st.warning(
            f"⚠️ 确定要永久删除角色「{char_detail['name']}」及其所有记忆、"
            f"对话和成长记录？**此操作不可撤销！**"
        )
        col_del_confirm, col_del_cancel = st.columns(2)
        with col_del_confirm:
            if st.button("✅ 确认删除", use_container_width=True, key="chat_delete_confirm_btn"):
                with st.spinner("正在级联删除角色数据..."):
                    del_result = delete_character(char_id)
                if del_result.get("error"):
                    st.error(f"删除失败：{del_result['detail']}")
                else:
                    st.success(del_result.get("detail", "删除成功"))
                    # 清理 session_state
                    st.session_state.chat_history = []
                    st.session_state.chat_loaded_char_id = None
                    if st.session_state.get("selected_character_id") == char_id:
                        st.session_state.selected_character_id = None
                    st.session_state.chat_confirm_delete = False
                    st.cache_data.clear()
                    st.rerun()
        with col_del_cancel:
            if st.button("❌ 取消", use_container_width=True, key="chat_delete_cancel_btn"):
                st.session_state.chat_confirm_delete = False
                st.rerun()
    else:
        if st.button("🗑️ 删除角色", use_container_width=True, key="chat_delete_btn", type="secondary"):
            st.session_state.chat_confirm_delete = True
            st.rerun()


# ============================================================
# 页面 3：角色状态面板
# ============================================================

def render_page_dashboard():
    """
    角色状态面板页面。

    UI 布局：
      ┌────────────────────────────────────┐
      │  标题：📊 角色状态面板             │
      │  [角色选择器]                       │
      │  ┌─────────────┬─────────────┐     │
      │  │ 人格面板    │ 当前状态    │     │
      │  │ 6个进度条   │ 位置/活动/  │     │
      │  │             │ 心情        │     │
      │  └─────────────┴─────────────┘     │
      │  ┌─────────────┬─────────────┐     │
      │  │ 记忆面板    │ 对话历史    │     │
      │  │ [类型筛选]  │ dataframe   │     │
      │  │ dataframe   │             │     │
      │  └─────────────┴─────────────┘     │
      │  ┌──────────────────────────┐      │
      │  │ 成长记录 (expander)      │      │
      │  │ - event_summary          │      │
      │  │ - personality_delta      │      │
      │  │ - new_memories           │      │
      │  └──────────────────────────┘      │
      └────────────────────────────────────┘

    为何将四个面板放在同一页面而非拆分：
      这是角色的"控制面板"——一次性展示所有维度的数据。
      分页会增加用户的认知负担，降低信息获取效率。
      通过 expander 折叠成长记录，避免信息过载。
    """
    st.title("📊 角色状态面板")
    st.markdown("选择一个角色查看其完整状态，包括人格属性、记忆、对话历史和成长记录。")

    # ---- 角色选择 ----
    char_id = render_character_selector(key_prefix="dashboard")
    if char_id is None:
        return

    # ---- 加载角色数据 ----
    char_detail = get_character(char_id)
    if char_detail.get("error"):
        st.error(f"无法加载角色信息：{char_detail['detail']}")
        return

    personality = safe_parse_json(char_detail.get("personality"))
    current_state = safe_parse_json(char_detail.get("current_state"))

    # ---- 上半部分：人格 + 当前状态 ----
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 人格属性")
        if personality:
            display_personality_bars(personality)
        else:
            st.info("暂无数据")

    with col_right:
        st.subheader("📍 当前状态")
        if current_state:
            st.markdown(f"**位置:** {current_state.get('location', '未知')}")
            st.markdown(f"**活动:** {current_state.get('activity', '空闲')}")
            st.markdown(f"**心情:** {current_state.get('mood', '平静')}")
        else:
            st.info("暂无数据")

        # 角色基本信息
        st.subheader("📋 基本信息")
        st.markdown(f"**名称：** {char_detail.get('name', '未知')}")
        st.markdown(f"**ID：** {char_detail.get('id', 'N/A')}")
        if char_detail.get("world_setting"):
            with st.expander("🌍 世界设定", expanded=False):
                st.markdown(char_detail["world_setting"])
        if char_detail.get("description"):
            st.caption(f"原始描述: {char_detail['description'][:200]}...")

        # 角色删除（带二次确认）
        st.markdown("---")
        if st.session_state.get("dashboard_confirm_delete"):
            st.warning(
                f"⚠️ 确定要永久删除角色「{char_detail['name']}」及其所有关联数据？"
                f"**此操作不可撤销！**"
            )
            col_del_cfm, col_del_ccl = st.columns(2)
            with col_del_cfm:
                if st.button("✅ 确认删除", use_container_width=True, key="dashboard_delete_confirm_btn"):
                    with st.spinner("正在级联删除角色数据..."):
                        del_result = delete_character(char_id)
                    if del_result.get("error"):
                        st.error(f"删除失败：{del_result['detail']}")
                    else:
                        st.success(del_result.get("detail", "删除成功"))
                        if st.session_state.get("selected_character_id") == char_id:
                            st.session_state.selected_character_id = None
                        st.session_state.dashboard_confirm_delete = False
                        st.cache_data.clear()
                        st.rerun()
            with col_del_ccl:
                if st.button("❌ 取消", use_container_width=True, key="dashboard_delete_cancel_btn"):
                    st.session_state.dashboard_confirm_delete = False
                    st.rerun()
        else:
            if st.button("🗑️ 删除角色", use_container_width=True, key="dashboard_delete_btn", type="secondary"):
                st.session_state.dashboard_confirm_delete = True
                st.rerun()

    st.divider()

    # ---- 下半部分：记忆 + 对话历史 ----
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("🧠 记忆列表")

        # 记忆类型筛选
        # 为何提供筛选：
        #   记忆有三种类型（conversation/event/growth），
        #   不同类型的记忆有不同的语义。筛选让用户快速聚焦。
        mem_type = st.selectbox(
            "筛选类型",
            ["全部", "对话记忆", "事件记忆", "成长记忆"],
            key="dashboard_mem_filter",
        )
        mem_type_map = {
            "全部": None,
            "对话记忆": "conversation",
            "事件记忆": "event",
            "成长记忆": "growth",
        }
        api_mem_type = mem_type_map.get(mem_type)

        memories = get_memories(char_id, memory_type=api_mem_type, limit=100)
        if memories:
            mem_df_data = []
            for m in memories:
                mem_df_data.append({
                    "ID": m.get("id"),
                    "内容": m.get("content", "")[:80] + ("..." if len(m.get("content", "")) > 80 else ""),
                    "重要度": m.get("importance", 5),
                    "类型": m.get("memory_type", "unknown"),
                    "创建时间": str(m.get("created_at", ""))[:19],
                })
            st.dataframe(mem_df_data, use_container_width=True, hide_index=True)
            st.caption(f"共 {len(memories)} 条记忆")
        else:
            st.info("暂无记忆记录")

    with col_right2:
        st.subheader("💬 对话历史")

        conversations = get_conversations(char_id, limit=50)
        if conversations:
            conv_data = []
            for c in conversations:
                conv_data.append({
                    "ID": c.get("id"),
                    "玩家": c.get("user_input", "")[:60] + ("..." if len(c.get("user_input", "")) > 60 else ""),
                    "NPC": c.get("npc_response", "")[:60] + ("..." if len(c.get("npc_response", "")) > 60 else ""),
                    "情绪": c.get("emotion", ""),
                    "时间": str(c.get("timestamp", ""))[:19],
                })
            st.dataframe(conv_data, use_container_width=True, hide_index=True)
            st.caption(f"共 {len(conversations)} 条对话")
        else:
            st.info("暂无对话记录")

    st.divider()

    # ---- 成长记录面板 ----
    st.subheader("📈 成长记录")
    growth_logs = get_growth_logs(char_id, limit=50)
    if growth_logs:
        for idx, gl in enumerate(growth_logs):
            with st.expander(
                f"🌱 成长 #{gl.get('id')} — {gl.get('event_summary', '无摘要')[:60]}..."
                f" | {str(gl.get('created_at', ''))[:19]}",
                expanded=(idx == 0),  # 第一条默认展开
            ):
                # 事件摘要
                st.markdown("**📝 事件摘要：**")
                st.markdown(gl.get("event_summary", "无"))

                # 人格变化
                delta = safe_parse_json(gl.get("personality_delta"))
                if delta:
                    st.markdown("**📊 人格变化 (Δ)：**")
                    delta_cols = st.columns(3)
                    for i, (key, cn_name) in enumerate(PERSONALITY_DIMS):
                        dv = delta.get(key, 0)
                        if dv != 0:
                            arrow = "↑" if dv > 0 else "↓"
                            color = "green" if dv > 0 else "red"
                            delta_cols[i % 3].markdown(f"{cn_name}: :{color}[{arrow}{abs(dv)}]")
                        else:
                            delta_cols[i % 3].markdown(f"{cn_name}: 0")

                # 新增记忆
                new_memories_raw = gl.get("new_memories", "[]")
                new_memories = safe_parse_json(new_memories_raw) if isinstance(new_memories_raw, str) else new_memories_raw
                if new_memories:
                    st.markdown("**🧠 新增记忆：**")
                    for nm in new_memories:
                        if isinstance(nm, dict):
                            st.markdown(f"- [{nm.get('importance', 5)}/10] {nm.get('content', '')}")

                # Growth LLM 原始响应
                if gl.get("growth_raw"):
                    with st.expander("🔍 Growth LLM 原始响应", expanded=False):
                        try:
                            st.json(json.loads(gl["growth_raw"]))
                        except json.JSONDecodeError:
                            st.text(gl["growth_raw"])

        st.caption(f"共 {len(growth_logs)} 条成长记录")
    else:
        st.info("暂无成长记录（请先在对话页触发角色成长）")


# ============================================================
# 页面 4：LLM 设置（参考 NextChat 的 settings 模式）
# ============================================================

def _get_provider_needs_key(provider_id: str, providers_meta: list) -> bool:
    """根据元信息查某个 provider 是否需要 api_key。"""
    for m in providers_meta:
        if m.get("id") == provider_id:
            return str(m.get("needs_key", "true")).lower() == "true"
    return True  # 默认需要 key（保守策略）


def render_page_settings():
    """
    LLM 设置页面。

    UI 布局（自上而下）：
      ┌────────────────────────────────────────────┐
      │  标题：⚙️ LLM 设置                         │
      │  当前激活 provider 状态条                   │
      │  ┌──────────────────────────────────────┐  │
      │  │ 📌 激活 Provider                     │  │
      │  │  [下拉选择] [切换]                    │  │
      │  └──────────────────────────────────────┘  │
      │  ┌──────────────────────────────────────┐  │
      │  │ 🔑 当前 Provider 配置                 │  │
      │  │  API Key: [password input]            │  │
      │  │  Base URL: [text input]               │  │
      │  │  Model: [text input]                  │  │
      │  │  [测试连接] [保存]                     │  │
      │  └──────────────────────────────────────┘  │
      │  ┌──────────────────────────────────────┐  │
      │  │ 🎛️ 默认调用参数                       │  │
      │  │  Temperature: [slider 0-2]            │  │
      │  │  Max Tokens: [number input]           │  │
      │  └──────────────────────────────────────┘  │
      │  ┌──────────────────────────────────────┐  │
      │  │ 📦 其他 Provider 配置（折叠）         │  │
      │  │  列出全部 6 个 provider 的脱敏状态    │  │
      │  └──────────────────────────────────────┘  │
      │  文件位置: <path>                          │
      └────────────────────────────────────────────┘
    """
    st.title("⚙️ LLM 设置")
    st.markdown(
        "管理 **大语言模型厂商** 与 **API Key**。"
        "所有更改会保存到 `usercontext/llm_settings.json`，**无需重启后端**即可生效。"
    )

    # ---- 拉取设置与厂商元信息 ----
    settings = get_llm_settings()
    if "error" in settings:
        st.error(f"❌ 无法加载设置：{settings['detail']}")
        st.info("请确认后端 uvicorn 已启动在 :8000")
        return

    providers_info = list_llm_providers()
    if "error" in providers_info:
        st.error(f"❌ 无法加载厂商列表：{providers_info['detail']}")
        return
    providers_meta = providers_info.get("providers", [])

    active_id = settings.get("active_provider", "agnes")
    active_name = settings.get("active_provider_name", active_id)
    active_cfg = settings.get("config", {})

    # ---- 状态条：当前激活 provider ----
    st.info(f"📌 当前激活：**{active_name}** (`{active_id}`)，模型 `{active_cfg.get('model', '-')}`")

    # ---- 区块 1：切换 Provider ----
    st.subheader("1️⃣ 切换 Provider")
    provider_options = [m["id"] for m in providers_meta]
    provider_labels = {
        m["id"]: f"{m['name']} ({m['id']})" for m in providers_meta
    }

    col1, col2 = st.columns([3, 1])
    with col1:
        new_active = st.selectbox(
            "选择要激活的 provider",
            provider_options,
            index=provider_options.index(active_id) if active_id in provider_options else 0,
            format_func=lambda x: provider_labels.get(x, x),
            key="settings_active_provider_select",
        )
    with col2:
        st.write("")  # 占位
        st.write("")
        if st.button("🔄 切换", use_container_width=True, key="settings_switch_btn"):
            if new_active == active_id:
                st.info("当前已是该 provider，无需切换")
            else:
                result = update_llm_settings(active_provider=new_active)
                if "error" in result:
                    st.error(f"切换失败：{result['detail']}")
                else:
                    st.success(f"✅ 已切换到 **{result.get('active_provider_name', new_active)}**")
                    time.sleep(0.5)
                    st.rerun()

    st.divider()

    # ---- 区块 2：当前 Provider 的详细配置 ----
    st.subheader(f"2️⃣ 当前 Provider 配置 — {active_name}")

    # API Key：脱敏显示在 caption 中，主输入框默认空（用户想改就输入新值）
    masked_key = active_cfg.get("api_key", "")
    key_caption = (
        f"当前已保存 key: `{masked_key}`"
        if masked_key and not masked_key.startswith("****")
        else "（当前 key 为空或已脱敏）"
    )

    needs_key = _get_provider_needs_key(active_id, providers_meta)
    if needs_key:
        st.caption(key_caption)
        new_api_key = st.text_input(
            "API Key（留空 = 不修改）",
            type="password",
            key=f"settings_api_key_{active_id}",
            placeholder="留空保持原值；填写新值则覆盖",
        )
    else:
        # Ollama 不需要 key
        st.caption("🔓 本地服务（Ollama）无需 API Key")
        new_api_key = None

    new_base_url = st.text_input(
        "Base URL",
        value=active_cfg.get("base_url", ""),
        key=f"settings_base_url_{active_id}",
    )
    new_model = st.text_input(
        "Model 名称",
        value=active_cfg.get("model", ""),
        key=f"settings_model_{active_id}",
    )

    # ---- 操作按钮：测试 + 保存 ----
    col_test, col_save = st.columns(2)
    with col_test:
        test_clicked = st.button("🧪 测试连接", use_container_width=True, key="settings_test_btn")
    with col_save:
        save_clicked = st.button("💾 保存配置", use_container_width=True, type="primary", key="settings_save_btn")

    # ---- 测试连接 ----
    if test_clicked:
        with st.spinner("正在连接 LLM..."):
            # 如果用户已填了新的 api_key（未保存），用它来测试
            test_kwargs = {
                "provider_id": active_id,
                "base_url": new_base_url or None,
                "model": new_model or None,
            }
            if needs_key and new_api_key:
                test_kwargs["api_key"] = new_api_key
            result = test_llm_connection(**test_kwargs)
        if "error" in result:
            st.error(f"❌ {result['detail']}")
        else:
            if result.get("success"):
                st.success(
                    f"✅ {result.get('message', '成功')}  "
                    f"延迟 **{result.get('latency_ms', '-')} ms**\n\n"
                    f"> {result.get('response_text', '')[:160]}"
                )
            else:
                st.error(f"❌ {result.get('message', '连接失败')}")

    # ---- 保存配置 ----
    if save_clicked:
        active_config_payload: Dict[str, Any] = {}
        if needs_key and new_api_key:
            # 用户填了新 key → 覆盖
            active_config_payload["api_key"] = new_api_key
        if new_base_url and new_base_url != active_cfg.get("base_url"):
            active_config_payload["base_url"] = new_base_url
        if new_model and new_model != active_cfg.get("model"):
            active_config_payload["model"] = new_model

        if not active_config_payload:
            st.info("ℹ️ 未检测到任何修改，无需保存")
        else:
            result = update_llm_settings(active_config=active_config_payload)
            if "error" in result:
                st.error(f"❌ 保存失败：{result['detail']}")
            else:
                st.success("✅ 配置已保存，下次对话/创建即生效")
                time.sleep(0.5)
                st.rerun()

    st.divider()

    # ---- 区块 3：默认调用参数 ----
    st.subheader("3️⃣ 默认调用参数")
    st.caption("新建对话/角色时使用的默认 temperature 和 max_tokens。导演/演员模块的固定温度不受影响。")

    cur_temp = float(settings.get("default_temperature", 0.7))
    cur_tokens = int(settings.get("default_max_tokens", 1000))

    col1, col2 = st.columns(2)
    with col1:
        new_temp = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=cur_temp,
            step=0.05,
            key="settings_temp_slider",
        )
    with col2:
        new_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=8000,
            value=cur_tokens,
            step=100,
            key="settings_tokens_input",
        )

    if st.button("💾 保存默认参数", key="settings_save_defaults_btn"):
        changed = {}
        if abs(new_temp - cur_temp) > 1e-6:
            changed["default_temperature"] = new_temp
        if new_tokens != cur_tokens:
            changed["default_max_tokens"] = new_tokens
        if not changed:
            st.info("ℹ️ 未检测到修改")
        else:
            result = update_llm_settings(**changed)
            if "error" in result:
                st.error(f"❌ {result['detail']}")
            else:
                st.success("✅ 默认参数已更新")
                time.sleep(0.4)
                st.rerun()

    st.divider()

    # ---- 区块 4：其他 Provider 概览（只读 + 脱敏）----
    st.subheader("4️⃣ 其他 Provider 概览")
    with st.expander("展开查看全部 provider 当前状态", expanded=False):
        for m in providers_meta:
            pid = m["id"]
            pname = m["name"]
            cfg = settings.get("providers", {}).get(pid, {})
            masked = cfg.get("api_key", "")
            has_key = bool(masked) and not masked.startswith("****")
            mark = "🟢" if (m["needs_key"] == "false" or has_key) else "🟡"
            active_mark = " ⭐（当前激活）" if pid == active_id else ""
            st.markdown(
                f"{mark} **{pname}** (`{pid}`){active_mark}  \n"
                f"&nbsp;&nbsp;• base_url: `{cfg.get('base_url', '-')}`  \n"
                f"&nbsp;&nbsp;• model: `{cfg.get('model', '-')}`  \n"
                f"&nbsp;&nbsp;• api_key: `{masked or '（空）'}`"
            )

    st.caption(f"📁 配置文件：`{settings.get('settings_file_path', '-')}`")


# ============================================================
# 页面 5：API 测试（参考 web-tools/react-vite 的 API Dashboard）
# ============================================================

def render_page_api_test():
    """
    API 连通性测试页面。

    三大能力（对应 web-tools/react-vite 的功能）：
      1. 一键拉取 /v1/models 列表（含耗时）
      2. 流式延迟测试（TTFT + 总延迟 + 响应片段 + 历史）
      3. 原始请求探针（debug 模式，查看完整 request/response）

    设计原则：
      - 全部从 LLM 设置页继承 provider / API Key（不重复输入）
      - 模型列表：可点击"应用到延迟测试"以快速填入
      - 历史记录：保留在 session_state，与 web-tools 行为一致
    """
    st.title("🧪 API 测试")
    st.markdown(
        "验证 **LLM Provider 连通性**。能力移植自 "
        "[joker1point/web-tools](https://github.com/joker1point/web-tools) 的 API Dashboard。"
    )

    # ---- 拉取设置 ----
    settings = get_llm_settings()
    if "error" in settings:
        st.error(f"❌ 无法加载设置：{settings['detail']}")
        return

    active_id = settings.get("active_provider", "agnes")
    active_name = settings.get("active_provider_name", active_id)
    active_cfg = settings.get("config", {})

    # ---- Provider 选择与快速概览 ----
    st.info(
        f"📌 当前激活：**{active_name}** (`{active_id}`) · "
        f"模型 `{active_cfg.get('model', '-')}` · "
        f"base_url `{active_cfg.get('base_url', '-')}`"
    )

    providers_info = list_llm_providers()
    providers_meta = providers_info.get("providers", []) if "error" not in providers_info else []
    provider_options = [m["id"] for m in providers_meta] if providers_meta else [active_id]
    provider_labels = {
        m["id"]: f"{m['name']} ({m['id']})" for m in providers_meta
    } if providers_meta else {active_id: active_id}

    target_pid = st.selectbox(
        "🔧 测试目标 provider（与设置页激活 provider 解耦）",
        provider_options,
        index=provider_options.index(active_id) if active_id in provider_options else 0,
        format_func=lambda x: provider_labels.get(x, x),
        key="apitest_target_provider",
    )
    # 切换 provider 后，对应 cfg 也要切换（用于展示给用户看）
    target_cfg = settings.get("providers", {}).get(target_pid, {})
    if not target_cfg:
        st.warning(f"⚠️ provider `{target_pid}` 没有配置")
        return

    st.divider()

    # ---- 标签页 ----
    tab_models, tab_latency, tab_probe = st.tabs(
        ["📜 Models 列表", "⚡ 流式延迟测试", "🔍 探针 (Debug)"]
    )

    # -------------------- Tab 1: Models 列表 --------------------
    with tab_models:
        st.subheader("1️⃣ 一键拉取 /v1/models")
        st.caption(
            "调用 provider 的 `GET /v1/models` 接口，返回该账号可用的全部模型。"
            "Anthropic 兼容 provider 会自动注入 `x-api-key` 头。"
        )

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            fetch_clicked = st.button(
                "🔄 拉取 Models",
                type="primary",
                use_container_width=True,
                key="apitest_fetch_models_btn",
            )
        with col_info:
            st.caption(f"目标：{provider_labels.get(target_pid, target_pid)}")

        if fetch_clicked:
            with st.spinner("⏳ 拉取中..."):
                result = list_models(provider_id=target_pid)
            if "error" in result:
                st.error(f"❌ 拉取失败：{result['detail']}")
            else:
                st.session_state["apitest_models"] = result

        result = st.session_state.get("apitest_models")
        if result and result.get("provider_id") == target_pid:
            models = result.get("models", [])
            duration = result.get("duration_ms", 0)
            st.success(
                f"✅ 共 **{len(models)}** 个模型，耗时 **{duration}ms**"
            )
            if models:
                # 用 dataframe 展示 + 选择列（点击后填入延迟测试的 model 字段）
                df_data = [
                    {
                        "模型 ID": m.get("id", ""),
                        "所有者": m.get("owned_by", "-") or "-",
                        "对象": m.get("object", "model"),
                    }
                    for m in models
                ]
                st.dataframe(
                    df_data,
                    use_container_width=True,
                    hide_index=True,
                    key="apitest_models_table",
                )

                # "应用到延迟测试"按钮组
                st.caption("💡 点击下方按钮可将模型 ID 填入延迟测试输入框")
                cols_per_row = 4
                for i in range(0, min(len(models), 12), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx >= len(models):
                            break
                        m = models[idx]
                        if col.button(
                            f"📌 {m.get('id', '')[:24]}",
                            key=f"apitest_pick_model_{idx}",
                            use_container_width=True,
                            help=m.get("id", ""),
                        ):
                            st.session_state["apitest_latency_model"] = m.get("id", "")
                            st.toast(f"已应用模型：{m.get('id', '')}", icon="✅")

    # -------------------- Tab 2: 流式延迟测试 --------------------
    with tab_latency:
        st.subheader("2️⃣ 流式延迟测试（TTFT + 总延迟）")
        st.caption(
            "发送 `stream=true` 的 chat 请求，测量首字节时间（TTFT）与总延迟。"
            "测试参数**不写盘**，仅用于本次连通性验证。"
        )

        # 持久化测试历史
        if "apitest_latency_history" not in st.session_state:
            st.session_state["apitest_latency_history"] = []

        col_msg, col_tok = st.columns([3, 1])
        with col_msg:
            test_msg = st.text_input(
                "📝 测试消息",
                value="Hi",
                max_chars=2000,
                key="apitest_latency_msg",
            )
        with col_tok:
            max_tok = st.number_input(
                "max_tokens",
                min_value=1,
                max_value=2048,
                value=16,
                step=1,
                key="apitest_latency_max_tokens",
            )

        # 模型字段：从 session_state 读取默认值（可被"应用"按钮覆盖）
        default_model = st.session_state.get(
            "apitest_latency_model", target_cfg.get("model", ""),
        )
        model_input = st.text_input(
            "🤖 模型（可手动修改，留空则用 provider 默认）",
            value=default_model,
            key="apitest_latency_model_input",
        )

        col_run, col_clear = st.columns([1, 1])
        with col_run:
            run_clicked = st.button(
                "🚀 运行延迟测试",
                type="primary",
                use_container_width=True,
                key="apitest_run_latency_btn",
            )
        with col_clear:
            if st.button(
                "🗑️ 清空历史",
                use_container_width=True,
                key="apitest_clear_latency_btn",
            ):
                st.session_state["apitest_latency_history"] = []
                st.rerun()

        if run_clicked:
            with st.spinner("⏳ 正在测试流式响应..."):
                result = test_latency(
                    provider_id=target_pid,
                    model=model_input.strip() or None,
                    test_message=test_msg,
                    max_tokens=int(max_tok),
                )
            if "error" in result:
                st.error(f"❌ 测试失败：{result['detail']}")
            else:
                # 写入历史（带时间戳 + provider/model/msg 元信息）
                from datetime import datetime
                history = st.session_state["apitest_latency_history"]
                history.insert(0, {
                    **result,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "test_message": test_msg,
                    "max_tokens": int(max_tok),
                })
                # 保留最近 20 条
                st.session_state["apitest_latency_history"] = history[:20]

        # 展示最新一次结果
        history = st.session_state.get("apitest_latency_history", [])
        if history:
            latest = history[0]
            st.divider()
            st.markdown("##### 📊 最新一次结果")
            cols = st.columns(4)
            status = latest.get("status", 0)
            ok = 200 <= status < 300 and not latest.get("error")
            cols[0].metric("状态码", f"{status}" if status else "-",
                          delta="OK" if ok else "FAIL",
                          delta_color="normal" if ok else "inverse")
            cols[1].metric("TTFT (ms)", latest.get("ttft_ms") or "-")
            cols[2].metric("总延迟 (ms)", latest.get("total_ms") or "-")
            cols[3].metric("SSE 块数", latest.get("chunks", 0))
            if latest.get("error"):
                st.error(f"❌ 错误：{latest['error']}")
            if latest.get("content"):
                st.caption("响应片段：")
                st.code(latest["content"], language="text")

            st.divider()
            st.markdown("##### 📜 历史记录（最近 20 次）")
            hist_rows = []
            for h in history:
                hist_rows.append({
                    "时间": h.get("timestamp", "-"),
                    "状态": h.get("status", 0),
                    "TTFT (ms)": h.get("ttft_ms") or "-",
                    "总延迟 (ms)": h.get("total_ms") or "-",
                    "块数": h.get("chunks", 0),
                    "模型": h.get("model", "-")[:32],
                    "消息": (h.get("test_message", "") or "")[:30],
                })
            st.dataframe(
                hist_rows,
                use_container_width=True,
                hide_index=True,
                key="apitest_latency_history_table",
            )

    # -------------------- Tab 3: 探针 --------------------
    with tab_probe:
        st.subheader("3️⃣ 原始请求探针（Debug 模式）")
        st.caption(
            "发送一次**非流式**请求，返回完整的 `request/response` 头/体（密钥已脱敏）。"
            "用于排查 provider 鉴权、路由、协议差异。"
        )

        col_pmsg, col_ptok = st.columns([3, 1])
        with col_pmsg:
            probe_msg = st.text_input(
                "📝 测试消息",
                value="Hi",
                max_chars=2000,
                key="apitest_probe_msg",
            )
        with col_ptok:
            probe_tok = st.number_input(
                "max_tokens",
                min_value=1,
                max_value=2048,
                value=16,
                step=1,
                key="apitest_probe_max_tokens",
            )

        if st.button(
            "🔬 启动探针",
            type="primary",
            use_container_width=True,
            key="apitest_probe_btn",
        ):
            with st.spinner("⏳ 探针运行中..."):
                result = probe_llm(
                    provider_id=target_pid,
                    test_message=probe_msg,
                    max_tokens=int(probe_tok),
                )
            if "error" in result:
                st.error(f"❌ 探针失败：{result['detail']}")
            else:
                st.session_state["apitest_probe_result"] = result

        probe = st.session_state.get("apitest_probe_result")
        if probe and probe.get("provider_id") == target_pid:
            err = probe.get("error")
            resp = probe.get("response", {})
            req = probe.get("request", {})
            if err:
                st.error(f"❌ {err}")
            else:
                st.success(
                    f"✅ HTTP {resp.get('status')} · 耗时 {resp.get('duration_ms')}ms"
                )

            with st.expander("📤 Request", expanded=True):
                st.markdown(f"**{req.get('method', 'POST')}** `{req.get('url', '-')}`")
                st.json({
                    "headers": req.get("headers", {}),
                    "body": req.get("body", {}),
                })

            with st.expander("📥 Response Headers", expanded=False):
                st.json(resp.get("headers", {}))

            with st.expander("📥 Response Body", expanded=True):
                body = resp.get("body", "")
                if isinstance(body, (dict, list)):
                    st.json(body)
                else:
                    st.code(body, language="text")


# ============================================================
# 主入口 + 侧边栏
# ============================================================

def main():
    """
    主函数：渲染侧边栏导航 → 分发到对应页面。

    为何不使用 Streamlit 的 pages/ 自动导航：
      pages/ 方案要求每个页面一个文件，三个页面需要共享
      selected_character_id 状态。在 st 1.29 中，跨页面
      session_state 共享有时不稳定，尤其在 rerun 时。
      手动 sidebar.radio 控制页面切换，所有组件在同一
      脚本上下文中，session_state 天然共享。
    """

    # ---- 侧边栏 ----
    with st.sidebar:
        # 直接使用 emoji 字符以避免 Streamlit magic 机制误判。
        st.markdown("# 🌱")
        st.title("CharacterSeed")

        # 后端健康检查
        # 为何每次 rerun 都检查一次：
        #   后端可能在中途崩溃或重启。实时检查比"启动时检查一次"更可靠。
        #   使用 st.cache_data(ttl=30) 缓存 30 秒，避免过度频繁的请求。
        @st.cache_data(ttl=30)
        def _cached_health_check():
            return check_backend_health()

        backend_ok = _cached_health_check()
        if backend_ok:
            st.success("🟢 后端连接正常")
        else:
            st.error(
                "🔴 后端未启动！\n\n"
                "请在另一个终端运行：\n"
                "```bash\n"
                "uvicorn backend.main:app --reload --port 8000\n"
                "```"
            )

        st.divider()

        # 页面导航
        # 为何用 radio 而非 selectbox：
        #   radio 让所有选项始终可见，用户一目了然有三个页面。
        #   selectbox 需要点击展开，对首次使用的用户不够友好。
        page = st.radio(
            "📂 功能导航",
            ["🌱 角色创建", "💬 对话交互", "📊 角色状态", "⚙️ LLM 设置", "🧪 API 测试"],
            key="nav_radio",
        )

        st.divider()

        # 应用统计信息
        characters = get_characters()
        st.caption(f"📦 数据库中有 **{len(characters)}** 个角色")

        if st.session_state.get("selected_character_id"):
            st.caption(f"当前选中角色 ID: **{st.session_state.selected_character_id}**")

    # ---- 页面分发 ----
    if page == "🌱 角色创建":
        render_page_create()
    elif page == "💬 对话交互":
        render_page_chat()
    elif page == "📊 角色状态":
        render_page_dashboard()
    elif page == "⚙️ LLM 设置":
        render_page_settings()
    elif page == "🧪 API 测试":
        render_page_api_test()


if __name__ == "__main__":
    main()
