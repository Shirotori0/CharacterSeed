"""
数据库迁移工具

用途：处理 schema 演进。当前内置两个迁移：
  - MIGRATION_V001_SESSIONS: 引入 ChatSession，给 Conversation 加 session_id
    并把存量"孤儿"对话回填到每个角色的"默认会话"。

设计原则：
  - 幂等：可重复执行，不会重复添加列/重复回填
  - 不依赖 alembic 等第三方库，纯 SQL 兼容性最大
  - 失败抛异常让启动失败，便于及早发现
"""
import logging
import sqlite3
from typing import List

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def _sqlite_columns(engine: Engine, table: str) -> List[str]:
    """读取 SQLite 表的列名列表（其它数据库 PRAGMA 行为可能不同）"""
    if not engine.url.get_backend_name().startswith("sqlite"):
        # 其它数据库暂时不处理，启动后端时跳过迁移
        return []
    with engine.connect() as conn:
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return [r[1] for r in rows]


def _sqlite_table_exists(engine: Engine, table: str) -> bool:
    if not engine.url.get_backend_name().startswith("sqlite"):
        return False
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:n"),
            {"n": table},
        ).fetchone()
    return row is not None


def migrate_v001_sessions(engine: Engine) -> dict:
    """
    迁移 v001：引入 ChatSession + Conversation.session_id + 回填

    步骤：
      1) 确保 chat_sessions 表存在（Base.metadata.create_all 已创建）
      2) 给 conversations 加 session_id 列（若已存在则跳过）
      3) 给每条 session_id IS NULL 的对话，分配到一个名为"默认会话"的 session
         （按 character_id 分组，每个角色一个默认 session）

    Returns:
        {"added_column": bool, "backfilled": int, "default_sessions_created": int}
    """
    result = {"added_column": False, "backfilled": 0, "default_sessions_created": 0}

    if not _sqlite_table_exists(engine, "conversations"):
        return result  # 全新库，不需要迁移

    # 1) 加列
    cols = _sqlite_columns(engine, "conversations")
    if "session_id" not in cols:
        logger.info("迁移 v001: 添加 conversations.session_id 列")
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE conversations ADD COLUMN session_id INTEGER "
                "REFERENCES chat_sessions(id) ON DELETE CASCADE"
            ))
        result["added_column"] = True
    else:
        logger.debug("迁移 v001: conversations.session_id 已存在，跳过")

    # 2) 回填
    with engine.connect() as conn:
        # 找所有存在孤儿对话的 character_id
        rows = conn.execute(text(
            "SELECT DISTINCT character_id FROM conversations WHERE session_id IS NULL"
        )).fetchall()
    char_ids = [r[0] for r in rows]
    if not char_ids:
        logger.debug("迁移 v001: 无孤儿对话，无需回填")
        return result

    logger.info("迁移 v001: 为 %d 个角色回填默认会话", len(char_ids))
    with engine.begin() as conn:
        for cid in char_ids:
            # 用最早一条对话的时间作为 created_at（让默认会话在列表里更靠下）
            earliest = conn.execute(text(
                "SELECT MIN(timestamp) FROM conversations "
                "WHERE character_id = :cid AND session_id IS NULL"
            ), {"cid": cid}).scalar()

            # 创建默认 session
            res = conn.execute(text(
                "INSERT INTO chat_sessions (character_id, title, created_at, updated_at) "
                "VALUES (:cid, :title, :ts, :ts)"
            ), {"cid": cid, "title": "默认会话", "ts": earliest})
            new_sid = res.lastrowid
            result["default_sessions_created"] += 1

            # 把该角色的所有孤儿对话指给新 session
            upd = conn.execute(text(
                "UPDATE conversations SET session_id = :sid "
                "WHERE character_id = :cid AND session_id IS NULL"
            ), {"sid": new_sid, "cid": cid})
            result["backfilled"] += upd.rowcount or 0

    logger.info(
        "迁移 v001 完成: 加列=%s, 回填=%d 条, 创建默认会话=%d",
        result["added_column"], result["backfilled"], result["default_sessions_created"],
    )
    return result


def migrate_v002_event_system(engine: Engine) -> dict:
    """
    迁移 v002：Event 表 + Character 新列 + GrowthLog 新列

    步骤：
      1. 创建 events 表（若已存在则跳过）
      2. Character 加列：day_number / speaking_style / values / habits / long_term_goal
      3. GrowthLog 加列：schedule_json / world_changes_json

    幂等设计：
      - 各步骤检查列是否存在后决定是否执行
      - events 表通过 sqlite_master 检查是否存在

    设计考量（Why this migration approach）：
      在 SQLite 中 ALTER TABLE 无法一次加多列，必须逐列操作。
      每列添加前用 PRAGMA table_info 检查存在性，确保迁移可重复执行。
    """
    result = {
        "events_table_created": False,
        "character_columns_added": {},
        "growthlog_columns_added": {},
    }

    if not engine.url.get_backend_name().startswith("sqlite"):
        return result

    # ---- 1) 建 events 表 ----
    if not _sqlite_table_exists(engine, "events"):
        logger.info("迁移 v002: 创建 events 表")
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id INTEGER NOT NULL REFERENCES characters(id),
                    day_number INTEGER NOT NULL DEFAULT 1,
                    order_index INTEGER NOT NULL DEFAULT 1,
                    event_type VARCHAR(30) NOT NULL DEFAULT 'schedule_action',
                    content TEXT NOT NULL,
                    metadata_json TEXT,
                    result_json TEXT,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    session_id INTEGER REFERENCES chat_sessions(id) ON DELETE SET NULL,
                    time_period VARCHAR(20),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            # 创建复合索引
            conn.execute(text("""
                CREATE INDEX ix_events_char_day_order
                ON events(character_id, day_number, order_index)
            """))
            conn.execute(text("""
                CREATE INDEX ix_events_char_day_status
                ON events(character_id, day_number, status)
            """))
            conn.execute(text("""
                CREATE INDEX ix_events_character_id
                ON events(character_id)
            """))
        result["events_table_created"] = True
    else:
        logger.debug("迁移 v002: events 表已存在，跳过")

    # ---- 2) Character 加列 ----
    char_cols = _sqlite_columns(engine, "characters")
    char_additions = {
        "day_number": "INTEGER DEFAULT 1",
        "speaking_style": "TEXT",
        "values": "TEXT",
        "habits": "TEXT",
        "long_term_goal": "TEXT",
    }
    with engine.begin() as conn:
        for col_name, col_type in char_additions.items():
            if col_name not in char_cols:
                logger.info("迁移 v002: characters 加列 %s", col_name)
                conn.execute(text(
                    f'ALTER TABLE characters ADD COLUMN "{col_name}" {col_type}'
                ))
                result["character_columns_added"][col_name] = True

    # ---- 3) GrowthLog 加列 ----
    gl_cols = _sqlite_columns(engine, "growth_logs")
    gl_additions = {
        "schedule_json": "TEXT",
        "world_changes_json": "TEXT",
    }
    with engine.begin() as conn:
        for col_name, col_type in gl_additions.items():
            if col_name not in gl_cols:
                logger.info("迁移 v002: growth_logs 加列 %s", col_name)
                conn.execute(text(
                    f'ALTER TABLE growth_logs ADD COLUMN "{col_name}" {col_type}'
                ))
                result["growthlog_columns_added"][col_name] = True

    logger.info("迁移 v002 完成: %s", result)
    return result


def migrate_v003_world_system(engine: Engine) -> dict:
    """
    迁移 v003：World + Scene + SceneChange 三表 + Character.world_id /
    current_scene_id / short_term_goals + 存量角色回填

    步骤：
      1. 创建 worlds / scenes / scene_changes 表（若已存在则跳过）
      2. Character 加列：world_id / current_scene_id / short_term_goals
      3. 创建场景变更索引
      4. 存量角色回填：为每个无 world_id 的角色创建默认 World + Scene

    幂等设计：
      - 各步骤检查表/列是否存在后决定是否执行
      - 存量回填仅对 world_id IS NULL 的角色执行（避免重复创建）

    存量回填策略：
      对每个无 world_id 的 character：
        1. INSERT INTO worlds (name, core_worldview)
           VALUES (角色名 + "的世界", world_setting 前 200 字)
        2. INSERT INTO scenes (world_id, name, scene_layer='conceptual',
           scene_type='region', description)
        3. INSERT INTO scenes (world_id, parent_scene_id, name,
           scene_layer='actual', scene_type='location', description)
           VALUES (from current_state.location 或 "初始位置")
        4. UPDATE characters SET world_id=?, current_scene_id=?
    """
    result = {
        "worlds_created": False,
        "scenes_created": False,
        "scene_changes_created": False,
        "character_columns_added": {},
        "characters_backfilled": 0,
    }

    if not engine.url.get_backend_name().startswith("sqlite"):
        return result

    # ---- 1) 建 worlds 表 ----
    if not _sqlite_table_exists(engine, "worlds"):
        logger.info("迁移 v003: 创建 worlds 表")
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE worlds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(100) NOT NULL,
                    core_worldview TEXT NOT NULL DEFAULT '',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
        result["worlds_created"] = True
    else:
        logger.debug("迁移 v003: worlds 表已存在，跳过")

    # ---- 2) 建 scenes 表 ----
    if not _sqlite_table_exists(engine, "scenes"):
        logger.info("迁移 v003: 创建 scenes 表")
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE scenes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    world_id INTEGER NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,
                    name VARCHAR(100) NOT NULL,
                    scene_layer VARCHAR(20) NOT NULL,
                    scene_type VARCHAR(30),
                    parent_scene_id INTEGER REFERENCES scenes(id) ON DELETE SET NULL,
                    description TEXT,
                    initial_description TEXT,
                    attributes_json TEXT,
                    created_day INTEGER DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text(
                "CREATE INDEX ix_scenes_world_id ON scenes(world_id)"
            ))
            conn.execute(text(
                "CREATE INDEX ix_scenes_parent_scene_id ON scenes(parent_scene_id)"
            ))
        result["scenes_created"] = True
    else:
        logger.debug("迁移 v003: scenes 表已存在，跳过")

    # ---- 3) 建 scene_changes 表 ----
    if not _sqlite_table_exists(engine, "scene_changes"):
        logger.info("迁移 v003: 创建 scene_changes 表")
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE scene_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scene_id INTEGER NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
                    growth_log_id INTEGER REFERENCES growth_logs(id) ON DELETE SET NULL,
                    change_type VARCHAR(20) NOT NULL,
                    description TEXT NOT NULL,
                    change_details_json TEXT,
                    day_number INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text(
                "CREATE INDEX ix_scene_changes_scene_id ON scene_changes(scene_id)"
            ))
            conn.execute(text(
                "CREATE INDEX ix_scene_changes_scene_day "
                "ON scene_changes(scene_id, day_number)"
            ))
        result["scene_changes_created"] = True
    else:
        logger.debug("迁移 v003: scene_changes 表已存在，跳过")

    # ---- 4) Character 加列 ----
    char_cols = _sqlite_columns(engine, "characters")
    char_additions = {
        "world_id": "INTEGER REFERENCES worlds(id) ON DELETE SET NULL",
        "current_scene_id": "INTEGER REFERENCES scenes(id) ON DELETE SET NULL",
        "short_term_goals": "TEXT",
    }
    with engine.begin() as conn:
        for col_name, col_type in char_additions.items():
            if col_name not in char_cols:
                logger.info("迁移 v003: characters 加列 %s", col_name)
                conn.execute(text(
                    f'ALTER TABLE characters ADD COLUMN "{col_name}" {col_type}'
                ))
                result["character_columns_added"][col_name] = True

    # ---- 5) 存量角色回填 ----
    if not _sqlite_table_exists(engine, "characters"):
        return result

    with engine.connect() as conn:
        orphan_rows = conn.execute(text(
            "SELECT id, name, world_setting, current_state "
            "FROM characters WHERE world_id IS NULL"
        )).fetchall()
    if not orphan_rows:
        logger.debug("迁移 v003: 无待回填角色")
        return result

    logger.info("迁移 v003: 回填 %d 个存量角色", len(orphan_rows))
    with engine.begin() as conn:
        for cid, cname, cworld, cstate in orphan_rows:
            world_name = f"{cname}的世界"
            worldview = (cworld or "一个充满未知的世界")[:200]

            # 创建默认 World
            wr = conn.execute(text(
                "INSERT INTO worlds (name, core_worldview) VALUES (:name, :wv)"
            ), {"name": world_name, "wv": worldview})
            world_id = wr.lastrowid

            # 创建概念场景（根节点）
            cr = conn.execute(text(
                "INSERT INTO scenes (world_id, name, scene_layer, scene_type, "
                "description, initial_description, created_day) "
                "VALUES (:wid, :name, 'conceptual', 'region', :desc, :desc, 1)"
            ), {"wid": world_id, "name": "起始之地", "desc": worldview[:200]})
            conceptual_id = cr.lastrowid

            # 从 current_state JSON 提取 location 文本
            import json as _json
            location_text = "初始位置"
            try:
                if cstate:
                    state_dict = _json.loads(cstate)
                    location_text = state_dict.get("location", "初始位置")
            except (ValueError, TypeError):
                location_text = cstate if cstate else "初始位置"

            # 创建实际场景
            ar = conn.execute(text(
                "INSERT INTO scenes (world_id, name, scene_layer, scene_type, "
                "parent_scene_id, description, initial_description, created_day) "
                "VALUES (:wid, :name, 'actual', 'location', :pid, :desc, :desc, 1)"
            ), {
                "wid": world_id, "name": location_text, "pid": conceptual_id,
                "desc": f"角色初始位置：{location_text}",
            })
            actual_id = ar.lastrowid

            # 更新 Character FK
            conn.execute(text(
                "UPDATE characters SET world_id = :wid, current_scene_id = :sid "
                "WHERE id = :cid"
            ), {"wid": world_id, "sid": actual_id, "cid": cid})
            result["characters_backfilled"] += 1

    logger.info(
        "迁移 v003 完成: worlds=%s, scenes=%s, changes=%s, 回填=%d",
        result["worlds_created"], result["scenes_created"],
        result["scene_changes_created"], result["characters_backfilled"],
    )
    return result


def run_all_migrations(engine: Engine) -> List[dict]:
    """
    按版本顺序执行所有迁移。在应用启动时调用一次。
    新增迁移时在此函数中追加。
    """
    history = []
    history.append({
        "version": "v001_sessions",
        **migrate_v001_sessions(engine),
    })
    history.append({
        "version": "v002_event_system",
        **migrate_v002_event_system(engine),
    })
    history.append({
        "version": "v003_world_system",
        **migrate_v003_world_system(engine),
    })
    history.append({
        "version": "v004_event_narrative_metadata",
        **migrate_v004_event_narrative(engine),
    })
    return history


def migrate_v004_event_narrative(engine: Engine) -> dict:
    """
    迁移 v004：Event 表 v1.6 叙事元数据字段

    新增列：
      - director_raw           TEXT   Director LLM 原始响应 JSON
      - actor_raw              TEXT   Actor LLM 原始响应 JSON
      - capabilities_applied   TEXT   角色选择的能力列表（JSON 数组）
      - emotion                VARCHAR(50)  角色处理事件时的情绪
      - expression             VARCHAR(100) 角色处理事件时的表情

    幂等设计：逐列检查是否存在，跳过已存在的列。
    """
    result = {
        "events_columns_added": {},
    }

    if not engine.url.get_backend_name().startswith("sqlite"):
        return result

    if not _sqlite_table_exists(engine, "events"):
        logger.debug("迁移 v004: events 表不存在，由 v002 创建时直接包含新列")
        return result

    cols = _sqlite_columns(engine, "events")
    additions = {
        "director_raw": "TEXT",
        "actor_raw": "TEXT",
        "capabilities_applied": "TEXT",
        "emotion": "VARCHAR(50)",
        "expression": "VARCHAR(100)",
    }
    with engine.begin() as conn:
        for col_name, col_type in additions.items():
            if col_name not in cols:
                logger.info("迁移 v004: events 加列 %s", col_name)
                conn.execute(text(
                    f'ALTER TABLE events ADD COLUMN "{col_name}" {col_type}'
                ))
                result["events_columns_added"][col_name] = True

    logger.info("迁移 v004 完成: %s", result)
    return result
