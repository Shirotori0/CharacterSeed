import json
from sqlalchemy.orm import Session
from backend.models import Character
from typing import Optional, List, Union

def get_character(db: Session, character_id: int):
    """获取单个角色"""
    return db.query(Character).filter(Character.id == character_id).first()

def get_characters(db: Session, skip: int = 0, limit: int = 100):
    """获取角色列表"""
    return db.query(Character).offset(skip).limit(limit).all()

def create_character(
    db: Session,
    name: str,
    description: Optional[str] = None,
    world_setting: Optional[str] = None,
    personality: Optional[Union[str, dict]] = None,
    current_state: Optional[Union[str, dict]] = None,
    creation_raw: Optional[str] = None,
    # Day4 新增：人格扩充字段（已由调用方序列化为 JSON 字符串传入）
    speaking_style: Optional[str] = None,
    values: Optional[str] = None,
    habits: Optional[str] = None,
    long_term_goal: Optional[str] = None,
    # v1.6 Phase 3 新增：短期目标（JSON 数组字符串）
    short_term_goals: Optional[str] = None,
):
    """
    创建角色
    
    personality 和 current_state 支持传入 dict（自动序列化为 JSON 字符串）
    或直接传入 JSON 字符串（向后兼容），统一在 CRUD 层完成序列化，
    避免调用方（API handler、测试脚本、其他模块）重复编写 json.dumps()。
    """
    # 统一将 dict 序列化为 JSON 字符串
    if isinstance(personality, dict):
        personality = json.dumps(personality, ensure_ascii=False)
    if isinstance(current_state, dict):
        current_state = json.dumps(current_state, ensure_ascii=False)

    db_character = Character(
        name=name,
        description=description,
        world_setting=world_setting,
        personality=personality,
        current_state=current_state,
        creation_raw=creation_raw,
        # Day4 新增：人格扩充字段
        speaking_style=speaking_style,
        values=values,
        habits=habits,
        long_term_goal=long_term_goal,
        # v1.6 Phase 3 新增：短期目标
        short_term_goals=short_term_goals,
    )
    db.add(db_character)
    db.commit()
    db.refresh(db_character)
    return db_character

def update_character(db: Session, character_id: int, **kwargs):
    """更新角色
    
    personality 和 current_state 支持传入 dict（自动序列化为 JSON 字符串），
    或直接传入 JSON 字符串（向后兼容）。
    """
    db_character = db.query(Character).filter(Character.id == character_id).first()
    if db_character:
        for key, value in kwargs.items():
            # 自动序列化 dict → JSON 字符串（与 create_character 保持一致）
            if key in ("personality", "current_state") and isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False)
            setattr(db_character, key, value)
        db.commit()
        db.refresh(db_character)
    return db_character

def delete_character(db: Session, character_id: int):
    """删除角色（仅删除主记录，不含级联）"""
    db_character = db.query(Character).filter(Character.id == character_id).first()
    if db_character:
        db.delete(db_character)
        db.commit()
        return True
    return False

def cascade_delete_character(db: Session, character_id: int) -> dict:
    """
    级联删除角色及其所有关联数据。
    
    按顺序删除：events → memories → conversations → growth_logs → characters，
    确保无孤儿记录残留。使用 query.delete() 批量删除关联数据，
    避免逐条加载再 delete 的性能开销。
    """
    from backend.models import Memory, Conversation, GrowthLog, Event

    db_character = db.query(Character).filter(Character.id == character_id).first()
    if not db_character:
        return {"deleted": False, "name": None}

    name = db_character.name

    # 按子表→主表顺序批量删除
    # Day4 新增：events 表级联清理
    events_count = db.query(Event).filter(Event.character_id == character_id).delete()
    mem_count = db.query(Memory).filter(Memory.character_id == character_id).delete()
    conv_count = db.query(Conversation).filter(Conversation.character_id == character_id).delete()
    growth_count = db.query(GrowthLog).filter(GrowthLog.character_id == character_id).delete()
    db.delete(db_character)
    db.commit()

    return {
        "deleted": True,
        "name": name,
        "events_deleted": events_count,
        "memories_deleted": mem_count,
        "conversations_deleted": conv_count,
        "growth_logs_deleted": growth_count,
    }
