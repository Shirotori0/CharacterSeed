# Changelog

## [Unreleased] — 2026-06-22 — Langfuse LLM Observability Integration

### ✨ New Features
- **端到端 LLM 可观测性**：集成 Langfuse v3（self-hosted via docker compose），所有 LLM 调用自动上报 trace 到 Langfuse UI
- **多 pipeline 追踪点**：在 `interaction.run` / `director.analyze` / `actor.generate` / `creation.run` / `growth.run` 5 个核心 pipeline 加 `@observe_safe` 装饰器
- **trace 元数据自动填充**：每次 chat/creation/growth 都自动带 `user_id` / `session_id` / `tags=[character_id=N, pipeline_name]` / `release`
- **drop-in OpenAI 替换**：`LLMService` 用 `get_openai_class()` 自动选择 `langfuse.openai.OpenAI` ↔ `openai.OpenAI`（健康检查失败时回退）
- **10 轮追踪演示脚本**：`scripts/demo_langfuse_tracing.py`，mock LLM + mock Langfuse 上报，输出 ASCII trace 树状图（1-2s 跑完 10 轮）
- **优雅降级**：Langfuse 不可达/key 无效时，主流程零影响；启动日志显示 `📊 Langfuse tracing: ⚪ 未启用` 提示

### 🔧 Changes
- `backend/services/observability.py`（新建，200 行）：`is_enabled / observe_safe / update_current_trace / flush / get_openai_class / score_current_trace` 6 个 API
- `backend/services/llm_service.py`：1 行改动用 `get_openai_class()` 替代硬编码 `openai.OpenAI`
- `backend/modules/interaction.py`：3 个 `@observe_safe` 装饰器（run / director.analyze / actor.generate）+ 2 个 `update_current_trace` 调用
- `backend/modules/creation.py`：2 个 `@observe_safe` 装饰器（run / call_llm as generation）+ 1 个 `update_current_trace` 调用
- `backend/modules/growth.py`：1 个 `@observe_safe` 装饰器（run）
- `backend/main.py`：启动日志显示 Langfuse 状态；`shutdown_event` 调 `flush()` 防丢 trace
- `backend/config.py`：+6 行 LANGFUSE_* 配置
- `.env.example`：+12 行 Langfuse 配置示例
- `requirements.txt`：+1 行 `langfuse>=2.40.0,<3.0.0`（锁版本，规避 v3 兼容问题）

### 📊 Observability 验证
启动后端跑 1 轮 chat，Langfuse UI 可见：
- **1 个 trace**：`name=interaction.run`
- **4 个 observation**：
  - `OpenAI-generation` × 2（Director + Actor 的 LLM 调用，由 langfuse.openai 自动记录）
  - `actor.generate` × 1（自定义 span）
  - `director.analyze` × 1（自定义 span）
- **trace 元数据**：`userId=anonymous`, `sessionId=100`, `tags=[character_id=4, director+actor, interaction]`, `release=docker-selfhost-2026-06-22`

### 🐛 Fixed
- **5 个部署踩坑**（详见 [langfuse-selfhost-deploy skill](file:///c:/Users/biren/.trae-cn/builtin/global/skills/langfuse-selfhost-deploy/SKILL.md)）：
  1. Base64 密码含 `+/=` 触发 "invalid port number in database URL"
  2. Postgres 旧 volume 残留导致认证失败（`docker compose down -v`）
  3. `langfuse.Langfuse().auth_check()` 在 uvicorn 进程里永远失败（需显式传 key/host）
  4. S3 SignatureDoesNotMatch（Langfuse v3 把 S3 secret 存在 `blob_storage_secrets` 表，默认 `miniosecret`，需让 MinIO 启动密码与 DB 初始化值一致）
  5. health_check 失败回退后 `@observe_safe` 变 identity（trace 永远不进上报队列）

### 📦 Deployment

#### 1. 启动 Langfuse 自部署版（6 容器）
```bash
cd ../langfuse-local  # 仓库同级目录
docker compose up -d
# 等待 90s 让 web/worker 启动 + 跑 background_migrations
```

#### 2. 配置 .env
```bash
# CharacterSeed/.env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-characterseed-demo-2026
LANGFUSE_SECRET_KEY=sk-lf-characterseed-demo-secret-2026
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_SAMPLE_RATE=1.0
LANGFUSE_RELEASE=docker-selfhost-2026-06-22
```

#### 3. 启动后端
```bash
# 启动日志会显示: 📊 Langfuse tracing: ✅ 已启用
uvicorn backend.main:app --reload
```

#### 4. 验证
```bash
# 跑 1 轮 chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"character_id":4,"message":"hello","session_id":null}'

# 等 15-30s（worker 消费 S3 → ClickHouse）
curl -u "pk-lf-characterseed-demo-2026:sk-lf-characterseed-demo-secret-2026" \
  http://localhost:3000/api/public/traces?limit=5
# 期望: count >= 1
```

### 🛠️ Resources
- **Langfuse UI**：http://localhost:3000（用 `admin@mylocal.local` / `AdminPass123` 登录）
- **掘金文章**：https://juejin.cn/post/7633462423300407336
- **部署避坑 Skill**：[langfuse-selfhost-deploy](file:///c:/Users/biren/.trae-cn/builtin/global/skills/langfuse-selfhost-deploy/SKILL.md)
- **10 轮追踪演示**：`python -m scripts.demo_langfuse_tracing`

### ⚠️ Backward Compatibility
- 完全向后兼容：未配 `LANGFUSE_ENABLED=true` 或环境变量缺失时，`@observe_safe` 退化为 identity 装饰器，行为与之前完全一致
- 现有调用方（main.py / interaction.py / creation.py / growth.py / llm_service.py）的 API 签名**零改动**
- 唯一运行时差异：每次 LLM 调用多 ~5ms（trace 上报异步，不阻塞主流程）
