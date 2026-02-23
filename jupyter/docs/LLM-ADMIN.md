# LLM Engine Management (Admin)

You can manage engines via the UI or the server endpoints. This is intentionally gated behind the env var `FUSE_LLM_ADMIN_ENABLED=1` for safety.

Server endpoints (admin-only):
- GET /fuse/api/llm/admin — list engines
- POST /fuse/api/llm/admin/<engine> — create/update engine (body = engine config)
- DELETE /fuse/api/llm/admin/<engine> — delete engine

Local config file is `jupyter/config/llm_config.json`. The admin endpoints update this file directly; in production, prefer a managed config store.

Security: only enable admin in trusted environments (set `FUSE_LLM_ADMIN_ENABLED=1`). All admin actions are audit logged to `jupyter/logs/llm_access.log`.
