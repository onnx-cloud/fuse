# Advanced LLM Features: Selectable Engines, Streaming (SSE), Logging & Rate Limiting

This document summarizes the enhancements added to the LLM integration:

- Selectable engines: Frontend queries `/fuse/api/llm/engines` to discover configured engines and their friendly labels; the chat UI shows a dropdown to pick which engine to use.
- Rate limiting: A simple in-memory per-IP limit (requests per minute) is enforced using the `FUSE_LLM_RATE_PER_MIN` environment variable (default 60). Responses include `X-RateLimit-Remaining` header.
- Audit logging: Each LLM call is appended to `jupyter/logs/llm_access.log` containing timestamp, client IP, engine, request payload (some fields), and status code. This file is created automatically and uses JSON-lines for ease of processing.
- Streaming (SSE): A streaming endpoint is scaffolded at `/fuse/api/llm/stream` (best-effort) and can be wired to compatible provider streaming responses. The chat widget supports receiving streaming updates via `EventSource` if the endpoint supports SSE.

Notes & limitations:
- Rate limiting is in-memory and per-server process; for production use behind multiple nodes, use a shared store (Redis) for counters.
- Audit logs are local files; consider shipping logs to a centralized logging/ELK system for analysis.
- Streaming implementation is provider-specific; for robust streaming support consider implementing a provider-specific adapter if you need chunked responses or SSE translations.

