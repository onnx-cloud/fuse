# LLM / Copilot Integration (Mini Chat) 🧭

This document describes the mini chat / copilot integration that ships with Fuse's Jupyter experience.
It uses a server-side proxy to call configured LLM endpoints (so API keys remain secret) and a lightweight JupyterLab widget to chat.

## Security model
- The frontend never contains secrets. The server reads secret environment variables named in the LLM config (e.g., `DEEPSEEK_API_KEY`).
- Only pre-configured LLM entries from `jupyter/config/llm_config.json` will be callable — frontends cannot call arbitrary external URLs.

## Config
Create `jupyter/config/llm_config.json` (or copy `jupyter/config/llm_config.json.example`) with structure:

```json
{
  "llm": {
    "think": {
      "model": "deepseek-thinking",
      "url": "https://api.deepseek.com/chat/completions",
      "secretEnv": "DEEPSEEK_API_KEY",
      "prompt": "You are a helpful assistant.",
      "label": "Deep Thoughts"
    }
  }
}
```

Fields:
- `model`: model name to send to provider
- `url`: endpoint to POST to (provider API)
- `secretEnv`: environment variable name containing API key
- `prompt`: optional system prompt to include
- `label`: friendly label shown in UI

## Server endpoint
- `POST /fuse/api/llm` accepts JSON: `{ "engine": "think", "messages": [{role:"user", content:"Hello"}], "stream": false }`
- Server validates `engine` exists in config, reads `secretEnv`, and POSTs to the configured `url` with Authorization header `Bearer <secret>`.
- Server returns the provider JSON response as-is.

## Frontend widget
- The extension provides command **Fuse: Open Copilot Chat** that opens a chat panel.
- The chat sends messages to `/fuse/api/llm` and renders returned messages.

## Running locally
1. Set your secret env var (example): `export DEEPSEEK_API_KEY=sk_...`
2. Put `jupyter/config/llm_config.json` in the repo, or set up environment-based configuration in your deployment.
3. Start Jupyter with the project config and open the command palette → **Fuse: Open Copilot Chat**.

## Notes & limitations
- This is a lightweight, secure proxy for basic chat integratation. Streaming responses are not implemented (the proxy currently handles non-stream responses only).
- The server performs basic signature behavior (Authorization header) and does not validate provider-specific schemas beyond performing a JSON POST.

If you'd like, I can:
- Add streaming support (SSE/websockets) for provider streaming responses.
- Add support for multiple LLM endpoints selectable in the UI.
- Add audit logging / rate limiting for the server proxy.

Pick one and I'll implement it.
