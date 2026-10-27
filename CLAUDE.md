# Ask a Satellite: notes for coding agents

- Public portfolio project. Do not reference any client work or proprietary methods from the author's consulting practice.
- Never invent results. Every number in docs, README, reports, or posts must come from an actual run. Leave `TBD` otherwise.
- Imagery credit on every image/report: "Contains modified Copernicus Sentinel-2 data (YEAR)".
- Secrets only via environment variables (`.env`, see `.env.example`). Never commit `.env`, data, or model weights.
- Do not commit or push without the author's explicit approval.
- Tooling: `uv sync`, `uv run pytest` (offline; `-m network` for live STAC tests), `uv run ruff check .`.
- Layout: core tools in `src/ask_a_satellite/tools/` (no LLM/MCP imports), MCP adapter in `mcp_server/`, agent loop in `agent/`. See `docs/architecture.md`.
