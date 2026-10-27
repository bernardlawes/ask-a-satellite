"""MCP server exposing the core tools to any MCP client (Milestone 3).

Each function in ``ask_a_satellite.tools`` is registered as an MCP tool with a JSON schema derived from
its pydantic models. Rasters stay server-side; tools exchange scene/crop IDs and small JSON payloads.
"""
