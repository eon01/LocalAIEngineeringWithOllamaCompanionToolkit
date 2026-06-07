# config.py
"""Central config. Loads .env once, exposes constants.

Real environment variables win over .env values, so a shell `export
OLLAMA_HOST=...` overrides whatever is in the file. That's what you
want when switching between local and a remote server.
"""

import os

from dotenv import load_dotenv

# override=False means real env vars beat the .env file.
load_dotenv(override=False)

OLLAMA_HOST: str = os.environ.get(
    "OLLAMA_HOST", "http://localhost:11434"
)
OLLAMA_MODEL: str = os.environ.get(
    "OLLAMA_MODEL", "gemma4:26b"
)

# When true, print extra diagnostic lines (e.g. memory write lifecycle).
# Errors are always shown regardless of this flag.
DEBUG: bool = os.environ.get("DEBUG", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Model used by mem0 to extract durable facts from conversations.
# Needs to be smart enough to distinguish "the user said X" from
# generic assistant text. A 2B model is too small for reliable
# extraction; 3B is the practical minimum.
EXTRACTION_MODEL: str = os.environ.get(
    "EXTRACTION_MODEL", "granite3.3:2b"
)

# Model used to turn text into vectors for similarity search.
EMBED_MODEL: str = os.environ.get(
    "EMBED_MODEL", "nomic-embed-text:v1.5"
)

# Collection name inside Chroma. One collection per app is fine.
COLLECTION_NAME: str = os.environ.get(
    "COLLECTION_NAME", "my_memories"
)

# Where mem0 keeps its vector store on disk. Survives restarts.
# Delete this directory to wipe all memories and start fresh.
MEMORY_DB_PATH: str = os.path.expanduser(
    os.environ.get("MEMORY_DB_PATH", "/var/data/ollama")
)

# Redis URL backing LangChain's global LLM-response cache. Repeating an
# identical prompt returns the cached reply instantly without hitting
# the model. Run `FLUSHALL` in redis-cli to wipe the cache.
REDIS_URL: str = os.environ.get(
    "REDIS_URL", "redis://localhost:6379"
)

# Time-to-live (seconds) for cached LLM responses. Set to 0 or negative
# to disable expiry. Default: 1 hour.
REDIS_CACHE_TTL: int = int(
    os.environ.get("REDIS_CACHE_TTL", "3600")
)

# Maximum cosine distance (Chroma raw, lower = more similar) for a memory
# to be considered relevant to the current query. mem0's own `threshold`
# is unusable - its normalized score saturates at 1.0 for irrelevant
# content. With nomic-embed-text:v1.5, relevant matches sit below ~1.0
# and noise sits above ~1.2. Lower this to be stricter.
MEMORY_RELEVANCE_THRESHOLD: float = float(
    os.environ.get("MEMORY_RELEVANCE_THRESHOLD", "1.0")
)

# Streamable-HTTP endpoint exposed by open-meteo-mcp when started with
# TRANSPORT=http. Override with OPEN_METEO_MCP_URL to point at a remote
# instance.
MCP_SERVER_URL: str = os.environ.get(
    "OPEN_METEO_MCP_URL", "http://localhost:3000/mcp"
)

# The MCP server exposes ~14 tools (weather_forecast, gem_weather,
# gfs_seamless, marine_weather, flood_forecast, ...). Small tool-calling
# models get overwhelmed by a long catalog of similar-looking names and
# start narrating tool calls in plain text instead of actually emitting
# them. Whitelist only what this REPL needs. Comma-separated in env.
ALLOWED_TOOLS: set[str] = {
    name.strip()
    for name in os.environ.get(
        "ALLOWED_TOOLS",
        "geocoding,weather_forecast,air_quality",
    ).split(",")
    if name.strip()
}

# The MCP weather tools require lat/lon, so the model has to call
# `geocoding` first when the user names a place. This system prompt
# nudges it toward that two-step pattern; without it small models tend
# to either invent coordinates or skip the geocoding call entirely.
TOOL_GUIDANCE: str = os.environ.get(
    "TOOL_GUIDANCE",
    "You have access to weather tools that require latitude and longitude. "
    "When the user names a place, first call the `geocoding` tool with that "
    "name, take the latitude/longitude from the first result, and then call "
    "the relevant weather tool (e.g. `weather_forecast`, `air_quality`) "
    "with ONLY the `latitude` and `longitude` arguments, do not pass "
    "`models`, `hourly`, `daily`, or any other parameter. "
    "Report the answer in plain English.",
)
