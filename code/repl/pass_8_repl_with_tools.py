# repl.py
"""A chat program that can fetch live data using small Python
helpers called "tools".

The model decides when a tool is needed (for example, when you ask
about weather), runs it, reads the result, and uses it to write the
reply. Long-term memory is preserved between sessions.
"""

import threading
import warnings

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
)
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_redis import RedisCache
from mem0 import Memory

from config import COLLECTION_NAME
from config import DEBUG
from config import EMBED_MODEL
from config import EXTRACTION_MODEL
from config import MEMORY_DB_PATH
from config import MEMORY_RELEVANCE_THRESHOLD
from config import OLLAMA_HOST
from config import OLLAMA_MODEL
from config import REDIS_CACHE_TTL
from config import REDIS_URL

# Same Redis cache as pass 6, repeats are instant.
set_llm_cache(
    RedisCache(redis_url=REDIS_URL, ttl=REDIS_CACHE_TTL)
)

# Silence langchain_redis's noisy "future warning", see pass 6.
warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects`.*",
)

USER_ID: str = ""
_write_threads: list[threading.Thread] = []


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
#
# Each function decorated with @tool becomes a tool the model can call.
# CRITICAL: the function's DOCSTRING is what the model reads to decide
# whether to call it. Write docstrings like a button label: short,
# specific, action-oriented.
#
# Whatever the tool returns is sent BACK to the model so it can use
# the result in its next reply.


def _get_coordinates(location: str) -> tuple[float, float]:
    """Helper: turn a place name into (latitude, longitude).

    The underscore prefix means "private", the model doesn't see
    this one. The two real tools below use it to look up coords
    before fetching weather data.
    """
    if DEBUG:
        print(
            f"Tool helper called: _get_coordinates({location})"
        )
    response = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        },
        timeout=10,
    )
    # If the server returned an error (4xx or 5xx), raise an
    # exception so the agent reports a readable error instead of
    # crashing on bad JSON.
    response.raise_for_status()
    data = response.json()
    if "results" in data and data["results"]:
        first = data["results"][0]
        return first["latitude"], first["longitude"]
    raise ValueError(
        f"Could not find coordinates for location: {location}"
    )


@tool
def get_air_quality(location: str) -> str:
    """Get current air quality (PM10 and PM2.5) for a named location."""
    if DEBUG:
        print(f"Tool called: get_air_quality({location})")
    latitude, longitude = _get_coordinates(location)
    response = httpx.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "pm10,pm2_5",
            "forecast_days": 1,
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    if (
        "hourly" in data
        and "pm10" in data["hourly"]
        and "pm2_5" in data["hourly"]
    ):
        pm10 = data["hourly"]["pm10"][
            0
        ]  # [0] = current hour
        pm2_5 = data["hourly"]["pm2_5"][0]
        result = f"PM10: {pm10} μg/m³, PM2.5: {pm2_5} μg/m³"
    else:
        result = "Air quality data not available"
    return f"Air quality in {location}: {result}"


@tool
def get_temperature(location: str) -> str:
    """Get the current temperature in Celsius for a named location."""
    if DEBUG:
        print(f"Tool called: get_temperature({location})")
    latitude, longitude = _get_coordinates(location)
    response = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m",
            "forecast_days": 1,
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    if (
        "hourly" in data
        and "temperature_2m" in data["hourly"]
    ):
        temperature = data["hourly"]["temperature_2m"][
            0
        ]  # [0] = current hour
        result = f"Temperature: {temperature} °C"
    else:
        result = "Temperature data not available"
    return f"Temperature in {location}: {result}"


# The list of tools we hand to the agent. Add new ones here and the
# model will automatically see them.
TOOLS = [get_air_quality, get_temperature]


# Without this, when a tool fails, models often "fill in" with made-up
# data ("typical temperature is about 20 °C"). That's worse than no
# answer. This system message tells the model to admit it doesn't know.
NO_FABRICATION = (
    "When a tool fails, tell the user the data is unavailable and stop. "
    "NEVER fabricate numbers, fall back to 'typical' values, or guess. "
    "It is better to say you don't know than to invent data."
)


# -----------------------------------------------------------------------------
# Memory layer (same as pass 7, see comments there)
# -----------------------------------------------------------------------------


def build_memory() -> Memory:
    """Set up mem0, see pass 7 for what each piece does."""
    config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": EXTRACTION_MODEL,
                "ollama_base_url": OLLAMA_HOST,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": EMBED_MODEL,
                "ollama_base_url": OLLAMA_HOST,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": COLLECTION_NAME,
                "path": MEMORY_DB_PATH,
            },
        },
    }
    return Memory.from_config(config)


def relevant_memories(
    memory: Memory,
    query: str,
    user_id: str,
    k: int = 5,
) -> str:
    """Find facts that look related to the user's current question.

    How it works: we turn the user's question into a vector and look
    for stored facts whose vectors are CLOSE to it (similar meaning).
    "Distance" measures how far apart they are; SMALL distance =
    similar. We only keep facts under MEMORY_RELEVANCE_THRESHOLD so
    that unrelated memories don't leak in.

    (Why not just use mem0's built-in search? Its scoring is broken
    for filtering, it returns 1.0 even for unrelated stuff. So we
    go straight to the Chroma database underneath.)
    """
    # Step 1: turn the user's question into a vector (a list of
    # numbers) using the same embedding model mem0 used when storing
    # facts. Same model = comparable vectors.
    embedding = memory.embedding_model.embed(
        query, memory_action="search"
    )

    # Step 2: ask Chroma for the closest stored facts. We ask for
    # k*4 candidates so we have spares after filtering by distance.
    # `where` restricts the search to this user only.
    res = memory.vector_store.collection.query(
        query_embeddings=[embedding],
        n_results=k * 4,
        where={"user_id": user_id},
    )

    # Chroma returns lists of lists (one inner list per query). We
    # only ran one query, so we grab the [0] inner list.
    distances = res.get("distances", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]

    # Step 3: walk through the candidates from closest to farthest,
    # keep up to `k` that pass our relevance threshold.
    facts: list[str] = []
    for dist, meta in zip(distances, metadatas):
        # SMALL distance = similar meaning. If a candidate is farther
        # than the threshold, it's not really related to the question.
        if dist >= MEMORY_RELEVANCE_THRESHOLD:
            continue
        # The actual fact text is stored in the metadata's "data" key.
        text = meta.get("data", "")
        if text:
            facts.append(text)
        # Got enough relevant facts, stop early.
        if len(facts) >= k:
            break

    # Nothing relevant? Return an empty string so the caller can skip
    # the memory injection entirely.
    if not facts:
        return ""

    # Format the kept facts as a bulleted list for the system message.
    return "Known facts about the user:\n" + "\n".join(
        f"- {f}" for f in facts
    )


def write_memory_async(
    memory: Memory, user_text: str, reply: str
) -> None:
    """Save a turn to memory in the background. See pass 7 for details."""

    def _run() -> None:
        try:
            if DEBUG:
                print(
                    "\nMemory write started...", flush=True
                )
            memory.add(
                messages=[
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply},
                ],
                user_id=USER_ID,
                infer=True,
            )
            if DEBUG:
                print(
                    "\nMemory write complete.", flush=True
                )
        except Exception as e:
            print(
                f"\n(memory write failed: {e})", flush=True
            )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _write_threads.append(t)


def read_input() -> str:
    """Read lines from stdin until the user submits an empty line."""
    lines: list[str] = []
    prompt = "You > "
    while True:
        line = input(prompt)
        if line == "":
            break
        lines.append(line)
        prompt = "      "
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------


def main() -> None:
    global USER_ID
    USER_ID = (
        input("Enter your user id: ").strip() or "default"
    )

    # The main chat model. Important: it must support "tool calling"
    # (not all models do). Check yours with `ollama show <model>` and
    # look for "tools" in the listed capabilities.
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        num_predict=2048,
    )

    summarizer = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        num_predict=2048,
    )

    # The agent now has tools wired in. On every turn it can:
    #   1. answer right away, OR
    #   2. call one of our tools, read the result, then answer.
    # That decision happens inside the agent loop.
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        middleware=[
            # When a tool raises an exception (network down, bad args,
            # etc.), don't crash. Instead, send the error TEXT back to
            # the model as the tool result so the model can apologise
            # or try a different approach.
            ToolRetryMiddleware(
                max_retries=0, on_failure="continue"
            ),
            SummarizationMiddleware(
                model=summarizer,
                trigger=("tokens", 2000),
                keep=("messages", 6),
            ),
        ],
    )

    print("Loading memory...")
    memory = build_memory()

    print(
        f"Chatting with {OLLAMA_MODEL} (with long-term memory and tools)."
    )
    print(
        f"Tools available: {', '.join(t.name for t in TOOLS)}"
    )
    print(
        "Hit Enter on an empty line to send. Type /bye to exit."
    )
    print(
        "Waiting for pending memory writes is automatic on /bye.\n"
    )

    while True:
        user = read_input()
        if user == "":
            continue

        if user.strip() == "/bye":
            pending = [
                t for t in _write_threads if t.is_alive()
            ]
            if pending:
                print(
                    f"Waiting for {len(pending)} memory write(s) to finish..."
                )
                for t in pending:
                    t.join()
            break

        memory_block = relevant_memories(
            memory, query=user, user_id=USER_ID
        )

        messages: list = [
            SystemMessage(content=NO_FABRICATION)
        ]
        if memory_block:
            messages.append(
                SystemMessage(content=memory_block)
            )
        messages.append(HumanMessage(content=user))

        # One user turn can now produce many internal events: the
        # model deciding to call a tool, the tool result coming back,
        # the model deciding to answer, etc. We only want to PRINT
        # the final natural-language reply, the rest happens behind
        # the scenes.
        print("Assistant > ", end="", flush=True)
        full_reply = ""
        for chunk, meta in agent.stream(
            {"messages": messages},
            stream_mode="messages",
        ):
            # Skip text from the summarizer model (see pass 7).
            if meta.get("langgraph_node") != "model":
                continue
            # Skip chunks that are tool-call planning (they have no
            # text). Only print real reply text.
            if (
                isinstance(chunk, AIMessage)
                and chunk.content
            ):
                print(chunk.content, end="", flush=True)
                full_reply += chunk.content
        print()

        # Save just the user's question and the final visible reply.
        # The tool calls under the hood are noise we don't need.
        write_memory_async(memory, user, full_reply)


if __name__ == "__main__":
    main()
