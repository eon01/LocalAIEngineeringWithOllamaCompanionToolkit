# repl.py
"""A chat program that remembers facts about you across sessions.

After every turn, durable facts ("the user has five dogs") are
extracted and saved to a local database. On future questions, the
matching facts are looked up and fed back to the model as context.
"""

import threading
import warnings

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
)
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
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

# Filled in once at startup when the user types their id. We keep it
# at module level so the background memory-writing threads can read
# it without us having to pass it around.
USER_ID: str = ""

# We write memories in the BACKGROUND (a separate thread) so they
# don't slow down the chat. This list keeps track of those threads so
# that when the user types /bye we can wait for them to finish
# instead of cutting them off mid-write.
_write_threads: list[threading.Thread] = []


# -----------------------------------------------------------------------------
# Memory layer
# -----------------------------------------------------------------------------


def build_memory() -> Memory:
    """Set up mem0. Everything runs locally, no cloud services.

    mem0 needs three pieces:
      - llm: extracts the durable facts from a conversation.
      - embedder: turns text into vectors (lists of numbers) so we
        can find facts that look similar to the user's question.
      - vector_store: where the facts and their vectors live on disk.
    """
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
    """Save the turn to long-term memory WITHOUT blocking the chat.

    mem0.add() can take a few seconds (it calls the model to extract
    facts). To keep the REPL snappy, we kick off the save in a
    background thread and let the user keep typing.
    """

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

    # Same agent as pass 6: chat model + summarization middleware.
    # No tools yet, pass 8 adds those.
    agent = create_agent(
        model=llm,
        middleware=[
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
        f"Chatting with {OLLAMA_MODEL} (with long-term memory)."
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
            # Wait for any background memory writes to finish before
            # we shut down, so we don't lose them.
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

        # Look up relevant facts for THIS question and pass them to
        # the model as a system message (background context).
        memory_block = relevant_memories(
            memory, query=user, user_id=USER_ID
        )

        messages: list = []
        if memory_block:
            messages.append(
                SystemMessage(content=memory_block)
            )
        messages.append(HumanMessage(content=user))

        print("Assistant > ", end="", flush=True)
        full_reply = ""
        for chunk, meta in agent.stream(
            {"messages": messages},
            stream_mode="messages",
        ):
            # Behind the scenes the agent uses two models: the main
            # chat model AND the summarizer. Both stream their text
            # through here. We only want to print the main one, so we
            # check the chunk's source node ("model" = main reply).
            if meta.get("langgraph_node") != "model":
                continue
            if (
                isinstance(chunk, AIMessage)
                and chunk.content
            ):
                print(chunk.content, end="", flush=True)
                full_reply += chunk.content
        print()

        # Skip saving empty replies, the embedding model errors out
        # on empty text, which would crash the background save.
        if full_reply.strip():
            write_memory_async(memory, user, full_reply)


if __name__ == "__main__":
    main()
