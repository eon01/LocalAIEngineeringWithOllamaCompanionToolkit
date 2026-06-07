# repl.py
"""A streaming chat program with summarization and a reply cache.

Past model replies are stored in Redis. If you ask the exact same
question again, the saved reply comes back instantly, with no call
to the model.
"""

import warnings

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
)
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_redis import RedisCache

from config import OLLAMA_HOST
from config import OLLAMA_MODEL
from config import REDIS_CACHE_TTL
from config import REDIS_URL

# Turn on caching for every model call in this program. The cache key
# is (the prompt + the model + a few settings), so if you change the
# model or the system prompt you'll get fresh replies. Entries also
# expire after REDIS_CACHE_TTL seconds. To wipe the whole cache, run
# `FLUSHALL` in redis-cli.
set_llm_cache(
    RedisCache(redis_url=REDIS_URL, ttl=REDIS_CACHE_TTL)
)

# langchain_redis prints a "future warning" on every cache hit. It's
# noisy and we can't fix it from here (the bug is in that library).
# Silence it until they release a fix.
warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects`.*",
)


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


def main() -> None:
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        num_predict=2048,
    )

    # Same setup as pass 5: a second model for writing summaries when
    # the history gets long. Reuses OLLAMA_MODEL for simplicity.
    summarizer = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        num_predict=2048,
    )

    # Same agent as pass 5: chat model + summarization middleware.
    # When history grows past 2000 tokens, summarize everything older
    # than the last 6 messages.
    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model=summarizer,
                trigger=("tokens", 2000),
                keep=("messages", 6),
            ),
        ],
    )

    print(
        f"Chatting with {OLLAMA_MODEL} (with summarization)."
    )
    print(
        "Hit Enter on an empty line to send. Type /bye to exit."
    )

    messages: list = []

    while True:
        user = read_input()
        if user == "":
            continue
        if user.strip() == "/bye":
            break

        messages.append(HumanMessage(content=user))

        # Stream the reply piece by piece. NOTE: on a cache hit you'll
        # see the whole reply appear at once instead of typing, that's
        # because cached replies don't need to be streamed.
        print("Assistant > ", end="", flush=True)
        full_reply = ""
        for chunk, _ in agent.stream(
            {"messages": messages},
            stream_mode="messages",
        ):
            if (
                isinstance(chunk, AIMessage)
                and chunk.content
            ):
                print(chunk.content, end="", flush=True)
                full_reply += chunk.content

        print()
        messages.append(AIMessage(content=full_reply))


if __name__ == "__main__":
    main()
