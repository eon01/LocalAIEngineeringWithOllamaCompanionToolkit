# repl.py
"""A streaming chat program that keeps long conversations going
without losing context.

When the history gets too long, the model is asked to summarize the
older messages. The summary then stands in for everything older, so
the chat can continue without forgetting what was discussed.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
)
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from config import OLLAMA_HOST
from config import OLLAMA_MODEL


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

    # A second model used ONLY for writing summaries. In real apps
    # people often pick a smaller, cheaper model here since summarizing
    # is easier than answering. For simplicity we reuse the same one.
    summarizer = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        num_predict=2048,
    )

    # An "agent" in LangChain is the chat model wrapped with extra
    # behavior. Here we add one piece of middleware:
    #   - trigger: when the conversation grows past 2000 tokens, run
    #     the summarizer.
    #   - keep: when summarizing, keep the most recent 6 messages
    #     unchanged and replace everything older with a summary.
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

        # agent.stream gives us the reply piece by piece. Each loop
        # iteration is one small chunk of text (just the new bit, not
        # the whole reply so far).
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
