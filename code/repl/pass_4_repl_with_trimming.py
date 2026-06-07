# repl.py
"""A streaming chat program that keeps the history from growing
without limit.

Once the conversation passes a size budget, the oldest messages are
dropped so the model doesn't get bogged down or hit its memory cap.
"""

from ollama import Client

from config import OLLAMA_HOST
from config import OLLAMA_MODEL

# Max total characters to keep in the history. Roughly 4 characters
# per token for English text, so 8000 chars ≈ 2000 tokens. Tune for
# your model and how much room you want to leave for replies.
MAX_HISTORY_CHARS = 8000


def trim_history(
    messages: list[dict], max_chars: int = MAX_HISTORY_CHARS
) -> list[dict]:
    """Drop the oldest user+assistant pairs until we're under the budget.

    Two rules:
    - If the very first message is a system message, KEEP it. System
      messages set the model's overall behavior, so we never drop them.
    - We drop messages in PAIRS (one user + one reply). Dropping just
      one half leaves an orphan that confuses the model.
    """
    # If there's a system message at the front, set it aside so we
    # don't accidentally drop it.
    has_system = (
        bool(messages)
        and messages[0].get("role") == "system"
    )
    head = messages[:1] if has_system else []
    body = messages[1:] if has_system else messages[:]

    def total_chars(msgs: list[dict]) -> int:
        return sum(len(m.get("content", "")) for m in msgs)

    # Keep dropping the oldest pair until we fit (or there's nothing
    # left to drop).
    while (
        total_chars(head + body) > max_chars
        and len(body) >= 2
    ):
        body = body[2:]

    return head + body


def read_input() -> str:
    """Let the user type a message across multiple lines.

    Press Enter on an EMPTY line to send. Same idea as pass 3.
    """
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
    # Long replies can take a while. Raise the default timeout to be safe.
    client = Client(host=OLLAMA_HOST, timeout=300)

    print(f"Chatting with {OLLAMA_MODEL}.")
    print(
        "Hit Enter on an empty line to send. Type /bye to exit."
    )

    messages: list[dict] = []

    while True:
        user = read_input()

        if user == "":
            continue
        if user.strip() == "/bye":
            break

        messages.append({"role": "user", "content": user})

        # Trim AFTER adding the user's new message but BEFORE calling
        # the model. That way, the user's latest message is always
        # kept, even if it alone would push us over the budget.
        messages = trim_history(messages)

        # stream=True: get the reply piece by piece (the "typing"
        # effect). We print each piece and also collect them all so
        # we can save the full reply to history at the end.
        print("Assistant > ", end="", flush=True)
        full_reply = ""
        for chunk in client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True,
        ):
            piece = chunk.message.content
            print(piece, end="", flush=True)
            full_reply += piece

        # Final newline so the next prompt starts on a fresh line.
        print()

        messages.append(
            {"role": "assistant", "content": full_reply}
        )


if __name__ == "__main__":
    main()
