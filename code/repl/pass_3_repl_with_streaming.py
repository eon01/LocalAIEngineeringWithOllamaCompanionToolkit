# repl.py
"""A chat program that prints the model's reply as it's being
generated (no waiting for the whole answer).

You can also type questions across multiple lines and send them by
hitting Enter on an empty line.
"""

from ollama import Client

from config import OLLAMA_HOST
from config import OLLAMA_MODEL


def read_input() -> str:
    """Let the user type a message that can span several lines.

    Each Enter starts a new line. An EMPTY line means "I'm done, send
    it." The first line shows 'You > ', extra lines show spaces so
    they line up underneath.
    """
    lines: list[str] = []
    prompt = "You > "
    while True:
        line = input(prompt)
        # Empty line = send. Works even on the very first line: just
        # gives back "" so the main loop can ignore it.
        if line == "":
            break
        lines.append(line)
        prompt = "      "
    return "\n".join(lines)


def main() -> None:
    # Long replies can take a while. Default timeout is short, so we
    # raise it to 300 seconds (5 minutes) to be safe.
    client = Client(host=OLLAMA_HOST, timeout=300)

    print(f"Chatting with {OLLAMA_MODEL}.")
    print(
        "Hit Enter on an empty line to send. Type /bye to exit."
    )

    messages: list[dict] = []

    while True:
        user = read_input()

        # If the user pressed Enter on an empty prompt, just loop back.
        if user == "":
            continue
        if user.strip() == "/bye":
            break

        messages.append({"role": "user", "content": user})

        # stream=True makes the model send back its reply one tiny
        # piece at a time instead of all at once at the end. We print
        # each piece as it arrives (the "typing" effect) and also
        # build up the full reply so we can save it to history.
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

        # Final newline so the next 'You > ' starts on a fresh line.
        print()

        messages.append(
            {"role": "assistant", "content": full_reply}
        )


if __name__ == "__main__":
    main()
