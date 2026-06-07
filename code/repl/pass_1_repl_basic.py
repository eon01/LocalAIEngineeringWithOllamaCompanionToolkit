# repl.py
"""A tiny chat program.

You type, the model replies. The model has no memory: every
question is treated as a fresh conversation.
"""

from ollama import Client

from config import OLLAMA_HOST
from config import OLLAMA_MODEL


def main() -> None:
    client = Client(host=OLLAMA_HOST)
    print(
        f"Chatting with {OLLAMA_MODEL}. Type /bye to exit."
    )

    while True:
        # Wait for the user to type something and press Enter.
        user = input("You > ")

        # Type /bye to quit.
        if user.strip() == "/bye":
            break

        # Send only this one message. Because we don't send any past
        # messages, the model has no memory, every turn is "fresh".
        # Pass 2 fixes that.
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": user}],
        )
        print(f"Assistant > {response.message.content}")


if __name__ == "__main__":
    main()
