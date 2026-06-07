# repl.py
"""A chat program that remembers what was said earlier in the
session.

Every turn, we send the whole conversation back to the model, so it
can reply with full context. Memory is lost when you exit.
"""

from ollama import Client

from config import OLLAMA_HOST
from config import OLLAMA_MODEL


def main() -> None:
    client = Client(host=OLLAMA_HOST)
    print(
        f"Chatting with {OLLAMA_MODEL}. Type /bye to exit."
    )

    # This list holds the whole conversation. Each turn we add the
    # user's message and the model's reply, then send the whole list
    # back on the next turn so the model has context.
    messages: list[dict] = []

    while True:
        user = input("You > ")
        if user.strip() == "/bye":
            break

        # Add the user's message to the history.
        messages.append({"role": "user", "content": user})

        # Send the whole history. The model uses every previous turn
        # as context for the new one.
        response = client.chat(
            model=OLLAMA_MODEL, messages=messages
        )

        # Add the model's reply too, so the next turn remembers what
        # the model just said. Forgetting this line is the most common
        # bug in chat loops, the model seems to "lose its train of
        # thought" because we never gave it its own past words.
        messages.append(
            {
                "role": "assistant",
                "content": response.message.content,
            }
        )

        print(f"Assistant > {response.message.content}")


if __name__ == "__main__":
    main()
