from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the result."""
    # Debugging
    print("I'm being called with", a, b)
    return a + b


llm = ChatOllama(
    model="granite3.3:2b", base_url="http://localhost:11434"
)
agent = create_agent(model=llm, tools=[add_numbers])

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 17 plus 25?",
            }
        ]
    }
)
print(response["messages"][-1].content)
