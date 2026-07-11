import ast
import operator

import streamlit as st
from ddgs import DDGS

from langchain.agents import create_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.tools import tool
from langchain_groq import ChatGroq


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Problem Solver",
    page_icon="👻"
)

st.title("AI Math and Search Assistant")


# =========================================================
# LOAD GROQ API KEY
# =========================================================

try:
    api_key = st.secrets["GROQ_API"]

except KeyError:
    st.error(
        "GROQ_API was not found in Streamlit Secrets. "
        "Open Manage app → Settings → Secrets and add it."
    )
    st.stop()


# =========================================================
# CREATE MODEL
# =========================================================

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0
)


# =========================================================
# SAFE MATH TOOL
# =========================================================

allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def evaluate_expression(node):
    """
    Recursively evaluate a safe mathematical expression.
    """

    if isinstance(node, ast.Expression):
        return evaluate_expression(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        operation = allowed_operators.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported mathematical operator.")

        left_value = evaluate_expression(node.left)
        right_value = evaluate_expression(node.right)

        return operation(left_value, right_value)

    if isinstance(node, ast.UnaryOp):
        operation = allowed_operators.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported unary operator.")

        return operation(evaluate_expression(node.operand))

    raise ValueError("Invalid mathematical expression.")


@tool
def math_tool(expression: str) -> str:
    """
    Calculate a basic arithmetic expression.

    Args:
        expression: A mathematical expression such as
                    "(20 + 5) * 2" or "2 ** 8".

    Returns:
        The calculated result.
    """

    try:
        parsed_expression = ast.parse(
            expression,
            mode="eval"
        )

        result = evaluate_expression(parsed_expression)

        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."

    except Exception as error:
        return f"Math error: {error}"


# =========================================================
# WEB SEARCH TOOL
# =========================================================

@tool
def search_tool(query: str) -> str:
    """
    Search the web for factual and current information.

    Args:
        query: A clear and concise web-search query.

    Returns:
        Titles, summaries and URLs from web-search results.
    """

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=5
                )
            )

        if not results:
            return "No search results were found."

        formatted_results = []

        for index, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            summary = result.get("body", "No summary")
            url = result.get("href", "No URL")

            formatted_results.append(
                f"Result {index}\n"
                f"Title: {title}\n"
                f"Summary: {summary}\n"
                f"URL: {url}"
            )

        return "\n\n".join(formatted_results)

    except Exception as error:
        return f"Web search failed: {error}"


# =========================================================
# CREATE AGENT
# =========================================================

tools = [
    search_tool,
    math_tool
]


agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
You are a helpful AI assistant.

You have access to two tools:

1. search_tool
   Use this for:
   - people and biographies
   - places
   - companies
   - sports
   - events
   - current information
   - factual web questions

2. math_tool
   Use this for arithmetic calculations.

Rules:

- Use only the tools listed above.
- Do not invent other tool names.
- For greetings and casual conversation, answer directly.
- Read the complete conversation before answering.
- Resolve words such as he, she, it and they using conversation history.
- For questions containing words such as current, latest,
  today or this year, use search_tool.
- After using a tool, provide a clear final answer.
"""
)


# =========================================================
# INITIALISE CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I can help you with mathematics, "
                "web searches and general questions."
            )
        }
    ]


# =========================================================
# DISPLAY EXISTING CHAT MESSAGES
# =========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# =========================================================
# RECEIVE USER INPUT
# =========================================================

user_input = st.chat_input(
    "Ask a question"
)


# =========================================================
# PROCESS USER INPUT
# =========================================================

if user_input:

    # Save the user message in session memory
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display the current user message
    with st.chat_message("user"):
        st.write(user_input)

    # Create and display assistant response
    with st.chat_message("assistant"):

        try:
            callback_handler = StreamlitCallbackHandler(
                st.container(),
                expand_new_thoughts=False
            )

            response = agent.invoke(
                {
                    "messages": st.session_state.messages
                },
                config={
                    "callbacks": [callback_handler]
                }
            )

            answer = response["messages"][-1].content

        except Exception as error:
            answer = (
                "The agent encountered an error while generating "
                f"the response.\n\nTechnical details: {error}"
            )

        st.write(answer)

    # Save the assistant response in session memory
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

