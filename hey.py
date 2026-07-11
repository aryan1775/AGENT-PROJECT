from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.chains import LLMMathChain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.prompts.prompt import PromptTemplate
from langchain.agents import create_agent
from langchain_community.tools import Tool
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()

#importing from env file

api_key=st.secrets["GROQ_API"]

#creating the model

model=ChatGroq(model="llama-3.1-8b-instant",api_key=api_key,temperature=0)

#creating the tools

@tool
def Math_tool(expression: str) -> str:
    """ Solves mathematical calculations. Input should be a valid Python mathematical expression. """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

#creating the math tool
import wikipediaapi
from langchain_core.tools import tool
from ddgs import DDGS

@tool
def search_tool(query:str) -> str:
    """Please use this tool to search the person , place or the information in the web in detail like in 300 words"""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    if not results:
        return "No search results found."

    output = []

    for result in results:
        output.append(
            f"Title: {result['title']}\n"
            f"Body: {result['body']}\n"
            f"URL: {result['href']}\n"
        )

    return "\n\n".join(output)

#creating the prompt and logic reasoning tool

prompt="""
You are the reasoning tool
"""

prompt_template=PromptTemplate(input_variables=["text"],template=prompt)

chain=LLMChain(llm=model,prompt=prompt_template)

@tool
def logic_reasoning(query:str) -> str:
    """You are the reasoning tool"""
    return chain.run(query)

# logic_reasoning=Tool(
#     name="Logic_reasoning_tool",
#     func=chain.run,
#     description="You are the reasoning tool"
# )

tools=[search_tool,Math_tool,logic_reasoning]

#create the agent
agent=create_agent(model=model,tools=tools,system_prompt="""
You are a helpful assistant.

You have access ONLY to these tools:
- search_tool
- math_tool
- logic_reasoning

Never call brave_search, web_search, google_search, browser_search, or any tool not listed above.

For factual biography questions, use wikipedia_tool.
For calculations, use math_tool.
For reasoning, use logic_reasoning.
""")

#streamlit

st.set_page_config(page_title="MATH PROBLEM SOLVER",page_icon="👻")
st.title("Ayy lowde")


groq_api_key=st.sidebar.text_input(label="Groq API key",type="password")

if not groq_api_key:
    st.info("Please providet the GROQ API key")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hey i am a Chatbot who will help you with Math problems"}
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


user_input=st.text_area("Please provide the scenario")

button=st.button("Generate Answer")


if button:
    if user_input:

        with st.spinner("Generate response.."):

            st.session_state.messages.append({"role": "user", "content":user_input})
            st.chat_message("user").write(user_input)


            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                },
                config={
                    "callbacks": [st_cb]
                }
            )
            answer = response["messages"][-1].content
            st.session_state.messages.append({"role":"assistant","content":answer})
            st.write("###response")
            st.success(answer)

    else:
      st.warning("kindly enter valid question")


