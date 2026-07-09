# =========================
# ARXIV TOOL
# =========================
from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools.arxiv import ArxivQueryRun
from streamlit import session_state
from langchain_community.callbacks import StreamlitCallbackHandler

arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=400
)

arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# =========================
# WIKIPEDIA TOOL
# =========================

import wikipedia

wikipedia.set_lang("en")

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=400
)

wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_wrapper
)

# =========================
# DUCKDUCKGO SEARCH TOOL
# =========================
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun(  name="web_search",
    description="""
    Search the internet for current information.
    Use this tool for people, events, companies, news,
    and general knowledge questions.
    """,)

# =========================
# LOAD PDF
# =========================
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("IF3KF6_2026_online.pdf")
docs = loader.load()

# =========================
# SPLIT DOCUMENTS
# =========================
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_documents(docs)

# =========================
# EMBEDDINGS
# =========================
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# VECTOR STORE
# =========================
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embedding
)

retriever = vectorstore.as_retriever()

# =========================
# RETRIEVER TOOL
# =========================
from langchain_core.tools import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "insurance_policy_search",
    "Search information from the uploaded insurance policy PDF."
)

# =========================
# LOAD ENV VARIABLES
# =========================
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API")



#strealit and adding the memory

from langchain_core.prompts import ChatPromptTemplate

import streamlit as st

st.sidebar.title("Settings")

if "messages" not in st.session_state:
    st.session_state["messages"]=[{
        "role":"assistant",
        "content":"i am chatbot am there to assist you"
    }]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="what is machine learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    # =========================
    # GROQ MODEL
    # =========================
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0
    )

    # =========================
    # TOOLS
    # =========================
    tools = [
        search_tool,
        arxiv_tool,
        retriever_tool
    ]

    # =========================
    # AGENT
    # =========================
    from langchain.agents import create_agent

    agent = create_agent(
        model=llm,
        tools=tools, system_prompt="""
    You are a helpful AI assistant.

    Rules:

    - For general knowledge questions use web_search.
    - For research questions use arxiv_tool.
    - For insurance PDF questions use insurance_policy_search.

    Always provide a final answer using the tool results.
    Do not say the tool failed unless it actually failed.
    """
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False
        )

        response = agent.invoke(
            {
                "messages": [
                    (msg["role"],msg["content"])
                    for msg in st.session_state.messages
                ]
            },
            config={
                "callbacks": [st_cb]
            }
        )

        answer = response["messages"][-1].content

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.write(answer)

