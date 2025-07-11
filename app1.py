# ---------- app.py -----------------------------------------------------------
"""
Streamlit app: Agentic RAG over PDFs with DeepSeek‑R1 (Ollama) + CrewAI
------------------------------------------------------------------------
• Upload a PDF ➜ it’s embedded with LangChain (FAISS + MiniLM)
• Ask questions ➜ Retriever agent + Response agent cooperate
• Uses local Ollama model via LiteLLM
"""

import os
import time
import gc
import base64
import tempfile
import streamlit as st
from typing import Optional

from crewai import Agent, Crew, Task, Process, LLM
from crewai.tools import BaseTool
from searxng_tool import SearxngTool

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

# --------------------------------------------------------------------------- #
#                              LLM INITIALISATION                             #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_llm():
    """
    Return a LiteLLM-wrapped Ollama model.
    api_type='ollama' + dummy api_key prevent 'list index out of range' error.
    """
    return LLM(
        model="ollama/deepseek-r1:7b",          # make sure you've pulled this
        base_url="http://localhost:11434",      # Ollama default endpoint
        api_type="ollama",
        api_key="dummy"                         # required positional arg
    )

# --------------------------------------------------------------------------- #
#                            PDF SEARCH TOOL (Pydantic)                       #
# --------------------------------------------------------------------------- #
class LangchainPDFSearchTool(BaseTool):
    """Search a PDF via embeddings + semantic similarity."""
    name: str = "PDF Search Tool"
    description: str = ("Search the contents of a PDF using embeddings "
                        "and semantic similarity.")
    file_path: str
    db: Optional[FAISS] = None  # will be filled in __init__

    def __init__(self, file_path: str):
        super().__init__(file_path=file_path)   # registers file_path with Pydantic
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = FAISS.from_documents(docs, embeddings)

    # CrewAI expects a synchronous _run
    def _run(self, query: str) -> str:
        if self.db is None:
            return "PDF not indexed."
        results = self.db.similarity_search(query, k=3)
        return "\n\n".join(doc.page_content for doc in results)


# --------------------------------------------------------------------------- #
#                           CREW / AGENTS / TASKS                             #
# --------------------------------------------------------------------------- #
def create_agents_and_tasks(pdf_tool):
    """Return a Crew ready to answer questions using PDF & web search."""
    web_tool = SearxngTool()  # requires SEARXNG_BASE_URL env var

    retriever_agent = Agent(
        role="retriever",
        goal=("Fetch the most relevant information for a user query, "
              "preferring the PDF tool, falling back to web search."),
        backstory="You are a meticulous analyst with a knack for finding facts.",
        tools=[t for t in (pdf_tool, web_tool) if t],
        llm=load_llm(),
        verbose=True,
    )

    synthesizer_agent = Agent(
        role="synthesizer",
        goal=("Compose a concise, helpful answer from the retrieved material. "
              "If no relevant info is found, apologise."),
        backstory="You are an eloquent communicator.",
        llm=load_llm(),
        verbose=True,
    )

    retrieval_task = Task(
        description="Retrieve material for the user query: {query}",
        expected_output="Relevant excerpts or notes.",
        agent=retriever_agent,
    )

    response_task = Task(
        description="Write the final answer for: {query}",
        expected_output=("A concise answer OR “I'm sorry, I couldn't find the "
                         "information you're looking for.”"),
        agent=synthesizer_agent,
    )

    return Crew(
        agents=[retriever_agent, synthesizer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=True,
    )

# --------------------------------------------------------------------------- #
#                               STREAMLIT UI                                  #
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Agentic RAG ", layout="wide")
st.title("Agentic Multimodal RAG- Deepseek+SearXNG")

# Session state setup
for key, default in (
    ("messages", []),
    ("pdf_tool",  None),
    ("crew",      None),
):
    if key not in st.session_state:
        st.session_state[key] = default

# Helper: display PDF preview
def show_pdf(file_bytes: bytes, filename: str):
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    st.markdown(
        f"#### Preview: {filename}\n"
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="600px"></iframe>',
        unsafe_allow_html=True
    )

# Sidebar: upload PDF
with st.sidebar:
    st.header("📑 Upload a PDF")
    file = st.file_uploader("Choose file", type="pdf")
    if file and st.session_state.pdf_tool is None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, file.name)
            with open(path, "wb") as f:
                f.write(file.getvalue())
            with st.spinner("Indexing..."):
                st.session_state.pdf_tool = LangchainPDFSearchTool(file_path=path)
        st.success("PDF indexed!")
        show_pdf(file.getvalue(), file.name)

    st.button("🗑️ Clear Chat", on_click=lambda: (st.session_state.update(messages=[]), gc.collect()))

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input box
query = st.chat_input(placeholder="Ask something about the PDF...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Create crew once PDF tool exists
    if st.session_state.crew is None and st.session_state.pdf_tool:
        st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""

        with st.spinner("Thinking..."):
            result = st.session_state.crew.kickoff(inputs={"query": query}).raw

        for line in result.splitlines():
            full += line + "\n"
            placeholder.markdown(full + "▌")
            time.sleep(0.05)
        placeholder.markdown(full)

    st.session_state.messages.append({"role": "assistant", "content": result})
# --------------------------------------------------------------------------- #
