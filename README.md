#  Enterprise-Grade Agentic RAG 
<img width="1663" height="777" alt="brave_screenshot_localhost" src="https://github.com/user-attachments/assets/39cab0a3-b3d7-4560-89f2-edb6dce960d7" />


> Inspired by [Daily Dose of Data Science](https://github.com/patchy631/ai-engineering-hub/tree/main/agentic_rag_deepseek)

---

##  Overview

A cutting-edge, production-ready **Agentic Retrieval-Augmented Generation (RAG)** system designed for pinpointing information from complex documents (e.g., research papers) and the web.

---

## 🔧 Key Features

- 🔍 **Multi-source Retrieval**: Combines **document embeddings** and **web search (SearXNG)** to fetch accurate answers.
- 🧠 **LLM**: Powered by **DeepSeek R1 & R1b** via [Ollama](https://ollama.com/library)
- 🤖 **Agents**: Built using **Crew AI** and **Langchain**
- 🖥️ **UI**: Interactive frontend using **Streamlit**
- 📄 **Upload Support**: Accepts complex document types like PDFs and research papers
- 🧭 **Autonomous Agent Routing**: Smart decision-making on whether to pull information from documents or the web
- Document example: <img width="1022" height="737" alt="image" src="https://github.com/user-attachments/assets/b114f9c5-aec9-4e9f-9bab-e798ed11ada6" />


---

## 🛠️ Tech Stack

| Component    | Tool/Framework        |
|--------------|------------------------|
| Language Model | DeepSeek R1 & R1b (via Ollama) |
| Search Engine | SearXNG (Dockerized)  |
| Agent Framework | Crew AI + Langchain  |
| Frontend      | Streamlit             |

---

## 📦 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/<your-repo-path>.git
cd <your-repo>

# Start SearXNG
docker compose up -d

# Start Streamlit app
streamlit run app.py
