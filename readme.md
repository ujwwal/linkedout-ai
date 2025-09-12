# 🚀 LinkedIn Post Generator (RAG + LangChain + FastAPI)

An AI-powered LinkedIn post generator that helps **copywriters** and **professionals** create high-quality posts.  
Built with **Azure OpenAI GPT-5**, **LangChain**, **FAISS**, and **FastAPI**.  

---

## ✨ Features

- 🤖 **AI-generated LinkedIn posts** tailored to users or clients.
- 📝 **Copywriter mode** → manage multiple clients, each with its own chat and memory.
- 🔄 **Memory-enabled AI** → remembers client tone, style, and past interactions.
- 🎯 **Pro vs Beginner Onboarding**:
  - Pro users → get **2 suggestions** initially, so the AI learns their style.
  - Beginner users → get **1 best suggestion** for simplicity.
- 📂 **RAG (Retrieval-Augmented Generation)** → GPT references top LinkedIn creator posts for style and tone.
- ⚡ **Fast and scalable backend** → powered by **FastAPI** and **FAISS**.

---

## 🛠️ Tech Stack

| Component | Technology | Why? |
|-----------|------------|------|
| Frontend | React/JS | User interface (handled by frontend team) |
| Backend | FastAPI (Python) | API layer between frontend and AI |
| LLM | Azure OpenAI GPT-5 | Generates LinkedIn posts |
| Framework | LangChain | Orchestrates GPT, memory, and retrieval |
| Vector DB | FAISS | Stores and retrieves embeddings of creator posts |
| Embeddings | Azure OpenAI Embeddings API | Converts posts & queries into vectors |

---

## 🔄 System Flow

```mermaid
flowchart TD
    A[User Query (Frontend)] --> B[FastAPI Backend]

    B --> C[LangChain Memory (Client History)]
    C --> D[Azure OpenAI Embeddings]
    D --> E[FAISS Vector DB]
    E --> F[Retrieve Top-K Similar Posts]

    C --> G[Prompt Assembly: System + Memory + Examples + Query]

    G --> H[Azure OpenAI GPT-5]
    H --> I[LangChain Memory Update]
    I --> J[Backend JSON Response]

    J --> K[Frontend Displays LinkedIn Post]
