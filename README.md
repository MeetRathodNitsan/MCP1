# 🧠 MCP AI Server — Modular Context Protocol for Intelligent Search

Welcome to the **MCP AI Server**, a powerful and modular tool that uses **RAG-based retrieval**, **Pinecone vector storage**, and **MCP** (Model Context Protocol) to create intelligent assistants capable of answering domain-specific questions from your own knowledge base.

![MCP + Claude + Pinecone](https://img.shields.io/badge/Built_with-MCP-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Language-Python%203.10%2B-blue?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

---

## 🚀 Features

✅ Local MCP server with FastAPI + Claude/ChatGPT integration  
✅ Embedding using `intfloat/multilingual-e5-large` (via SentenceTransformer)  
✅ Fast vector search with Pinecone  
✅ Documented `tools` exposed to clients like **Claude** and **Cursor IDE**  
✅ Secure `.env` usage for managing API keys  
✅ Clean, extensible architecture

---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone git@github.com:MeetRathodNitsan/MCP1.git
cd MCP1
```
### 2. Create a Virtual Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
OPENAI_API_KEY=your-api-key...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=your-env
```

### 5. How to use it
```bash
uv --directory F:/Project run main.py
```
