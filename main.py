import os
import logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from duckduckgo_search import DDGS
from rag import EmbedData, Retriever
import sys


load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.info("main.py loaded")


llm = OpenAI(model="gpt-3.5-turbo")
mcp = FastMCP("ml_faq_server")


@mcp.tool()
def machine_learning_faq_retrieval_tool(query: str) -> dict:
    retriever = Retriever("testmcp", EmbedData())
    context = retriever.search(query)
    prompt = PromptTemplate(
        "Answer the following question based on the context below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    )
    full_prompt = prompt.format(context=context, question=query)
    response = llm.complete(full_prompt)
    return {"result": response.text.strip()}

@mcp.tool()
def pinecone_web_search_tool(query: str) -> dict:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=10):
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", "")
            })
    return {"result": results}


__mcp__ = mcp
__mcp_info__ = {
    "name": "ML FAQ + Web Agent",
    "description": "Search ML FAQs and the web using Pinecone + DDG",
    "endpoint": "http://localhost:3000"
}

if __name__ == "__main__":
    try:
        # Send only to stderr so Claude doesn't see it
        print("PYTHON:", sys.executable, file=sys.stderr)
        print("PATH:", sys.path, file=sys.stderr)
        mcp.run()
    except Exception as e:
        import traceback
        print("❌ MCP SERVER CRASHED ❌", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
