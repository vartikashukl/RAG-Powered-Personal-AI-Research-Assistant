# agent.py
import os
from typing import Tuple, List
from dotenv import load_dotenv

# Use Google Gemini instead of OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from rag_pipeline import get_retriever

# Load environment variables
load_dotenv()

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize LLM using Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # changed from "chat-bison" to "gemini-pro"
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

# Initialize retriever placeholder
retriever = None

def init_retriever(k: int = 5):
    """
    Initialize or reset the retriever with desired top-k.
    """
    global retriever
    try:
        retriever = get_retriever(k=k)
        # Check if retriever is valid (not None and is callable or has get_relevant_documents)
        if retriever is None:
            raise RuntimeError("FAISS index not found or incomplete. Please build it first using build_faiss_index().")
        # Optionally, check for a method typical of retrievers
        if not (callable(getattr(retriever, "get_relevant_documents", None)) or callable(getattr(retriever, "retrieve", None))):
            retriever = None
            raise RuntimeError("Retriever object is invalid. Please rebuild the FAISS index.")
        print(f"✅ Retriever initialized with k={k}")
    except Exception as e:
        print(f"❌ Error initializing retriever: {e}")
        retriever = None

def ask(query: str) -> Tuple[str, List[str]]:
    """
    Process a user query: retrieves relevant docs, runs LLM, returns answer and sources.

    Args:
        query: The user question.

    Returns:
        answer: The LLM-generated answer string.
        sources: List of source identifiers used for the answer.
    """
    global retriever, llm, memory
    
    if retriever is None:
        print("🔄 Initializing retriever...")
        init_retriever()
        
    if retriever is None:
        return "❌ Error: No documents loaded. Please upload documents first.", []

    try:
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Run the chain
        result = qa_chain({"query": query})
        answer = result.get("result", "")
        docs: List[Document] = result.get("source_documents", [])

        # Update memory
        memory.save_context({"input": query}, {"output": answer})

        # Extract unique sources from metadata
        sources = []
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            if src not in sources:
                sources.append(src)

        return answer, sources
    
    except Exception as e:
        print(f"❌ Error in ask(): {e}")
        return f"Error processing query: {str(e)}", []