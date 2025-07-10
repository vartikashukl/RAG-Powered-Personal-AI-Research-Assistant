# ✅ FIXED rag_pipeline.py
import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

from utils.loaders import load_pdf, load_docx, load_txt, load_wikipedia
from utils.chunkers import chunk_documents

def build_faiss_index(
    sources: List[str],
    use_wikipedia: bool = False,
    wiki_topic: str = "",
    index_path: str = "data/faiss_index"
):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # This is correct for embeddings
        google_api_key=gemini_api_key
    )

    documents = []
    for source in sources:
        try:
            if source.endswith(".pdf"):
                documents.extend(load_pdf(source))
            elif source.endswith(".docx"):
                documents.extend(load_docx(source))
            elif source.endswith(".txt"):
                documents.extend(load_txt(source))
        except Exception as e:
            print(f"❌ Error loading {source}: {e}")

    if use_wikipedia and wiki_topic:
        try:
            wiki_docs = load_wikipedia(wiki_topic)
            documents.extend(wiki_docs)
        except Exception as e:
            print(f"❌ Wikipedia load failed: {e}")

    if not documents:
        raise ValueError("No documents loaded")

    splits = chunk_documents(documents)
    if not splits:
        raise ValueError("No text chunks created")

    os.makedirs(index_path, exist_ok=True)
    faiss_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")

    try:
        if os.path.exists(faiss_file) and os.path.exists(pkl_file):
            vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            vs.add_documents(splits)
        else:
            vs = FAISS.from_documents(splits, embeddings)
        vs.save_local(index_path)
    except Exception as e:
        print(f"❌ Error saving FAISS index: {e}")
        raise

def get_retriever(index_path: str = "data/faiss_index", k: int = 5):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # This is correct for embeddings
        google_api_key=gemini_api_key
    )

    faiss_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")

    if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
        raise ValueError("FAISS index not found or incomplete")

    vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": k})

# NOTE: If you see errors about "models/gemini-pro not found", 
# check your chat model instantiation (likely in agent.py) and use "gemini-pro" instead of "models/gemini-pro".
