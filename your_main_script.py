# utils/loaders.py

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.docstore.document import Document
from typing import List
import os

try:
    from langchain_community.utilities import WikipediaAPIWrapper  # ✅ Required for Wikipedia
except ImportError as e:
    raise ImportError(
        "The 'wikipedia' Python package is required for WikipediaAPIWrapper. "
        "Please install it with `pip install wikipedia`."
    ) from e

def load_pdf(path: str) -> List[Document]:
    """Load and return documents from a PDF file."""
    loader = PyPDFLoader(path)
    docs = loader.load()
    docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
    if not docs:
        raise ValueError(f"No valid text found in PDF file: {path}")
    return docs

def load_docx(path: str) -> List[Document]:
    """Load and return documents from a DOCX file."""
    loader = Docx2txtLoader(path)
    docs = loader.load()
    docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
    if not docs:
        raise ValueError(f"No valid text found in DOCX file: {path}")
    return docs

def load_txt(path: str) -> List[Document]:
    """Load and return documents from a plain text file."""
    loader = TextLoader(path)
    docs = loader.load()
    docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
    if not docs:
        raise ValueError(f"No valid text found in TXT file: {path}")
    return docs

def load_wikipedia(topic: str, max_chars: int = 10000) -> List[Document]:
    """Fetch and return a Wikipedia page as a Document."""
    try:
        wiki = WikipediaAPIWrapper()
        content = wiki.run(topic)[:max_chars]  # ✅ Limit characters
        if not content or not content.strip():
            raise ValueError(f"No content found for Wikipedia topic: {topic}")
        return [Document(page_content=content, metadata={"source": f"Wikipedia:{topic}"})]
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Wikipedia topic '{topic}': {e}\n"
            "Make sure the 'wikipedia' package is installed: pip install wikipedia"
        )

def ensure_faiss_index(index_path: str, build_index_func):
    """Ensure the FAISS index exists, and build it if it doesn't."""
    if not os.path.exists(index_path):
        print(f"FAISS index not found at {index_path}. Building index...")
        build_index_func()
    else:
        print(f"FAISS index found at {index_path}.")