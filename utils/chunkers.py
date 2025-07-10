# utils/chunkers.py
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks."""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    try:
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"❌ Error chunking documents: {e}")
        return []