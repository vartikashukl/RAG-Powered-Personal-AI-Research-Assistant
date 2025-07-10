# ✅ FIXED main.py
import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import build_faiss_index, get_retriever
from agent import ask, init_retriever, memory

load_dotenv()

os.makedirs("data", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)
os.makedirs("utils", exist_ok=True)

st.set_page_config(page_title="RAG-Powered AI Research Assistant", layout="wide", page_icon="🧠")

st.sidebar.header("📊 Index Management")

faiss_index_path = "data/faiss_index"
faiss_files = ["index.faiss", "index.pkl"]
faiss_exists = all(os.path.exists(os.path.join(faiss_index_path, f)) for f in faiss_files)

with st.sidebar.expander("🔍 Debug: Environment Variables"):
    gemini_key = os.getenv("GEMINI_API_KEY")
    st.write(f"{'✅' if gemini_key else '❌'} GEMINI_API_KEY: {gemini_key[:10]+'...' if gemini_key else 'Not found'}")

uploaded_files = st.sidebar.file_uploader("Upload Documents (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
wiki_topic = st.sidebar.text_input("Wikipedia Topic", value="")

if st.sidebar.button("Build/Update Index"):
    if not uploaded_files and not wiki_topic:
        st.sidebar.error("Upload files or enter a Wikipedia topic.")
    else:
        temp_paths = []
        for file in uploaded_files:
            save_path = os.path.join("data", file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            temp_paths.append(save_path)

        try:
            build_faiss_index(temp_paths, use_wikipedia=bool(wiki_topic), wiki_topic=wiki_topic)
            init_retriever()
            st.sidebar.success("✅ Index built/updated successfully.")
            faiss_exists = True
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Failed to build index: {e}")

if not faiss_exists:
    if st.sidebar.button("🔄 Auto-build with Wikipedia AI"):
        with st.spinner("🔄 Building default index..."):
            try:
                build_faiss_index([], use_wikipedia=True, wiki_topic="Artificial intelligence")
                init_retriever()
                faiss_exists = True
                st.sidebar.success("✅ Default Wikipedia index built!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Auto-build failed: {e}")

st.title("🧠 RAG-Powered AI Research Assistant")

if not faiss_exists:
    st.warning("⚠️ **No FAISS index found!** Please build the index first by:")
    st.write("1. Uploading documents in the sidebar, or")
    st.write("2. Entering a Wikipedia topic, or")
    st.write("3. Using the auto-build button")
    st.stop()

if not hasattr(st.session_state, 'retriever_initialized'):
    try:
        init_retriever()
        st.session_state.retriever_initialized = True
    except Exception as e:
        st.error(f"❌ Failed to initialize retriever: {e}")
        st.stop()

if hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
    st.subheader("💬 Chat History")
    for message in memory.chat_memory.messages:
        prefix = "👩" if getattr(message, "type", "") == "human" else "🧠"
        st.markdown(f"**{prefix} {message.type.title() if hasattr(message, 'type') else 'Message'}:** {message.content}")

st.subheader("❓ Ask a Question")
query = st.text_input("Enter your question:", placeholder="e.g., What is artificial intelligence?")
col1, col2 = st.columns([1, 4])

with col1:
    send_button = st.button("🚀 Send Query", type="primary")
with col2:
    if st.button("🗑️ Clear Chat"):
        memory.clear()
        st.rerun()

if send_button and query:
    with st.spinner("🤔 Processing your question..."):
        answer, sources = ask(query)
        st.subheader("📜 Answer")
        st.markdown(answer)

        if sources:
            st.subheader("📚 Sources")
            for i, src in enumerate(sources, 1):
                st.markdown(f"{i}. {src}")
        else:
            st.info("No sources found for this query.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join("data", "logs", f"chat_{timestamp}.txt"), "w") as f:
            f.write(f"Timestamp: {timestamp}\nQuery: {query}\nAnswer: {answer}\nSources: {sources}\n")
        st.rerun()

st.markdown("---")
st.markdown("💡 **Tip:** Upload documents or use Wikipedia to get started!")
