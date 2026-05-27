import streamlit as st
import tempfile
import os
from ingest import ingest_pdf, clear_collection
from query import answer

st.set_page_config(page_title="DocWizard", page_icon="📄")
st.title("DocWizard")
st.caption("Ask questions about your PDFs — powered by local Ollama")

with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )
    if uploaded_files and st.button("Ingest PDFs"):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                count = ingest_pdf(tmp_path)
            os.unlink(tmp_path)
            st.success(f"{uploaded_file.name}: {count} new chunks added")

    st.divider()
    st.subheader("Manage DB")
    if st.button("🗑️ Clear all chunks", type="secondary"):
        removed = clear_collection()
        st.session_state.messages = []
        st.success(f"Cleared {removed} chunks. Chat history reset.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question about your PDFs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
