import streamlit as st
import tempfile

from document_loader import load_documents
from vector_store import create_vector_store
from chatbot import build_qa_chain


# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="AI Document Chatbot", layout="wide")

st.title("📄 AI Document Chatbot with Source Citations")
st.write("Upload documents, ask questions, get answers with sources!")


# -------------------------------
# Chat History Setup
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


# -------------------------------
# File Upload Section
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF or TXT documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# Sidebar File Display
st.sidebar.title("📂 Uploaded Files")
if uploaded_files:
    for f in uploaded_files:
        st.sidebar.write("📄", f.name)


# -------------------------------
# Process Documents Button
# -------------------------------
if uploaded_files and st.session_state.qa_chain is None:

    if st.button("⚡ Process Documents"):

        all_docs = []

        with st.spinner("Processing documents..."):

            for file in uploaded_files:
                # Save file temporarily
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(file.read())

                # Load + split documents with metadata
                docs = load_documents(temp_file.name, file.name)
                all_docs.extend(docs)

            # Create Vector Store
            db = create_vector_store(all_docs)

            # Build QA Chain
            st.session_state.qa_chain = build_qa_chain(db)

        st.success("✅ Documents processed successfully! You can now chat below.")


# -------------------------------
# Display Chat Messages
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -------------------------------
# Chat Input Section
# -------------------------------
if st.session_state.qa_chain:

    query = st.chat_input("Ask something about your uploaded documents...")

    if query:

        # Store user message
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                result = st.session_state.qa_chain(query)

                answer = result["result"]
                sources = result["source_documents"]

                # Display answer
                st.write("### ✅ Answer")
                st.write(answer)

                # Display sources
                st.write("### 📌 Sources")

                for doc in sources:
                    source_file = doc.metadata.get("source", "Unknown File")
                    page = doc.metadata.get("page", "N/A")

                    st.markdown(
                        f"**File:** `{source_file}` | **Page:** `{page}`"
                    )

                    st.write(doc.page_content[:300] + "...")
                    st.write("------")

        # Store assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


# -------------------------------
# Summarize + Compare Buttons
# -------------------------------
if st.session_state.qa_chain:

    st.sidebar.title("🛠 Actions")

    if st.sidebar.button("📝 Summarize Documents"):
        with st.spinner("Generating summary..."):
            summary = st.session_state.qa_chain(
                "Give a detailed summary of all uploaded documents."
            )
        st.sidebar.success("Done!")