from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(file_path, file_name):

    docs = []

    # ✅ PDF Loading (Safe)
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()

    # ✅ TXT Loading (Cloud-safe)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        docs = [Document(page_content=text, metadata={"source": file_name})]

    # Add metadata (file name)
    for doc in docs:
        doc.metadata["source"] = file_name

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(docs)
