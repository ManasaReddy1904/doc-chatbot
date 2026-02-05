from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_vector_store(documents):

    # ✅ Safety check
    if not documents or len(documents) == 0:
        raise ValueError("No valid text found in uploaded documents.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(documents, embeddings)

    return db
