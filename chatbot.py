from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq

def build_qa_chain(vector_db):

    llm = ChatGroq(
        model="llama3-8b-8192"
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa
