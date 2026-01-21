from embeddings import load_vector_store
from llm import load_llm


def answer_question(
    question: str,
    index_path="vector_index",
    k=6
):
    try:
        # Load vector store
        vectorstore = load_vector_store(index_path)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        # Retrieve relevant documents
        docs = retriever.invoke(question)

        if not docs:
            return "No relevant context found."

        # Build context
        context = "\n\n".join(doc.page_content for doc in docs)

        # Load LLM
        llm = load_llm()

        # Construct prompt explicitly
        prompt = f"""
                    You are an assistant that answers questions using ONLY the context below.
                    If the answer is not in the context, say "I don't know".

                    Context:
                    {context}

                    Question:
                    {question}
                """

        # Invoke LLM
        response = llm.invoke(prompt)

        return response

    except Exception as e:
        raise RuntimeError(f"RAG pipeline failed: {str(e)}")
