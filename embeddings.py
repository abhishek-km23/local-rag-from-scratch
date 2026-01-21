from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def build_vector_store(chunks, index_path="vector_index"):
    try:
        if not chunks:
            raise ValueError("No chunks provided for embedding")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Persist index to disk
        vectorstore.save_local(index_path)

        return vectorstore

    except Exception as e:
        raise RuntimeError(f"Error while building vector store: {str(e)}")


def load_vector_store(index_path="vector_index"):
    try:
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Vector index not found at '{index_path}'. Build it first."
            )

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # We explicitly allow deserialization because:
        # - the index was created locally by us
        # - the source is trusted
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    except Exception as e:
        raise RuntimeError(f"Error while loading vector store: {str(e)}")

