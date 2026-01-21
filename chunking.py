from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    try:
        if not documents:
            raise ValueError("No documents provided for chunking")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise RuntimeError("Chunking produced no chunks")

        return chunks

    except Exception as e:
        raise RuntimeError(f"Error during document chunking: {str(e)}")
