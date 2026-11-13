from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.loader import load

def split():
    # loading documents
    docs=load()

    # defining splitter
    splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=256,
        chunk_overlap=50,
    )

    # splitting
    chunks=splitter.split_documents(docs)

    return chunks
