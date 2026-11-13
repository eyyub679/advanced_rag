from langchain_chroma import Chroma
from rag.chunks import split
from rag.embedding import get_embedding
from pathlib import Path
import pickle
from rank_bm25 import BM25Okapi # expects a list of lists of string

# the absolute directory of this project
PROJECT_DIR=Path(__file__).resolve().parent.parent
chunks=split()

def ingest_chroma():
    embedding_fn=get_embedding()
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name="rag_langgraph",
        persist_directory=f"{PROJECT_DIR}/chroma" # instead of this i should have used a persistent client
    )

def ingest_bm25():
    text=[doc.page_content.lower().split() for doc in chunks]
    bm25=BM25Okapi(text)
    with open(PROJECT_DIR / "bm25.pkl","wb") as file:
        pickle.dump((bm25,chunks),file)

if __name__ == "__main__":
    # ingest_chroma()
    ingest_bm25()