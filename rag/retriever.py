from langchain_chroma import Chroma
from rag.embedding import get_embedding
from pathlib import Path
import pickle
import numpy as np

#absolute path of this project
PROJECT_DIR=Path(__file__).resolve().parent.parent

# load variables
with open(PROJECT_DIR / "bm25.pkl","rb") as file:
    bm25,chunks=pickle.load(file)

# vector database
embedding_fn=get_embedding()
vectorstore=Chroma(
        collection_name="rag_langgraph",
        embedding_function=embedding_fn,
        persist_directory=f"{PROJECT_DIR}/chroma" # creating an explicit persistent client and passing it into here is better, but argument is client
    )

def retrieve_documents(query):
    # retrieval
    retriever=vectorstore.as_retriever(
        search_kwargs={"k":20}
    )
    docs_dense=retriever.invoke(query)
    query_text=query.lower().split()
    scores=bm25.get_scores(query_text)
    indices=np.argsort(scores)[::-1][:20]
    docs_sparse=[chunks[i] for i in indices]

    rrf_scores={}
    docs={}
    k=60 # default
    for rank,doc in enumerate(docs_dense,start=1):
        # we will use doc ids for keys to our dict (to distinguish docs)
        id=hash(doc.page_content) # not movie_id cuz different chunks can have the same id (because of splitter)
        rrf_scores[id]=1/(k+rank)
        docs[id]=doc
    for rank,doc in enumerate(docs_sparse,start=1):
        id=hash(doc.page_content)
        score=1/(k+rank)
        if id in rrf_scores:
            rrf_scores[id]+=score
        else:
            docs[id]=doc
            rrf_scores[id]=score
    indices=sorted(rrf_scores,key=lambda x: rrf_scores[x],reverse=True)[:20]
    documents=[docs[id] for id in indices]
    return documents
    