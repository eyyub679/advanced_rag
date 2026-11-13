from sentence_transformers import CrossEncoder
reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
import numpy as np

def rerank(documents,query):
    """reranks documents and returns top 5"""
    pairs=[(query,doc.page_content) for doc in documents]
    scores=reranker.predict(pairs)
    indices=np.argsort(scores)[::-1][:5]
    relevant_docs=[documents[i] for i in indices]
    return relevant_docs