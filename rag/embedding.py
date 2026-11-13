from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={
            "normalize_embeddings":True
        }
    )