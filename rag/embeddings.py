from sentence_transformers import SentenceTransformer
from rag.chunker import create_chunks

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Vector store
vector_store = []


def generate_embedding(text):
    """
    Generate embedding using local transformer model
    """
    embedding = model.encode(text)
    return embedding


def build_vector_store():
    """
    Create embeddings for document chunks
    """
    chunks = create_chunks()

    for chunk in chunks:

        embedding = generate_embedding(chunk["content"])

        vector_store.append({
            "title": chunk["title"],
            "content": chunk["content"],
            "embedding": embedding
        })

    return vector_store