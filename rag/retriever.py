import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rag.embeddings import build_vector_store

# Build vector store
vector_store = build_vector_store()


def retrieve_chunks(query_embedding, top_k=3):

    similarities = []

    for item in vector_store:

        doc_embedding = np.array(item["embedding"]).reshape(1, -1)
        query_embedding = np.array(query_embedding).reshape(1, -1)

        score = cosine_similarity(query_embedding, doc_embedding)[0][0]

        similarities.append({
            "title": item["title"],
            "content": item["content"],
            "score": score
        })

    # Sort by similarity score
    similarities.sort(key=lambda x: x["score"], reverse=True)

    return similarities[:top_k]