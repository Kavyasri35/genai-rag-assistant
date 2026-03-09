import json

CHUNK_SIZE = 120
OVERLAP = 20


def load_documents(file_path="docs.json"):
    with open(file_path, "r") as f:
        return json.load(f)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size
        chunk = words[start:end]

        chunks.append(" ".join(chunk))

        start += chunk_size - overlap

    return chunks


def create_chunks():

    documents = load_documents()

    all_chunks = []

    for doc in documents:

        chunks = chunk_text(doc["content"])

        for chunk in chunks:
            all_chunks.append({
                "title": doc["title"],
                "content": chunk
            })

    return all_chunks