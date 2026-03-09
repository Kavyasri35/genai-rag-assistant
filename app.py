from flask import Flask, request, jsonify, render_template
from rag.embeddings import generate_embedding
from rag.retriever import retrieve_chunks

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.json
    message = data.get("message")

    if not message:
        return jsonify({"error": "Message required"}), 400

    # Convert question to embedding
    query_embedding = generate_embedding(message)

    # Retrieve similar chunks
    chunks = retrieve_chunks(query_embedding)

    if not chunks:
        reply = "Sorry, I couldn't find relevant information in the documents."
    else:
        reply = chunks[0]["content"]

    return jsonify({
        "reply": reply,
        "retrievedChunks": len(chunks)
    })


if __name__ == "__main__":
    app.run(debug=True)