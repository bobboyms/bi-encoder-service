from utils import normalize
from flask import Flask, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer('./model')
model.max_seq_length = 256


@app.route('/')
def hello_geek():
    sentence = "olá mundo doido"
    sentence_embeddings = model.encode([sentence], convert_to_numpy=True)

    return jsonify({
        "statusCode": 200,
        "embedding": normalize(sentence_embeddings[0])
    })


if __name__ == "__main__":
    # app.run(host="localhost", port=8000, debug=True)
    app.run(debug=True)
