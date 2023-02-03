from utils import normalize
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer('./model')
model.max_seq_length = 256


@app.route('/sentence-embedding', methods=['POST'])
def hello_geek():
    request_data = request.get_json()
    sentences = request_data['sentences']
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)

    embeddings = []
    length = len(sentence_embeddings)
    for i in range(length):
        embeddings.append(normalize(sentence_embeddings[i]))

    return jsonify({
        "length": length,
        "embeddings": embeddings
    }), 200, {'ContentType': 'application/json'}


if __name__ == "__main__":
    # app.run(host="localhost", port=8000, debug=True)
    app.run(debug=True)
