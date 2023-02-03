# bi-encoder-service
Service responsible for generating embeddings taking a list of strings as input parameter. Embeddings are useful for calculating semantic similarity between 2 texts, using cosine similarity.

## Generate docker image

first clone this repo, enter into the folder and execute docker build command

```
docker built -t bi-encoder-service
```

## Run docker image
```
docker run -p 3000:5000 bi-encoder-service
```

## How use the service

```
curl --location --request POST 'http://localhost:3000/sentence-embedding' \
--header 'Content-Type: application/json' \
--data-raw '{
    "sentences" : ["Hello","world","my friend"]
}'
```
