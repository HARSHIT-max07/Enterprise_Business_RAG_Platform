from ollama import embed

response = embed(
    model="nomic-embed-text",
    input="Apache Kafka is a distributed event streaming platform"
)

embedding = response["embeddings"][0]

print("Type:", type(embedding))
print("Dimensions:", len(embedding))
print("First 10 values:")
print(embedding[:10])