from ollama import embeddings
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

question = input("Ask a question: ")

response = embeddings(
    model="nomic-embed-text",
    prompt=question
)

query_vector = response["embedding"]

results = client.query_points(
    collection_name="pdf_knowledge_base",
    query=query_vector,
    limit=3
)

print("\nTOP MATCHES:\n")

for point in results.points:
    print("=" * 80)
    print(f"Score: {point.score:.4f}")
    print(point.payload["text"])