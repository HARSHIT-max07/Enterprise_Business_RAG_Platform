from qdrant_client import QdrantClient
import ollama

client = QdrantClient("localhost", port=6333)

while True:

    question = input("\nAsk Question: ")

    query_embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=question
    )["embedding"]

    results = client.query_points(
        collection_name="enterprise_docs",
        query=query_embedding,
        limit=5
    ).points

    print("\nTop Results:\n")

    for i, result in enumerate(results, start=1):

        print(f"Result {i}")
        print(f"Source : {result.payload['source']}")
        print(f"Page   : {result.payload['page']}")
        print(f"Score  : {round(result.score, 4)}")
        print()

        print(result.payload["text"][:300])
        print("\n" + "-" * 60)