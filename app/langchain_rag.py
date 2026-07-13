from ollama import embeddings, chat
from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=6333
)

while True:

    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

    query_embedding = embeddings(
        model="nomic-embed-text",
        prompt=question
    )["embedding"]

    results = client.query_points(
        collection_name="langchain_pdf",
        query=query_embedding,
        limit=3
    )

    context = ""

    for point in results.points:
        context += point.payload["text"] + "\n\n"

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the provided context.

If the answer is not available,
say you could not find it in the document.

Context:
{context}

Question:
{question}

Answer:
"""

    response = chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(response["message"]["content"])

    print("\nSOURCES")

    for point in results.points:

        print(
            f"- {point.payload['source']} "
            f"(Page {point.payload['page']})"
        )