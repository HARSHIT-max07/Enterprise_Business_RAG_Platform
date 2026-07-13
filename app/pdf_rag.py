from ollama import embeddings, chat
from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)

while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    # Generate query embedding
    response = embeddings(
        model="nomic-embed-text",
        prompt=question
    )

    query_vector = response["embedding"]

    # Retrieve relevant chunks
    results = client.query_points(
        collection_name="pdf_knowledge_base",
        query=query_vector,
        limit=3
    )

    context = "\n\n".join(
        point.payload["text"]
        for point in results.points
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not found in the context,
say: "I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    # Ask Llama 3
    answer = chat(
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

    print(answer["message"]["content"])