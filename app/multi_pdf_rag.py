from qdrant_client import QdrantClient
import ollama

client = QdrantClient("localhost", port=6333)

while True:

    question = input("\nAsk Question: ")

    # Generate query embedding
    query_embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=question
    )["embedding"]

    # Retrieve top chunks
    results = client.query_points(
        collection_name="enterprise_docs",
        query=query_embedding,
        limit=5
    ).points

    # Build context
    context = "\n\n".join(
        result.payload["text"]
        for result in results
    )

    # Prompt for Llama 3
    prompt = f"""
You are a helpful enterprise knowledge assistant.

Answer the question ONLY using the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate answer
    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print("\nSOURCES:")
    
    seen = set()

    for result in results:

        source = result.payload["source"]
        page = result.payload["page"]

        key = (source, page)

        if key not in seen:
            seen.add(key)
            print(f"- {source} (Page {page})")

    print("=" * 80)