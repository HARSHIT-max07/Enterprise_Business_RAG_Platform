from ollama import chat

response = chat(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": "What is Apache Kafka?"
        }
    ]
)

print(response["message"]["content"])