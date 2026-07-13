import uuid
import ollama
import os

os.environ["USER_AGENT"] = "Enterprise-RAG-Platform/1.0"
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


# -----------------------------
# Config
# -----------------------------

URLS = [
    "https://spark.apache.org/docs/latest/",
    "https://kafka.apache.org/documentation/",
    "https://nightlies.apache.org/flink/flink-docs-release-1.20/"
]

COLLECTION_NAME = "enterprise_docs"


# -----------------------------
# Qdrant Connection
# -----------------------------

client = QdrantClient(
    host="localhost",
    port=6333,
    timeout=120
)


# -----------------------------
# Text Splitter
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


all_points = []
total_chunks = 0


# -----------------------------
# Process URLs
# -----------------------------

for url in URLS:

    print(f"\nLoading: {url}")

    loader = WebBaseLoader(url)

    docs = loader.load()

    chunks = splitter.split_documents(docs)

    print(f"Chunks: {len(chunks)}")

    for chunk in chunks:

        text = chunk.page_content.strip()

        if len(text) < 50:
            continue

        embedding = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )["embedding"]

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": text,
                "source": url,
                "page": "web"
            }
        )

        all_points.append(point)

    total_chunks += len(chunks)


# -----------------------------
# Upload In Batches
# -----------------------------

batch_size = 100

for i in range(0, len(all_points), batch_size):

    batch = all_points[i:i + batch_size]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )

    print(
        f"Uploaded {min(i + batch_size, len(all_points))}/{len(all_points)}"
    )


print("\nWeb Ingestion Complete")
print(f"URLs Processed: {len(URLS)}")
print(f"Total Chunks: {total_chunks}")