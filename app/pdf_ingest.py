from pypdf import PdfReader
from ollama import embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import re

# --------------------------
# PDF Extraction
# --------------------------

pdf_path = "data/raw/document.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

# --------------------------
# Clean Text
# --------------------------

text = re.sub(r"\s+", " ", text)

# --------------------------
# Chunking with Overlap
# --------------------------

chunk_size = 500
overlap = 100

chunks = []

start = 0

while start < len(text):
    end = start + chunk_size

    chunk = text[start:end]

    chunks.append(chunk)

    start += chunk_size - overlap

print(f"Total Chunks: {len(chunks)}")

# --------------------------
# Qdrant Setup
# --------------------------

client = QdrantClient("localhost", port=6333)

collection_name = "pdf_knowledge_base"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    )
)

# --------------------------
# Generate Embeddings
# --------------------------

points = []

for idx, chunk in enumerate(chunks):

    response = embeddings(
        model="nomic-embed-text",
        prompt=chunk
    )

    vector = response["embedding"]

    points.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload={
                "text": chunk
            }
        )
    )

# --------------------------
# Upload to Qdrant
# --------------------------

client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"\nStored {len(points)} chunks in Qdrant successfully!")