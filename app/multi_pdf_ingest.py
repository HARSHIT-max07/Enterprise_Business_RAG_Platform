import os
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import ollama


# -----------------------------
# Qdrant Setup
# -----------------------------

client = QdrantClient("localhost", port=6333)

collection_name = "enterprise_docs"

# Delete old collection if exists
try:
    client.delete_collection(collection_name)
except:
    pass

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    )
)

print("Collection created")


# -----------------------------
# Load PDFs
# -----------------------------

pdf_folder = "data/raw/enterprise_docs"

pdf_files = [
    file
    for file in os.listdir(pdf_folder)
    if file.endswith(".pdf")
]

print(f"\nFound {len(pdf_files)} PDFs\n")


# -----------------------------
# Chunking Setup
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

all_points = []

total_chunks = 0


# -----------------------------
# Process PDFs
# -----------------------------

for pdf_file in pdf_files:

    pdf_path = os.path.join(pdf_folder, pdf_file)

    print(f"Processing {pdf_file}")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    chunks = splitter.split_documents(documents)

    print(f"Chunks: {len(chunks)}")

    for chunk in chunks:

        page = chunk.metadata.get("page", 0)

        text = chunk.page_content.strip()

        # Skip cover pages
        if page == 0:
            continue

        # Skip very small chunks
        if len(text) < 100:
            continue

        # Skip garbage chunks
        garbage_terms = [
            "table of contents",
            "contents",
            "all rights reserved",
            "copyright",
            "notices"
        ]

        if any(
            term in text.lower()
            for term in garbage_terms
        ):
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
                "source": pdf_file,
                "page": chunk.metadata.get("page", 0)
            }
        )

        all_points.append(point)

    total_chunks += len(chunks)


# -----------------------------
# Upload in Batches
# -----------------------------

batch_size = 100

for i in range(0, len(all_points), batch_size):

    batch = all_points[i:i + batch_size]

    client.upsert(
        collection_name=collection_name,
        points=batch
    )

    print(
        f"Uploaded {min(i + batch_size, len(all_points))}/{len(all_points)} vectors"
    )

print("\nUpload Complete")
print(f"Total PDFs: {len(pdf_files)}")
print(f"Total Chunks: {total_chunks}")