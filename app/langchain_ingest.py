from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ollama import embeddings

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

# Load PDF
loader = PyPDFLoader("data/raw/document.pdf")

documents = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

# Connect Qdrant
client = QdrantClient(
    host="localhost",
    port=6333
)

collection_name = "langchain_pdf"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )

points = []

for idx, chunk in enumerate(chunks):

    response = embeddings(
        model="nomic-embed-text",
        prompt=chunk.page_content
    )

    vector = response["embedding"]

    points.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload={
                "text": chunk.page_content,
                "page": chunk.metadata["page"],
                "source": chunk.metadata["source"]
            }
        )
    )

client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"Stored {len(points)} chunks successfully!")