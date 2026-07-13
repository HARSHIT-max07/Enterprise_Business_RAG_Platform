import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ollama import Client


# ==========================================
# OLLAMA CLIENT
# ==========================================

ollama_client = Client(
    host="http://localhost:11434"
)

# ==========================================
# QDRANT
# ==========================================

client = QdrantClient(
    host="localhost",
    port=6333
)


# ==========================================
# INGEST PDF
# ==========================================

def ingest_pdf(
    pdf_path,
    collection_name="business_docs"
):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    points = []

    for chunk in chunks:

        text = chunk.page_content

        embedding = ollama_client.embeddings(
            model="nomic-embed-text",
            prompt=text
        )["embedding"]

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": text,
                    "source": pdf_path.split("\\")[-1],
                    "page": chunk.metadata.get(
                        "page",
                        0
                    )
                }
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    return len(chunks)