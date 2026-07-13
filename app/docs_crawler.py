import os
import uuid
import requests
import ollama

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

os.environ["USER_AGENT"] = "Enterprise-RAG-Crawler/1.0"

SEED_URL = "https://iceberg.apache.org/docs/latest/"

MAX_PAGES = 20

COLLECTION_NAME = "enterprise_docs"

BATCH_SIZE = 100


# --------------------------------------------------
# QDRANT
# --------------------------------------------------

client = QdrantClient(
    host="localhost",
    port=6333,
    timeout=120
)


# --------------------------------------------------
# TEXT SPLITTER
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


# --------------------------------------------------
# STEP 1: CRAWL URLS
# --------------------------------------------------

visited = set()
to_visit = [SEED_URL]

print("\nStarting Crawl...\n")

while to_visit and len(visited) < MAX_PAGES:

    url = to_visit.pop(0)

    if url in visited:
        continue

    try:

        print(f"Crawling: {url}")

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Enterprise-RAG-Crawler"
            }
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        visited.add(url)

        for link in soup.find_all("a", href=True):

            full_url = urljoin(url, link["href"])

            # remove anchors
            full_url = full_url.split("#")[0]

            # only crawl same domain
            if (
                urlparse(full_url).netloc
                == urlparse(SEED_URL).netloc
            ):

                if (
                    full_url not in visited
                    and full_url not in to_visit
                ):
                    to_visit.append(full_url)

    except Exception as e:

        print(f"Error crawling {url}")
        print(e)


print("\nCrawling Complete")
print(f"Pages Found: {len(visited)}")


# --------------------------------------------------
# STEP 2: LOAD CONTENT
# --------------------------------------------------

all_points = []

total_chunks = 0

for page_url in visited:

    try:

        print(f"\nLoading Content: {page_url}")

        loader = WebBaseLoader(page_url)

        docs = loader.load()

        chunks = splitter.split_documents(docs)

        print(f"Chunks: {len(chunks)}")

        total_chunks += len(chunks)

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
                    "source": page_url,
                    "page": "web"
                }
            )

            all_points.append(point)

    except Exception as e:

        print(f"Failed Loading: {page_url}")
        print(e)


# --------------------------------------------------
# STEP 3: UPLOAD TO QDRANT
# --------------------------------------------------

print("\nUploading Vectors...\n")

for i in range(0, len(all_points), BATCH_SIZE):

    batch = all_points[i:i + BATCH_SIZE]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )

    print(
        f"Uploaded {min(i + BATCH_SIZE, len(all_points))}/{len(all_points)}"
    )


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n===================================")
print("Crawler Ingestion Complete")
print("===================================")

print(f"Pages Crawled : {len(visited)}")
print(f"Total Chunks  : {total_chunks}")
print(f"Vectors Added : {len(all_points)}")