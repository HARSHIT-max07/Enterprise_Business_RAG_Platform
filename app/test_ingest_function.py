from ingest_utils import ingest_pdf

chunks = ingest_pdf("data/raw/docker.pdf")

print(f"Added {chunks} chunks")