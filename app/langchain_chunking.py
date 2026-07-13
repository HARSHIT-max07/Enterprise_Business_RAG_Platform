from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("data/raw/document.pdf")

documents = loader.load()

print(f"Pages Loaded: {len(documents)}")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

print("\nFIRST CHUNK:\n")
print(chunks[0].page_content)

print("\nMETADATA:\n")
print(chunks[0].metadata)