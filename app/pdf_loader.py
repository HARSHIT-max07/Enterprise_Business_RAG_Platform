from pypdf import PdfReader

pdf_path = "data/raw/document.pdf"

reader = PdfReader(pdf_path)

print(f"Total Pages: {len(reader.pages)}")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("\nTEXT EXTRACTED SUCCESSFULLY\n")

print(text[:3000])