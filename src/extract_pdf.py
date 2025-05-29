import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        all_text.append(text)

    return all_text

# Example usage 
pdf_path = r'C:\Users\ishu\Downloads\Telegram Desktop\UPSC GS 2025.pdf'
pages_text = extract_text_from_pdf(pdf_path)

# Print each page's text
for i, text in enumerate(pages_text):
    print(f"\n--- Page {i+1} ---\n{text}")
