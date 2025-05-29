# ocr_utils.py
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import easyocr

# Load once globally
reader = easyocr.Reader(['en'], gpu=False)  
def extract_text_with_easyocr(pdf_bytes: bytes, dpi: int = 300) -> str:

    text_blocks = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_np = np.array(img)

        # OCR
        results = reader.readtext(img_np, detail=0)
        page_text = " ".join(results)
        text_blocks.append(page_text.strip())

    return "\n".join(text_blocks).strip()
