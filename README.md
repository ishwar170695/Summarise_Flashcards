# Summarise Flashcards

A Streamlit web app that extracts text from YouTube transcripts, PDFs, or images using OCR, summarizes the content, and generates Anki-style flashcards with spaced repetition metadata.

## Features

- 📄 Preprocess noisy transcripts or OCR output
- ✂️ Chunk long text intelligently
- 🧠 Summarize chunks using facebook/bart-large-cnn
- ❓ Generate Q&A pairs using google/flan-t5-base for flashcards
- 🗂️ Export to Anki `.apkg` deck using `genanki`

## How to Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # Optional if not using the auto-download method
streamlit run src/app.py
