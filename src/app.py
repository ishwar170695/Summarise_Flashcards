import streamlit as st
import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"  # Avoid torch class probing issue

from preprocessing import preprocess_text
from pipeline import summarize_chunks, qa_tokenizer, generate_questions
from transcript import fetch_youtube_transcript
from ocr import extract_text_with_easyocr 

st.set_page_config(page_title="Study Summarizer", layout="centered")
st.title("📚 Study Summarizer")
st.markdown("Upload a file, paste a transcript, or fetch from YouTube. Get clean AI-generated summaries and questions.")

def load_text_input():
    input_type = st.radio("Choose input type:", [
        "Upload PDF/Text File", 
        "Paste Transcript Text", 
        "Fetch YouTube Transcript by URL"
    ])
    text = ""

    if input_type == "Upload PDF/Text File":
        uploaded_file = st.file_uploader("Upload file", type=["pdf", "txt"])

        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                import fitz
                pdf_bytes = uploaded_file.read()

                # Try extracting text
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    text = "\n".join([page.get_text() for page in doc])

                # OCR fallback
                if not text.strip():
                    st.warning("No extractable text found. Running OCR...")
                    text = extract_text_with_easyocr(pdf_bytes)

            else:  # .txt
                text = uploaded_file.read().decode("utf-8")

    elif input_type == "Paste Transcript Text":
        text = st.text_area("Paste transcript text here", height=250)

    elif input_type == "Fetch YouTube Transcript by URL":
        url = st.text_input("Enter YouTube video URL")
        if url:
            with st.spinner("Fetching transcript..."):
                transcript = fetch_youtube_transcript(url)
            if transcript:
                st.success("Transcript fetched successfully!")
                text = transcript
            else:
                st.error("Failed to fetch transcript. Check the video URL or transcript availability.")

    return text.strip()

with st.form("summarize_form"):
    text_input = load_text_input()
    submitted = st.form_submit_button("Generate Summary + Questions")

    if submitted:
        if not text_input:
            st.warning("Please provide input via upload, paste, or YouTube URL.")
        else:
            with st.spinner("Processing..."):
                chunks = preprocess_text(text_input, model_tokenizer=qa_tokenizer)
                summaries = summarize_chunks(chunks)
                questions = generate_questions(chunks, num_questions=3)

            st.subheader("🧠 Summaries")
            for i, summary in enumerate(summaries, 1):
                st.markdown(f"**Summary {i}:**")
                st.success(summary)

            st.subheader("❓ Generated Questions")
            if questions:
                for i, q in enumerate(questions, 1):
                    st.markdown(f"**Q{i}:** {q}")
            else:
                st.info("No questions generated. Try using a more informative input.")
