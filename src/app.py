import streamlit as st
import os
import io
from fpdf import FPDF

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

from preprocessing import preprocess_text
from pipeline import summarize_chunks, flan_tokenizer, generate_questions, bart_tokenizer
from transcript import fetch_youtube_transcript
from ocr import extract_text_with_easyocr
from flashcards import build_anki_deck

st.set_page_config(page_title="Study Summarizer", layout="centered")
st.title("📚 Study Summarizer")
st.markdown("Upload a file, paste a transcript, or fetch from YouTube. Get clean AI-generated summaries and questions.")

input_type = st.radio("Choose input type:", ["Upload PDF/Text File", "Paste Transcript Text", "Fetch YouTube Transcript by URL"])
text_input = ""

if input_type == "Upload PDF/Text File":
    uploaded_file = st.file_uploader("Upload file", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            import fitz
            pdf_bytes = uploaded_file.read()
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                text_input = "\n".join([page.get_text() for page in doc])
            if not text_input.strip():
                st.warning("No extractable text found. Running OCR...")
                text_input = extract_text_with_easyocr(pdf_bytes)
        else:
            text_input = uploaded_file.read().decode("utf-8")

elif input_type == "Paste Transcript Text":
    text_input = st.text_area("Paste transcript text here", height=250)

elif input_type == "Fetch YouTube Transcript by URL":
    url = st.text_input("Enter YouTube video URL")

# Initialize session state to persist data across reruns
if "questions" not in st.session_state:
    st.session_state["questions"] = []
if "summaries" not in st.session_state:
    st.session_state["summaries"] = []
if "anki_bytes" not in st.session_state:
    st.session_state["anki_bytes"] = None

with st.form("summarize_form"):
    submitted = st.form_submit_button("Generate Summary + Questions")
    if submitted:
        if input_type == "Fetch YouTube Transcript by URL":
            if not url or not url.strip():
                st.warning("Please enter a valid YouTube URL.")
                st.stop()
            with st.spinner("Fetching YouTube transcript..."):
                text_input = fetch_youtube_transcript(url)
            if not text_input:
                st.error("Failed to fetch transcript. Check the video URL or transcript availability.")
                st.stop()
        else:
            if not text_input:
                st.warning("Please provide input via upload or paste.")
                st.stop()

        # Now process text_input (whether from YouTube or upload/paste)
        with st.spinner("Processing..."):
            summary_chunks = preprocess_text(text_input, model_tokenizer=bart_tokenizer)
            question_chunks = preprocess_text(text_input, model_tokenizer=flan_tokenizer)

            st.session_state["summaries"] = summarize_chunks(summary_chunks)
            st.session_state["questions"] = generate_questions(question_chunks, num_questions=3)

            if st.session_state["questions"]:
                st.session_state["anki_bytes"] = build_anki_deck(
                    st.session_state["questions"], deck_name="Study Summarizer Deck"
                )
            else:
                st.session_state["anki_bytes"] = None


# Display summaries
if st.session_state["summaries"]:
    st.subheader("🧠 Summaries")
    for i, summary in enumerate(st.session_state["summaries"], 1):
        st.markdown(f"**Summary {i}:**")
        st.success(summary)

    # Combine summaries text for downloads
    all_summaries_text = "\n\n".join([f"Summary {i}:\n{summary}" for i, summary in enumerate(st.session_state["summaries"], 1)])

    # TXT download button
    st.download_button(
        label="📝 Download Summaries as .txt",
        data=all_summaries_text,
        file_name="study_summaries.txt",
        mime="text/plain"
    )

    # PDF generation
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "Study Summarizer - Summaries", ln=True, align="C")

        def chapter_body(self, text):
            # Replace problematic chars with safe ones for latin-1
            safe_text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
            self.set_font("Arial", "", 11)
            self.multi_cell(0, 10, safe_text)
            self.ln()

    pdf = PDF()
    pdf.add_page()
    pdf.chapter_body(all_summaries_text)

    # Get PDF as bytes, properly encoded
    pdf_bytes_str = pdf.output(dest="S").encode("latin1")
    pdf_bytes = io.BytesIO(pdf_bytes_str)
    pdf_bytes.seek(0)

    st.download_button(
        label="📄 Download Summaries as .pdf",
        data=pdf_bytes,
        file_name="study_summaries.pdf",
        mime="application/pdf"
    )

# Display questions and Anki deck download
if st.session_state["questions"]:
    st.subheader("❓ Generated Questions")
    for i, (q, a) in enumerate(st.session_state["questions"], 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")

    if st.session_state["anki_bytes"]:
        st.subheader("🧾 Export Flashcards")
        st.download_button(
            label="📥 Download Anki Deck",
            data=st.session_state["anki_bytes"],
            file_name="study_flashcards.apkg",
            mime="application/octet-stream"
        )
