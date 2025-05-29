from transformers import pipeline, BartTokenizerFast, BartForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer
import spacy
from difflib import SequenceMatcher
from preprocessing import preprocess_text  

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model=bart_model, tokenizer=bart_tokenizer)

flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
flan_pipeline = pipeline("text2text-generation", model=flan_model, tokenizer=flan_tokenizer)

nlp = spacy.load("en_core_web_sm")

# Summarize text chunks
def summarize_chunks(chunks):
    summaries = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        try:
            summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False, num_beams=4)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"[!] Error summarizing chunk {i + 1}: {e}")
    return summaries

# Similarity filter to avoid duplicate questions
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Highlight sentence for Flan-T5 question generation
def highlight_sentence_for_qg(context: str, sentence: str) -> str:
    return f"generate question: {context.replace(sentence, f'<hl> {sentence} <hl>', 1)}"

# Generate questions and answers from summarized text
def generate_questions(chunks, num_questions=3):
    qa_pairs = []
    seen_questions = []

    for chunk in chunks:
        sentences = [sent.text.strip() for sent in nlp(chunk).sents if sent.text.strip()]
        count = 0

        for sent in sentences:
            if count >= num_questions:
                break

            prompt = highlight_sentence_for_qg(chunk, sent)
            try:
                q_output = flan_pipeline(prompt, max_new_tokens=64, do_sample=False)
                question = q_output[0]["generated_text"].strip()
            except Exception:
                continue

            # Filters
            if (
                any(similar(question.lower(), q.lower()) > 0.80 for q in seen_questions)
                or not question.endswith("?")
                or len(question.split()) < 4
                or "sample text" in question.lower()
                or "which of the following" in question.lower()
            ):
                continue

            # Generate answer
            a_prompt = f"Provide a direct and informative answer to the question below, based on the provided context.\n\nContext: {chunk}\nQuestion: {question}\nAnswer:"
            try:
                a_output = flan_pipeline(a_prompt, max_new_tokens=64, do_sample=False)
                answer = a_output[0]['generated_text'].strip()
            except Exception:
                continue

            if not answer or len(answer.split()) < 2:
                continue

            qa_pairs.append((question, answer))
            seen_questions.append(question)
            count += 1

    return qa_pairs

# Complete pipeline
def run_pipeline(text):
    chunks = preprocess_text(text, model_tokenizer=flan_tokenizer)
    summaries = summarize_chunks(chunks)
    qa_pairs = generate_questions(summaries, num_questions=3)
    return summaries, qa_pairs

