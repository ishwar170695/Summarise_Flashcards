from transformers import pipeline, BartTokenizerFast, BartForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer
import spacy
from difflib import SequenceMatcher
from preprocessing import preprocess_text

# Initialize models
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

qa_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
qa_model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")
qa_pipeline = pipeline("text2text-generation", model=qa_model, tokenizer=qa_tokenizer)

nlp = spacy.load("en_core_web_sm")


def summarize_chunks(chunks):
    summaries = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            print(f"Skipping empty chunk {i+1}")
            continue

        inputs = tokenizer(chunk, max_length=1024, truncation=True, return_tensors="pt")
        truncated_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

        summary = summarizer(truncated_text, max_length=150, min_length=40, do_sample=False)
        text = summary[0]['summary_text']
        summaries.append(text)
        print(f"\nSummary {i+1}:\n{text}\n{'='*80}")
    return summaries


def highlight_sentence_for_qg(context: str, sentence: str) -> str:
    return f"generate question: {context.replace(sentence, f'<hl> {sentence} <hl>')}"


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def generate_questions(chunks, num_questions=3):
    questions = []
    seen_questions = []

    for chunk in chunks:
        sentences = [sent.text.strip() for sent in nlp(chunk).sents if sent.text.strip()]
        count = 0

        for sent in sentences:
            if count >= num_questions:
                break

            prompt = highlight_sentence_for_qg(chunk, sent)
            outputs = qa_pipeline(prompt, max_new_tokens=64, do_sample=False)
            generated_question = outputs[0]["generated_text"].strip()

            # Filters for quality and duplicates
            if not generated_question.endswith("?") or len(generated_question.split()) < 4:
                continue
            if any(similar(generated_question, q) > 0.8 for q in seen_questions):
                continue
            if "sample text" in generated_question.lower():  # Remove boilerplate
                continue

            questions.append(generated_question)
            seen_questions.append(generated_question)
            count += 1

    return questions


def run_pipeline(text):
    chunks = preprocess_text(text, model_tokenizer=qa_tokenizer)
    summaries = summarize_chunks(chunks)
    questions = generate_questions(summaries, num_questions=3)  # Use summaries as input here
    return summaries, questions


# Example usage:
text = """
This is a sample text for testing the summarization and question generation pipeline. It contains multiple sentences that will be processed to generate summaries and questions.
AI is transforming the way we interact with technology, making it more intuitive and efficient. The future of AI holds immense potential for innovation and growth.
Ai is a powerful tool that can enhance productivity and creativity across various fields. It is important to understand its capabilities and limitations.
Healthcare is one of the sectors where AI can make a significant impact, improving patient outcomes and streamlining operations.
modern AI systems are designed to learn from data, adapt to new information, and provide insights that were previously unattainable.
another important aspect of AI is its ethical implications, which require careful consideration and regulation to ensure responsible use.
social media platforms are increasingly using AI to personalize user experiences, recommend content, and detect harmful behavior.
zero-shot learning is a fascinating area of AI research, allowing models to perform tasks without explicit training on those tasks.
xAI is also being used in creative fields, such as art and music, where it can assist artists in generating new ideas and compositions.
data privacy is a critical concern in the age of AI, as vast amounts of personal information are collected and analyzed.
virtual assistants powered by AI are becoming more prevalent, helping users manage their daily tasks and schedules.
"""

# Run pipeline
summaries, questions = run_pipeline(text)
print("Summaries:", summaries)
print("Questions:", questions)
