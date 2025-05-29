import re
import spacy
from typing import List, Optional
from transformers import PreTrainedTokenizer

# Load spaCy English model once
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str, max_tokens_per_chunk: int = 400, model_tokenizer: Optional[PreTrainedTokenizer] = None, verbose: bool = False) -> List[str]:
    def regex_cleaner(text: str) -> str:
        text = re.sub(r"\[.*?\]", "", text)  # Remove [bracketed text]
        text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text)  # Remove timestamps
        text = re.sub(r"\b(uh|um|erm|you know|like|so|well)\b[,\s]*", "", text, flags=re.IGNORECASE)  # Filler words
        text = re.sub(r"\b(or|and|but)\b(?=\s*[.,!?])", "", text, flags=re.IGNORECASE)  # Remove hanging conjunctions
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)  # Remove repeated words
        text = re.sub(r"\.\.\.", " ... ", text)  # Normalize ellipsis
        text = re.sub(r"([.!?]){2,}", r"\1", text)  # Normalize repeated punctuation
        text = re.sub(r",\s*,+", ",", text)  # Normalize commas
        text = re.sub(r"\s+([?.!,])", r"\1", text)  # Remove space before punctuation
        text = re.sub(r"\s{2,}", " ", text)  # Remove extra spaces
        text = re.sub(r"\b(and|or|but)\b\s+\b(and|or|but)\b", r"\1", text)  # Remove double conjunctions
        return text.strip()

    cleaned_text = regex_cleaner(text)
    doc = nlp(cleaned_text)

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in sentences:
        if model_tokenizer:
            sent_len = len(model_tokenizer.tokenize(sent))
        else:
            sent_len = len(nlp(sent))  # Approximate token count

        if current_tokens + sent_len > max_tokens_per_chunk:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sent]
            current_tokens = sent_len
        else:
            current_chunk.append(sent)
            current_tokens += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    if verbose:
        print(f"Processed into {len(chunks)} chunks with max {max_tokens_per_chunk} tokens each.")
    
    return chunks
