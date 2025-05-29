import re
import spacy
from typing import List, Optional, Literal
from transformers import PreTrainedTokenizerBase

# Load spaCy model globally
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str,max_tokens_per_chunk: int = 400,model_tokenizer: Optional[PreTrainedTokenizerBase] = None,cleaning_level: Literal["light", "strict"] = "strict",max_chunks: Optional[int] = None,verbose: bool = False,min_tokens_per_chunk: int = 15 ) -> List[str]:

    def regex_cleaner(text: str) -> str:
        text = re.sub(r"\[.*?\]", "", text)  # Remove bracketed text
        text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text)  # Timestamps
        text = re.sub(r"\.\.\.", " ... ", text)  # Normalize ellipsis
        text = re.sub(r"([.!?]){2,}", r"\1", text)  # Normalize punctuation

        if cleaning_level == "strict":
            filler_words = r"\b(uh|um|erm|you know|like|so|well)\b[,\s]*"
            text = re.sub(filler_words, "", text, flags=re.IGNORECASE)
            text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)  # Repeated words
            text = re.sub(r"\b(and|or|but)\b(?=\s*[.,!?])", "", text, flags=re.IGNORECASE)  # Hanging conjunctions
            text = re.sub(r"\b(and|or|but)\b\s+\b(and|or|but)\b", r"\1", text, flags=re.IGNORECASE)  # Double conjunctions

        text = re.sub(r",\s*,+", ",", text)  # Normalize commas
        text = re.sub(r"\s+([?.!,])", r"\1", text)  # Space before punctuation
        text = re.sub(r"\s{2,}", " ", text)  # Extra spaces

        return text.strip()

    cleaned_text = regex_cleaner(text)
    doc = nlp(cleaned_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in sentences:
        try:
            sent_tokens = len(model_tokenizer.tokenize(sent)) if model_tokenizer else len(nlp(sent))
        except Exception:
            sent_tokens = len(nlp(sent))

        if sent_tokens > max_tokens_per_chunk:
            if verbose:
                print(f"[SKIP] Sentence too long ({sent_tokens} tokens): {sent[:60]}...")
            continue

        if current_tokens + sent_tokens > max_tokens_per_chunk:
            if current_tokens >= min_tokens_per_chunk:
                chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sent]
            current_tokens = sent_tokens
        else:
            current_chunk.append(sent)
            current_tokens += sent_tokens

    if current_chunk and current_tokens >= min_tokens_per_chunk:
        chunks.append(" ".join(current_chunk).strip())
    if max_chunks:
        chunks = chunks[:max_chunks]
    if verbose:
        print(f"[INFO] Processed into {len(chunks)} chunk(s), max {max_tokens_per_chunk} tokens, min {min_tokens_per_chunk} tokens each.")

    return chunks
