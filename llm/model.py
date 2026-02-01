from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline
)
import torch

# ────────────────────────────────────────────────
# Model 1: Email Summarization
# ────────────────────────────────────────────────
SUMMARIZER_MODEL_NAME = "IrisWiris/email-summarizer"

summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME)

def summarize_email_direct(text: str) -> str:
    inputs = summarizer_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = summarizer_model.generate(
            **inputs,
            max_length=100,
            min_length=30,
            num_beams=4,
            early_stopping=True
        )
    return summarizer_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ────────────────────────────────────────────────
# Model 2: Spam Classification
# ────────────────────────────────────────────────
SPAM_MODEL_NAME = "AntiSpamInstitute/spam-detector-bert-MoE-v2.2"

spam_tokenizer = AutoTokenizer.from_pretrained(SPAM_MODEL_NAME)
spam_model = AutoModelForSequenceClassification.from_pretrained(SPAM_MODEL_NAME)

spam_classifier = pipeline(
    "text-classification",
    model=spam_model,
    tokenizer=spam_tokenizer,
    device=-1,                
    return_all_scores=False   # we just want the top label
)

# Legacy Flan-T5 (you can keep or remove)
FLAN_MODEL_NAME = "google/flan-t5-base"
flan_tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL_NAME)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL_NAME)

def run_flan(prompt: str) -> str:
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = flan_model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=3,
            early_stopping=True
        )
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)


# Export useful constants
MODEL_VERSIONS = {
    "summary": SUMMARIZER_MODEL_NAME,
    "spam": SPAM_MODEL_NAME,
    "flan-t5": FLAN_MODEL_NAME
}