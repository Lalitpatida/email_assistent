# Intelligent Email Assistant API

Modern email processing API powered by specialized machine learning models for spam detection and email summarization, with a legacy prompt-based action suggestion system.

## Features

- **Spam classification** — using a fine-tuned BERT Mixture-of-Experts model  
- **Email summarization** — using a dedicated seq2seq model fine-tuned on email data  
- **Legacy action suggestion** — zero-shot classification (REPLY / ARCHIVE / FLAG) using Flan-T5  
- RESTful API built with **FastAPI**  
- All predictions are logged to a PostgreSQL database  
- Clear separation between modern task-specific models and legacy prompt-based approach

## Architecture Overview
Client ──► FastAPI ─┬─► POST /email/classify-spam
├─► POST /email/summarize
└─► POST /email/action (legacy)
│
▼
Email Service Layer
│       │       │
┌─────────┼───────┼───────┘
│         │       │
Spam Classifier   Summarizer   Flan-T5 + Prompts (legacy)
(BERT MoE)     (seq2seq)         (google/flan-t5-base)
│         │               │
└─────────┼───────────────┘
│
▼
PostgreSQL
(prediction logs)



The system uses **specialized fine-tuned models** for spam detection and summarization (better accuracy & efficiency), while keeping the original **Flan-T5 + prompt** approach for action classification and comparison.

## Models Used

| Task              | Model ID / Repository                               | Model Type            | Notes                                 |
|-------------------|------------------------------------------------------|-----------------------|---------------------------------------|
| Spam Detection    | `AntiSpamInstitute/spam-detector-bert-MoE-v2.2`     | BERT Mixture-of-Experts | Fine-tuned for spam/ham classification |
| Summarization     | `IrisWiris/email-summarizer`                        | Seq2Seq (T5-family)   | Fine-tuned specifically on emails     |
| Action (legacy)   | `google/flan-t5-base`                               | Flan-T5               | Zero-shot via strict prompting        |

## LLM / Model Integration Approach

- **Spam & Summarization**: direct inference using **specialized fine-tuned models** (no prompting)  
- **Legacy action classification**: **prompt engineering** + Flan-T5 (zero-shot)  
- All models run **locally** using Hugging Face `transformers` library — **no external API calls**  
- Models are loaded once at application startup (global scope in `llm/models.py`)  
- Inference is currently **synchronous** (suitable for low-to-moderate load)

## Prompt Design (used only in legacy Flan-T5 path)

Very concise and strict prompts are used to improve reliability with the smaller Flan-T5 model:

```text
# Spam classification prompt
Classify the following email as SPAM or NOT_SPAM.

Email:
{email}

Answer only SPAM or NOT_SPAM.


# Summarization prompt
You are an email summarization system.

Summarize the email in ONE short sentence.
Do NOT repeat the email.
Do NOT add new information.

Email:
{email}

# Action suggestion prompt (legacy)
Based on the email below, suggest one action:
REPLY, ARCHIVE, or FLAG.

Email:
{email}

Answer with only one word.

How to Run the Project

Prerequisites
Python 3.9+
PostgreSQL (local or cloud)

# Steps

Clone the repository
git clone <your-repository-url>
cd intelligent-email-assistant
Create and activate virtual environment

# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Create .env file in the root directory
DATABASE_URL=postgresql://username:password@localhost:5432/email_assistant

Create the database in PostgreSQL
CREATE DATABASE email_assistant;

Start the API server
uvicorn main:app --reload --port 8000

API will be available at:
→ http://localhost:8000
→ Interactive docs: http://localhost:8000/docs

Project Structure
.
├── main.py                     # FastAPI app entry point
├── database.py                 # SQLAlchemy engine & session
├── models.py                   # SQLAlchemy table definitions
├── schemas.py                  # Pydantic models for API
├── requirements.txt
├── .env.example                (optional)
├── llm/
│   ├── models.py               # Model loading & inference
│   └── prompts.py              # Legacy prompt templates
├── routes/
│   └── email_routes.py         # API endpoints
└── services/
    └── email_service.py        # Business logic & DB operations


Known Limitations & Next Steps
Current Limitations

Synchronous model inference (may block under high load)
No authentication / API key protection
No automatic input truncation / length validation
Legacy action classification (Flan-T5) has limited reliability
No caching of results
Minimal error recovery when models fail to load or run
Only PostgreSQL is supported (but easy to extend)

Suggested Improvements / Roadmap

Add async endpoints + batch inference support
Replace Flan-T5 action classifier with a fine-tuned small model
Add JWT authentication or API key middleware
Implement rate limiting (e.g. slowapi)
Add input size validation + smart truncation
Write unit/integration tests for model outputs & API
Add inference time & model performance logging / metrics
Export models to ONNX or TorchScript for faster startup/inference
Add language detection and multi-language handling
Build a simple web UI (Streamlit / Gradio / React) for demo & testing