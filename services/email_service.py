from sqlalchemy.orm import Session
from llm.model import (
    summarize_email_direct,
    spam_classifier,
    run_flan,
    MODEL_VERSIONS
)
from models import EmailPrediction


def normalize_spam_label(label: str) -> str:
    label = label.lower()
    if "spam" in label:
        return "SPAM"
    return "NOT_SPAM"


def summarize_email(email_text: str) -> str:
    try:
        return summarize_email_direct(email_text)
    except Exception as e:
        return f"[Error: {str(e)}]"


def classify_spam(email_text: str) -> str:
    try:
        result = spam_classifier(email_text)
        label = result[0]['label']
        return normalize_spam_label(label)
    except Exception as e:
        return f"[Spam classification error: {str(e)}]"


def process_email_legacy(email_text: str, task_type: str, db: Session):
    """Original flan-t5 based processing (kept for comparison)"""
    from llm.prompts import spam_prompt, summary_prompt, action_prompt

    if task_type == "spam":
        prompt = spam_prompt(email_text)
    elif task_type == "summary":
        prompt = summary_prompt(email_text)
    elif task_type == "action":
        prompt = action_prompt(email_text)
    else:
        raise ValueError("Invalid task type")

    raw_output = run_flan(prompt)
    
    # simple normalization
    text = raw_output.strip().upper()
    if task_type == "spam":
        return "SPAM" if "SPAM" in text else "NOT_SPAM"
    if task_type == "action":
        for act in ["REPLY", "ARCHIVE", "FLAG"]:
            if act in text:
                return act
        return "ARCHIVE"
    
    return text


def save_prediction(db: Session, email_text: str, task_type: str, result: str, model_version: str):
    record = EmailPrediction(
        email_text=email_text,
        task_type=task_type,
        response=result,
        model_version=model_version
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return result

# from llm.model import run_llm, MODEL_VERSION
# from llm.prompts import spam_prompt, summary_prompt, action_prompt
# from models import EmailPrediction
# from sqlalchemy.orm import Session


# def normalize_output(task_type: str, text: str) -> str:
#     text = text.strip()

#     if task_type == "spam":
#         text = text.upper()
#         return "SPAM" if "SPAM" in text else "NOT_SPAM"

#     if task_type == "action":
#         text = text.upper()
#         for action in ["REPLY", "ARCHIVE", "FLAG"]:
#             if action in text:
#                 return action
#         return "ARCHIVE"
        
#     return text



# def process_email(email_text: str, task_type: str, db: Session):
#     if task_type == "spam":
#         prompt = spam_prompt(email_text)
#     elif task_type == "summary":
#         prompt = summary_prompt(email_text)
#     elif task_type == "action":
#         prompt = action_prompt(email_text)
#     else:
#         raise ValueError("Invalid task type")

#     raw_output = run_llm(prompt)
#     final_output = normalize_output(task_type, raw_output)

#     record = EmailPrediction(
#         email_text=email_text,
#         task_type=task_type,
#         response=final_output,
#         model_version=MODEL_VERSION
#     )

#     db.add(record)
#     db.commit()
#     db.refresh(record)

#     return final_output
