from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
from schemas import EmailRequest, EmailResponse, SpamResponse, SummaryResponse
from services.email_service import (
    summarize_email,
    classify_spam,
    save_prediction,
    process_email_legacy
)

router = APIRouter(prefix="/email", tags=["email"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── New Endpoints ────────────────────────────────────────

@router.post("/summarize", response_model=SummaryResponse)
def summarize(request: EmailRequest, db: Session = Depends(get_db)):
    if request.task_type != "summary":
        raise HTTPException(400, "This endpoint only accepts task_type = 'summary'")

    result = summarize_email(request.email_text)
    
    save_prediction(
        db=db,
        email_text=request.email_text,
        task_type="summary",
        result=result,
        model_version="IrisWiris/email-summarizer"
    )
    
    return SummaryResponse(
        task_type="summary",
        summary=result
    )


@router.post("/classify-spam", response_model=SpamResponse)
def classify_spam_endpoint(request: EmailRequest, db: Session = Depends(get_db)):
    if request.task_type != "spam":
        raise HTTPException(400, "This endpoint only accepts task_type = 'spam'")

    result = classify_spam(request.email_text)
    save_prediction(
        db=db,
        email_text=request.email_text,
        task_type="spam",
        result=result,
        model_version="AntiSpamInstitute/spam-detector-bert-MoE-v2.2"
    )
    
    return SpamResponse(
        task_type="spam",
        is_spam=result
    )


# ─── Original endpoint (Flan-T5) ──────────────────────────

@router.post("/action", response_model=EmailResponse)
def process_email_api(request: EmailRequest, db: Session = Depends(get_db)):
    try:
        result = process_email_legacy(request.email_text, request.task_type, db)
        
        # Save with Flan-T5 version
        save_prediction(
            db=db,
            email_text=request.email_text,
            task_type=request.task_type,
            result=result,
            model_version="google/flan-t5-base"
        )
        
        return EmailResponse(
            task_type=request.task_type,
            result=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))