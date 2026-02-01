from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from database import Base

class EmailPrediction(Base):
    __tablename__ = "email_predictions"

    id = Column(Integer, primary_key=True, index=True)
    email_text = Column(Text, nullable=False)
    task_type = Column(String(50))
    response = Column(Text)
    model_version = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
