from fastapi import FastAPI
from database import Base, engine
from routes.email_routes import router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Intelligent Email Assistant")

app.include_router(router)
