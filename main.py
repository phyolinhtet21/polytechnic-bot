from fastapi import FastAPI
from pydantic import BaseModel
from bot import get_bot

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/chat")
def chat(req: ChatRequest):
    bot = get_bot(req.session_id)
    result = bot.ask(req.message)
    return {
        "answer": result["answer"],
        "model":  result["model"]
    }

@app.post("/reset")
def reset(req: ChatRequest):
    bot = get_bot(req.session_id)
    bot.reset()
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "running"}