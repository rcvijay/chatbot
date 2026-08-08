import os
import time
import httpx
import psycopg2
from pgvector.psycopg2 import register_vector
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Corporate LAN RAG Backend")

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_URL = os.getenv("DATABASE_URL", "postgresql://chat_user:chat_password@db:5432/chatbot_db")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

# Helper: Connect to Database with Retry Logic
def get_db_connection():
    for _ in range(10):
        try:
            conn = psycopg2.connect(DB_URL)
            return conn
        except psycopg2.OperationalError:
            time.sleep(2)
    raise Exception("Could not connect to PostgreSQL database.")

# Startup: Initialize pgvector extension & database tables
@app.on_event("startup")
def startup_db_init():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

class ChatRequest(BaseModel):
    message: str
    model: str = "llama3"

@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    conn = get_db_connection()
    cur = conn.cursor()
    
    # 1. Save user query to DB history
    cur.execute("INSERT INTO chat_history (role, content) VALUES (%s, %s);", ("user", payload.message))
    conn.commit()

    # 2. Query Ollama model
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": payload.model,
                    "prompt": payload.message,
                    "stream": False
                }
            )
            res_data = response.json()
            ai_text = res_data.get("response", "No response generated.")
        except Exception as e:
            cur.close()
            conn.close()
            raise HTTPException(status_code=500, detail=f"Ollama backend error: {str(e)}")

    # 3. Save assistant response to DB history
    cur.execute("INSERT INTO chat_history (role, content) VALUES (%s, %s);", ("assistant", ai_text))
    conn.commit()
    cur.close()
    conn.close()

    return {"response": ai_text}

@app.get("/api/history")
def get_history():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT role, content, created_at FROM chat_history ORDER BY id ASC LIMIT 50;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"role": r[0], "content": r[1], "timestamp": str(r[2])} for r in rows]