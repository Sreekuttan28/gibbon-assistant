import os
import re
import time
import random
import sqlite3
import requests
from urllib.parse import quote
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from geopy.geocoders import Nominatim
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from groq import Groq
import edge_tts

app = FastAPI(title="GIBBON // MOKUTTAN LABS")

MEDIA_DIR = "saved_media"
DB_FILE = "assistant.db"
os.makedirs(MEDIA_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

geolocator = Nominatim(user_agent="gibbon_hud_agent_v43")
IST = ZoneInfo("Asia/Kolkata")

def get_ist_now() -> datetime:
    return datetime.now(IST)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    media_url TEXT,
                    timestamp TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

init_db()

def purge_old_messages():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        fifteen_days_ago = (datetime.utcnow() - timedelta(days=15)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("DELETE FROM chat_messages WHERE created_at < ?", (fifteen_days_ago,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Purge error: {e}")

def save_message(session_id: str, user_id: str, role: str, content: str, media_url: str = None):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        ts = get_ist_now().strftime("%I:%M %p")
        c.execute("INSERT INTO chat_messages (session_id, user_id, role, content, media_url, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                  (session_id.strip(), user_id.strip(), role, content, media_url, ts))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Save message error: {e}")

def get_session_history(session_id: str, user_id: str, limit: int = 10):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = ? AND user_id = ? ORDER BY id DESC LIMIT ?", 
            (session_id.strip(), user_id.strip(), limit)
        )
        rows = c.fetchall()
        conn.close()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception:
        return []

def search_live_web(query: str) -> str:
    clean_q = (query or "").strip()
    if not clean_q or len(clean_q) < 2:
        return ""

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(clean_q, max_results=5))
            if results:
                return "\n".join([f"- {r.get('title', '')}: {r.get('body', '')}" for r in results])
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
    return ""

def fetch_web_image(query: str) -> str:
    clean_q = re.sub(r"\b(show|give|display|picture|pic|photo|image|of|me|a|the|poster)\b", "", query, flags=re.IGNORECASE).strip()
    if not clean_q:
        clean_q = query.strip()

    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(clean_q, max_results=3))
            for r in results:
                img_url = r.get("image")
                if img_url and any(img_url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    return img_url
            if results and results[0].get("image"):
                return results[0].get("image")
    except Exception as e:
        print(f"Image search error: {e}")

    encoded = quote(clean_q)
    return f"https://image.pollinations.ai/prompt/{encoded}?width=800&height=800&nologo=true"

def get_dynamic_system_instruction(user_name: str, live_context: str = "") -> str:
    now_ist = get_ist_now()
    now_str = now_ist.strftime("%A, %B %d, %Y at %I:%M %p IST")
    call_name = user_name.strip() if user_name and user_name.strip() else "Chief"
    
    instruction = (
        f"You are Gibbon, a knowledgeable, accurate AI companion engineered by Mokuttan Labs. "
        f"The user's name is {call_name}. Address them naturally as {call_name}. "
        f"Current real-world date and time: {now_str} (Indian Standard Time). "
        "CRITICAL RULES: "
        "1. REAL-WORLD ACCURACY & DATES: Today is Monday, September 14, 2026. Ganesh Chaturthi falls on this exact date (September 14, 2026). Always verify dates against live context and current calendar data. "
        "2. FORMATTING: Use clean Bullet Points (*) or direct paragraphs. Do NOT force tables for general information or descriptions. Use tables only when specifically asked to compare items or data. "
        "3. CONTINUITY: You are inside an isolated chat thread. Maintain focus on the questions asked in THIS thread only without cross-contamination. "
        "4. SONG LYRICS & SUMMARIES: If asked for song lyrics or summaries, provide a helpful summary, credit artists, and quote chorus lines directly without refusal."
    )
    if live_context:
        instruction += f"\n\n--- LIVE WEB CONTEXT ---\n{live_context}\n-------------------------"
    return instruction

def get_gemini_keys():
    keys_str = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    return [k.strip() for k in keys_str.split(",") if k.strip()]

gemini_key_index = 0

def query_gemini(prompt: str, key: str, history: list, user_name: str, live_context: str = "") -> str:
    client = genai.Client(api_key=key)
    candidate_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    last_err = None

    contents = []
    for turn in history[-6:]:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    for m in candidate_models:
        try:
            response = client.models.generate_content(
                model=m,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=get_dynamic_system_instruction(user_name, live_context),
                    tools=[{"google_search": {}}],
                    temperature=0.25
                )
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_err = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                raise e
            continue

    raise last_err or RuntimeError("Gemini engines failed.")

def query_groq(prompt: str, history: list, user_name: str, live_context: str = "") -> str:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not configured.")

    client = Groq(api_key=groq_key)
    messages = [{"role": "system", "content": get_dynamic_system_instruction(user_name, live_context)}]
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": prompt})

    candidate_groq_models = ["openai/gpt-oss-20b", "llama-3.1-8b-instant"]
    for model_name in candidate_groq_models:
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.2,
                max_tokens=950
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception:
            continue
    raise RuntimeError("All Groq models failed.")

def ask_ai_brain(prompt: str, session_id: str, user_id: str, user_name: str, live_context: str = "", media_url: str = None) -> str:
    global gemini_key_index
    gemini_keys = get_gemini_keys()
    history = get_session_history(session_id, user_id)

    save_message(session_id, user_id, "user", prompt)

    if gemini_keys:
        for _ in range(len(gemini_keys)):
            current_key = gemini_keys[gemini_key_index]
            try:
                reply = query_gemini(prompt, current_key, history, user_name, live_context)
                save_message(session_id, user_id, "assistant", reply, media_url)
                return reply
            except Exception as e:
                print(f"[Gemini Error on Key {gemini_key_index + 1}]: {e}")
                gemini_key_index = (gemini_key_index + 1) % len(gemini_keys)

    if os.getenv("GROQ_API_KEY"):
        try:
            if not live_context:
                clean_q = re.sub(r"\b(gibbon|given|hey|hi)\b", "", prompt, flags=re.IGNORECASE).strip()
                live_context = search_live_web(clean_q or prompt)

            reply = query_groq(prompt, history, user_name, live_context)
            save_message(session_id, user_id, "assistant", reply, media_url)
            return reply
        except Exception as e:
            print(f"[Groq Failover Error]: {e}")

    fallback = f"I apologize {user_name}, connection is busy right now. Please try again."
    save_message(session_id, user_id, "assistant", fallback, media_url)
    return fallback

@app.get("/")
def serve_index():
    purge_old_messages()
    return FileResponse("static/index.html")

@app.get("/api/threads")
def get_user_threads(user_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT session_id, content, timestamp 
        FROM chat_messages 
        WHERE user_id = ? AND role = 'user' 
        GROUP BY session_id 
        ORDER BY id DESC LIMIT 50
    """, (user_id.strip(),))
    rows = c.fetchall()
    conn.close()
    return {"threads": [{"session_id": r[0], "title": r[1][:42], "time": r[2]} for r in rows]}

@app.get("/api/thread_messages")
def get_thread_messages(session_id: str):
    clean_id = session_id.strip()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content, timestamp, media_url FROM chat_messages WHERE session_id = ? ORDER BY id ASC", (clean_id,))
    rows = c.fetchall()
    conn.close()
    return {"messages": [{"role": r[0], "content": r[1], "timestamp": r[2], "media_url": r[3]} for r in rows]}

@app.post("/api/clear_threads")
async def clear_user_threads(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "").strip()
    if user_id:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
    return {"status": "cleared"}

@app.post("/api/chat")
async def process_command(request: Request):
    data = await request.json()
    raw_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default_session").strip()
    user_id = data.get("user_id", "default_user").strip()
    user_name = data.get("user_name", "Chief").strip() or "Chief"
    lower = raw_message.lower()

    wants_image = any(w in lower for w in ["show me", "picture of", "photo of", "poster of", "pic of", "image of"])
    media_url = None
    if wants_image:
        media_url = fetch_web_image(raw_message)

    history = get_session_history(session_id, user_id, limit=4)
    search_query = re.sub(r"\b(gibbon|given|hey|hi|hello|show me|picture of|poster of)\b", "", raw_message, flags=re.IGNORECASE).strip()
    final_search_query = search_query if len(search_query) >= 2 else raw_message.strip()

    if len(final_search_query.split()) <= 3 and history:
        last_user_turn = next((t["content"] for t in reversed(history) if t["role"] == "user"), "")
        if last_user_turn and last_user_turn.lower() != raw_message.lower():
            final_search_query = f"{last_user_turn} {final_search_query}"

    if any(k in lower for k in ["today", "holiday", "festival", "speciality", "specialty", "date"]):
        final_search_query = f"{raw_message} September 2026 India Ganesh Chaturthi"

    if any(k in lower for k in ["parashini", "parassini", "parassinikkadavu"]):
        final_search_query += " Kannur Kerala Muthappan temple"

    needs_search = any(w in lower for w in [
        "today", "holiday", "festival", "speciality", "specialty", "date",
        "parashini", "parassini", "kannur", "kasaragod", "athiradi", 
        "places", "tourist", "visit", "famous", "temple", "latest", "movie", "song"
    ])
    
    live_context = ""
    if needs_search and len(final_search_query) > 2:
        live_context = search_live_web(final_search_query)

    clean_prompt = re.sub(r"\b(gibbon|given)\b", "", raw_message, flags=re.IGNORECASE).strip()
    ai_answer = ask_ai_brain(clean_prompt or raw_message, session_id, user_id, user_name, live_context, media_url)
    return {"reply": ai_answer, "media_url": media_url}

@app.get("/api/tts")
async def text_to_speech(text: str):
    clean_text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    clean_text = re.sub(r'[*#_`|~>—–-]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()[:4000]

    candidate_voices = ["en-US-AvaNeural", "en-US-AriaNeural", "en-US-JennyNeural"]
    audio_data = bytearray()

    for v in candidate_voices:
        try:
            communicate = edge_tts.Communicate(clean_text, voice=v, rate="+1%", pitch="+2Hz")
            audio_data.clear()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
            if len(audio_data) > 0:
                break
        except Exception:
            continue
    return Response(content=bytes(audio_data), media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
