import os
import re
import time
import random
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import quote

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from duckduckgo_search import DDGS
from google import genai
from google.genai import types
from groq import Groq
import edge_tts


# ============================================================
# APP SETUP & PATHS
# ============================================================

app = FastAPI(title="GIBBON // MOKUTTAN LABS")

MEDIA_DIR = "saved_media"
os.makedirs(MEDIA_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

IST = ZoneInfo("Asia/Kolkata")

def get_ist_now() -> datetime:
    return datetime.now(IST)

def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# DATABASE (SUPABASE POSTGRESQL)
# ============================================================

def get_db():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is missing!")
    return psycopg2.connect(db_url, cursor_factory=RealDictCursor)

def init_db():
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                media_url TEXT,
                timestamp TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Init error: {e}")

init_db()

def purge_old_messages():
    try:
        conn = get_db()
        c = conn.cursor()
        cutoff = (get_utc_now() - timedelta(days=15)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("DELETE FROM chat_messages WHERE created_at < %s", (cutoff,))
        deleted = c.rowcount
        conn.commit()
        conn.close()
        if deleted:
            print(f"[DB] Purged {deleted} expired records (>15 days).")
    except Exception as e:
        print(f"[DB] Purge error: {e}")

def save_message(session_id: str, user_id: str, role: str, content: str, media_url: str = None):
    try:
        conn = get_db()
        c = conn.cursor()
        ts = get_ist_now().strftime("%I:%M %p")
        c.execute("""
            INSERT INTO chat_messages (session_id, user_id, role, content, media_url, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (session_id.strip(), user_id.strip(), role, content, media_url, ts))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Save error: {e}")

def get_session_history(session_id: str, user_id: str, limit: int = 8):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            SELECT role, content FROM chat_messages
            WHERE session_id = %s AND user_id = %s
            ORDER BY id DESC LIMIT %s
        """, (session_id.strip(), user_id.strip(), limit))
        rows = c.fetchall()
        conn.close()
        rows.reverse()
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    except Exception as e:
        print(f"[DB] History fetch error: {e}")
        return []


# ============================================================
# SEARCH & MEDIA SERVICES
# ============================================================

def search_live_web(query: str) -> str:
    clean_q = (query or "").strip()
    if not clean_q or len(clean_q) < 2:
        return ""

    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_key:
        print("[SEARCH] FIRECRAWL_API_KEY not configured.")
        return ""

    url = "https://api.firecrawl.dev/v2/search"
    headers = {
        "Authorization": f"Bearer {firecrawl_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": clean_q,
        "limit": 4,
        "scrapeOptions": {
            "formats": ["markdown"]
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=12)
        if response.status_code == 200:
            data = response.json()
            results = data.get("data", [])
            blocks = []
            for item in results:
                title = item.get("title", "No Title")
                content = item.get("markdown") or item.get("description", "")
                clean_content = content.replace("\n", " ").strip()[:1400]
                if clean_content:
                    blocks.append(f"- {title}: {clean_content}")
            return "\n\n".join(blocks)
        else:
            print(f"[SEARCH] Firecrawl error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[SEARCH] Firecrawl request exception: {e}")

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
        print(f"[IMAGE] Search error: {e}")

    encoded = quote(clean_q)
    return f"https://image.pollinations.ai/prompt/{encoded}?width=800&height=800&nologo=true"


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

def get_dynamic_system_instruction(user_name: str, live_context: str = "") -> str:
    now_ist = get_ist_now()
    now_str = now_ist.strftime("%A, %d %B %Y at %I:%M %p IST")
    call_name = user_name.strip() if user_name else "there"

    instruction = f"""
You are Gibbon, a helpful, highly accurate AI companion engineered by MOKUTTAN LABS.
The user's name is {call_name}. Address them naturally. Never call them "Chief" unless their name is explicitly Chief.

Current real-world date and time: {now_str} (Indian Standard Time).

CORE GUIDELINES:
1. FACTUAL ACCURACY: Evaluate all current events, real-time facts, and dates strictly against the current timestamp and provided LIVE WEB CONTEXT. Do not invent or guess terms of office, elections, or dates.
2. NATURAL CONVERSATION: Be concise, clear, and supportive. Answer directly without reciting operational rules or meta-commentary.
3. FORMATTING: Use clean bullet points (*) for lists and highlights. Avoid markdown tables unless structured comparison is requested.
4. CONTINUITY: You remember past messages in this thread. When asked to elaborate or follow up, expand seamlessly on the preceding discussion.
"""
    if live_context:
        instruction += f"\n--- LIVE WEB CONTEXT (REAL-TIME DATA) ---\n{live_context}\n----------------------------------------"

    return instruction


# ============================================================
# AI INFERENCE ENGINES
# ============================================================

def get_gemini_keys():
    keys = []
    for k in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]:
        v = os.getenv(k)
        if v:
            keys.append(v.strip())
    return keys

gemini_key_index = 0

def query_gemini(prompt: str, history: list, user_name: str, live_context: str = "") -> str:
    global gemini_key_index
    keys = get_gemini_keys()
    if not keys:
        print("[GEMINI] No API keys configured.")
        return ""

    collapsed_contents = []
    for turn in history[-6:]:
        role = "user" if turn["role"] == "user" else "model"
        txt = (turn.get("content") or "").strip()
        if not txt:
            continue
        if collapsed_contents and collapsed_contents[-1].role == role:
            collapsed_contents[-1].parts[0].text += f"\n\n{txt}"
        else:
            collapsed_contents.append(types.Content(role=role, parts=[types.Part.from_text(text=txt)]))

    if collapsed_contents and collapsed_contents[-1].role == "user":
        collapsed_contents[-1].parts[0].text += f"\n\n{prompt}"
    else:
        collapsed_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    system_instruction = get_dynamic_system_instruction(user_name, live_context)
    
    # Updated to the required 2026 models based on your logs
    candidate_models = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]

    for _ in range(len(keys)):
        active_key = keys[gemini_key_index]
        try:
            client = genai.Client(api_key=active_key)
            for model_name in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=collapsed_contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.25
                        )
                    )
                    if response and response.text and response.text.strip():
                        return response.text.strip()
                except Exception as model_err:
                    print(f"[GEMINI] Model {model_name} failed: {model_err}")
                    continue
        except Exception as client_err:
            print(f"[GEMINI] Key {gemini_key_index + 1} failed: {client_err}")
            gemini_key_index = (gemini_key_index + 1) % len(keys)

    return ""

def query_groq(prompt: str, history: list, user_name: str, live_context: str = "") -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[GROQ] GROQ_API_KEY is not set.")
        return ""

    try:
        client = Groq(api_key=api_key)
        system_instruction = get_dynamic_system_instruction(user_name, live_context)
        messages = [{"role": "system", "content": system_instruction}]

        for turn in history[-6:]:
            if turn.get("role") in ["user", "assistant"]:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": prompt})

        candidate_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        for m in candidate_models:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=0.25,
                    max_tokens=1500
                )
                if resp.choices[0].message.content:
                    return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[GROQ] Model {m} error: {e}")
                continue
    except Exception as e:
        print(f"[GROQ] Client error: {e}")

    return ""

def ask_ai_brain(prompt: str, session_id: str, user_id: str, user_name: str, live_context: str = "", media_url: str = None) -> str:
    history = get_session_history(session_id, user_id, limit=8)
    save_message(session_id, user_id, "user", prompt)

    # 1. Primary engine: Gemini
    reply = query_gemini(prompt, history, user_name, live_context)

    # 2. Fallback engine: Groq
    if not reply:
        reply = query_groq(prompt, history, user_name, live_context)

    # 3. Connection safety catch
    if not reply or not reply.strip():
        reply = f"I apologize {user_name}, I ran into a network interruption. Please try sending that again."

    save_message(session_id, user_id, "assistant", reply, media_url)
    return reply


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def serve_index():
    purge_old_messages()
    return FileResponse("static/index.html")

@app.get("/api/threads")
def get_user_threads(user_id: str):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            SELECT session_id, MAX(id) AS latest_id
            FROM chat_messages
            WHERE user_id = %s
            GROUP BY session_id
            ORDER BY latest_id DESC
            LIMIT 50
        """, (user_id.strip(),))
        thread_rows = c.fetchall()

        output = []
        for row in thread_rows:
            sid = row["session_id"]
            c.execute("""
                SELECT content, timestamp FROM chat_messages
                WHERE user_id = %s AND session_id = %s AND role = 'user'
                ORDER BY id ASC LIMIT 1
            """, (user_id.strip(), sid))
            first = c.fetchone()
            output.append({
                "session_id": sid,
                "title": first["content"][:42] if first and first.get("content") else "New conversation",
                "time": first["timestamp"] if first and first.get("timestamp") else ""
            })
        conn.close()
        return {"threads": output}
    except Exception as e:
        print(f"[THREADS] Error: {e}")
        return {"threads": []}

@app.get("/api/thread_messages")
def get_thread_messages(session_id: str, user_id: str = ""):
    try:
        conn = get_db()
        c = conn.cursor()
        if user_id:
            c.execute("""
                SELECT role, content, timestamp, media_url
                FROM chat_messages
                WHERE session_id = %s AND user_id = %s
                ORDER BY id ASC
            """, (session_id.strip(), user_id.strip()))
        else:
            c.execute("""
                SELECT role, content, timestamp, media_url
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY id ASC
            """, (session_id.strip(),))
        rows = c.fetchall()
        conn.close()
        return {
            "messages": [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "media_url": r["media_url"]
                }
                for r in rows
            ]
        }
    except Exception as e:
        print(f"[THREAD_MESSAGES] Error: {e}")
        return {"messages": []}

@app.post("/api/clear_threads")
async def clear_user_threads(request: Request):
    try:
        data = await request.json()
        user_id = data.get("user_id", "").strip()
        if not user_id:
            return {"status": "error", "message": "user_id required"}
        conn = get_db()
        c = conn.cursor()
        c.execute("DELETE FROM chat_messages WHERE user_id = %s", (user_id,))
        deleted = c.rowcount
        conn.commit()
        conn.close()
        return {"status": "cleared", "deleted": deleted}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/chat")
async def process_command(request: Request):
    try:
        data = await request.json()
        raw_message = str(data.get("message", "")).strip()
        session_id = str(data.get("session_id", "")).strip() or f"session_{int(time.time() * 1000)}"
        user_id = str(data.get("user_id", "")).strip() or "default_user"
        user_name = str(data.get("user_name", "")).strip() or "Chief"

        if not raw_message:
            return {"reply": "Tell me what's on your mind.", "media_url": None, "session_id": session_id}

        purge_old_messages()
        lower = raw_message.lower()

        # Image generation detection
        wants_image = any(w in lower for w in ["show me", "picture of", "photo of", "poster of", "pic of", "image of"])
        media_url = fetch_web_image(raw_message) if wants_image else None

        # Clean search query
        clean_q = re.sub(r"\b(gibbon|given|hey|hi|hello|show me|picture of|poster of)\b", "", raw_message, flags=re.IGNORECASE).strip()
        search_query = clean_q if len(clean_q) >= 2 else raw_message

        # Append current calendar anchor for time-sensitive topics
        if any(k in lower for k in ["today", "now", "current", "latest", "news", "score", "cm", "pm", "date"]):
            current_date_str = get_ist_now().strftime("%d %B %Y")
            search_query = f"{search_query} {current_date_str}"

        # Retrieve live context through Firecrawl for informational queries
        conversational_greetings = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye", "good night", "good morning", "yo", "sup"}
        live_context = ""
        if lower not in conversational_greetings and len(search_query) > 2:
            live_context = search_live_web(search_query)

        # Generate response
        clean_prompt = re.sub(r"\b(gibbon|given)\b", "", raw_message, flags=re.IGNORECASE).strip() or raw_message
        ai_answer = ask_ai_brain(clean_prompt, session_id, user_id, user_name, live_context, media_url)

        return {"reply": ai_answer, "media_url": media_url, "session_id": session_id}

    except Exception as e:
        print(f"[CHAT] Handler exception: {e}")
        return {"reply": "An error occurred while processing your request. Please try again.", "media_url": None}

@app.get("/api/tts")
async def text_to_speech(text: str):
    clean_text = re.sub(r'```.*?```', '', text or "", flags=re.DOTALL)
    clean_text = re.sub(r'[*#_`|~>—–-]', ' ', clean_text)
    clean_text = re.sub(r'https?://\S+', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()[:4000]

    if not clean_text:
        return Response(content=b"", media_type="audio/mpeg")

    candidate_voices = ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural"]
    for v in candidate_voices:
        try:
            audio_data = bytearray()
            communicate = edge_tts.Communicate(clean_text, voice=v, rate="+1%", pitch="+2Hz")
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
            if audio_data:
                return Response(content=bytes(audio_data), media_type="audio/mpeg")
        except Exception:
            continue

    return Response(content=b"", media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn
    # Bound properly to the dynamic port required by Render
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
