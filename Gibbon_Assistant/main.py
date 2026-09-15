import os
import re
import time
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
# SUPABASE POSTGRESQL (15-DAY PERSISTENCE)
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
            );
            CREATE INDEX IF NOT EXISTS idx_chat_user_session 
            ON chat_messages(user_id, session_id);
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Init error: {e}")

init_db()

def purge_old_messages():
    """Automatically purge records older than 15 days."""
    try:
        conn = get_db()
        c = conn.cursor()
        cutoff = (get_utc_now() - timedelta(days=15)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("DELETE FROM chat_messages WHERE created_at < %s", (cutoff,))
        deleted = c.rowcount
        conn.commit()
        conn.close()
        if deleted > 0:
            print(f"[DB] Cleaned up {deleted} messages older than 15 days.")
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
# SEARCH & MEDIA SERVICES (OPTIMIZED FOR SPEED)
# ============================================================

def search_live_web(query: str) -> str:
    clean_q = (query or "").strip()
    if not clean_q or len(clean_q) < 2:
        return ""

    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_key:
        return ""

    url = "https://api.firecrawl.dev/v1/search"
    headers = {
        "Authorization": f"Bearer {firecrawl_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": clean_q,
        "limit": 2,
        "scrapeOptions": {
            "formats": ["markdown"]
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=7)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                results = data.get("data", [])
                blocks = []
                for item in results:
                    title = item.get("title", "No Title")
                    content = item.get("markdown") or item.get("description", "")
                    clean_content = content.replace("\n", " ").strip()[:900]
                    if clean_content:
                        blocks.append(f"- {title}: {clean_content}")
                return "\n\n".join(blocks)
    except Exception as e:
        print(f"[SEARCH] Error: {e}")

    return ""

def fetch_web_image(query: str) -> str:
    clean_q = re.sub(r"\b(show|give|display|picture|pic|photo|image|of|me|a|the|poster)\b", "", query, flags=re.IGNORECASE).strip()
    if not clean_q:
        clean_q = query.strip()

    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(clean_q, max_results=2))
            for r in results:
                img_url = r.get("image")
                if img_url and any(img_url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    return img_url
            if results and results[0].get("image"):
                return results[0].get("image")
    except Exception:
        pass

    encoded = quote(clean_q)
    return f"https://image.pollinations.ai/prompt/{encoded}?width=800&height=800&nologo=true"

# ============================================================
# SYSTEM INSTRUCTION & SAFETY GUARDRAILS
# ============================================================

def get_dynamic_system_instruction(user_name: str, live_context: str = "", client_time: str = "") -> str:
    now_str = client_time.strip() if client_time else get_ist_now().strftime("%A, %d %B %Y at %I:%M %p IST")
    call_name = user_name.strip() if user_name else "friend"

    instruction = f"""
You are Gibbon, an intelligent AI companion built by MOKUTTAN LABS.
Current user: {call_name}.
Current verified date and time: {now_str}.

CORE RULES & SAFETY GUARDRAILS:
1. TYPOS & MANGLISH INTERPRETATION: The user may use Manglish (Malayalam in English script like 'enthokkeyundu', 'sugamane', 'evideya'), informal slang, or misspellings. Infer the core intent smoothly without criticizing grammar.
2. ZERO GUESSING ON LIVE FACTS: For live sports scores, tech announcements, government positions, or current news, rely STRICTLY on the 'LIVE WEB CONTEXT' below. If context is empty or missing, state candidly: "I don't have verified real-time data for that right now." Never fabricate players, matches, or election outcomes.
3. MEDICAL & DRUG SAFETY: You are not a doctor. If asked for medicine names, dosages, or treatments, you MUST NOT prescribe or recommend specific drugs. Instruct the user clearly to consult a medical professional or visit a hospital.
4. FINANCIAL & INVESTMENT SAFETY: You are not a certified financial advisor. For stock market or crypto questions, summarize factual market movements from the live context if present, but always add a reminder to consult a registered financial advisor before investing.
5. PERSONAL BOUNDARIES: If a user sends sexually explicit or harassing messages, maintain professional dignity. Politely refuse to participate in explicit scenarios.
6. STYLE: Keep responses direct, well-structured, and helpful. Use clean bullet points (*) when explaining complex lists.
"""
    if live_context:
        instruction += f"\n--- LIVE WEB CONTEXT ---\n{live_context}\n-----------------------"

    return instruction

# ============================================================
# AI ENGINES (GEMINI 3.6/3.5 + GROQ FALLBACK)
# ============================================================

def get_gemini_keys():
    keys = []
    for k in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]:
        v = os.getenv(k)
        if v:
            keys.append(v.strip())
    return keys

gemini_key_index = 0

def query_gemini(prompt: str, history: list, user_name: str, live_context: str = "", client_time: str = "") -> str:
    global gemini_key_index
    keys = get_gemini_keys()
    if not keys:
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

    system_instruction = get_dynamic_system_instruction(user_name, live_context, client_time)
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
                except Exception:
                    continue
        except Exception:
            gemini_key_index = (gemini_key_index + 1) % len(keys)

    return ""

def query_groq(prompt: str, history: list, user_name: str, live_context: str = "", client_time: str = "") -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""

    try:
        client = Groq(api_key=api_key)
        system_instruction = get_dynamic_system_instruction(user_name, live_context, client_time)
        messages = [{"role": "system", "content": system_instruction}]

        for turn in history[-6:]:
            if turn.get("role") in ["user", "assistant"]:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": prompt})

        candidate_models = ["openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b"]
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
            except Exception:
                continue
    except Exception:
        pass

    return ""

def ask_ai_brain(prompt: str, session_id: str, user_id: str, user_name: str, live_context: str = "", media_url: str = None, client_time: str = "") -> str:
    history = get_session_history(session_id, user_id, limit=8)
    save_message(session_id, user_id, "user", prompt)

    reply = query_gemini(prompt, history, user_name, live_context, client_time)
    if not reply:
        reply = query_groq(prompt, history, user_name, live_context, client_time)

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
    except Exception:
        return {"threads": []}

@app.get("/api/thread_messages")
def get_thread_messages(session_id: str, user_id: str = ""):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            SELECT role, content, timestamp, media_url
            FROM chat_messages
            WHERE session_id = %s AND user_id = %s
            ORDER BY id ASC
        """, (session_id.strip(), user_id.strip()))
        rows = c.fetchall()
        conn.close()
        return {
            "messages": [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"], "media_url": r["media_url"]} for r in rows]
        }
    except Exception:
        return {"messages": []}

@app.post("/api/clear_threads")
async def clear_user_threads(request: Request):
    """Permanently purges all chat history for this specific user_id."""
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
        user_id = str(data.get("user_id", "")).strip() or "default_device"
        user_name = str(data.get("user_name", "")).strip() or "friend"
        client_time = str(data.get("client_time", "")).strip()

        if not raw_message:
            return {"reply": "Tell me what's on your mind.", "media_url": None, "session_id": session_id}

        lower = raw_message.lower()

        # Image generation detection
        wants_image = any(w in lower for w in ["show me", "picture of", "photo of", "poster of", "pic of", "image of"])
        media_url = fetch_web_image(raw_message) if wants_image else None

        clean_q = re.sub(r"\b(gibbon|given|hey|hi|hello|show me|picture of|poster of)\b", "", raw_message, flags=re.IGNORECASE).strip()
        search_query = clean_q if len(clean_q) >= 2 else raw_message

        # Date-sensitive keywords
        time_keywords = [
            "today", "yesterday", "tomorrow", "now", "current", "latest", "news", 
            "score", "match", "vs", "won", "result", "update", "market", "stock", 
            "price", "cm", "pm", "date", "recent", "new", "release"
        ]
        if any(k in lower for k in time_keywords):
            date_anchor = client_time if client_time else get_ist_now().strftime("%d %B %Y")
            search_query = f"{search_query} (Searched on: {date_anchor})"

        # Immediate fast-path bypass for common greetings, casual expressions, and Manglish
        fast_bypass = {
            "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye", 
            "good night", "good morning", "yo", "sup", "sugamano", "sugamane", 
            "da", "enthokkeyundu", "entha visesham", "enna und", "evide", 
            "how are you", "whatsup", "whats up"
        }

        live_context = ""
        # Search live web only when substantive factual context is required
        if lower not in fast_bypass and len(search_query) > 3:
            live_context = search_live_web(search_query)

        clean_prompt = re.sub(r"\b(gibbon|given)\b", "", raw_message, flags=re.IGNORECASE).strip() or raw_message
        ai_answer = ask_ai_brain(clean_prompt, session_id, user_id, user_name, live_context, media_url, client_time)

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
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
