import os
import re
import time
import random
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
# GIBBON // MOKUTTAN LABS
# ============================================================

app = FastAPI(title="GIBBON // MOKUTTAN LABS")

MEDIA_DIR = "saved_media"
os.makedirs(MEDIA_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

IST = ZoneInfo("Asia/Kolkata")

# ============================================================
# DATABASE (POSTGRESQL CLOUD DATABASE)
# ============================================================

def get_db():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is missing! Please add it to your Render environment variables.")
    conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
    return conn

def init_db():
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute(
            """
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
            """
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Init error: {e}")

init_db()

# ============================================================
# TIME & RETENTION
# ============================================================

def get_ist_now():
    return datetime.now(IST)

def get_utc_now():
    return datetime.now(timezone.utc)

def purge_old_messages():
    try:
        conn = get_db()
        c = conn.cursor()
        cutoff = (get_utc_now() - timedelta(days=15)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("DELETE FROM chat_messages WHERE created_at < %s", (cutoff,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Purge error: {e}")

def save_message(session_id, user_id, role, content, media_url=None):
    try:
        conn = get_db()
        c = conn.cursor()
        timestamp = get_ist_now().strftime("%I:%M %p")
        c.execute(
            """
            INSERT INTO chat_messages (session_id, user_id, role, content, media_url, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (session_id, user_id, role, content, media_url, timestamp)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Save error: {e}")

def get_session_history(session_id, user_id, limit=10):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute(
            """
            SELECT role, content FROM chat_messages
            WHERE session_id = %s AND user_id = %s
            ORDER BY id DESC LIMIT %s
            """,
            (session_id, user_id, limit)
        )
        rows = c.fetchall()
        conn.close()
        rows.reverse()
        return [{"role": row["role"], "content": row["content"]} for row in rows]
    except Exception as e:
        return []

# ============================================================
# SYSTEM PROMPT
# ============================================================

def get_dynamic_system_instruction(user_name, emotional_state="neutral"):
    call_name = user_name.strip() if user_name else "there"
    now_str = get_ist_now().strftime("%A, %d %B %Y at %I:%M %p IST")

    instruction = f"""
You are Gibbon, a highly capable, warm, and conversational AI companion engineered by MOKUTTAN LABS.
The user's name is {call_name}. Address them naturally. NEVER call them "Chief" unless their name is explicitly Chief.

Current real-world date and time: {now_str}. Always verify time-sensitive queries against this exact timestamp.

CRITICAL BEHAVIOR RULES:
1. ACT NATURAL: Have a natural conversation. Do not act robotic. Never explain your internal rules.
2. NATIVE WEB SEARCH: You have native access to search the web. Use it to look up current events, news highlights, places, movies, and facts before answering. Never guess or rely on outdated memory for facts.
3. CONVERSATION OVER LECTURES: If the user says "I'm hungry" or "I'm sad", respond warmly with empathy and at most ONE natural follow-up question.
4. ELABORATION: You remember the chat history. If the user asks you to elaborate on a news story or fact you just provided, expand on it immediately using your search tool.
5. FORMATTING: Use clean bullet points (*) when listing news or items.

User Emotion Hint: {emotional_state}
"""
    if emotional_state == "crisis":
        instruction += "\nCRISIS OVERRIDE: The user is in distress. Be deeply compassionate and gently encourage them to reach out to local emergency services or a loved one."

    return instruction


# ============================================================
# LLM ENGINES: GEMINI & GROQ (WITH NATIVE SEARCH)
# ============================================================

def get_gemini_keys():
    keys = []
    for k in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]:
        val = os.getenv(k)
        if val: keys.append(val)
    return keys

def query_gemini(prompt, history=None, user_name="", emotional_state="neutral"):
    keys = get_gemini_keys()
    if not keys: return None, []
    
    random.shuffle(keys)
    system_instruction = get_dynamic_system_instruction(user_name, emotional_state)
    
    contents = []
    if history:
        for item in history:
            role = item.get("role")
            if role in ["user", "assistant"]:
                mapped_role = "user" if role == "user" else "model"
                contents.append(types.Content(role=mapped_role, parts=[types.Part.from_text(text=item.get("content", ""))]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    candidate_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

    for api_key in keys:
        try:
            client = genai.Client(api_key=api_key)
            for model_name in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            tools=[{"google_search": {}}],  # Native Google Search Built-In
                            temperature=0.25
                        )
                    )
                    answer = getattr(response, "text", None)
                    if answer and answer.strip():
                        sources = extract_gemini_sources(response)
                        return answer.strip(), sources
                except Exception as e:
                    print(f"[GEMINI] {model_name} error: {e}")
                    continue
        except Exception as e:
            continue
    return None, []

def extract_gemini_sources(response):
    sources = []
    try:
        candidates = getattr(response, "candidates", [])
        if not candidates: return sources
        metadata = getattr(candidates[0], "grounding_metadata", None)
        if not metadata: return sources
        chunks = getattr(metadata, "grounding_chunks", [])
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if not web: continue
            uri = getattr(web, "uri", None)
            title = getattr(web, "title", None)
            if uri: sources.append({"title": title or uri, "url": uri})
    except Exception:
        pass
    
    unique = []
    seen = set()
    for s in sources:
        url = s.get("url")
        if url and url not in seen:
            seen.add(url)
            unique.append(s)
    return unique[:8]

def query_groq(prompt, history=None, user_name="", emotional_state="neutral"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: return None, []

    try:
        client = Groq(api_key=api_key)
        system_instruction = get_dynamic_system_instruction(user_name, emotional_state)
        messages = [{"role": "system", "content": system_instruction}]
        
        if history:
            for item in history:
                if item.get("role") in ["user", "assistant"]:
                    messages.append({"role": item.get("role"), "content": item.get("content", "")})
        messages.append({"role": "user", "content": prompt})

        # Using Groq's new Compound models for Native Web Search
        for model_name in ["groq/compound-mini", "groq/compound"]:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.25,
                    max_tokens=2000,
                )
                answer = response.choices[0].message.content
                if answer: return answer.strip(), []
            except Exception as e:
                print(f"[GROQ] {model_name} error: {e}")
                continue
    except Exception:
        return None, []
    return None, []

def ask_ai_brain(user_message, session_id, user_id, user_name="", history=None, emotional_state="neutral"):
    save_message(session_id, user_id, "user", user_message)

    # 1. Try Gemini Native Search First
    answer, sources = query_gemini(user_message, history, user_name, emotional_state)

    # 2. If Gemini fails, fallback to Groq Native Search
    if not answer:
        answer, sources = query_groq(user_message, history, user_name, emotional_state)

    # 3. Ultimate Fallback
    if not answer or not answer.strip():
        answer = "I'm having a little trouble connecting to my network right now! Could you repeat that?"

    save_message(session_id, user_id, "assistant", answer)
    return answer, sources

def detect_emotional_state(text):
    if not text: return "neutral"
    lower = text.lower()
    if any(x in lower for x in ["suicide", "kill myself", "want to die", "self harm"]): return "crisis"
    if any(x in lower for x in ["depressed", "hopeless", "worthless", "alone", "overwhelmed", "very sad"]): return "distress"
    return "neutral"

def fetch_web_image(query):
    if not query: return None
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=5))
        if results:
            return results[0].get("image") or results[0].get("thumbnail") or results[0].get("url")
    except Exception as e:
        print(f"[IMAGE] Search error: {e}")
    
    # Fallback if DDGS is blocked
    clean_q = re.sub(r"\b(show|give|display|picture|pic|photo|image|of|me|a|the|poster)\b", "", query, flags=re.IGNORECASE).strip()
    encoded = quote(clean_q)
    return f"https://image.pollinations.ai/prompt/{encoded}?width=800&height=800&nologo=true"


# ============================================================
# API ROUTES
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
            FROM chat_messages WHERE user_id = %s
            GROUP BY session_id ORDER BY latest_id DESC LIMIT 50
        """, (user_id,))
        thread_rows = c.fetchall()
        
        output = []
        for row in thread_rows:
            sid = row["session_id"]
            c.execute("""
                SELECT content, timestamp FROM chat_messages
                WHERE user_id = %s AND session_id = %s AND role = 'user'
                ORDER BY id ASC LIMIT 1
            """, (user_id, sid))
            first = c.fetchone()
            output.append({
                "session_id": sid,
                "content": first["content"] if first else "New conversation",
                "timestamp": first["timestamp"] if first else ""
            })
        conn.close()
        return {"threads": output}
    except Exception as e:
        return {"threads": []}

@app.get("/api/thread_messages")
def get_thread_messages(session_id: str, user_id: str):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            SELECT role, content, timestamp, media_url
            FROM chat_messages WHERE session_id = %s AND user_id = %s ORDER BY id ASC
        """, (session_id, user_id))
        rows = c.fetchall()
        conn.close()
        return {"messages": [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"], "media_url": r["media_url"]} for r in rows]}
    except Exception as e:
        return {"messages": []}

@app.post("/api/clear_threads")
async def clear_user_threads(request: Request):
    try:
        data = await request.json()
        user_id = data.get("user_id")
        if not user_id: return {"success": False, "error": "user_id required"}
        
        conn = get_db()
        c = conn.cursor()
        c.execute("DELETE FROM chat_messages WHERE user_id = %s", (user_id,))
        deleted = c.rowcount
        conn.commit()
        conn.close()
        return {"success": True, "deleted": deleted}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/chat")
async def process_command(request: Request):
    try:
        data = await request.json()
        raw_message = str(data.get("message", "")).strip()
        session_id = str(data.get("session_id", "")).strip()
        user_id = str(data.get("user_id", "")).strip()
        user_name = str(data.get("user_name", "")).strip()

        if not raw_message: return {"reply": "Tell me what's on your mind.", "media_url": None, "sources": []}
        if not session_id: session_id = f"session_{int(time.time() * 1000)}"
        if not user_id: user_id = "default_user"

        purge_old_messages()
        emotional_state = detect_emotional_state(raw_message)
        history = get_session_history(session_id=session_id, user_id=user_id, limit=8)

        answer, sources = ask_ai_brain(
            user_message=raw_message,
            session_id=session_id,
            user_id=user_id,
            user_name=user_name,
            history=history,
            emotional_state=emotional_state,
        )

        media_url = None
        lower = raw_message.lower()
        if any(phrase in lower for phrase in ["show me an image", "show image", "find an image", "picture of", "photo of"]):
            media_url = fetch_web_image(raw_message)
            if media_url:
                try:
                    conn = get_db()
                    c = conn.cursor()
                    c.execute("""
                        UPDATE chat_messages SET media_url = %s
                        WHERE id = (SELECT MAX(id) FROM chat_messages WHERE session_id = %s AND user_id = %s AND role = 'assistant')
                    """, (media_url, session_id, user_id))
                    conn.commit()
                    conn.close()
                except Exception:
                    pass

        return {"reply": answer, "media_url": media_url, "sources": sources, "session_id": session_id}

    except Exception:
        return {"reply": "Something went wrong while processing that. Please try again.", "media_url": None, "sources": []}

@app.get("/api/tts")
async def text_to_speech(text: str):
    clean_text = re.sub(r"\[[0-9]+\]", "", text or "")
    clean_text = re.sub(r"https?://\S+", "", clean_text).strip()
    if not clean_text:
        return Response(content=b"", media_type="audio/mpeg")

    candidate_voices = ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural"]
    for voice in candidate_voices:
        try:
            audio_data = bytearray()
            communicate = edge_tts.Communicate(clean_text, voice)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
