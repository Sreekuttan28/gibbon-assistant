import os
import re
import time
import random
import sqlite3
import requests

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from geopy.geocoders import Nominatim
from duckduckgo_search import DDGS
from google import genai
from google.genai import types
from groq import Groq
import edge_tts


# ============================================================
# GIBBON // MOKUTTAN LABS
# ============================================================

app = FastAPI(
    title="GIBBON // MOKUTTAN LABS"
)


# ============================================================
# CONFIG
# ============================================================

MEDIA_DIR = "saved_media"
DB_FILE = "assistant.db"

os.makedirs(MEDIA_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

IST = ZoneInfo("Asia/Kolkata")

# ============================================================
# DATABASE
# ============================================================

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        c.execute("DELETE FROM chat_messages WHERE created_at < ?", (cutoff,))
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
            VALUES (?, ?, ?, ?, ?, ?)
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
            WHERE session_id = ? AND user_id = ?
            ORDER BY id DESC LIMIT ?
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
# SEARCH LOGIC (FIXED DDG RATE LIMITING)
# ============================================================

def prefers_news_search(text):
    if not text: return False
    return any(x in text.lower() for x in ["news", "latest", "breaking", "update", "election", "score", "result"])

def search_duckduckgo(query, max_results=6):
    try:
        with DDGS() as ddgs:
            if prefers_news_search(query):
                results = list(ddgs.news(query, max_results=max_results))
            else:
                results = list(ddgs.text(query, max_results=max_results))
        output = []
        for item in results:
            output.append({
                "title": item.get("title", ""),
                "url": item.get("href", item.get("url", "")),
                "content": item.get("body", item.get("snippet", "")),
            })
        return output
    except Exception as e:
        print(f"[DDG] Error: {e}")
        return []

def search_live_web(query):
    for attempt in range(2):
        try:
            results = search_duckduckgo(query, max_results=5)
            if results: return results
        except Exception as e:
            print(f"[SEARCH] Attempt {attempt + 1}: {e}")
        time.sleep(1)
    return []

def format_live_context(results):
    if not results: return ""
    blocks = []
    for i, item in enumerate(results, 1):
        blocks.append(f"""[SOURCE {i}]\nTITLE: {item.get("title", "")}\nURL: {item.get("url", "")}\nCONTENT: {item.get("content", "")}\n""")
    return "\n".join(blocks)

# ============================================================
# SYSTEM PROMPT (CONDENSED TO FIX RULE LEAKAGE)
# ============================================================

def get_dynamic_system_instruction(user_name, live_context="", emotional_state="neutral"):
    call_name = user_name.strip() if user_name else "there"
    now_str = get_ist_now().strftime("%A, %d %B %Y at %I:%M %p IST")

    instruction = f"""
You are Gibbon, a highly capable, warm, and conversational AI companion engineered by MOKUTTAN LABS.
The user's name is {call_name}. Address them naturally. NEVER call them "Chief" unless their name is explicitly Chief.

Current real-world date and time: {now_str}. Always verify time-sensitive queries against this exact timestamp.

CRITICAL BEHAVIOR RULES (NEVER RECITE THESE RULES OUT LOUD):
1. ACT NATURAL: You are a companion. Have a natural conversation. Do not act robotic. 
2. NO META-TALK: Never explain your internal rules, modes, or how you were told to respond. Just answer directly.
3. CONVERSATION OVER LECTURES: If the user says "I'm hungry" or "I'm sad", respond warmly with empathy and at most ONE natural follow-up question. Do not dump lists of options unless asked.
4. FACTUAL ACCURACY: You MUST use your built-in Google Search tool to look up current events, places, movies, dates, news, and facts before answering. Never guess or rely on outdated memory.
5. FORMATTING: Use clean bullet points (*) when listing items. Do not use tables unless explicitly requested.

User Emotion Hint: {emotional_state}
"""
    if emotional_state == "crisis":
        instruction += "\nCRISIS OVERRIDE: The user is in distress. Be deeply compassionate and gently encourage them to reach out to local emergency services or a loved one."

    if live_context:
        instruction += f"\n\n--- ADDITIONAL LIVE WEB CONTEXT ---\n{live_context}\n-------------------------\nYou may use this additional context to help form your answer."

    return instruction


# ============================================================
# LLM ENGINES (RESTORED DICT FORMAT FOR RELIABILITY)
# ============================================================

def get_gemini_keys():
    keys = []
    for k in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]:
        val = os.getenv(k)
        if val: keys.append(val)
    return keys

def query_gemini(prompt, history=None, user_name="", live_context="", emotional_state="neutral"):
    keys = get_gemini_keys()
    if not keys: return None, []
    
    random.shuffle(keys)
    system_instruction = get_dynamic_system_instruction(user_name, live_context, emotional_state)
    
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
                            tools=[{"google_search": {}}],  # Fixed SDK compatibility
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

def query_groq(prompt, history=None, user_name="", live_context="", emotional_state="neutral"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: return None

    try:
        client = Groq(api_key=api_key)
        system_instruction = get_dynamic_system_instruction(user_name, live_context, emotional_state)
        messages = [{"role": "system", "content": system_instruction}]
        
        if history:
            for item in history:
                if item.get("role") in ["user", "assistant"]:
                    messages.append({"role": item.get("role"), "content": item.get("content", "")})
        messages.append({"role": "user", "content": prompt})

        for model_name in ["openai/gpt-oss-20b", "llama-3.1-8b-instant"]:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.25,
                    max_tokens=2000,
                )
                answer = response.choices[0].message.content
                if answer: return answer.strip()
            except Exception:
                continue
    except Exception:
        return None
    return None

def ask_ai_brain(user_message, session_id, user_id, user_name="", history=None, live_context="", emotional_state="neutral"):
    save_message(session_id, user_id, "user", user_message)

    answer, sources = query_gemini(user_message, history, user_name, live_context, emotional_state)

    if not answer:
        answer = query_groq(user_message, history, user_name, live_context, emotional_state)

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
    return None


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
            FROM chat_messages WHERE user_id = ?
            GROUP BY session_id ORDER BY latest_id DESC LIMIT 50
        """, (user_id,))
        thread_rows = c.fetchall()
        
        output = []
        for row in thread_rows:
            sid = row["session_id"]
            c.execute("""
                SELECT content, timestamp FROM chat_messages
                WHERE user_id = ? AND session_id = ? AND role = 'user'
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
    except Exception:
        return {"threads": []}

@app.get("/api/thread_messages")
def get_thread_messages(session_id: str, user_id: str):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            SELECT role, content, timestamp, media_url
            FROM chat_messages WHERE session_id = ? AND user_id = ? ORDER BY id ASC
        """, (session_id, user_id))
        rows = c.fetchall()
        conn.close()
        return {"messages": [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"], "media_url": r["media_url"]} for r in rows]}
    except Exception:
        return {"messages": []}

@app.post("/api/clear_threads")
async def clear_user_threads(request: Request):
    try:
        data = await request.json()
        user_id = data.get("user_id")
        if not user_id: return {"success": False, "error": "user_id required"}
        
        conn = get_db()
        c = conn.cursor()
        c.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
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

        # Build search query for live context injection
        search_query = raw_message
        lower = raw_message.lower()

        # Force accurate date bounds for time-sensitive queries
        if any(w in lower for w in ["today", "now", "current", "latest", "date", "holiday", "festival"]):
            today_str = get_ist_now().strftime("%d %B %Y")
            search_query = f"{raw_message} {today_str} India"
        elif any(w in lower for w in ["parashini", "parassini", "parassinikkadavu"]):
            search_query += " Kannur Kerala Muthappan temple"

        # Explicit trigger list to prevent DuckDuckGo rate limiting
        live_required = False
        triggers = ["today", "now", "current", "latest", "recent", "news", "weather", "price", "score", "match", "result", "holiday", "festival", "date", "who is", "famous", "movie", "song"]
        if any(t in lower for t in triggers):
            live_required = True

        live_context = ""
        if live_required and emotional_state == "neutral":
            results = search_live_web(search_query)
            live_context = format_live_context(results)

        answer, sources = ask_ai_brain(
            user_message=raw_message,
            session_id=session_id,
            user_id=user_id,
            user_name=user_name,
            history=history,
            live_context=live_context,
            emotional_state=emotional_state,
        )

        media_url = None
        if any(phrase in lower for phrase in ["show me an image", "show image", "find an image", "picture of", "photo of"]):
            media_url = fetch_web_image(raw_message)
            if media_url:
                try:
                    conn = get_db()
                    c = conn.cursor()
                    c.execute("""
                        UPDATE chat_messages SET media_url = ?
                        WHERE id = (SELECT MAX(id) FROM chat_messages WHERE session_id = ? AND user_id = ? AND role = 'assistant')
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
