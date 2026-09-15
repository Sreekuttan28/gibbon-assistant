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

geolocator = Nominatim(user_agent="gibbon_hud_agent_v45")
IST = ZoneInfo("Asia/Kolkata")


def get_ist_now() -> datetime:
    return datetime.now(IST)


# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LIVE SEARCH — this is the part that determines factual accuracy.
# Two search modes: general "text" search, and "news" search for anything
# time-sensitive (current office-holders, scores, releases, prices, etc).
# Both retry once on failure instead of silently returning empty context,
# and every result line carries its source so the model (and the user, if
# it ever prints sources) can tell where a claim came from.
# ---------------------------------------------------------------------------

TIME_SENSITIVE_PATTERNS = [
    "current", "latest", "today", "now", "recent", "this year", "who is the",
    "chief minister", "cm of", "president of", "prime minister", "ceo of",
    "score", "match result", "result", "won", "release date", "news",
    "update", "price of", "stock", "weather", "election", "when did",
    "how old is", "net worth", "died", "passed away", "launch", "released"
]

NEWS_PREFERRED_PATTERNS = [
    "news", "latest", "today", "score", "result", "update", "election",
    "breaking", "announcement", "launch", "released"
]


def is_time_sensitive_query(text: str) -> bool:
    lower = (text or "").lower()
    return any(p in lower for p in TIME_SENSITIVE_PATTERNS)


def prefers_news_search(text: str) -> bool:
    lower = (text or "").lower()
    return any(p in lower for p in NEWS_PREFERRED_PATTERNS)


def search_live_web(query: str, prefer_news: bool = False) -> str:
    clean_q = (query or "").strip()
    if not clean_q or len(clean_q) < 2:
        return ""

    for attempt in range(2):
        try:
            with DDGS() as ddgs:
                if prefer_news:
                    news_results = list(ddgs.news(clean_q, max_results=6))
                    if news_results:
                        lines = []
                        for r in news_results:
                            title = r.get("title", "")
                            body = r.get("body", "") or r.get("excerpt", "")
                            date = r.get("date", "")
                            source = r.get("source", "") or r.get("url", "")
                            lines.append(f"- [{date}] {title}: {body} (Source: {source})")
                        return "\n".join(lines)

                text_results = list(ddgs.text(clean_q, max_results=6))
                if text_results:
                    lines = []
                    for r in text_results:
                        title = r.get("title", "")
                        body = r.get("body", "")
                        href = r.get("href", "")
                        lines.append(f"- {title}: {body} (Source: {href})")
                    return "\n".join(lines)
            # No results but no exception either — don't keep retrying.
            return ""
        except Exception as e:
            print(f"[Search Error attempt {attempt + 1}] query='{clean_q}' err={e}")
            time.sleep(0.6)
            continue
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


# ---------------------------------------------------------------------------
# EMOTIONAL STATE DETECTION
# Keyword-based, deliberately simple and fast (no extra API round-trip).
# "crisis" triggers a hard-coded safety-net resource message appended to
# whatever the model says, so it never depends solely on the model
# following the system prompt correctly.
# ---------------------------------------------------------------------------

CRISIS_KEYWORDS = [
    "kill myself", "want to die", "end my life", "suicide", "suicidal",
    "no reason to live", "hurt myself", "self harm", "self-harm",
    "can't go on", "cant go on", "better off dead", "ending it all",
    "don't want to live", "dont want to live"
]

DISTRESS_KEYWORDS = [
    "sad", "depressed", "depression", "anxious", "anxiety", "stressed",
    "stress", "heartbroken", "lonely", "hopeless", "overwhelmed", "crying",
    "cried", "breakup", "broke up", "lost my job", "grief", "grieving",
    "miss him", "miss her", "panic attack", "worthless", "exhausted",
    "tired of everything", "give up", "hate my life", "no one cares",
    "feeling low", "not okay", "not ok", "struggling", "scared", "afraid",
    "angry at myself", "failed", "failure", "disappointed in myself"
]

CRISIS_RESOURCE_MSG = (
    "\n\nIf things feel like too much right now, please know support is available. "
    "In India you can call the KIRAN mental health helpline at 1800-599-0019 (toll-free, 24/7) "
    "or iCall at 9152987821. If you're outside India, please reach out to your local emergency "
    "number or a crisis helpline. You don't have to go through this alone."
)


def detect_emotional_state(text: str) -> str:
    lower = (text or "").lower()
    if any(k in lower for k in CRISIS_KEYWORDS):
        return "crisis"
    if any(k in lower for k in DISTRESS_KEYWORDS):
        return "distress"
    return "neutral"


# ---------------------------------------------------------------------------
# SYSTEM INSTRUCTION
# ---------------------------------------------------------------------------

def get_dynamic_system_instruction(user_name: str, live_context: str = "", emotional_state: str = "neutral") -> str:
    now_ist = get_ist_now()
    now_str = now_ist.strftime("%A, %B %d, %Y at %I:%M %p IST")
    call_name = user_name.strip() if user_name and user_name.strip() else "Chief"

    instruction = (
        f"You are Gibbon, a knowledgeable, accurate AI companion engineered by Mokuttan Labs. "
        f"The user's name is {call_name}. Address them naturally by their name ({call_name}) and NEVER refer to them as 'Chief' unless their name is explicitly Chief. "
        f"Current real-world date and time: {now_str} (Indian Standard Time). "
        "CRITICAL RULES: "
        "1. FACTUAL ACCURACY: Base answers on the LIVE WEB CONTEXT below whenever it is present — it reflects the real current state of the world, which is more recent and more reliable than your own training data, especially for things like current office-holders, scores, prices, and recent events. If the LIVE WEB CONTEXT contradicts what you think you know, trust the LIVE WEB CONTEXT. If no LIVE WEB CONTEXT is provided for a question about current people, events, scores, or prices, say plainly that you couldn't verify the latest information rather than guessing. "
        "2. FORMATTING: Use clean bullet points (*) or direct paragraphs. Do NOT force tables for general information or descriptions. Use tables only when specifically asked to compare items or data. "
        "3. CONTINUITY: You are inside an isolated chat thread. Maintain focus on the questions asked in THIS thread only without cross-contamination. "
        "4. SONG LYRICS: Never reproduce full song lyrics or large verbatim chunks of them. If asked for lyrics, credit the artist and song, give a short thematic summary in your own words, and point the user to a licensed lyrics platform or streaming service for the exact text."
    )

    if emotional_state == "crisis":
        instruction += (
            " 5. EMOTIONAL PRIORITY — CRISIS: The user's message suggests they may be in real emotional pain or crisis. "
            "Respond with warmth and calm before anything else. Do not lecture, minimize, or rush to solutions. "
            "Validate what they're feeling, encourage them to reach out to someone they trust or a professional, "
            "and keep your tone steady and caring through the rest of the reply."
        )
    elif emotional_state == "distress":
        instruction += (
            " 5. EMOTIONAL PRIORITY: The user's message suggests they're going through a difficult moment. "
            "Lead with empathy and genuine warmth before offering information or advice. Keep the tone calm and supportive throughout."
        )

    if live_context:
        instruction += f"\n\n--- LIVE WEB CONTEXT ---\n{live_context}\n-------------------------"
    return instruction


# ---------------------------------------------------------------------------
# MODEL BACKENDS
# ---------------------------------------------------------------------------

def get_gemini_keys():
    keys_str = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    return [k.strip() for k in keys_str.split(",") if k.strip()]


gemini_key_index = 0


def query_gemini(prompt: str, key: str, history: list, user_name: str, live_context: str = "", emotional_state: str = "neutral") -> str:
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
                    system_instruction=get_dynamic_system_instruction(user_name, live_context, emotional_state),
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


def query_groq(prompt: str, history: list, user_name: str, live_context: str = "", emotional_state: str = "neutral") -> str:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not configured.")

    client = Groq(api_key=groq_key)
    messages = [{"role": "system", "content": get_dynamic_system_instruction(user_name, live_context, emotional_state)}]
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


def ask_ai_brain(prompt: str, session_id: str, user_id: str, user_name: str, live_context: str = "",
                  media_url: str = None, emotional_state: str = "neutral") -> str:
    global gemini_key_index
    gemini_keys = get_gemini_keys()
    history = get_session_history(session_id, user_id)

    save_message(session_id, user_id, "user", prompt)

    reply = None

    if gemini_keys:
        for _ in range(len(gemini_keys)):
            current_key = gemini_keys[gemini_key_index]
            try:
                reply = query_gemini(prompt, current_key, history, user_name, live_context, emotional_state)
                break
            except Exception as e:
                print(f"[Gemini Error on Key {gemini_key_index + 1}]: {e}")
                gemini_key_index = (gemini_key_index + 1) % len(gemini_keys)

    if reply is None and os.getenv("GROQ_API_KEY"):
        try:
            if not live_context:
                clean_q = re.sub(r"\bgibbon\b", "", prompt, flags=re.IGNORECASE).strip()
                prefer_news = prefers_news_search(prompt)
                live_context = search_live_web(clean_q or prompt, prefer_news=prefer_news)
            reply = query_groq(prompt, history, user_name, live_context, emotional_state)
        except Exception as e:
            print(f"[Groq Failover Error]: {e}")

    if reply is None:
        reply = f"I apologize {user_name}, connection is busy right now. Please try again."

    # Safety net: guarantee crisis resources are present regardless of what
    # the model produced, in case it didn't follow the system instruction.
    if emotional_state == "crisis" and "1800-599-0019" not in reply:
        reply += CRISIS_RESOURCE_MSG

    save_message(session_id, user_id, "assistant", reply, media_url)
    return reply


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

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

    emotional_state = detect_emotional_state(raw_message)

    wants_image = any(w in lower for w in ["show me", "picture of", "photo of", "poster of", "pic of", "image of"])
    media_url = None
    if wants_image:
        media_url = fetch_web_image(raw_message)

    history = get_session_history(session_id, user_id, limit=4)
    search_query = re.sub(r"\b(gibbon|hey|hi|hello|show me|picture of|poster of)\b", "", raw_message, flags=re.IGNORECASE).strip()
    final_search_query = search_query if len(search_query) >= 2 else raw_message.strip()

    if len(final_search_query.split()) <= 3 and history:
        last_user_turn = next((t["content"] for t in reversed(history) if t["role"] == "user"), "")
        if last_user_turn and last_user_turn.lower() != raw_message.lower():
            final_search_query = f"{last_user_turn} {final_search_query}"

    if any(k in lower for k in ["today", "holiday", "festival", "speciality", "specialty", "date", "now", "current"]):
        current_date_query = get_ist_now().strftime("%B %Y")
        final_search_query = f"{raw_message} {current_date_query} India"

    if any(k in lower for k in ["parashini", "parassini", "parassinikkadavu"]):
        final_search_query += " Kannur Kerala Muthappan temple"

    # UNIVERSAL SEARCH TRIGGER: fetch live context for anything that isn't a
    # bare conversational greeting, and for short-but-time-sensitive queries
    # (e.g. "cm of kerala") route to news search instead of general text search.
    conversational_greetings = ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodnight", "good morning", "yo", "sup", "yes", "no"]

    live_context = ""
    should_search = (
        lower not in conversational_greetings
        and (len(final_search_query) > 2 or is_time_sensitive_query(raw_message))
        and emotional_state == "neutral"  # don't derail an emotional message with a web lookup
    )
    if should_search:
        prefer_news = prefers_news_search(raw_message)
        live_context = search_live_web(final_search_query, prefer_news=prefer_news)

    clean_prompt = re.sub(r"\bgibbon\b", "", raw_message, flags=re.IGNORECASE).strip()
    ai_answer = ask_ai_brain(
        clean_prompt or raw_message, session_id, user_id, user_name,
        live_context, media_url, emotional_state
    )
    return {"reply": ai_answer, "media_url": media_url}


@app.get("/api/tts")
async def text_to_speech(text: str):
    clean_text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    clean_text = re.sub(r'[*#_`|~>—–-]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()[:4000]

    candidate_voices = ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural"]
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
