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

def get_current_year() -> int:
    """Always derive the year from the clock. Never hardcode a year anywhere in this file."""
    return get_ist_now().year

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
# SEARCH — MULTI-SOURCE, DATE-ANCHORED, RECENCY-FILTERED
# ============================================================
#
# Design notes:
# - Year/date are always computed live (get_current_year / get_ist_now), never hardcoded.
# - We try a real search API first if a key is configured (Brave / Tavily / Serper —
#   whichever env var is present), because these return actual freshness metadata and
#   are far more reliable than scraping. Set ONE of BRAVE_API_KEY / TAVILY_API_KEY /
#   SERPER_API_KEY in your environment to enable this tier. If none are set, we skip
#   straight to the fallbacks below — the bot still works, just with weaker recency.
# - Firecrawl is kept as a second-tier fallback.
# - DDGS is the last resort, restricted to the past month (timelimit="m") so it can't
#   hand back 2024 cache hits for a 2026 query.
# - Every source gets the same date-anchored query string.

def _anchor_query(query: str) -> str:
    year = get_current_year()
    if str(year) in query:
        return query
    return f"{query} {year}"

def _search_brave(query: str) -> str:
    key = os.getenv("BRAVE_API_KEY")
    if not key:
        return ""
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": key},
            params={"q": query, "count": 5, "freshness": "pm"},  # past month
            timeout=6,
        )
        if resp.status_code != 200:
            return ""
        results = resp.json().get("web", {}).get("results", [])
        blocks = []
        for item in results[:5]:
            title = item.get("title", "No Title")
            desc = (item.get("description") or "").strip()
            age = item.get("age", "")
            if desc:
                blocks.append(f"- {title} [{age}]: {desc}")
        return "\n\n".join(blocks)
    except Exception as e:
        print(f"[SEARCH] Brave error: {e}")
        return ""

def _search_tavily(query: str) -> str:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return ""
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": query, "max_results": 5, "days": 30},
            timeout=6,
        )
        if resp.status_code != 200:
            return ""
        results = resp.json().get("results", [])
        blocks = []
        for item in results[:5]:
            title = item.get("title", "No Title")
            content = (item.get("content") or "").strip()[:600]
            if content:
                blocks.append(f"- {title}: {content}")
        return "\n\n".join(blocks)
    except Exception as e:
        print(f"[SEARCH] Tavily error: {e}")
        return ""

def _search_serper(query: str) -> str:
    key = os.getenv("SERPER_API_KEY")
    if not key:
        return ""
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": 5, "tbs": "qdr:m"},  # past month
            timeout=6,
        )
        if resp.status_code != 200:
            return ""
        data = resp.json()
        blocks = []
        # Direct answer box, if present, is usually the most reliable single fact
        if data.get("answerBox"):
            ab = data["answerBox"]
            snippet = ab.get("answer") or ab.get("snippet")
            if snippet:
                blocks.append(f"- DIRECT ANSWER: {snippet}")
        for item in data.get("organic", [])[:4]:
            title = item.get("title", "No Title")
            snippet = (item.get("snippet") or "").strip()
            date = item.get("date", "")
            if snippet:
                blocks.append(f"- {title} [{date}]: {snippet}")
        return "\n\n".join(blocks)
    except Exception as e:
        print(f"[SEARCH] Serper error: {e}")
        return ""

def _search_firecrawl(query: str) -> str:
    key = os.getenv("FIRECRAWL_API_KEY")
    if not key:
        return ""
    try:
        resp = requests.post(
            "https://api.firecrawl.dev/v1/search",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"query": query, "limit": 3, "scrapeOptions": {"formats": ["markdown"]}},
            timeout=7,
        )
        if resp.status_code != 200:
            return ""
        data = resp.json()
        results = data.get("data", []) if isinstance(data, dict) else []
        blocks = []
        for item in results:
            title = item.get("title", "No Title")
            content = item.get("markdown") or item.get("description", "")
            clean_content = content.replace("\n", " ").strip()[:900]
            if clean_content:
                blocks.append(f"- {title}: {clean_content}")
        return "\n\n".join(blocks)
    except Exception as e:
        print(f"[SEARCH] Firecrawl error: {e}")
        return ""

def _search_ddgs(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5, timelimit="m"))
            blocks = [f"- {r.get('title','')}: {r.get('body','')}" for r in results if r.get("body")]
            return "\n\n".join(blocks)
    except Exception as e:
        print(f"[SEARCH] DDGS error: {e}")
        return ""

def search_live_web(query: str, client_time: str = "") -> str:
    """
    Runs the date-anchored query through each configured search tier in order,
    stopping at the first one that returns usable content. Returns "" if every
    tier fails — the caller MUST treat "" as "no verified data", not as license
    to guess.
    """
    clean_q = (query or "").strip()
    if not clean_q or len(clean_q) < 2:
        return ""

    targeted_query = _anchor_query(clean_q)

    for engine in (_search_brave, _search_tavily, _search_serper, _search_firecrawl, _search_ddgs):
        result = engine(targeted_query)
        if result:
            return result

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
# CATEGORY DETECTION — only gate advice where bad guesses hurt
# ============================================================
#
# We do NOT slap a "consult a professional" disclaimer on every message — that
# trains people to tune it out. It's reserved for medical, financial, and legal
# topics, where Gibbon gives a genuinely useful, informative answer first and
# closes with a light, natural nudge to see a professional for anything
# diagnostic, prescriptive, or binding. Lifestyle, emotional, tech, books, and
# general knowledge questions get answered directly and warmly, like a friend
# who actually knows things — no disclaimers bolted on.

CATEGORY_KEYWORDS = {
    "medical": [
        "symptom", "disease", "medicine", "medication", "dosage", "dose", "tablet",
        "treatment", "diagnosis", "surgery", "fever", "infection", "pain", "prescription",
        "side effect", "injury", "mental health crisis", "chest pain", "allergy",
    ],
    "financial": [
        "invest", "investment", "stock", "mutual fund", "crypto", "loan", "tax",
        "trading", "insurance policy", "emi", "portfolio", "retirement fund", "ipo",
    ],
    "legal": [
        "lawsuit", "legal notice", "contract clause", "divorce", "fir", "court case",
        "legal action", "sue", "lawyer", "property dispute",
    ],
}

def detect_category(query: str) -> str:
    lower = query.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            return cat
    return "general"

# ============================================================
# TIME-SENSITIVITY DETECTION — decides whether we search at all
# ============================================================
#
# Replaces the old "bypass list" approach. Instead of a shrinking blacklist of
# greetings, we use a small allowlist of signals that a question is actually
# about real-world current state. This is intentionally broad — false
# positives (searching when we didn't strictly need to) are cheap; false
# negatives (not searching when we needed live data) are what caused the
# hallucinations in the first place.

PURE_CHITCHAT = {
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye",
    "good night", "good morning", "yo", "sup", "sugamano", "sugamane",
    "da", "enthokkeyundu", "entha visesham", "enna und", "evide",
    "how are you", "whatsup", "whats up", "gm", "gn",
}

TIME_SENSITIVE_HINTS = [
    "today", "yesterday", "tomorrow", "now", "current", "currently", "latest",
    "news", "score", "match", "vs", "won", "result", "update", "market", "stock",
    "price", "cm", "pm", "president", "prime minister", "chief minister", "ceo",
    "date", "recent", "new", "release", "who is", "when is", "when did",
    "weather", "election", "live",
]

def needs_live_search(raw_message: str, lower: str) -> bool:
    if lower in PURE_CHITCHAT:
        return False
    if len(raw_message.strip()) <= 2:
        return False
    # Anything that isn't obviously pure chit-chat gets searched by default.
    # The TIME_SENSITIVE_HINTS list just makes the intent explicit for logging.
    return True

# ============================================================
# SYSTEM INSTRUCTION — warm "best friend" persona + safety rules
# ============================================================

def get_dynamic_system_instruction(user_name: str, live_context: str, client_time: str, category: str) -> str:
    now_str = client_time.strip() if client_time else get_ist_now().strftime("%A, %d %B %Y at %I:%M %p IST")
    call_name = user_name.strip() if user_name else "friend"

    instruction = f"""
You are Gibbon, built by MOKUTTAN LABS. You talk like a genuinely good friend to {call_name} —
warm, direct, present. Not a customer-support bot, not a lecture. You listen to what's actually
being asked, you're honest even when the truth is inconvenient, and you don't pad answers with
filler or over-hedge on things that don't need hedging.

Current user: {call_name}.
Current verified date and time: {now_str}.

CORE RULES:

1. TYPOS & MANGLISH: The user may write Manglish (Malayalam in English script like
   'enthokkeyundu', 'sugamane', 'evideya'), slang, or typos. Read the intent, don't nitpick grammar.

2. ZERO GUESSING ON LIVE FACTS. This is the most important rule. For anything involving current
   office-holders, scores, prices, news, or dates: rely STRICTLY on the LIVE WEB CONTEXT below.
   - If LIVE WEB CONTEXT is present, ground your answer in it and only it for the live facts.
   - If it says "NO LIVE DATA RETRIEVED", say plainly you couldn't confirm the current answer and
     offer what you're confident is still true generally (e.g. how a process works), without
     inventing a name, score, or date. Never present a guess as a fact. Saying "I'm not sure,
     let's check" is a good answer. A confident wrong answer is a failure.

3. MEDICAL: You're not a doctor and this app does not have a "{category}"-specific override to
   ignore that. You CAN explain how a condition generally works, what symptoms typically mean, and
   what questions to ask a doctor — genuinely useful, not evasive. You must NOT name specific drugs
   or dosages to take, or rule conditions in/out. Close with a brief, natural nudge to see a doctor
   for anything diagnostic or prescriptive — one line, not a paragraph, and not on every message.

4. FINANCIAL: Not a registered financial advisor. Explain concepts, summarize verified market facts
   from LIVE WEB CONTEXT when present, help someone think through tradeoffs — but don't tell them
   what to buy/sell/invest in. Close with a light nudge toward a registered advisor for anything
   that commits real money.

5. LEGAL: Explain how something generally works, what documents/steps are typically involved — but
   don't tell someone their case will win or draft binding legal strategy. Nudge toward a lawyer for
   anything that could go to court or become binding.

6. Everything else — lifestyle, emotional support, technology, books, general knowledge,
   intellectual discussion — just answer it well, like a friend who actually knows the subject.
   No disclaimers bolted onto questions that don't need them.

7. EMOTIONAL CONVERSATIONS: Listen first. Reflect what you're hearing before jumping to advice.
   If someone describes something that sounds like a real crisis (self-harm, suicidal thoughts,
   danger from another person), take it seriously, respond with care, and gently point them to
   reaching out to someone who can actually help right now (a trusted person, a local helpline) —
   don't just hand them a generic disclaimer and move on.

8. PERSONAL BOUNDARIES: If messages turn sexually explicit or harassing, stay warm but firm and
   decline to engage with that content.

9. STYLE: Talk like a person, not a report. Short paragraphs. Use bullet points only when listing
   genuinely distinct items, not as a crutch. It's fine to have a point of view.
"""

    if live_context:
        instruction += f"\n--- LIVE WEB CONTEXT (retrieved just now, {get_ist_now().strftime('%d %b %Y %H:%M IST')}) ---\n{live_context}\n--- END LIVE WEB CONTEXT ---"
    else:
        instruction += (
            "\n--- NO LIVE DATA RETRIEVED ---\n"
            "No current web data could be retrieved for this query. If the question depends on "
            "real-time facts (current office-holders, live scores, prices, breaking news), you do "
            "NOT have that information right now — say so plainly instead of answering from memory. "
            "Do not present a recalled/trained fact as if it were current."
        )

    return instruction

# ============================================================
# AI ENGINES (GEMINI + GROQ FALLBACK)
# ============================================================

def get_gemini_keys():
    keys = []
    for k in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]:
        v = os.getenv(k)
        if v:
            keys.append(v.strip())
    return keys

gemini_key_index = 0

def query_gemini(prompt: str, history: list, user_name: str, live_context: str, client_time: str, category: str) -> str:
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

    system_instruction = get_dynamic_system_instruction(user_name, live_context, client_time, category)
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
                            temperature=0.25,
                        ),
                    )
                    if response and response.text and response.text.strip():
                        return response.text.strip()
                except Exception:
                    continue
        except Exception:
            gemini_key_index = (gemini_key_index + 1) % len(keys)

    return ""

def query_groq(prompt: str, history: list, user_name: str, live_context: str, client_time: str, category: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""

    try:
        client = Groq(api_key=api_key)
        system_instruction = get_dynamic_system_instruction(user_name, live_context, client_time, category)
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
                    max_tokens=1500,
                )
                if resp.choices[0].message.content:
                    return resp.choices[0].message.content.strip()
            except Exception:
                continue
    except Exception:
        pass

    return ""

def ask_ai_brain(prompt: str, session_id: str, user_id: str, user_name: str,
                  live_context: str, media_url: str, client_time: str, category: str) -> str:
    history = get_session_history(session_id, user_id, limit=8)
    save_message(session_id, user_id, "user", prompt)

    reply = query_gemini(prompt, history, user_name, live_context, client_time, category)
    if not reply:
        reply = query_groq(prompt, history, user_name, live_context, client_time, category)

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
                "time": first["timestamp"] if first and first.get("timestamp") else "",
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
            "messages": [
                {"role": r["role"], "content": r["content"], "timestamp": r["timestamp"], "media_url": r["media_url"]}
                for r in rows
            ]
        }
    except Exception:
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

        live_context = ""
        if needs_live_search(raw_message, lower):
            live_context = search_live_web(search_query, client_time)

        category = detect_category(raw_message)

        clean_prompt = re.sub(r"\b(gibbon|given)\b", "", raw_message, flags=re.IGNORECASE).strip() or raw_message
        ai_answer = ask_ai_brain(clean_prompt, session_id, user_id, user_name, live_context, media_url, client_time, category)

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
