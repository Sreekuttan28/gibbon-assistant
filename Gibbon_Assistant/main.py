import os
import re
import time
import random
import sqlite3
import requests
from urllib.parse import quote
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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

geolocator = Nominatim(user_agent="gibbon_hud_agent_v30")
IST = ZoneInfo("Asia/Kolkata")

def get_ist_now() -> datetime:
    return datetime.now(IST)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Message store with 15-day purge capability
    c.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    text TEXT,
                    remind_at TEXT,
                    status TEXT DEFAULT 'pending'
                )''')
    conn.commit()
    conn.close()

init_db()

def purge_old_messages():
    """Purges chat history older than 15 days to respect strict privacy."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        fifteen_days_ago = (datetime.utcnow() - timedelta(days=15)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("DELETE FROM chat_messages WHERE created_at < ?", (fifteen_days_ago,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Purge error: {e}")

def save_message(session_id: str, user_id: str, role: str, content: str):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        ts = get_ist_now().strftime("%I:%M %p")
        c.execute("INSERT INTO chat_messages (session_id, user_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                  (session_id, user_id, role, content, ts))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Save message error: {e}")

def get_session_history(session_id: str, limit: int = 12):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id DESC LIMIT ?", (session_id, limit))
        rows = c.fetchall()
        conn.close()
        # Return in ascending order for LLM context
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception:
        return []

def fetch_url_content(url: str) -> str:
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=12)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            for tag in soup(["script", "style", "nav", "noscript", "svg", "iframe"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            return re.sub(r"\s+", " ", text).strip()[:4500]
    except Exception as e:
        print(f"Scraper error: {e}")
    return ""

def search_live_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if results:
                return "\n".join([f"- {r.get('title', '')}: {r.get('body', '')}" for r in results])
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
    return ""

def get_dynamic_system_instruction(user_name: str, live_context: str = "") -> str:
    now_ist = get_ist_now()
    now_str = now_ist.strftime("%A, %B %d, %Y at %I:%M %p IST")
    call_name = user_name.strip() if user_name and user_name.strip() else "Chief"
    
    instruction = (
        f"You are Gibbon, a friendly, 99% accurate personal AI assistant engineered by Mokuttan Labs. "
        f"The user's name is {call_name}. Address them naturally. "
        f"The current real-world date and time is {now_str} (Indian Standard Time). "
        "CRITICAL RULES: "
        "1. ACCURACY: Provide exact, verified real-world facts. Never guess phone numbers, addresses, or release dates. "
        "2. CONTINUITY: Maintain complete context of previous messages in this conversation. "
        "3. UNIVERSAL ENGLISH: Use clear, simple, everyday English so that anyone can understand effortlessly. "
        "4. FORMATTING: When presenting data, lists, or comparisons, use clean markdown tables."
    )
    if live_context:
        instruction += f"\n\n--- LIVE SEARCH & GROUNDING DATA ---\n{live_context}\n----------------------------------"
    return instruction

def get_gemini_keys():
    keys_str = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    return [k.strip() for k in keys_str.split(",") if k.strip()]

gemini_key_index = 0

def query_gemini(prompt: str, key: str, history: list, user_name: str, live_context: str = "") -> str:
    client = genai.Client(api_key=key)
    candidate_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    last_err = None

    for m in candidate_models:
        try:
            chat = client.chats.create(
                model=m,
                config=types.GenerateContentConfig(
                    system_instruction=get_dynamic_system_instruction(user_name, live_context),
                    tools=[{"google_search": {}}]
                )
            )

            for turn in history:
                if turn["role"] == "user":
                    try:
                        chat.send_message(turn["content"])
                    except Exception:
                        pass

            response = chat.send_message(prompt)
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
    for turn in history:
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

def ask_ai_brain(prompt: str, session_id: str, user_id: str, user_name: str, live_context: str = "") -> str:
    global gemini_key_index
    gemini_keys = get_gemini_keys()
    history = get_session_history(session_id)

    if gemini_keys:
        for _ in range(len(gemini_keys)):
            current_key = gemini_keys[gemini_key_index]
            try:
                reply = query_gemini(prompt, current_key, history, user_name, live_context)
                save_message(session_id, user_id, "user", prompt)
                save_message(session_id, user_id, "assistant", reply)
                return reply
            except Exception:
                gemini_key_index = (gemini_key_index + 1) % len(gemini_keys)

    if os.getenv("GROQ_API_KEY"):
        try:
            reply = query_groq(prompt, history, user_name, live_context)
            save_message(session_id, user_id, "user", prompt)
            save_message(session_id, user_id, "assistant", reply)
            return reply
        except Exception as e:
            print(f"Groq failover error: {e}")

    return f"I apologize {user_name}, our network link is momentarily busy. Please ask again in a few seconds."

def generate_image_with_fallback(clean_prompt: str) -> str:
    encoded = quote(clean_prompt)
    seed = random.randint(1000, 999999)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&seed={seed}&model=flux&nologo=true"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 5000:
            filename = f"gibbon_gen_{int(time.time())}.png"
            with open(os.path.join(MEDIA_DIR, filename), "wb") as f:
                f.write(resp.content)
            return filename
    except Exception:
        pass
    return None

def get_live_forecast(city_name: str, user_name: str) -> str:
    call_name = user_name.strip() if user_name and user_name.strip() else "Chief"
    try:
        loc = geolocator.geocode(city_name, timeout=10)
        if not loc:
            return f"{call_name}, I could not locate coordinates for {city_name}."

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={loc.latitude}&longitude={loc.longitude}"
            f"&current=temperature_2m,wind_speed_10m"
            f"&daily=temperature_2m_max,temperature_2m_min"
            f"&timezone=Asia%2FKolkata"
        )
        res = requests.get(url, timeout=10).json()
        current = res.get("current") or res.get("current_weather", {})
        temp = current.get("temperature_2m", current.get("temperature", "--"))
        wind = current.get("wind_speed_10m", current.get("windspeed", "--"))
        daily = res.get("daily", {})
        max_t = daily.get("temperature_2m_max", [temp])[0]
        min_t = daily.get("temperature_2m_min", [temp])[0]

        city_clean = loc.address.split(",")[0]
        return f"In {city_clean}, it is currently {temp}°C with wind speeds at {wind} km/h. Today's high is {max_t}°C and low is {min_t}°C."
    except Exception:
        return f"Unable to retrieve live forecast right now, {call_name}."

@app.get("/")
def serve_index():
    purge_old_messages()
    return FileResponse("static/index.html")

@app.get("/api/threads")
def get_user_threads(user_id: str):
    """Returns all active conversation threads for this user."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT session_id, content, timestamp 
        FROM chat_messages 
        WHERE user_id = ? AND role = 'user' 
        GROUP BY session_id 
        ORDER BY id DESC LIMIT 20
    """, (user_id,))
    rows = c.fetchall()
    conn.close()
    return {"threads": [{"session_id": r[0], "title": r[1][:38], "time": r[2]} for r in rows]}

@app.get("/api/thread_messages")
def get_thread_messages(session_id: str):
    """Fetches all WhatsApp-style message logs for this specific chat thread."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content, timestamp FROM chat_messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    return {"messages": [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]}

@app.post("/api/chat")
async def process_command(request: Request):
    data = await request.json()
    raw_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default_session")
    user_id = data.get("user_id", "default_user")
    user_name = data.get("user_name", "Chief").strip() or "Chief"
    lower = raw_message.lower()

    if any(k in lower for k in ["what time is it", "current time", "what's the time", "tell me the time", "time now"]):
        now_time = get_ist_now().strftime("%I:%M %p")
        reply = f"The current time is {now_time} IST, {user_name}."
        save_message(session_id, user_id, "user", raw_message)
        save_message(session_id, user_id, "assistant", reply)
        return {"reply": reply}

    if any(k in lower for k in ["what date is it", "today's date", "what is the date", "what day is today"]):
        now_date = get_ist_now().strftime("%A, %B %d, %Y")
        reply = f"Today is {now_date}, {user_name}."
        save_message(session_id, user_id, "user", raw_message)
        save_message(session_id, user_id, "assistant", reply)
        return {"reply": reply}

    if any(k in lower for k in ["date and time", "time and date"]):
        now_full = get_ist_now().strftime("%A, %B %d, %Y at %I:%M %p")
        reply = f"It is currently {now_full} IST, {user_name}."
        save_message(session_id, user_id, "user", raw_message)
        save_message(session_id, user_id, "assistant", reply)
        return {"reply": reply}

    if any(k in lower for k in ["generate image", "create image", "draw", "render image"]):
        clean_idea = re.sub(r"\b(gibbon|given|generate an image of|generate image of|create an image of|draw|render)\b", "", raw_message, flags=re.IGNORECASE).strip()
        filename = generate_image_with_fallback(clean_idea or "futuristic cyberpunk neon core")
        if filename:
            reply = f"Here is the picture I created for you, {user_name}!"
            save_message(session_id, user_id, "user", raw_message)
            save_message(session_id, user_id, "assistant", reply)
            return {"reply": reply, "media_url": f"/media/{filename}"}

    if any(k in lower for k in ["forecast", "weather", "temperature", "rain"]):
        match = re.search(r"(?:in|for|at)\s+([a-zA-Z\s]+)", lower)
        city = match.group(1).strip() if match else "Bengaluru"
        reply = get_live_forecast(city, user_name)
        save_message(session_id, user_id, "user", raw_message)
        save_message(session_id, user_id, "assistant", reply)
        return {"reply": reply}

    # URL Scraping & Live Search Grounding
    live_context = ""
    url_match = re.search(r'(https?://[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)', raw_message)
    if url_match and ("." in url_match.group(1)) and not url_match.group(1).endswith("."):
        url = url_match.group(1).rstrip(",.?!")
        page_text = fetch_url_content(url)
        if page_text:
            live_context = f"CONTENT SCRAPED DIRECTLY FROM {url}:\n{page_text}"

    if not live_context:
        search_query = re.sub(r"\b(gibbon|given|hey|hi)\b", "", raw_message, flags=re.IGNORECASE).strip()
        live_context = search_live_web(search_query)

    clean_prompt = re.sub(r"\b(gibbon|given)\b", "", raw_message, flags=re.IGNORECASE).strip()
    ai_answer = ask_ai_brain(clean_prompt or raw_message, session_id, user_id, user_name, live_context)
    return {"reply": ai_answer}

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
