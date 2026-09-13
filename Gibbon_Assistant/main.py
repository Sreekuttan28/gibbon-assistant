import os
import re
import time
import random
import sqlite3
import requests
from collections import defaultdict
from urllib.parse import quote
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from geopy.geocoders import Nominatim
from duckduckgo_search import DDGS
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

geolocator = Nominatim(user_agent="gibbon_hud_agent_v20")
IST = ZoneInfo("Asia/Kolkata")

# Isolated in-memory conversation turns per user/device session
user_sessions = defaultdict(list)

def get_ist_now() -> datetime:
    return datetime.now(IST)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
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

def get_dynamic_system_instruction(user_name: str = "Chief") -> str:
    now_ist = get_ist_now()
    now_str = now_ist.strftime("%A, %B %d, %Y at %I:%M %p IST")
    name = user_name.strip() if user_name and user_name.strip() else "Chief"
    return (
        f"You are Gibbon, a crisp, modern, intelligent female personal AI assistant engineered by Mokuttan Labs. "
        f"Always address the user politely by their chosen name: '{name}'. "
        f"The current real-world date and time is {now_str} (Indian Standard Time). "
        "CRITICAL TIMING & SCHEDULE RULES: "
        "1. When calculating sleep, waking, or working hours, calculate step-by-step arithmetic before answering. "
        "2. If the user works late (e.g. until 2:00 AM), sleep must begin shortly after wrapping up (e.g. 2:30 AM). "
        "3. A standard 7.5 to 8 hour sleep block after 2:30 AM means waking between 10:00 AM and 10:30 AM. Never suggest waking at 6:30 AM after late work. "
        "4. Keep answers warm, accurate, and concise (under 3 sentences unless explaining a complex idea)."
    )

def get_gemini_keys():
    keys_str = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    return [k.strip() for k in keys_str.split(",") if k.strip()]

gemini_key_index = 0

def query_gemini(prompt: str, key: str, session_id: str, user_name: str) -> str:
    client = genai.Client(api_key=key)
    candidate_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    last_err = None

    history = user_sessions[session_id]

    for m in candidate_models:
        try:
            chat = client.chats.create(
                model=m,
                config=types.GenerateContentConfig(
                    system_instruction=get_dynamic_system_instruction(user_name),
                    tools=[{"google_search": {}}]
                )
            )

            for turn in history[-4:]:
                if turn["role"] == "user":
                    try:
                        chat.send_message(turn["content"])
                    except Exception:
                        pass

            response = chat.send_message(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            err_str = str(e)
            print(f"Gemini {m} error: {err_str}")
            last_err = e
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                raise e
            continue

    raise last_err or RuntimeError("Gemini models failed.")

def query_groq(prompt: str, session_id: str, user_name: str) -> str:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not set.")

    client = Groq(api_key=groq_key)
    messages = [{"role": "system", "content": get_dynamic_system_instruction(user_name)}]
    for turn in user_sessions[session_id][-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": prompt})

    candidate_groq_models = ["openai/gpt-oss-20b", "llama-3.1-8b-instant"]
    last_err = None

    for model_name in candidate_groq_models:
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.4,
                max_tokens=280
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as err:
            last_err = err
            continue

    raise last_err or RuntimeError("All Groq models failed.")

def ask_ai_brain(prompt: str, session_id: str, user_name: str) -> str:
    global gemini_key_index
    gemini_keys = get_gemini_keys()

    if gemini_keys:
        for _ in range(len(gemini_keys)):
            current_key = gemini_keys[gemini_key_index]
            try:
                reply = query_gemini(prompt, current_key, session_id, user_name)
                user_sessions[session_id].append({"role": "user", "content": prompt})
                user_sessions[session_id].append({"role": "assistant", "content": reply})
                return reply
            except Exception as e:
                gemini_key_index = (gemini_key_index + 1) % len(gemini_keys)

    if os.getenv("GROQ_API_KEY"):
        try:
            reply = query_groq(prompt, session_id, user_name)
            user_sessions[session_id].append({"role": "user", "content": prompt})
            user_sessions[session_id].append({"role": "assistant", "content": reply})
            return reply
        except Exception as groq_err:
            print(f"Groq failover exception: {groq_err}")

    name = user_name or "Chief"
    return f"Apologies {name}, both primary and backup cognitive links are temporarily rate-limited. Please allow 30 seconds."

def generate_image_with_fallback(clean_prompt: str) -> str:
    encoded = quote(clean_prompt)
    seed = random.randint(1000, 999999)
    endpoints = [
        f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&seed={seed}&model=flux&nologo=true",
        f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&seed={seed}&nologo=true"
    ]

    for url in endpoints:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 5000:
                filename = f"gibbon_gen_{int(time.time())}.png"
                filepath = os.path.join(MEDIA_DIR, filename)
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                return filename
        except Exception:
            continue
    return None

def engineer_prompt_for_creator(raw_idea: str, session_id: str, user_name: str) -> str:
    instruction = (
        f"Turn this concept into a studio-grade cinematic image & video prompt: '{raw_idea}'. "
        "Format: Provide 1 clean, high-detail visual prompt with lighting, camera lens, resolution, and aesthetic details. Keep it under 50 words."
    )
    return ask_ai_brain(instruction, session_id, user_name)

def get_live_forecast(city_name: str, user_name: str) -> str:
    name = user_name or "Chief"
    try:
        loc = geolocator.geocode(city_name, timeout=10)
        if not loc:
            return f"{name}, I could not pinpoint coordinates for {city_name}."

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={loc.latitude}&longitude={loc.longitude}"
            f"&current_weather=true&daily=temperature_2m_max,temperature_2m_min"
            f"&timezone=Asia%2FKolkata"
        )
        data = requests.get(url, timeout=10).json()
        current = data.get("current_weather", {})
        temp = current.get("temperature")
        wind = current.get("windspeed")
        daily = data.get("daily", {})
        max_t = daily.get("temperature_2m_max", [temp])[0]
        min_t = daily.get("temperature_2m_min", [temp])[0]

        city_clean = loc.address.split(",")[0]
        return (
            f"Current temperature in {city_clean} is {temp}°C with wind speeds at {wind} km/h. "
            f"Today's forecast peaks at {max_t}°C with a low of {min_t}°C."
        )
    except Exception:
        return f"Telemetry failed to fetch live weather metrics, {name}."

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/clear")
async def clear_conversation(request: Request):
    data = await request.json()
    session_id = data.get("session_id", "default")
    user_name = data.get("user_name", "Chief")
    user_sessions[session_id] = []
    return {"reply": f"Memory cleared, {user_name}. Ready for a new directive."}

@app.post("/api/chat")
async def process_command(request: Request):
    data = await request.json()
    raw_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    user_name = data.get("user_name", "").strip() or "Chief"
    lower = raw_message.lower()

    if any(k in lower for k in ["what time is it", "current time", "what's the time", "tell me the time", "time now"]):
        now_time = get_ist_now().strftime("%I:%M %p")
        return {"reply": f"The current time is {now_time} IST, {user_name}."}

    if any(k in lower for k in ["what date is it", "today's date", "what is the date", "what day is today"]):
        now_date = get_ist_now().strftime("%A, %B %d, %Y")
        return {"reply": f"Today is {now_date}, {user_name}."}

    if any(k in lower for k in ["date and time", "time and date", "what date and time"]):
        now_full = get_ist_now().strftime("%A, %B %d, %Y at %I:%M %p")
        return {"reply": f"It is currently {now_full} IST, {user_name}."}

    if any(k in lower for k in ["clear history", "reset memory", "forget everything", "new conversation"]):
        user_sessions[session_id] = []
        return {"reply": f"Memory cleared, {user_name}. We are on a clean slate."}

    if any(greet in lower for greet in ["hello", "hi", "hey", "wake up"]):
        clean_check = re.sub(r"\b(gibbon|given|hey|hi|hello|wake up)\b", "", lower).strip()
        if len(clean_check) < 2:
            greetings = [
                f"Hello {user_name}! Mokuttan Labs core online. How may I assist you today?",
                f"Systems nominal, {user_name}. Standing by for your instructions.",
                f"Gibbon ready, {user_name}! What's on your mind?",
                f"Welcome back, {user_name}. How can I help?"
            ]
            return {"reply": random.choice(greetings)}

    if any(k in lower for k in ["prompt for", "make a prompt", "create a prompt", "video prompt", "midjourney prompt"]):
        idea = re.sub(r"\b(gibbon|given|prompt for|make a prompt for|create a prompt for|video prompt for|generate prompt for)\b", "", raw_message, flags=re.IGNORECASE).strip()
        engineered = engineer_prompt_for_creator(idea or raw_message, session_id, user_name)
        return {"reply": f"Here is your optimized cinematic prompt, {user_name}:\n\n\"{engineered}\""}

    elif any(k in lower for k in ["generate image", "create image", "draw", "render image", "make an image"]):
        clean_idea = re.sub(r"\b(gibbon|given|generate an image of|generate image of|create an image of|draw|render|make an image of)\b", "", raw_message, flags=re.IGNORECASE).strip()
        filename = generate_image_with_fallback(clean_idea or "futuristic cyberpunk neon core")
        if filename:
            return {
                "reply": f"Visual synthesis complete, {user_name}! Rendered based on '{clean_idea}'.",
                "media_type": "image",
                "media_url": f"/media/{filename}"
            }
        else:
            fallback_prompt = engineer_prompt_for_creator(clean_idea, session_id, user_name)
            return {
                "reply": f"Direct renderer was busy, {user_name}. Here is a production-grade prompt for your scene:\n\n\"{fallback_prompt}\""
            }

    elif any(k in lower for k in ["forecast", "weather", "temperature", "rain"]):
        match = re.search(r"(?:in|for|at)\s+([a-zA-Z\s]+)", lower)
        target_city = match.group(1).strip() if match else "Bangalore"
        report = get_live_forecast(target_city, user_name)
        return {"reply": report}

    elif "remind me to" in lower or "remind me" in lower:
        task = re.sub(r"\b(gibbon|given|remind me to|remind me)\b", "", lower).strip()
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO reminders (user_id, text, remind_at) VALUES (?, ?, ?)", 
                  (session_id, task, get_ist_now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        return {"reply": f"Logged to your reminders, {user_name}: '{task}'."}

    elif "reminders" in lower or "my plans" in lower or "schedule" in lower:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT text FROM reminders WHERE user_id = ? AND status = 'pending'", (session_id,))
        rows = c.fetchall()
        conn.close()
        if rows:
            tasks = ", ".join([r[0] for r in rows])
            return {"reply": f"Your pending schedule, {user_name}: {tasks}."}
        return {"reply": f"Your schedule is clear, {user_name}. No pending tasks."}

    clean_prompt = re.sub(r"\b(gibbon|given)\b", "", raw_message, flags=re.IGNORECASE).strip()
    ai_answer = ask_ai_brain(clean_prompt or raw_message, session_id, user_name)
    return {"reply": ai_answer}

@app.get("/api/tts")
async def text_to_speech(text: str):
    spoken_text = text[:360]
    candidate_voices = ["en-US-AvaNeural", "en-US-AriaNeural", "en-US-JennyNeural"]
    audio_data = bytearray()

    for voice_name in candidate_voices:
        try:
            communicate = edge_tts.Communicate(
                spoken_text, 
                voice=voice_name, 
                rate="+1%", 
                pitch="+2Hz",
                volume="+0%"
            )
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
