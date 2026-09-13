import os
import re
import time
import random
import sqlite3
import requests
from urllib.parse import quote
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
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

geolocator = Nominatim(user_agent="gibbon_hud_agent_v22")
IST = ZoneInfo("Asia/Kolkata")

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

user_conversation_histories = defaultdict(list)

def search_live_web(query: str) -> str:
    """Searches live DuckDuckGo web for real-time grounding."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
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
        f"You are Gibbon, a friendly, modern personal AI assistant engineered by Mokuttan Labs. "
        f"The user's name is {call_name}. Address the user naturally by {call_name}. "
        f"The current real-world date and time is {now_str} (Indian Standard Time). "
        "CRITICAL GUIDELINES: "
        "1. Write in clear, straightforward English that is effortless to understand for everyone. "
        "2. When asked for recent tech, products, or current news, base your answers on verified real-world facts rather than outdated historical data. "
        "3. Explain things completely and logically. Do not artificially truncate your response. "
        "4. TIMING & SCHEDULES: Calculate step-by-step arithmetic. If someone sleeps at 2:30 AM, 7.5 to 8 hours of sleep means waking between 10:00 AM and 10:30 AM."
    )
    if live_context:
        instruction += f"\n\nLIVE SEARCH GROUNDING DATA:\n{live_context}"
    return instruction

def get_gemini_keys():
    keys_str = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    return [k.strip() for k in keys_str.split(",") if k.strip()]

gemini_key_index = 0

def query_gemini(prompt: str, key: str, user_id: str, user_name: str, live_context: str = "") -> str:
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

            history = user_conversation_histories[user_id]
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
            print(f"Gemini {m} failed: {err_str}")
            last_err = e
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                raise e
            continue

    raise last_err or RuntimeError("Gemini models failed.")

def query_groq(prompt: str, user_id: str, user_name: str, live_context: str = "") -> str:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not set.")

    client = Groq(api_key=groq_key)
    messages = [{"role": "system", "content": get_dynamic_system_instruction(user_name, live_context)}]
    history = user_conversation_histories[user_id]
    for turn in history[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": prompt})

    candidate_groq_models = ["openai/gpt-oss-20b", "llama-3.1-8b-instant"]
    last_err = None

    for model_name in candidate_groq_models:
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.3,
                max_tokens=900
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as err:
            print(f"Groq {model_name} attempt failed: {err}")
            last_err = err
            continue

    raise last_err or RuntimeError("All Groq models failed.")

def ask_ai_brain(prompt: str, user_id: str, user_name: str, live_context: str = "") -> str:
    global gemini_key_index
    gemini_keys = get_gemini_keys()

    if gemini_keys:
        for _ in range(len(gemini_keys)):
            current_key = gemini_keys[gemini_key_index]
            try:
                reply = query_gemini(prompt, current_key, user_id, user_name, live_context)
                user_conversation_histories[user_id].append({"role": "user", "content": prompt})
                user_conversation_histories[user_id].append({"role": "assistant", "content": reply})
                return reply
            except Exception as e:
                print(f"Gemini key {gemini_key_index + 1} exhausted: {e}")
                gemini_key_index = (gemini_key_index + 1) % len(gemini_keys)

    if os.getenv("GROQ_API_KEY"):
        try:
            reply = query_groq(prompt, user_id, user_name, live_context)
            user_conversation_histories[user_id].append({"role": "user", "content": prompt})
            user_conversation_histories[user_id].append({"role": "assistant", "content": reply})
            return reply
        except Exception as groq_err:
            print(f"Groq failover exception: {groq_err}")

    call_name = user_name.strip() if user_name and user_name.strip() else "Chief"
    return f"I am sorry {call_name}, the AI connection is busy right now. Please try again in a few seconds."

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

def engineer_prompt_for_creator(raw_idea: str, user_id: str, user_name: str) -> str:
    instruction = (
        f"Turn this idea into a clear, high-quality image and video prompt: '{raw_idea}'. "
        "Keep the words simple, describe lighting, colors, camera view, and details in simple English under 50 words."
    )
    return ask_ai_brain(instruction, user_id, user_name)

def get_live_forecast(city_name: str, user_name: str) -> str:
    call_name = user_name.strip() if user_name and user_name.strip() else "Chief"
    try:
        loc = geolocator.geocode(city_name, timeout=10)
        if not loc:
            return f"{call_name}, I could not pinpoint coordinates for {city_name}."

        # Reliable Open-Meteo Current & Daily API query
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={loc.latitude}&longitude={loc.longitude}"
            f"&current=temperature_2m,wind_speed_10m"
            f"&daily=temperature_2m_max,temperature_2m_min"
            f"&timezone=Asia%2FKolkata"
        )
        res = requests.get(url, timeout=10).json()

        # Handle both modern 'current' and fallback 'current_weather'
        current_data = res.get("current") or res.get("current_weather", {})
        temp = current_data.get("temperature_2m")
        if temp is None:
            temp = current_data.get("temperature", "--")

        wind = current_data.get("wind_speed_10m")
        if wind is None:
            wind = current_data.get("windspeed", "--")

        daily_data = res.get("daily", {})
        max_temps = daily_data.get("temperature_2m_max", [])
        min_temps = daily_data.get("temperature_2m_min", [])

        max_t = max_temps[0] if max_temps else temp
        min_t = min_temps[0] if min_temps else temp

        city_clean = loc.address.split(",")[0]
        return (
            f"Right now in {city_clean}, the temperature is {temp}°C with wind speed around {wind} km/h. "
            f"Today's high is {max_t}°C and the low is {min_t}°C."
        )
    except Exception as e:
        print(f"Weather error: {e}")
        return f"Could not check the live weather right now, {call_name}."

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/clear")
async def clear_conversation(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default_user")
    user_name = data.get("user_name", "Chief").strip() or "Chief"
    user_conversation_histories[user_id] = []
    return {"reply": f"Your chat memory is cleared, {user_name}! Ready for your next question."}

@app.post("/api/chat")
async def process_command(request: Request):
    data = await request.json()
    raw_message = data.get("message", "").strip()
    user_id = data.get("user_id", "default_user")
    user_name = data.get("user_name", "Chief").strip() or "Chief"
    lower = raw_message.lower()

    if any(k in lower for k in ["what time is it", "current time", "what's the time", "tell me the time", "time now"]):
        now_time = get_ist_now().strftime("%I:%M %p")
        return {"reply": f"The current time is {now_time} IST, {user_name}."}

    if any(k in lower for k in ["what date is it", "today's date", "what is the date", "what day is today", "what's the date"]):
        now_date = get_ist_now().strftime("%A, %B %d, %Y")
        return {"reply": f"Today is {now_date}, {user_name}."}

    if any(k in lower for k in ["date and time", "time and date", "what date and time"]):
        now_full = get_ist_now().strftime("%A, %B %d, %Y at %I:%M %p")
        return {"reply": f"It is currently {now_full} IST, {user_name}."}

    if any(k in lower for k in ["clear history", "reset memory", "forget everything", "new conversation"]):
        user_conversation_histories[user_id] = []
        return {"reply": f"Your chat memory is cleared, {user_name}. We have a clean start."}

    if any(greet in lower for greet in ["hello", "hi", "hey", "wake up"]):
        clean_check = re.sub(r"\b(gibbon|given|hey|hi|hello|wake up)\b", "", lower).strip()
        if len(clean_check) < 2:
            return {"reply": f"Hello {user_name}! Mokuttan Labs core is ready. What can I do for you today?"}

    if any(k in lower for k in ["prompt for", "make a prompt", "create a prompt", "video prompt", "midjourney prompt"]):
        idea = re.sub(r"\b(gibbon|given|prompt for|make a prompt for|create a prompt for|video prompt for|generate prompt for)\b", "", raw_message, flags=re.IGNORECASE).strip()
        engineered = engineer_prompt_for_creator(idea or raw_message, user_id, user_name)
        return {"reply": f"Here is a simple and clear prompt you can copy and use, {user_name}:\n\n\"{engineered}\""}

    elif any(k in lower for k in ["generate image", "create image", "draw", "render image", "make an image"]):
        clean_idea = re.sub(r"\b(gibbon|given|generate an image of|generate image of|create an image of|draw|render|make an image of)\b", "", raw_message, flags=re.IGNORECASE).strip()
        filename = generate_image_with_fallback(clean_idea or "futuristic glowing core")
        if filename:
            return {
                "reply": f"Here is the picture I created for you, {user_name}!",
                "media_type": "image",
                "media_url": f"/media/{filename}"
            }
        else:
            fallback_prompt = engineer_prompt_for_creator(clean_idea, user_id, user_name)
            return {
                "reply": f"The direct image maker was busy right now, {user_name}. But here is a ready-to-use prompt you can use:\n\n\"{fallback_prompt}\""
            }

    elif any(k in lower for k in ["forecast", "weather", "temperature", "rain"]):
        match = re.search(r"(?:in|for|at)\s+([a-zA-Z\s]+)", lower)
        target_city = match.group(1).strip() if match else "Bengaluru"
        report = get_live_forecast(target_city, user_name)
        return {"reply": report}

    elif "remind me to" in lower or "remind me" in lower:
        task = re.sub(r"\b(gibbon|given|remind me to|remind me)\b", "", lower).strip()
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO reminders (user_id, text, remind_at) VALUES (?, ?, ?)", 
                  (user_id, task, get_ist_now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        return {"reply": f"Saved in your reminders, {user_name}: '{task}'."}

    elif "reminders" in lower or "my plans" in lower or "schedule" in lower:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT text FROM reminders WHERE user_id = ? AND status = 'pending'", (user_id,))
        rows = c.fetchall()
        conn.close()
        if rows:
            tasks = ", ".join([r[0] for r in rows])
            return {"reply": f"Here are your pending reminders, {user_name}: {tasks}."}
        return {"reply": f"Your reminder list is completely empty right now, {user_name}."}

    # AUTOMATIC REAL-TIME GROUNDING FOR LATEST INFORMATION
    live_context = ""
    needs_live_data = any(w in lower for w in [
        "latest", "newest", "current", "release", "released", "launch", 
        "price", "who is", "news", "specs", "phone", "iphone", "apple", "samsung"
    ])

    if needs_live_data:
        search_query = re.sub(r"\b(gibbon|given|hey|hi)\b", "", raw_message, flags=re.IGNORECASE).strip()
        current_year = get_ist_now().strftime("%Y")
        live_context = search_live_web(f"{search_query} {current_year}")

    clean_prompt = re.sub(r"\b(gibbon|given)\b", "", raw_message, flags=re.IGNORECASE).strip()
    ai_answer = ask_ai_brain(clean_prompt or raw_message, user_id, user_name, live_context)
    return {"reply": ai_answer}

@app.get("/api/tts")
async def text_to_speech(text: str):
    # Strip markdown symbols, pipes, bullets, code fences, and dashes so neural audio reads seamlessly
    clean_text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    clean_text = re.sub(r'[*#_`|~>—–-]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # Support full-length detailed answers without premature truncation
    spoken_text = clean_text[:4000]

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
        except Exception as e:
            print(f"TTS attempt error: {e}")
            continue

    return Response(content=bytes(audio_data), media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
