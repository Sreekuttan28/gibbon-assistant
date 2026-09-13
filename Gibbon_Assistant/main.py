import os
import re
import time
import random
import sqlite3
import requests
from urllib.parse import quote
from datetime import datetime
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

geolocator = Nominatim(user_agent="gibbon_hud_agent_v17")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    remind_at TEXT,
                    status TEXT DEFAULT 'pending'
                )''')
    conn.commit()
    conn.close()

init_db()

def get_dynamic_system_instruction() -> str:
    now_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    return (
        "You are Gibbon, a bright, melodic, and intelligent personal AI assistant with a natural human female voice. "
        "Engineered and deployed by Mokuttan Labs. Always address the user as Chief. "
        "Keep answers warm, clear, conversational, and under 3 sentences. "
        f"The current real-world date and time is {now_str}. "
        "Maintain context of earlier questions, recommendations, and conversation history."
    )

def get_gemini_keys():
    keys_str = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    return [k.strip() for k in keys_str.split(",") if k.strip()]

gemini_key_index = 0
conversation_history = []

def reset_memory():
    global conversation_history
    conversation_history = []

def query_gemini(prompt: str, key: str) -> str:
    client = genai.Client(api_key=key)
    candidate_models = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
    last_err = None

    for m in candidate_models:
        try:
            chat = client.chats.create(
                model=m,
                config=types.GenerateContentConfig(
                    system_instruction=get_dynamic_system_instruction(),
                    tools=[{"google_search": {}}]
                )
            )

            for turn in conversation_history[-4:]:
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

def query_groq(prompt: str) -> str:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not set.")

    client = Groq(api_key=groq_key)
    messages = [{"role": "system", "content": get_dynamic_system_instruction()}]
    for turn in conversation_history[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": prompt})

    candidate_groq_models = [
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant"
    ]
    last_err = None

    for model_name in candidate_groq_models:
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.6,
                max_tokens=220
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as err:
            print(f"Groq {model_name} attempt failed: {err}")
            last_err = err
            continue

    raise last_err or RuntimeError("All Groq models failed.")

def ask_ai_brain(prompt: str) -> str:
    global gemini_key_index, conversation_history
    gemini_keys = get_gemini_keys()

    if gemini_keys:
        for _ in range(len(gemini_keys)):
            current_key = gemini_keys[gemini_key_index]
            try:
                reply = query_gemini(prompt, current_key)
                conversation_history.append({"role": "user", "content": prompt})
                conversation_history.append({"role": "assistant", "content": reply})
                return reply
            except Exception as e:
                print(f"Gemini key {gemini_key_index + 1} exhausted: {e}")
                gemini_key_index = (gemini_key_index + 1) % len(gemini_keys)

    if os.getenv("GROQ_API_KEY"):
        try:
            print("Routing to Groq failover engine...")
            reply = query_groq(prompt)
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as groq_err:
            print(f"Groq failover exception: {groq_err}")

    return "Apologies Chief, both primary and backup cognitive links are temporarily rate-limited. Please allow 30 seconds."

def compute_route_and_distance(origin_str: str, dest_str: str) -> dict:
    try:
        loc1 = geolocator.geocode(origin_str, timeout=10)
        loc2 = geolocator.geocode(dest_str, timeout=10)
        if not loc1 or not loc2:
            return {"error": f"Coordinates unverified for '{origin_str}' or '{dest_str}'."}

        url = (
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{loc1.longitude},{loc1.latitude};{loc2.longitude},{loc2.latitude}"
            f"?overview=false&steps=true"
        )
        res = requests.get(url, timeout=12).json()
        if res.get("code") != "Ok":
            return {"error": "Routing calculation failed on highway grid."}

        route = res["routes"][0]
        distance_km = round(route["distance"] / 1000, 1)
        duration_hrs = round(route["duration"] / 3600, 1)

        key_roads = []
        for step in route["legs"][0]["steps"]:
            name = step.get("name")
            if name and name not in key_roads and not name.startswith("Unnamed"):
                key_roads.append(name)

        summary_route = " ➔ ".join(key_roads[:3]) if key_roads else "Direct highway"
        gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={loc1.latitude},{loc1.longitude}&destination={loc2.latitude},{loc2.longitude}"

        return {
            "origin": loc1.address.split(",")[0],
            "destination": loc2.address.split(",")[0],
            "distance_km": distance_km,
            "duration_hrs": duration_hrs,
            "key_route": summary_route,
            "map_url": gmaps_url
        }
    except Exception as e:
        return {"error": str(e)}

def get_live_forecast(city_name: str) -> str:
    try:
        loc = geolocator.geocode(city_name, timeout=10)
        if not loc:
            return f"Chief, I could not pinpoint coordinates for {city_name}."

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={loc.latitude}&longitude={loc.longitude}"
            f"&current_weather=true&daily=temperature_2m_max,temperature_2m_min,weathercode"
            f"&timezone=auto"
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
    except Exception as e:
        return "Telemetry failed to fetch live weather metrics, Chief."

def generate_free_image(prompt: str) -> str:
    try:
        encoded_prompt = quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
        response = requests.get(url, timeout=45)
        if response.status_code == 200:
            filename = f"gibbon_img_{int(time.time())}.png"
            filepath = os.path.join(MEDIA_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            return filename
    except Exception as e:
        print(f"Image error: {e}")
    return None

def search_live_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return " | ".join([f"{r.get('title', '')}: {r.get('body', '')}" for r in results])
    except Exception as e:
        print(f"Search error: {e}")
    return "Could not retrieve live search data right now."

GREETING_RESPONSES = [
    "Hello Chief! Mokuttan Labs core is online. What can I do for you today?",
    "Systems are ready, Chief. Mokuttan Labs standing by for your instructions.",
    "Gibbon online. How may I assist you today, Chief?",
    "Ready when you are, Chief! What's on your mind?"
]

THINKING_PREFIXES = [
    "Checking that now, Chief... ",
    "On it, Chief. ",
    "Looking that up for you... ",
    "One moment, Chief. "
]

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/clear")
def clear_conversation():
    reset_memory()
    return {"reply": "Memory cleared, Chief. Ready for a new directive."}

@app.post("/api/chat")
async def process_command(request: Request):
    data = await request.json()
    raw_message = data.get("message", "").strip()
    lower = raw_message.lower()

    if any(k in lower for k in ["what time is it", "current time", "what's the time", "tell me the time"]):
        now_time = datetime.now().strftime("%I:%M %p")
        return {"reply": f"The current time is {now_time}, Chief."}

    if any(k in lower for k in ["what date is it", "today's date", "what is the date", "what day is today", "what's the date"]):
        now_date = datetime.now().strftime("%A, %B %d, %Y")
        return {"reply": f"Today is {now_date}, Chief."}

    if any(k in lower for k in ["clear history", "reset memory", "forget everything", "new conversation", "clear conversation"]):
        reset_memory()
        return {"reply": "Memory cleared, Chief. We are on a clean slate."}

    if any(greet in lower for greet in ["hello", "hi", "hey", "wake up", "good morning", "good evening"]):
        clean_check = re.sub(r"\b(gibbon|given|hey|hi|hello|good morning|good evening|good afternoon|wake up)\b", "", lower).strip()
        if len(clean_check) < 2:
            return {"reply": random.choice(GREETING_RESPONSES)}

    if any(k in lower for k in ["forecast", "weather", "temperature", "rain"]):
        match = re.search(r"(?:in|for|at)\s+([a-zA-Z\s]+)", lower)
        target_city = match.group(1).strip() if match else "Bangalore"
        report = get_live_forecast(target_city)
        return {"reply": f"{random.choice(THINKING_PREFIXES)}{report}"}

    elif "distance" in lower or "route" in lower or "how far" in lower:
        match = re.search(r"from\s+([a-zA-Z0-9\s,]+?)\s+to\s+([a-zA-Z0-9\s,]+)", lower)
        if not match:
            match = re.search(r"between\s+([a-zA-Z0-9\s,]+?)\s+and\s+([a-zA-Z0-9\s,]+)", lower)

        if match:
            origin = match.group(1).strip()
            destination = match.group(2).replace("?", "").strip()
            nav = compute_route_and_distance(origin, destination)
            if "error" in nav:
                return {"reply": f"Navigation error: {nav['error']}"}

            prefix = random.choice(THINKING_PREFIXES)
            reply_text = (
                f"{prefix}The road distance from {nav['origin']} to {nav['destination']} "
                f"is {nav['distance_km']} km. Travel time is approximately {nav['duration_hrs']} hours via {nav['key_route']}."
            )
            return {"reply": reply_text, "map_link": nav["map_url"]}
        return {"reply": "Please specify both the origin and destination, Chief. Example: 'Distance from Bangalore to Mysore'."}

    elif "remind me to" in lower or "remind me" in lower:
        task = re.sub(r"\b(gibbon|given|remind me to|remind me)\b", "", lower).strip()
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO reminders (text, remind_at) VALUES (?, ?)", 
                  (task, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        return {"reply": f"Logged to your reminders, Chief: '{task}'."}

    elif "reminders" in lower or "my plans" in lower or "schedule" in lower:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT text FROM reminders WHERE status = 'pending'")
        rows = c.fetchall()
        conn.close()
        if rows:
            tasks = ", ".join([r[0] for r in rows])
            return {"reply": f"Your pending schedule, Chief: {tasks}."}
        return {"reply": "Your schedule is clear, Chief. No pending tasks."}

    elif any(k in lower for k in ["generate image", "create an image", "draw", "make an image"]):
        prompt = re.sub(r"\b(gibbon|given|generate an image of|generate image of|create an image of|draw|make an image of)\b", "", lower).strip()
        filename = generate_free_image(prompt)
        if filename:
            return {
                "reply": f"Visual synthesis complete, Chief. Saved as {filename}.",
                "media_type": "image",
                "media_url": f"/media/{filename}"
            }
        return {"reply": "Image rendering encountered an issue. Please try again."}

    elif any(k in lower for k in ["ticket", "flight", "bus", "train", "fare", "cheap price", "compare"]):
        search_query = re.sub(r"\b(gibbon|given)\b", "", raw_message, flags=re.IGNORECASE).strip()
        search_summary = search_live_web(f"{search_query} fare price booking")
        prefix = random.choice(THINKING_PREFIXES)
        return {"reply": f"{prefix}Here is the latest fare info: {search_summary[:280]}..."}

    clean_prompt = re.sub(r"\b(gibbon|given)\b", "", raw_message, flags=re.IGNORECASE).strip()
    ai_answer = ask_ai_brain(clean_prompt or raw_message)
    return {"reply": ai_answer}

@app.get("/api/tts")
async def text_to_speech(text: str):
    spoken_text = text[:360]
    # AvaNeural provides organic breath, melodic cadence, and human warmth
    # Pitch tuned to +2Hz to give it a lighter, musical quality
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
            print(f"TTS {voice_name} error: {e}")
            continue

    return Response(content=bytes(audio_data), media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
