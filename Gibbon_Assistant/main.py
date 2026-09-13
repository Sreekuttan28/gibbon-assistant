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
import edge_tts

app = FastAPI(title="GIBBON AKA ASSISTANT")

MEDIA_DIR = "saved_media"
DB_FILE = "assistant.db"
os.makedirs(MEDIA_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

geolocator = Nominatim(user_agent="gibbon_hud_agent_v5")

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

SYSTEM_INSTRUCTION = (
    "You are Gibbon, a gentle, intelligent, and loyal personal AI assistant modeled after JARVIS. "
    "Always address the user as Master. Keep answers soft, natural, intelligent, and under 3 sentences. "
    "Crucially, maintain context of earlier questions, recommendations, and conversation history."
)

chat_session = None
ai_client = None

def init_chat_session():
    """Initializes the multi-turn session with model fallback protection."""
    global chat_session, ai_client
    current_key = os.getenv("GEMINI_API_KEY")
    if not current_key:
        print("GEMINI_API_KEY is not set in environment.")
        chat_session = None
        return "ERROR: GEMINI_API_KEY environment variable is not set in Render."

    try:
        ai_client = genai.Client(api_key=current_key)
    except Exception as e:
        print(f"Client init failed: {e}")
        chat_session = None
        return f"Client creation error: {str(e)}"

    # Try preferred modern models with fallback
    candidate_models = ["gemini-2.5-flash", "gemini-1.5-flash"]
    last_err = None

    for model_name in candidate_models:
        try:
            chat_session = ai_client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                )
            )
            print(f"Chat session active using model: {model_name}")
            return None
        except Exception as e:
            print(f"Failed initiating {model_name}: {e}")
            last_err = e

    chat_session = None
    return f"Failed connecting to Gemini: {str(last_err)}"

init_chat_session()

GREETING_RESPONSES = [
    "Hey Master, welcome back. All systems are serene and standing by.",
    "Online at your command, Master. What would you like to explore today?",
    "Gibbon core initialized, Master. How may I assist you?",
    "Good to see you, Master. Systems check out clean. What's on your mind?"
]

THINKING_PREFIXES = [
    "Checking that for you, Master... ",
    "On it, Master. ",
    "Scanning the data stream... ",
    "Right away, Master. "
]

def ask_ai_brain(prompt: str) -> str:
    """Answers using multi-turn conversation memory with direct error diagnostics."""
    global chat_session, ai_client

    if not os.getenv("GEMINI_API_KEY"):
        return "Master, GEMINI_API_KEY is not set in Render's Environment settings."

    if chat_session is None:
        err = init_chat_session()
        if err:
            return f"Master, authentication issue: {err}"

    try:
        response = chat_session.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Direct Gemini exception: {type(e).__name__}: {e}")
        # Try re-initializing once on failure
        err = init_chat_session()
        if err:
            return f"Master, connection issue: {err}"
        try:
            response = chat_session.send_message(prompt)
            return response.text.strip()
        except Exception as retry_err:
            return f"Neural link error [{type(retry_err).__name__}]: {str(retry_err)}"

def compute_route_and_distance(origin_str: str, dest_str: str) -> dict:
    """Calculates road distance, travel duration, and route via OpenStreetMap & OSRM."""
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
    """Fetches real-time weather and forecast via Open-Meteo."""
    try:
        loc = geolocator.geocode(city_name, timeout=10)
        if not loc:
            return f"Master, I could not pinpoint coordinates for {city_name}."

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
        print(f"Weather error: {e}")
        return "Telemetry failed to fetch live weather metrics, Master."

def generate_free_image(prompt: str) -> str:
    """Generates an image via Pollinations.ai and saves locally."""
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
    """Searches DuckDuckGo for live tickets or pricing."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return " | ".join([f"{r.get('title', '')}: {r.get('body', '')}" for r in results])
    except Exception as e:
        print(f"Search error: {e}")
    return "Could not retrieve live search data right now."

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/chat")
async def process_command(request: Request):
    data = await request.json()
    raw_message = data.get("message", "").strip()
    lower = raw_message.lower()

    # 1. Reset / Clear Memory
    if any(k in lower for k in ["clear history", "reset memory", "forget everything", "new conversation"]):
        init_chat_session()
        return {"reply": "Memory matrix cleared, Master. We are operating on a clean slate."}

    # 2. Greetings
    if any(greet in lower for greet in ["hello", "hi", "hey", "wake up", "good morning", "good evening"]):
        clean_check = re.sub(r"\b(gibbon|given|hey|hi|hello|good morning|good evening|good afternoon|wake up)\b", "", lower).strip()
        if len(clean_check) < 2:
            return {"reply": random.choice(GREETING_RESPONSES)}

    # 3. Weather & Forecast
    if any(k in lower for k in ["forecast", "weather", "temperature", "rain"]):
        match = re.search(r"(?:in|for|at)\s+([a-zA-Z\s]+)", lower)
        target_city = match.group(1).strip() if match else "Bangalore"
        report = get_live_forecast(target_city)
        return {"reply": f"{random.choice(THINKING_PREFIXES)}{report}"}

    # 4. Distance & Route
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
                f"{prefix}Road distance from {nav['origin']} to {nav['destination']} "
                f"is {nav['distance_km']} km. Estimated travel time is {nav['duration_hrs']} hours via {nav['key_route']}."
            )
            return {"reply": reply_text, "map_link": nav["map_url"]}
        return {"reply": "Please specify origin and destination, Master. Example: 'Distance from Bangalore to Mysore'."}

    # 5. Reminders
    elif "remind me to" in lower or "remind me" in lower:
        task = re.sub(r"\b(gibbon|given|remind me to|remind me)\b", "", lower).strip()
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO reminders (text, remind_at) VALUES (?, ?)", 
                  (task, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        return {"reply": f"Logged to your memory queue, Master: '{task}'."}

    elif "reminders" in lower or "my plans" in lower or "schedule" in lower:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT text FROM reminders WHERE status = 'pending'")
        rows = c.fetchall()
        conn.close()
        if rows:
            tasks = ", ".join([r[0] for r in rows])
            return {"reply": f"Your current pending schedule, Master: {tasks}."}
        return {"reply": "Your schedule is clear right now, Master. No pending reminders."}

    # 6. Image Generation
    elif any(k in lower for k in ["generate image", "create an image", "draw", "make an image"]):
        prompt = re.sub(r"\b(gibbon|given|generate an image of|generate image of|create an image of|draw|make an image of)\b", "", lower).strip()
        filename = generate_free_image(prompt)
        if filename:
            return {
                "reply": f"Visual synthesis complete, Master. Rendered as {filename}.",
                "media_type": "image",
                "media_url": f"/media/{filename}"
            }
        return {"reply": "Image rendering pipeline encountered an issue. Please try again."}

    # 7. Live Fare Search
    elif any(k in lower for k in ["ticket", "flight", "bus", "train", "fare", "cheap price", "compare"]):
        search_query = re.sub(r"\b(gibbon|given)\b", "", raw_message, flags=re.IGNORECASE).strip()
        search_summary = search_live_web(f"{search_query} fare price booking")
        prefix = random.choice(THINKING_PREFIXES)
        return {"reply": f"{prefix}Here is the latest fare info: {search_summary[:280]}..."}

    # 8. General Knowledge & Follow-ups
    clean_prompt = re.sub(r"\b(gibbon|given)\b", "", raw_message, flags=re.IGNORECASE).strip()
    ai_answer = ask_ai_brain(clean_prompt or raw_message)
    return {"reply": ai_answer}

@app.get("/api/tts")
async def text_to_speech(text: str):
    communicate = edge_tts.Communicate(
        text, 
        voice="en-GB-LibbyNeural", 
        rate="-6%", 
        pitch="-3Hz",
        volume="-20%"
    )
    audio_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])
    return Response(content=bytes(audio_data), media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
