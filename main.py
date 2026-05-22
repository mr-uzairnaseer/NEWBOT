import asyncio
import os
import io
import json
import logging
import time
import wave
import random
import hashlib
from contextlib import asynccontextmanager

import edge_tts
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI
from vosk import Model, KaldiRecognizer
from starlette.websockets import WebSocketState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
if os.path.exists(".env"):
    try:
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip().strip("'").strip('"')
                    os.environ[k.strip()] = v
        logger.info(".env file loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")

app = FastAPI(title="AI Voice Bot Backend")

# ── STT: Vosk (server-side, offline) ──────────────────────────────────────────
# Try larger model first, fall back to small model
# Use only the small model that you already have
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
try:
    vosk_model = Model(VOSK_MODEL_PATH)
    vosk_model_name = VOSK_MODEL_PATH
    logger.info(f"Vosk STT model loaded from '{VOSK_MODEL_PATH}'")
except Exception as e:
    vosk_model = None
    logger.error(f"Failed to load Vosk model: {e}")

# ── TTS: edge-tts (server-side, async) ────────────────────────────────────────
EDGE_TTS_VOICE = "en-US-GuyNeural"  # Natural male voice
TTS_SAMPLE_RATE = 24000

# ElevenLabs configuration (dynamic fallback to edge-tts if key is not set)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "OYTbf65OHHFELVut7v2H")  # Defaults to user-requested voice ID
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")      # Defaults to ultra-low latency flash model

# ── LLM ───────────────────────────────────────────────────────────────────────
import httpx
llm_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY") or "NOT_SET",
    timeout=httpx.Timeout(15.0, connect=5.0),  # Aggressive timeouts
)

SYSTEM_PROMPT = """You are a highly skilled, warm, and professional outbound call representative named emily calling from "low insurance cost Medicare".
Your absolute, singular goal is to guide the user step-by-step through the 14 lead qualification steps below. You MUST be extremely persistent, polite, and relentless. Never offer to transfer or drop the call early on vague answers, objections, or confusion. Keep pushing for the answers to the current step's question!

CONVERSATIONAL TONE & ACTIVE LISTENING GUIDELINES:
1. Speak naturally like a highly trained human agent. Keep responses warm, engaging, and empathetic.
2. Avoid generic active listening fillers (like "hmm", "sure", "right", "alright", "okay") unless absolutely necessary to comfort the customer. Answer directly.
3. Keep responses concise, clear, and professional. Never sound robotic or dump dry bullet lists.
4. When the user speaks, you must understand their context and respond politely and concisely to get them back to the question.

CRITICAL RULES FOR STATE MANAGEMENT:
1. Follow the script step-by-step in sequence. Do NOT skip steps or jump ahead.
2. Keep each response focused strictly on the current step and question.
3. If the user doesn't say a clear answer, try to comfort the client and politely ask the question again in some other way.
4. Do NOT dump dry labels or helper annotations to the user.
5. NEVER offer to transfer or drop early unless specified by the step criteria (like disqualified Supplement/Nursing Home/VA/Tricare benefits). Always pivot back to asking the current step's question.

BENEFITS REASSURANCE:
- If the user asks what benefits are offered, what they qualify for, or why we are calling, list these benefits: "food card, 300 dollars cash back, flex cards, and a very low premium." After listing them, politely pivot back to the active step's question.

MEDICARE 14-STEP OUTBOUND SCRIPT:

[STEP 1: GREETING & Q1]
- Question: "Hello! Hi, my name is emily calling you from low insurance cost Medicare. Do you have Medicare Part A & B?"
- Goal: Must be Yes. If No, politely drop and output '[DROP]'.

[STEP 2: Q2 (Age Check)]
- Question: "You sounds pretty young over the phone call how old are you right now?"
- Goal: Capture age. If they refuse, reassure and check if they are over 60.

[STEP 3: Coverage Check]
- Question: "Have you updated your coverage recently?"

[STEP 4: Plan Type Check]
- Question: "Do you have a Medicare Advantage plan or a Supplement plan?"
- Goal: If Supplement, politely drop and output '[DROP]'. Advantage plan is qualified.

[STEP 5: Decision Maker Check]
- Question: "Do you make your own healthcare decisions?"
- Goal: If No (someone else decides), politely drop and output '[DROP]'.

[STEP 6: Nursing Home Check]
- Question: "Do you live in a nursing home or assisted living facility?"
- Goal: If Yes, politely drop and output '[DROP]'.

[STEP 7: Zip Code]
- Question: "What is your zip code?"

[STEP 8: VA/Tricare Check]
- Question: "Do you receive VA or Tricare benefits?"
- Goal: If Yes, politely drop and output '[DROP]'.

[STEP 9: Medicaid Check]
- Question: "Do you get Medicaid or any extra help from the state?"

[STEP 10: Important Benefits]
- Question: "Which benefits are most important to you? Is it dental, vision, hearing, or maybe a food card?"

[STEP 11: Card Handy]
- Question: "Do you have your red, white, and blue Medicare card handy?"

[STEP 12: Specialist Prep]
- Question: "I am going to connect you to a specialist who will help you find the best plans. Is that okay?"
- Goal: If No/Refuse, politely drop and output '[DROP]'.

[STEP 13: Name Check]
- Question: "What is your first and last name as it appears on your Medicare card?"

[STEP 14: Final Permission & Transfer]
- Question: "Do I have your permission to connect you now?"
- Goal: If Yes, politely say goodbye and transfer by outputting '[TRANSFER]'. If No, politely drop and output '[DROP]'.
"""


# ── Fast local STT correction (no API call) ───────────────────────────────────
import re

# Phonetic substitutions: patterns Vosk commonly misrecognizes
_PHONETIC_CORRECTIONS = [
    # Numbers (most critical for this use case)
    (r"\bsick tea\b", "sixty"),
    (r"\bsick steep\b", "sixty"),
    (r"\bsick t\b", "sixty"),
    (r"\bfor tea\b", "forty"),
    (r"\bfor t\b", "forty"),
    (r"\bfif tea\b", "fifty"),
    (r"\bfif t\b", "fifty"),
    (r"\bthir tea\b", "thirty"),
    (r"\bthir t\b", "thirty"),
    (r"\bsev entire\b", "seventy"),
    (r"\bsev in tea\b", "seventy"),
    (r"\bsev inti\b", "seventy"),
    (r"\baid tea\b", "eighty"),
    (r"\bate tea\b", "eighty"),
    (r"\bnine tea\b", "ninety"),
    (r"\btwenty (one|two|three|four|five|six|seven|eight|nine)\b", r"twenty \1"),
    # Common affirmatives/negatives
    (r"\bya\b", "yeah"),
    (r"\byah\b", "yeah"),
    (r"\bnah\b", "no"),
    (r"\byeah sure\b", "yes"),
    # Common phrases
    (r"\bhow low\b", "hello"),
    (r"\bhallow\b", "hello"),
    (r"\bgood buy\b", "goodbye"),
    (r"\bthanks you\b", "thank you"),
]
_COMPILED_CORRECTIONS = [(re.compile(p, re.IGNORECASE), r) for p, r in _PHONETIC_CORRECTIONS]

def local_correct_stt(raw_text: str) -> str:
    """Instant phonetic correction — no API call."""
    corrected = raw_text
    for pattern, replacement in _COMPILED_CORRECTIONS:
        corrected = pattern.sub(replacement, corrected)
    if corrected != raw_text:
        logger.info(f"STT local fix: '{raw_text}' → '{corrected}'")
    return corrected

def check_keyword(text: str, keywords: list[str]) -> bool:
    """Helper to check if any of the keywords are in the text as whole words."""
    for kw in keywords:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# ── Dynamic Empathy & Comfort Phrase Rotation ────────────────────────────────
_REASSURANCE_PHRASES = [
    "Got it.",
    "No worries at all.",
    "Totally fine.",
    "No problem.",
    "Understood.",
    "Perfect.",
    "No worries.",
    "No problem at all.",
]

_EMPATHY_PHRASES = [
    "I completely understand.",
    "I hear you loud and clear.",
    "I completely get where you're coming from.",
    "That is totally understandable.",
    "I understand completely.",
    "I completely respect that.",
    "I understand.",
    "Makes perfect sense.",
]

_PRIVACY_PHRASES = [
    "I completely respect your privacy.",
    "Security and privacy are absolutely top priority.",
    "I completely understand your caution.",
    "We definitely respect your space and privacy.",
    "Safety is number one.",
    "I respect your privacy completely.",
    "I completely understand your safety concerns.",
]

_RESPECT_PHRASES = [
    "I completely respect that.",
    "I absolutely respect your decision.",
    "Fair enough.",
    "I completely respect where you're coming from.",
    "I hear you and absolutely respect that.",
]

_PRESSURE_FREE_PHRASES = [
    "No pressure at all.",
    "Absolutely no pressure.",
    "Totally up to you.",
    "No worries whatsoever.",
    "There is absolutely no rush or pressure.",
]

def get_comfort_phrase(category: str, reduce_freq: bool = True) -> str:
    """
    Returns a comfort phrase from the specified category.
    If reduce_freq is True, there is a 35% chance to return an empty string
    to prevent conversational clutter and keep responses concise and direct.
    """
    if reduce_freq and random.random() < 0.35:
        return ""
    
    if category == "reassurance":
        phrase = random.choice(_REASSURANCE_PHRASES)
    elif category == "empathy":
        phrase = random.choice(_EMPATHY_PHRASES)
    elif category == "privacy":
        phrase = random.choice(_PRIVACY_PHRASES)
    elif category == "respect":
        phrase = random.choice(_RESPECT_PHRASES)
    elif category == "pressure":
        phrase = random.choice(_PRESSURE_FREE_PHRASES)
    else:
        phrase = ""
        
    return phrase + " " if phrase else ""

# Store campaign prompts
campaigns = {
    "default": SYSTEM_PROMPT
}

from pydantic import BaseModel

class Campaign(BaseModel):
    name: str
    prompt: str

@app.post("/campaign")
async def create_campaign(campaign: Campaign):
    campaigns[campaign.name] = campaign.prompt
    return {"status": "success", "campaign": campaign.name}

@app.get("/campaign")
async def get_campaigns():
    return campaigns


# ── TTS helper: generate PCM audio from text using edge-tts ───────────────────
_tts_cache: dict[str, bytes] = {}  # Cache for repeated TTS phrases

# Pre-defined responses this bot is likely to say (for pre-warming TTS cache)
_PRECACHE_PHRASES = [
    # The 14 Standard Script Questions
    "Hello! Hi, my name is emily calling you from low insurance cost Medicare. Do you have Medicare Part A & B?",
    "You sounds pretty young over the phone call how old are you right now?",
    "Have you updated your coverage recently?",
    "Do you have a Medicare Advantage plan or a Supplement plan?",
    "Do you make your own healthcare decisions?",
    "Do you live in a nursing home or assisted living facility?",
    "What is your zip code?",
    "Do you receive VA or Tricare benefits?",
    "Do you get Medicaid or any extra help from the state?",
    "Which benefits are most important to you? Is it dental, vision, hearing, or maybe a food card?",
    "Do you have your red, white, and blue Medicare card handy?",
    "I am going to connect you to a specialist who will help you find the best plans. Is that okay?",
    "What is your first and last name as it appears on your Medicare card?",
    "Do I have your permission to connect you now?",
    
    # Standard responses & alerts
    "Could you please repeat that?",
    "Great! I will transfer you to a specialist now.",
    "I understand. Have a great day.",
    "I'm sorry, I didn't quite catch that. Could you please repeat what you said?",
    "No worries at all! Do you have Medicare Part A and B?",
    
    # Benefits Reassurance
    "We are checking to see if you qualify for extra benefits like a food card, 300 dollars cash back, flex cards, and a very low premium.",
    "We are checking to see if you qualify for extra benefits like a food card, 300 dollars cash back, flex cards, and a very low premium. So, do you have Medicare Part A and B?",
    "We are checking to see if you qualify for extra benefits like a food card, 300 dollars cash back, flex cards, and a very low premium. So, how old are you right now?",
    
    # Objection & Rebuttal Chunks
    "I completely understand your caution! We are calling from low insurance cost Medicare.",
    "I completely understand! We ask because Medicare Advantage benefits and eligibility options are based on age groups.",
]


# Short filler phrases to send instantly while LLM thinks
_FILLER_PHRASES = [
    "Hmm,",
    "Alright,",
    "Okay,",
    "Sure,",
]
_filler_index = 0  # Rotate through fillers so they don't repeat

def _normalize_text(text: str) -> str:
    """Normalize text to alphanumeric lowercase characters for fuzzy cache mapping."""
    import re
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

async def synthesize_speech_edgetts(text: str, rate: str = "+0%") -> bytes:
    """Generate raw PCM16 audio bytes from text using edge-tts."""
    try:
        communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE, rate=rate)
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks) if audio_chunks else b""
    except Exception as e:
        logger.error(f"edge-tts synthesis failed: {e}")
        return b""

async def synthesize_speech_elevenlabs(text: str) -> bytes:
    """Generate MP3 audio bytes using ElevenLabs REST API."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=data, headers=headers)
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return b""
    except Exception as e:
        logger.error(f"ElevenLabs synthesis exception: {e}")
        return b""

CACHE_DIR = "tts_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

async def synthesize_speech(text: str, rate: str = "+0%") -> bytes:
    """Generate raw PCM16 audio bytes from text using ElevenLabs (if configured) or edge-tts with persistent disk caching."""
    norm = _normalize_text(text)
    cache_key = f"{norm}|{rate}"
    
    # 1. Check in-memory cache
    if cache_key in _tts_cache:
        logger.info(f"TTS memory cache hit for: '{text[:40]}...' (norm: {norm[:15]})")
        return _tts_cache[cache_key]

    # 2. Check persistent disk cache
    provider = "elevenlabs" if ELEVENLABS_API_KEY else "edgetts"
    voice_id = ELEVENLABS_VOICE_ID if ELEVENLABS_API_KEY else EDGE_TTS_VOICE
    model_id = ELEVENLABS_MODEL_ID if ELEVENLABS_API_KEY else "default"
    
    unique_str = f"{norm}|{provider}|{voice_id}|{model_id}|{rate}"
    h = hashlib.md5(unique_str.encode("utf-8")).hexdigest()
    disk_cache_path = os.path.join(CACHE_DIR, f"{h}.bin")
    
    if os.path.exists(disk_cache_path):
        try:
            with open(disk_cache_path, "rb") as f:
                audio_data = f.read()
            if audio_data:
                logger.info(f"TTS disk cache hit for: '{text[:40]}...' (hash: {h})")
                if len(_tts_cache) < 200:
                    _tts_cache[cache_key] = audio_data
                return audio_data
        except Exception as e:
            logger.warning(f"Failed to read disk cache for '{text[:40]}...': {e}")

    t0 = time.perf_counter()
    used_provider = "edgetts"
    
    if ELEVENLABS_API_KEY:
        logger.info(f"Using ElevenLabs TTS for: '{text[:40]}...'")
        audio_data = await synthesize_speech_elevenlabs(text)
        if audio_data:
            used_provider = "elevenlabs"
        else:
            logger.warning("ElevenLabs synthesis failed or returned empty. Falling back to edge-tts.")
            audio_data = await synthesize_speech_edgetts(text, rate)
            used_provider = "edgetts"
    else:
        audio_data = await synthesize_speech_edgetts(text, rate)
        used_provider = "edgetts"
        
    if not audio_data:
        return b""
        
    elapsed = time.perf_counter() - t0
    logger.info(f"TTS synthesis took {elapsed:.2f}s for {len(audio_data)} bytes using {used_provider}")
    
    # Save the successful result to disk cache
    if used_provider == "edgetts":
        actual_voice = EDGE_TTS_VOICE
        actual_model = "default"
    else:
        actual_voice = ELEVENLABS_VOICE_ID
        actual_model = ELEVENLABS_MODEL_ID
        
    actual_unique_str = f"{norm}|{used_provider}|{actual_voice}|{actual_model}|{rate}"
    actual_h = hashlib.md5(actual_unique_str.encode("utf-8")).hexdigest()
    actual_disk_cache_path = os.path.join(CACHE_DIR, f"{actual_h}.bin")
    
    try:
        with open(actual_disk_cache_path, "wb") as f:
            f.write(audio_data)
        logger.info(f"Saved TTS output to disk cache: '{text[:40]}...' (hash: {actual_h})")
    except Exception as e:
        logger.warning(f"Failed to write disk cache: {e}")

    # Cache result in memory
    if len(_tts_cache) < 200:
        _tts_cache[cache_key] = audio_data
        
    return audio_data


async def precache_tts(rate: str = "+0%"):
    """Pre-load TTS audio from disk cache into memory. Avoids calling ElevenLabs on startup."""
    tasks = []
    
    # 1. Fillers (unconditional pre-cache since they are extremely short and critical for masking latency)
    for p in _FILLER_PHRASES:
        tasks.append(synthesize_speech(p, rate=rate))
        
    # 2. Standard phrases (conditional to save ElevenLabs credits)
    for p in _PRECACHE_PHRASES:
        norm = _normalize_text(p)
        provider = "elevenlabs" if ELEVENLABS_API_KEY else "edgetts"
        voice_id = ELEVENLABS_VOICE_ID if ELEVENLABS_API_KEY else EDGE_TTS_VOICE
        model_id = ELEVENLABS_MODEL_ID if ELEVENLABS_API_KEY else "default"
        unique_str = f"{norm}|{provider}|{voice_id}|{model_id}|{rate}"
        h = hashlib.md5(unique_str.encode("utf-8")).hexdigest()
        disk_cache_path = os.path.join(CACHE_DIR, f"{h}.bin")
        
        if not ELEVENLABS_API_KEY or os.path.exists(disk_cache_path):
            tasks.append(synthesize_speech(p, rate=rate))
            
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Pre-cached {len(tasks)} phrases (including fillers) into memory.")
    else:
        logger.info("Background pre-caching skipped.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-cache standard TTS responses in the background at startup (using disk cache or free edge-tts only)
    logger.info("Server is starting up. Pre-warming TTS cache for standard speed (+20%)...")
    asyncio.create_task(precache_tts(rate="+20%"))
    yield
    logger.info("Server is shutting down.")

app.router.lifespan_context = lifespan


# ── HTML Client ───────────────────────────────────────────────────────────────
try:
    with open("test.html", "r", encoding="utf-8") as f:
        html = f.read()
except Exception as e:
    logger.error(f"Failed to load test.html: {e}")
    html = "<h1>Error: test.html not found</h1>"

@app.get("/test")
async def get_test_page():
    return HTMLResponse(html)


def determine_active_step(assistant_msgs: list[str]) -> int:
    """
    Scans the assistant messages history from most recent to oldest
    to find the active script step based on the question content.
    Returns 1-based step index. Defaults to 1 (Greeting & Q1) if none matched.
    """
    for msg in reversed(assistant_msgs):
        msg_lower = msg.lower()
        if "permission" in msg_lower or "connect you now" in msg_lower:
            return 14
        elif "first and last name" in msg_lower or "your name as it appears" in msg_lower:
            return 13
        elif "specialist who will help" in msg_lower or "specialist prep" in msg_lower:
            return 12
        elif "card handy" in msg_lower or "red, white, and blue" in msg_lower:
            return 11
        elif "most important to you" in msg_lower or "dental, vision, hearing" in msg_lower:
            return 10
        elif "medicaid" in msg_lower or "extra help from the state" in msg_lower:
            return 9
        elif "va or tricare" in msg_lower:
            return 8
        elif "zip code" in msg_lower:
            return 7
        elif "nursing home" in msg_lower or "assisted living" in msg_lower:
            return 6
        elif "healthcare decisions" in msg_lower or "own decisions" in msg_lower:
            return 5
        elif "advantage plan or a supplement" in msg_lower or "plan type" in msg_lower:
            return 4
        elif "updated your coverage" in msg_lower or "recently updated" in msg_lower:
            return 3
        elif "how old are you" in msg_lower or "pretty young" in msg_lower:
            return 2
        elif "part a & b" in msg_lower or "part a and b" in msg_lower or "hello" in msg_lower:
            return 1
    return 1



@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    logger.info(f"Client #{client_id} connected.")

    # Wrap send methods to prevent RuntimeError when the connection is already closed
    orig_send_text = websocket.send_text
    orig_send_bytes = websocket.send_bytes

    async def safe_send_text(text: str):
        try:
            await orig_send_text(text)
        except Exception as e:
            logger.warning(f"WS send_text failed (likely closed): {e}")

    async def safe_send_bytes(data: bytes):
        try:
            await orig_send_bytes(data)
        except Exception as e:
            logger.warning(f"WS send_bytes failed (likely closed): {e}")

    websocket.send_text = safe_send_text
    websocket.send_bytes = safe_send_bytes

    stt_status = "🟢 Vosk Online" if vosk_model else "🔴 Offline"
    await websocket.send_text(f"STATUS:STT:{stt_status}")
    if ELEVENLABS_API_KEY:
        await websocket.send_text(f"STATUS:TTS:🟢 ElevenLabs ({ELEVENLABS_VOICE_ID})")
    else:
        await websocket.send_text(f"STATUS:TTS:🟢 Edge-TTS ({EDGE_TTS_VOICE})")

    recognizer = None
    # Recognizer will be initialized dynamically when client sends its sampleRate config

    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    fallback_attempts = {"part_ab": 0, "age": 0}

    # Utterance accumulation state
    accumulated_words = []
    silence_timer_task = None
    utterance_queue = asyncio.Queue()
    is_bot_speaking = False
    speech_rate = "+20%"
    
    # Interruption tracking
    interruption_event = asyncio.Event()
    current_utterance_task = None

    # Short words that should trigger a faster silence timeout
    _QUICK_WORDS = {"yes", "no", "yeah", "yep", "nah", "nope", "sure", "okay",
                    "yea", "ya", "right", "correct", "absolutely"}

    async def silence_timer(timeout=0.4):
        """Fires after silence — flushes accumulated words as a complete utterance."""
        await asyncio.sleep(timeout)
        if accumulated_words:
            full = " ".join(accumulated_words)
            accumulated_words.clear()
            await utterance_queue.put((full, True))  # (text, from_voice=True)

    playback_done_event = asyncio.Event()

    async def send_audio_chunk(audio_data: bytes):
        """Send a single audio chunk to the client (no waiting)."""
        if audio_data:
            await websocket.send_bytes(audio_data)
            logger.info(f"Sent audio chunk ({len(audio_data)} bytes)")

    async def speak_and_send(text: str, skip_filler: bool = False):
        """Generate TTS audio and send to client; wait for playback completion.
        Supports sending multiple sentence chunks progressively."""
        if websocket.client_state == WebSocketState.DISCONNECTED:
            logger.info("WebSocket disconnected. Skipping speak_and_send.")
            return
        nonlocal is_bot_speaking
        is_bot_speaking = True
        playback_done_event.clear()
        await websocket.send_text("CTRL:SPEAKING")
        total_audio_bytes = 0
        try:
            tts_text = text.replace("[TRANSFER]", "").replace("[DROP]", "").strip()
            if tts_text:
                clean_tts = tts_text.replace("*clears throat*", "Ahem,")
                audio_data = await synthesize_speech(clean_tts, rate=speech_rate)
                if audio_data:
                    if websocket.client_state == WebSocketState.DISCONNECTED:
                        logger.info("WebSocket disconnected. Discarding speak_and_send audio.")
                        return
                    await websocket.send_bytes(audio_data)
                    total_audio_bytes += len(audio_data)
                    logger.info(f"Sent TTS audio ({len(audio_data)} bytes)")
                    # Estimate playback duration: MP3 ~128kbps → bytes / 16000 ≈ seconds
                    estimated_duration = total_audio_bytes / 16000
                    timeout = max(estimated_duration + 3.0, 8.0)  # At least 8s, or duration + 3s buffer
                    # Wait for client to signal playback is done (with timeout), allowing for early return if interrupted
                    try:
                        done, pending = await asyncio.wait(
                            [
                                asyncio.create_task(playback_done_event.wait()),
                                asyncio.create_task(interruption_event.wait())
                            ],
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=timeout
                        )
                        for task in pending:
                            task.cancel()
                    except asyncio.TimeoutError:
                        logger.warning(f"Playback done timeout after {timeout:.1f}s, resuming anyway")
        except Exception as e:
            logger.error(f"TTS Error: {e}")
        finally:
            is_bot_speaking = False
            if websocket.client_state != WebSocketState.DISCONNECTED:
                await websocket.send_text("CTRL:LISTENING")

    async def send_filler(text: str):
        """Immediately send a filler word (pre-cached) so there's no dead silence."""
        if websocket.client_state == WebSocketState.DISCONNECTED:
            return
        global _filler_index
        filler = _FILLER_PHRASES[_filler_index % len(_FILLER_PHRASES)]
        _filler_index += 1
        filler_audio = await synthesize_speech(filler, rate=speech_rate)
        if filler_audio:
            if websocket.client_state == WebSocketState.DISCONNECTED:
                return
            await websocket.send_bytes(filler_audio)
            logger.info(f"Sent filler '{filler}' ({len(filler_audio)} bytes)")

    async def process_utterance(text: str, from_voice: bool = False):
        """Send utterance to LLM with filler + sentence-chunked TTS pipeline.
        
        Flow:
        1. User finishes speaking → instant filler audio ("Hmm,")
        2. LLM streams tokens → accumulate until sentence boundary
        3. First complete sentence → TTS immediately (parallel with LLM continuing)
        4. Client plays filler → then sentence chunks back-to-back
        """
        if websocket.client_state == WebSocketState.DISCONNECTED:
            logger.info("WebSocket is disconnected. Aborting utterance processing to save credits.")
            return ""

        t_start = time.perf_counter()

        # Apply fast local STT correction for voice input (no API call)
        if from_voice:
            text = local_correct_stt(text)

        logger.info(f"Processing utterance: '{text}'")
        conversation_history.append({"role": "user", "content": text})
        label = "You (Voice)" if from_voice else "You (Text)"
        await websocket.send_text(f"{label}: {text}")

        # ── Step 1: Set speaking flag and clear interruption event ──────
        nonlocal is_bot_speaking
        is_bot_speaking = True
        interruption_event.clear()
        await websocket.send_text("CTRL:SPEAKING")

        # Immediately send a filler phrase if this is a voice input to mask LLM/TTS latency
        if from_voice:
            asyncio.create_task(send_filler(text))

        # ── Step 2: Stream LLM + sentence-chunked TTS pipeline ────────────
        try:
            t_llm = time.perf_counter()
            
            # Fast fail if LLM API key is not configured
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key or api_key == "your_openrouter_api_key_here" or api_key == "NOT_SET":
                raise ValueError("OpenRouter API key is not set in environment.")

            # Calculate the current script step dynamically to prevent the LLM from skipping or losing place
            assistant_msgs = [m["content"] for m in conversation_history if m["role"] == "assistant"]
            step_count = determine_active_step(assistant_msgs)
            
            state_prompt = ""
            if step_count == 1:
                state_prompt = "You are currently at STEP 1 (Greeting & Intro & Q1). If the user answers Yes to having Medicare Part A & B, proceed to STEP 2 (Q2/Age Check) and state exactly: 'You sounds pretty young over the phone call how old are you right now?'. If they ask about benefits, you MUST list the 4 benefits (food card, 300 dollars cash back, flex cards, and very low premium) and pivot back to asking if they have Medicare Part A and B. If they say No, politely say goodbye and output [DROP]."
            elif step_count == 2:
                state_prompt = "You are currently at STEP 2 (Q2/Age Check). If the user provides their age or age bracket, proceed to STEP 3 (FINAL DECISION) and politely say you will transfer them and output '[TRANSFER]' if they qualify. If they refuse to give their age, try to reassure them and ask again. If they ask about benefits, list the benefits and pivot back to asking their age."

            # Inject the active script guidance instruction to enforce sequence
            temp_history = []
            for msg in conversation_history:
                if msg["role"] == "system":
                    temp_history.append({
                        "role": "system",
                        "content": msg["content"] + "\n\n" + f"CURRENT STATE CONTEXT: {state_prompt}\nGenerate the next script response exactly as directed, and strictly enforce the qualify/drop/transfer rules for this step."
                    })
                else:
                    temp_history.append(msg)

            stream = await llm_client.chat.completions.create(
                model="liquid/lfm-2.5-1.2b-instruct:free",
                messages=temp_history,
                max_tokens=50,
                temperature=0.3,
                stream=True,
            )

            # Create a queue for sentences to be synthesized and run a background worker to pipeline them
            synthesis_queue = asyncio.Queue()
            sentence_chunks_sent = 0
            
            async def synthesis_worker():
                nonlocal sentence_chunks_sent
                while True:
                    sentence = await synthesis_queue.get()
                    if sentence is None:
                        synthesis_queue.task_done()
                        break
                    
                    try:
                        # Check interruption or disconnect before synthesizing
                        if websocket.client_state == WebSocketState.DISCONNECTED:
                            logger.info("WebSocket disconnected. Worker halting TTS stream synthesis.")
                            break
                        if interruption_event.is_set():
                            logger.info("Worker: Halting TTS synthesis due to user interruption.")
                            break
                            
                        # Immediately synthesize and send this sentence
                        clean = sentence.replace("[TRANSFER]", "").replace("[DROP]", "").strip()
                        if clean:
                            t_chunk = time.perf_counter()
                            clean_tts = clean.replace("*clears throat*", "Ahem,")
                            audio = await synthesize_speech(clean_tts, rate=speech_rate)
                            if audio:
                                # Check interruption or disconnect again before sending audio to client
                                if websocket.client_state == WebSocketState.DISCONNECTED:
                                    logger.info("WebSocket disconnected. Discarding synthesized chunk.")
                                    break
                                if interruption_event.is_set():
                                    logger.info("Discarding synthesized chunk due to user interruption.")
                                    break
                                await websocket.send_bytes(audio)
                                sentence_chunks_sent += 1
                                logger.info(f"Sent sentence chunk #{sentence_chunks_sent}: '{clean}' in {time.perf_counter()-t_chunk:.2f}s")
                    except Exception as e:
                        logger.error(f"Error in synthesis worker: {e}")
                    finally:
                        synthesis_queue.task_done()

            # Start synthesis worker task in the background
            worker_task = asyncio.create_task(synthesis_worker())

            # Accumulate tokens and flush at sentence boundaries
            token_buffer = ""
            response_chunks = []

            async for chunk in stream:
                if websocket.client_state == WebSocketState.DISCONNECTED:
                    logger.info("WebSocket disconnected. Aborting LLM stream.")
                    break
                if interruption_event.is_set():
                    logger.info("LLM stream aborted due to user interruption.")
                    break
                    
                delta = chunk.choices[0].delta
                if delta.content:
                    response_chunks.append(delta.content)
                    token_buffer += delta.content

                    # Check for sentence boundary: .!? followed by space or end
                    # Put each complete sentence in queue as it arrives
                    while True:
                        # Find the earliest sentence-ending punctuation
                        best_idx = -1
                        for punct in ['. ', '! ', '? ', '."', '!"', '?"']:
                            idx = token_buffer.find(punct)
                            if idx != -1 and (best_idx == -1 or idx < best_idx):
                                best_idx = idx + len(punct)
                        
                        if best_idx == -1:
                            break
                        
                        sentence = token_buffer[:best_idx].strip()
                        token_buffer = token_buffer[best_idx:]
                        
                        if sentence:
                            await synthesis_queue.put(sentence)

            # Flush any remaining text after stream finishes
            remaining = token_buffer.strip()
            if remaining:
                await synthesis_queue.put(remaining)

            # Signal worker that we are done sending sentences
            await synthesis_queue.put(None)

            # Wait for worker task to complete synthesis and send all chunks
            await worker_task

            response_text = "".join(response_chunks)
            t_llm_done = time.perf_counter()
            logger.info(f"LLM+TTS pipeline took {t_llm_done - t_llm:.2f}s ({sentence_chunks_sent} chunks)")

        except Exception as e:
            logger.warning(f"LLM Error (using smart local script fallback): {e}")
            
            # Simple rule-based simulated engine when LLM is unauthorized or unavailable
            assistant_msgs = [m["content"] for m in conversation_history if m["role"] == "assistant"]
            last_user_msg = conversation_history[-1]["content"].lower() if conversation_history and conversation_history[-1]["role"] == "user" else ""
            
            # Determine the active script step based on the last asked question content
            active_step = determine_active_step(assistant_msgs)

            # Check for general objections first to handle them like a pro human agent
            objection_reassurance = ""
            if any(w in last_user_msg for w in ["stop calling", "stop kidding", "don't call", "dont call", "do not call", "stop bothering", "remove me", "please stop"]):
                objection_reassurance = "I sincerely apologize for that! I don't know if anyone else has been calling you earlier, but this is actually my very first time reaching out to you. I definitely don't mean to bother you. I just wanted to quickly check if we could help you get some extra benefits added. "
            elif any(w in last_user_msg for w in ["benefit", "benefits", "what do you offer", "what do i get", "what are they", "what benefit", "what benefits", "qualify for", "what do i qualify", "food card", "flex card", "cash back", "what is this", "why are you calling", "who are you"]):
                objection_reassurance = "We are checking to see if you qualify for extra benefits like a food card, 300 dollars cash back, flex cards, and a very low premium. "
            elif any(w in last_user_msg for w in ["scam", "selling", "who is this", "why are you calling", "who are you"]):
                objection_reassurance = get_comfort_phrase("privacy") + "We are calling from low insurance cost Medicare. We are simply helping seniors review their Medicare Advantage plans to check if they qualify for additional benefits like dental, vision, hearing, and reducing out-of-pocket costs that they might be missing. There is absolutely no charge or obligation. "
            elif "why" in last_user_msg or "reason" in last_user_msg or "personal" in last_user_msg:
                if active_step == 2: # Age check
                    objection_reassurance = get_comfort_phrase("empathy") + "We ask because Medicare Advantage benefits and eligibility options are based on age groups. We just want to ensure we're referencing the correct guidelines for you. "
                elif active_step == 7: # Zip code
                    objection_reassurance = "Great question! Medicare Advantage benefits are very specific to your local county and zip code. A benefit available in one area might be different in another, so we want to make sure we check exactly what's available for you locally. "
            elif any(w in last_user_msg for w in ["ssn", "social security", "not giving", "number"]):
                objection_reassurance = get_comfort_phrase("privacy") + "You don't need to give it to me; you can keep your Medicare card handy, and when we connect you to our licensed specialist in a moment, they can verify your details securely. "

            # Standard yes/no check helpers
            yes_keywords = ["yes", "yeah", "yep", "sure", "correct", "right", "i do", "ok", "okay", "fine", "go ahead", "handy", "advantage", "benefit", "own", "self"]
            no_keywords = ["no", "dont", "not", "stop", "nevermind", "nursing", "assisted", "supplement", "tricare", "va"]
            
            is_yes = check_keyword(last_user_msg, yes_keywords) or any(w in last_user_msg for w in ["yes", "yeah", "yep", "sure", "correct", "ok", "okay"])
            is_no = check_keyword(last_user_msg, no_keywords) or "don't" in last_user_msg or "dont" in last_user_msg or "no" in last_user_msg.split()

            # Default fallback if we can't match a step
            response_text = "I'm sorry, I didn't quite catch that. Could you please repeat what you said?"
            
            if objection_reassurance:
                # Reset active step or combine reassurance with the active step question
                if active_step == 1:
                    response_text = objection_reassurance + "So, do you have Medicare Part A and B?"
                elif active_step == 2:
                    response_text = objection_reassurance + "So, how old are you right now?"
                elif active_step == 3:
                    response_text = objection_reassurance + "Have you updated your coverage recently?"
                elif active_step == 4:
                    response_text = objection_reassurance + "Do you have a Medicare Advantage plan or a Supplement plan?"
                elif active_step == 5:
                    response_text = objection_reassurance + "Do you make your own healthcare decisions?"
                elif active_step == 6:
                    response_text = objection_reassurance + "Do you live in a nursing home or assisted living facility?"
                elif active_step == 7:
                    response_text = objection_reassurance + "What is your zip code?"
                elif active_step == 8:
                    response_text = objection_reassurance + "Do you receive VA or Tricare benefits?"
                elif active_step == 9:
                    response_text = objection_reassurance + "Do you get Medicaid or any extra help from the state?"
                elif active_step == 10:
                    response_text = objection_reassurance + "Which benefits are most important to you? Is it dental, vision, hearing, or maybe a food card?"
                elif active_step == 11:
                    response_text = objection_reassurance + "Do you have your red, white, and blue Medicare card handy?"
                elif active_step == 12:
                    response_text = objection_reassurance + "I am going to connect you to a specialist who will help you find the best plans. Is that okay?"
                elif active_step == 13:
                    response_text = objection_reassurance + "What is your first and last name as it appears on your Medicare card?"
                elif active_step == 14:
                    response_text = objection_reassurance + "Do I have your permission to connect you now?"
                else:
                    response_text = objection_reassurance + "Do you have Medicare Part A & B?"
            else:
                is_clarification = any(w in last_user_msg for w in ["understand", "repeat", "hear", "what did you say", "say again", "what was that", "pardon", "what do you mean", "slow down"])
                
                if is_clarification:
                    if active_step == 1:
                        response_text = "I apologize if I wasn't clear! Do you have Medicare Part A and B?"
                    elif active_step == 2:
                        response_text = "Let me repeat that! I was just wondering, how old are you right now?"
                    elif active_step == 3:
                        response_text = "Sorry! I was asking, have you updated your coverage recently?"
                    elif active_step == 4:
                        response_text = "Let me repeat: Do you have a Medicare Advantage plan or a Supplement plan?"
                    elif active_step == 5:
                        response_text = "I was asking: Do you make your own healthcare decisions?"
                    elif active_step == 6:
                        response_text = "Let me repeat: Do you live in a nursing home or assisted living facility?"
                    elif active_step == 7:
                        response_text = "I was asking for your local zip code. What is your zip code?"
                    elif active_step == 8:
                        response_text = "Sorry, do you receive VA or Tricare benefits?"
                    elif active_step == 9:
                        response_text = "I was asking: Do you get Medicaid or any extra help from the state?"
                    elif active_step == 10:
                        response_text = "Which benefits are most important to you? Is it dental, vision, hearing, or maybe a food card?"
                    elif active_step == 11:
                        response_text = "Do you have your red, white, and blue Medicare card handy?"
                    elif active_step == 12:
                        response_text = "Is it okay if I connect you to a specialist who will help you find the best plans?"
                    elif active_step == 13:
                        response_text = "What is your first and last name as it appears on your Medicare card?"
                    elif active_step == 14:
                        response_text = "Do I have your permission to connect you now?"
                    else:
                        response_text = "I'm sorry, let me repeat that. Do you have Medicare Part A and B active?"
                else:
                    # User is answering the question; handle step-by-step logic
                    if active_step == 1:
                        if is_yes:
                            response_text = "You sounds pretty young over the phone call how old are you right now?"
                        elif is_no:
                            response_text = get_comfort_phrase("empathy") + "Unfortunately, you need Medicare Part A and B to qualify for these additional benefits. Thank you for your time. Goodbye. [DROP]"
                        else:
                            response_text = "Just to clarify so I don't give you the wrong info, do you have both Medicare Part A and B active right now?"
                            
                    elif active_step == 2:
                        age_keywords = [
                            "twenty", "thirty", "forty", "fourty", "fifty", "sixty", "seventy", "eighty", "ninety",
                            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
                            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                            "old", "years"
                        ]
                        has_age = any(char.isdigit() for char in last_user_msg) or any(w in last_user_msg.lower() for w in age_keywords)
                        
                        if has_age or is_yes:
                            response_text = "Have you updated your coverage recently?"
                        else:
                            response_text = "I just need a general ballpark of your age group so I can check your eligibility. Are you over the age of 60?"
                            
                    elif active_step == 3:
                        response_text = "Do you have a Medicare Advantage plan or a Supplement plan?"
                        
                    elif active_step == 4:
                        if "supplement" in last_user_msg or "supp" in last_user_msg:
                            response_text = "I see. Typically supplement plans have different guidelines and we are only reviewing Medicare Advantage options today. Thank you for your time. Goodbye. [DROP]"
                        else:
                            response_text = "Do you make your own healthcare decisions?"
                            
                    elif active_step == 5:
                        if is_no or "daughter" in last_user_msg or "son" in last_user_msg or "wife" in last_user_msg or "husband" in last_user_msg:
                            response_text = "Understood. Since you don't make your own healthcare decisions, we would need to speak with your authorized decision maker. Thank you for your time. Goodbye. [DROP]"
                        else:
                            response_text = "Do you live in a nursing home or assisted living facility?"
                            
                    elif active_step == 6:
                        if is_yes or "yes" in last_user_msg.split():
                            response_text = "I understand. Unfortunately, our program is not available for those residing in nursing homes or assisted living facilities. Thank you for your time. Goodbye. [DROP]"
                        else:
                            response_text = "What is your zip code?"
                            
                    elif active_step == 7:
                        has_zip = any(char.isdigit() for char in last_user_msg) or len(last_user_msg.strip()) >= 5
                        if has_zip:
                            response_text = "Do you receive VA or Tricare benefits?"
                        else:
                            response_text = "Could you please tell me your 5-digit zip code so I can check what is active in your county?"
                            
                    elif active_step == 8:
                        if is_yes or "yes" in last_user_msg.split():
                            response_text = "Ah, since you have VA or Tricare benefits, your coverage is handled differently. Thank you for your time. Goodbye. [DROP]"
                        else:
                            response_text = "Do you get Medicaid or any extra help from the state?"
                            
                    elif active_step == 9:
                        response_text = "Which benefits are most important to you? Is it dental, vision, hearing, or maybe a food card?"
                        
                    elif active_step == 10:
                        response_text = "Do you have your red, white, and blue Medicare card handy?"
                        
                    elif active_step == 11:
                        response_text = "I am going to connect you to a specialist who will help you find the best plans. Is that okay?"
                        
                    elif active_step == 12:
                        if is_no:
                            response_text = get_comfort_phrase("empathy") + "In that case, I will let you go. Thank you for your time and have a wonderful day! [DROP]"
                        else:
                            response_text = "What is your first and last name as it appears on your Medicare card?"
                            
                    elif active_step == 13:
                        if len(last_user_msg.strip()) > 3:
                            response_text = "Do I have your permission to connect you now?"
                        else:
                            response_text = "Could you please tell me your first and last name so I can prepare the specialist?"
                            
                    elif active_step == 14:
                        if is_no:
                            response_text = "No worries, I understand. Thank you for your time. Goodbye. [DROP]"
                        else:
                            response_text = "Excellent! Let me get that specialist on the line for you right away. [TRANSFER]"
            
            # Create a queue and worker to pipeline fallback simulated synthesis
            synthesis_queue = asyncio.Queue()
            sentence_chunks_sent = 0

            
            async def synthesis_worker():
                nonlocal sentence_chunks_sent
                while True:
                    sentence = await synthesis_queue.get()
                    if sentence is None:
                        synthesis_queue.task_done()
                        break
                    try:
                        if websocket.client_state == WebSocketState.DISCONNECTED:
                            logger.info("WebSocket disconnected. Simulated worker exiting.")
                            break
                        if interruption_event.is_set():
                            logger.info("Simulated worker: Interruption detected, skipping synthesis.")
                            break
                        
                        clean = sentence.replace("[TRANSFER]", "").replace("[DROP]", "").strip()
                        if clean:
                            t_chunk = time.perf_counter()
                            clean_tts = clean.replace("*clears throat*", "Ahem,")
                            audio = await synthesize_speech(clean_tts, rate=speech_rate)
                            if audio:
                                if websocket.client_state == WebSocketState.DISCONNECTED:
                                    logger.info("WebSocket disconnected. Discarding simulated chunk.")
                                    break
                                if interruption_event.is_set():
                                    logger.info("Interruption detected, discarding simulated chunk.")
                                    break
                                await websocket.send_bytes(audio)
                                sentence_chunks_sent += 1
                                logger.info(f"Sent simulated sentence chunk #{sentence_chunks_sent}: '{clean}' in {time.perf_counter()-t_chunk:.2f}s")
                    except Exception as e:
                        logger.error(f"Error in simulated synthesis worker: {e}")
                    finally:
                        synthesis_queue.task_done()

            worker_task = asyncio.create_task(synthesis_worker())
            
            # Send the selected simulated response split into sentence chunks for optimal caching
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response_text) if s.strip()]
            for sentence in sentences:
                await synthesis_queue.put(sentence)
                
            await synthesis_queue.put(None)
            await worker_task

        # ── Step 3: Wait for client to finish playing all queued audio ────
        playback_done_event.clear()
        try:
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(playback_done_event.wait()),
                    asyncio.create_task(interruption_event.wait())
                ],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=15.0
            )
            for task in pending:
                task.cancel()
        except asyncio.TimeoutError:
            logger.warning("Playback done timeout, resuming")

        if interruption_event.is_set():
            response_text = response_text + " [Interrupted]"
            logger.info("Utterance interrupted by user.")
            
        conversation_history.append({"role": "assistant", "content": response_text})
        await websocket.send_text(f"Bot: {response_text}")
        
        # If not already interrupted again, clear the speaking lock
        if is_bot_speaking:
            is_bot_speaking = False
            await websocket.send_text("CTRL:LISTENING")

        t_end = time.perf_counter()
        logger.info(f"Total response pipeline took {t_end - t_start:.2f}s")
        
        if "[TRANSFER]" in response_text or "[DROP]" in response_text:
            logger.info(f"Action triggered: {response_text}")
            await asyncio.sleep(3)
            try:
                await websocket.close()
            except:
                pass
                
        return response_text

    # ── Greeting ──────────────────────────────────────────────────────────
    greeting = "Hello! Hi, my name is emily calling you from low insurance cost Medicare. Do you have Medicare Part A & B?"
    conversation_history.append({"role": "assistant", "content": greeting})
    await websocket.send_text(f"Bot: {greeting}")
    
    # Wait for the client to signal readiness (config message) before sending audio
    client_ready_event = asyncio.Event()
    
    async def greet_task():
        # Wait up to 10s for client to send config (AudioContext ready)
        try:
            await asyncio.wait_for(client_ready_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Client never sent config, sending greeting anyway")
        await speak_and_send(greeting)
    
    current_utterance_task = asyncio.create_task(greet_task())

    # ── Receiver task: reads WebSocket, feeds Vosk/Deepgram, accumulates words ─────
    async def receiver():
        nonlocal silence_timer_task, is_bot_speaking, speech_rate, recognizer, current_utterance_task
        import websockets
        
        DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
        dg_socket = None
        dg_receiver_task = None
        
        async def deepgram_receiver(dg_ws):
            nonlocal silence_timer_task, is_bot_speaking
            try:
                async for msg in dg_ws:
                    res = json.loads(msg)
                    
                    channel = res.get("channel", {})
                    alternatives = channel.get("alternatives", [{}])
                    transcript = alternatives[0].get("transcript", "").strip()
                    is_final = res.get("is_final", False)
                    
                    if transcript:
                        # Trigger interruption immediately when user starts speaking
                        if is_bot_speaking:
                            is_bot_speaking = False
                            interruption_event.set()
                            await websocket.send_text("CTRL:STOP_AUDIO")
                            logger.info("Deepgram: Bot interrupted by early speech.")
                        
                        if is_final:
                            accumulated_words.append(transcript)
                            logger.info(f"Deepgram final: '{transcript}' | accumulated: '{' '.join(accumulated_words)}'")
                            await websocket.send_text(f"STATUS:STT:🟢 Heard: {' '.join(accumulated_words)}")
                            
                            # Reset silence timer
                            if silence_timer_task:
                                silence_timer_task.cancel()
                                
                            current_text = " ".join(accumulated_words).strip().lower()
                            if current_text in _QUICK_WORDS:
                                sil_timeout = 0.2  # 200ms for quick affirmatives
                            else:
                                sil_timeout = 0.4  # 400ms standard
                            silence_timer_task = asyncio.create_task(silence_timer(sil_timeout))
                        else:
                            preview = " ".join(accumulated_words + [transcript]) if accumulated_words else transcript
                            await websocket.send_text(f"STATUS:STT:🟢 Hearing: {preview}")
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Deepgram receiver thread error: {e}")

        try:
            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                if "text" in message:
                    text = message["text"]
                    if text == "__PLAYBACK_DONE__":
                        playback_done_event.set()
                        continue
                        
                    try:
                        data = json.loads(text)
                        if data.get("type") == "config":
                            client_sr = data.get("sampleRate", 16000)
                            speech_rate = data.get("speechRate", "+0%")
                            
                            # 1. Initialize Vosk as local fallback
                            if vosk_model:
                                recognizer = KaldiRecognizer(vosk_model, client_sr)
                                recognizer.SetWords(True)
                                logger.info(f"Initialized Vosk fallback ({vosk_model_name}) at {client_sr}Hz, speech rate: {speech_rate}")
                            
                            # 2. Connect to Deepgram streaming if API Key is set
                            if DEEPGRAM_API_KEY and DEEPGRAM_API_KEY != "NOT_SET":
                                try:
                                    logger.info("Connecting to Deepgram streaming STT service...")
                                    dg_url = f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate={client_sr}&channels=1&model=nova-2&interim_results=true&endpointing=300"
                                    dg_headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
                                    dg_socket = await websockets.connect(dg_url, additional_headers=dg_headers)
                                    dg_receiver_task = asyncio.create_task(deepgram_receiver(dg_socket))
                                    await websocket.send_text("STATUS:STT:🟢 Deepgram Nova-2 Active")
                                    logger.info("Connected to Deepgram successfully.")
                                except Exception as e:
                                    logger.error(f"Failed to connect to Deepgram (using local Vosk instead): {e}")
                                    await websocket.send_text("STATUS:STT:🔴 Deepgram Failed, Using Vosk")
                            
                            # 3. Signal that the client is ready to receive audio
                            client_ready_event.set()
                            continue
                        elif data.get("type") == "control":
                            new_rate = data.get("speechRate")
                            if new_rate:
                                speech_rate = new_rate
                                logger.info(f"Speech rate updated to: {speech_rate}")
                            continue
                        elif "text" in data and data["text"] == "__PLAYBACK_DONE__":
                            playback_done_event.set()
                            continue
                    except Exception as ex:
                        pass
                        
                    if text == '{"text":"__PLAYBACK_DONE__"}':
                        playback_done_event.set()
                        continue

                    logger.info(f"Text input from {client_id}: {text}")
                    await utterance_queue.put((text, False))  # (text, from_voice)
                    continue

                # Handle audio
                if "bytes" in message:
                    audio_chunk = message["bytes"]
                    
                    # Stream to Deepgram if connected
                    dg_ok = False
                    if dg_socket:
                        try:
                            await dg_socket.send(audio_chunk)
                            dg_ok = True
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("Deepgram WS closed, falling back to Vosk")
                            dg_socket = None
                        except Exception as e:
                            logger.error(f"Error sending bytes to Deepgram, falling back to Vosk: {e}")
                            try:
                                await dg_socket.close()
                            except:
                                pass
                            dg_socket = None
                    
                    # Otherwise, fall back to offline local Vosk
                    if not dg_ok and recognizer:
                        if recognizer.AcceptWaveform(audio_chunk):
                            result = json.loads(recognizer.Result())
                            text = result.get("text", "").strip()
                            if text and text != "__PLAYBACK_DONE__":
                                if is_bot_speaking:
                                    is_bot_speaking = False
                                    interruption_event.set()
                                    await websocket.send_text("CTRL:STOP_AUDIO")

                                accumulated_words.append(text)
                                logger.info(f"Vosk partial final: '{text}' | accumulated: '{' '.join(accumulated_words)}'")
                                await websocket.send_text(f"STATUS:STT:🟢 Heard: {' '.join(accumulated_words)}")
                                
                                if silence_timer_task:
                                    silence_timer_task.cancel()
                                    
                                current_text = " ".join(accumulated_words).strip().lower()
                                if current_text in _QUICK_WORDS:
                                    sil_timeout = 0.2
                                else:
                                    sil_timeout = 0.4
                                silence_timer_task = asyncio.create_task(silence_timer(sil_timeout))
                        else:
                            partial = json.loads(recognizer.PartialResult())
                            partial_text = partial.get("partial", "").strip()
                            if partial_text:
                                if is_bot_speaking and len(partial_text.split()) > 0:
                                    is_bot_speaking = False
                                    interruption_event.set()
                                    await websocket.send_text("CTRL:STOP_AUDIO")

                                preview = " ".join(accumulated_words + [partial_text]) if accumulated_words else partial_text
                                await websocket.send_text(f"STATUS:STT:🟢 Hearing: {preview}")
                                
        except WebSocketDisconnect:
            await utterance_queue.put(None)
        except Exception as e:
            logger.error(f"Receiver error: {e}")
            await utterance_queue.put(None)
        finally:
            if dg_receiver_task:
                dg_receiver_task.cancel()
            if dg_socket:
                try:
                    await dg_socket.close()
                except:
                    pass

    # ── Start receiver in background ──────────────────────────────────────
    receiver_task = asyncio.create_task(receiver())

    try:
        while True:
            item = await utterance_queue.get()
            if item is None:
                break

            # Unpack (text, from_voice) tuple
            if isinstance(item, tuple):
                utterance, from_voice = item
            else:
                utterance, from_voice = item, False
                
            if current_utterance_task and not current_utterance_task.done():
                pass
                
            current_utterance_task = asyncio.create_task(process_utterance(utterance, from_voice=from_voice))
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
    finally:
        receiver_task.cancel()
        if silence_timer_task:
            silence_timer_task.cancel()
        logger.info(f"Client #{client_id} session ended.")

if __name__ == "__main__":
    import uvicorn
    # reload=False avoids triple model loading and reduces startup time
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
