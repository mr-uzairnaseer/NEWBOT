import os
import hashlib
import time
import httpx
import logging
import asyncio
import edge_tts
from config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID, EDGE_TTS_VOICE

logger = logging.getLogger(__name__)

CACHE_DIR = "tts_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

_tts_cache: dict[str, bytes] = {}

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
    
    # Reassurance Comfort Phrases
    "Got it.", "No worries at all.", "Totally fine.", "No problem.", 
    "Understood.", "Perfect.", "No worries.", "No problem at all.",
    
    # Empathy Comfort Phrases
    "I completely understand.", "I hear you loud and clear.", 
    "I completely get where you're coming from.", "That is totally understandable.", 
    "I understand completely.", "I completely respect that.", "I understand.", "Makes perfect sense.",
    
    # Privacy Comfort Phrases
    "I completely respect your privacy.", "Security and privacy are absolutely top priority.", 
    "I completely understand your caution.", "We definitely respect your space and privacy.", 
    "Safety is number one.", "I respect your privacy completely.", "I completely understand your safety concerns.",
    
    # Respect Comfort Phrases
    "I completely respect that.", "I absolutely respect your decision.", "Fair enough.", 
    "I completely respect where you're coming from.", "I hear you and absolutely respect that.",
    
    # Pressure-Free Comfort Phrases
    "No pressure at all.", "Absolutely no rush.", "Totally up to you.", 
    "No worries whatsoever.", "There is absolutely no rush or pressure.",
    
    # Local Router Dialogue Chunks
    "My name is Emily calling from low insurance cost Medicare. We are simply helping seniors review their Medicare options. ",
    "Like I mentioned, my name is Emily and I am calling from low insurance cost Medicare. We are just helping seniors check their eligibility. ",
    "I am just a representative from low insurance cost Medicare checking basic eligibility so we don't waste your time. ",
    "We are based in houston, Texas, calling from low insurance cost Medicare. ",
    "Part A covers hospital stays, and Part B covers doctor visits. They make up standard Medicare. ",
    
    "We are reaching out to local seniors to help them review if they are eligible for additional benefits like dental, vision, hearing, and food card allowances. ",
    "We are just checking local seniors' eligibility so they don't miss out on premium benefits like flex cards or cash back. ",
    
    "No, I am a live representative calling from low insurance cost Medicare. We never ask for any private SSN numbers on this call. We just want to check. ",
    
    "We check for allowances like the food card, three hundred dollars cash back, flex cards, and very low premiums. ",
    "Those are premium benefits that add on top of standard Medicare to cover dental, vision, and groceries. ",
    
    "You don't need to give it to me! You can keep your card handy and verify it securely in a moment. ",
    
    "If you are sixty-five or older, you usually have Part A and B active. Do you receive those benefits, or have a red, white, and blue card?",
    "If you visit a doctor, is that covered by standard Medicare? That usually means both parts are active.",
    "Let's do this—I can get a specialist on the line who can quickly verify that. How old are you right now?",
    
    "We ask because eligibility is based on age. Are you sixty or older right now?",
    "Let's get that specialist on the line right away to verify. [TRANSFER]",
    "Let's get that specialist on the line right away to verify.",
    
    "Let me repeat. ",
    "I completely respect that. Just to be absolutely sure,",
    "Understood. Just to make sure we don't waste your time,",
    "Fair enough. Just to be absolutely certain,",
    "Got it. Just to verify,",
    "No problem. Just to make sure we are on the same page,",
    
    "Do you have Medicare Part A & B?",
    "Are both your Part A and Part B active right now?",
    "Do you have the red, white, and blue Medicare card handy?",
    
    "How old are you right now?",
    "What is your age group right now?",
    "Are you generally over the age of sixty?",
    
    "I see. You need Medicare Part A and B to qualify. Have a wonderful day! [DROP]",
    "I see. Unfortunately, you must be sixty or older to qualify. Have a wonderful day! [DROP]",
    "Unfortunately, you must be sixty or older to qualify. Have a wonderful day! [DROP]",
    "No worries, you don't have to give the exact age. ",
    "Excellent! Let me get that specialist on the line for you right away. [TRANSFER]",
    "Excellent! Let me get that specialist on the line for you right away.",
    
    # New phrases for greeting, spouse and bot age handlers
    "Hi there! Yes, this is Emily calling from low insurance cost Medicare.",
    "Do you have Medicare Part A and B active?",
    "Haha, I get that a lot! I'm twenty-four, but I promise I'm fully qualified.",
    "Ah, got it! That makes complete sense.",
    "If they are around, they can verify it, but just to check for your own eligibility,",
    "Yes, I am still here!",
]

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
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}?optimize_streaming_latency=3"
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

async def synthesize_speech(text: str, rate: str = "+0%") -> bytes:
    """Generate raw PCM16 audio bytes from text using ElevenLabs (if configured) or edge-tts with persistent disk caching."""
    norm = _normalize_text(text)
    
    # Check what provider/voice we would naturally use
    provider = "elevenlabs" if ELEVENLABS_API_KEY else "edgetts"
    voice_id = ELEVENLABS_VOICE_ID if ELEVENLABS_API_KEY else EDGE_TTS_VOICE
    model_id = ELEVENLABS_MODEL_ID if ELEVENLABS_API_KEY else "default"
    
    # 1. Unified Cache Key to prevent cross-over
    cache_key = f"{norm}|{rate}|{provider}|{voice_id}|{model_id}"
    
    # Check in-memory cache
    if cache_key in _tts_cache:
        logger.debug(f"TTS memory cache hit for: '{text[:40]}...' (norm: {norm[:15]})")
        return _tts_cache[cache_key]

    # 2. Check persistent disk cache
    unique_str = f"{norm}|{provider}|{voice_id}|{model_id}|{rate}"
    h = hashlib.md5(unique_str.encode("utf-8")).hexdigest()
    disk_cache_path = os.path.join(CACHE_DIR, f"{h}.bin")
    
    if os.path.exists(disk_cache_path):
        try:
            with open(disk_cache_path, "rb") as f:
                audio_data = f.read()
            if audio_data:
                logger.debug(f"TTS disk cache hit for: '{text[:40]}...' (hash: {h})")
                if len(_tts_cache) < 200:
                    _tts_cache[cache_key] = audio_data
                return audio_data
        except Exception as e:
            logger.warning(f"Failed to read disk cache for '{text[:40]}...': {e}")

    # 3. Perform live synthesis
    t0 = time.perf_counter()
    used_provider = "edgetts"
    
    if ELEVENLABS_API_KEY:
        logger.debug(f"Using ElevenLabs TTS for: '{text[:40]}...'")
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
    logger.debug(f"TTS synthesis took {elapsed:.2f}s for {len(audio_data)} bytes using {used_provider}")
    
    # Save the successful result to disk cache using actual final voice/model
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
        logger.debug(f"Saved TTS output to disk cache: '{text[:40]}...' (hash: {actual_h})")
    except Exception as e:
        logger.warning(f"Failed to write disk cache: {e}")

    # Cache result in memory
    if len(_tts_cache) < 200:
        _tts_cache[cache_key] = audio_data
        
    return audio_data

async def precache_tts(rate: str = "+0%"):
    """Pre-load TTS audio from disk cache into memory. Avoids calling ElevenLabs on startup."""
    import re
    
    # Extract all individual sentences from the precache phrases
    individual_sentences = set()
    for p in _PRECACHE_PHRASES:
        split_s = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()]
        for s in split_s:
            individual_sentences.add(s)
            
    tasks = []
    
    # Standard phrases (synthesized and stored persistently if not already cached)
    for p in individual_sentences:
        tasks.append(synthesize_speech(p, rate=rate))
            
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Pre-cached {len(tasks)} phrases into memory.")
    else:
        logger.info("Background pre-caching skipped.")
