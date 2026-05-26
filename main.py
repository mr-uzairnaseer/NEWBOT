import asyncio
import os
import json
import logging
import time
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState
from vosk import Model, KaldiRecognizer

import config
import dialogue
import tts
import brain

# Enterprise Database Integrations
import asyncpg
import redis.asyncio as aioredis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Voice Bot Backend")

# ── STT: Vosk (server-side, offline fallback) ─────────────────────────────────
try:
    vosk_model = Model(config.VOSK_MODEL_PATH)
    vosk_model_name = config.VOSK_MODEL_PATH
    logger.info(f"Vosk STT model loaded from '{config.VOSK_MODEL_PATH}'")
except Exception as e:
    vosk_model = None
    logger.error(f"Failed to load Vosk model: {e}")


# ── Campaign Management ────────────────────────────────────────────────────────
campaigns = {
    "default": config.SYSTEM_PROMPT
}

# ── Live Enterprise settings ──────────────────────────────────────────────────
ACTIVE_SYSTEM_PROMPT = config.SYSTEM_PROMPT
ACTIVE_MODEL = config.LLM_MODEL
ACTIVE_SPEECH_SPEED = "+20%"
BOT_ACTIVE = True

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


# ── Lifespan Context for Startup Pre-caching & DB Init ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server is starting up. Pre-warming TTS cache for standard speed (+20%)...")
    asyncio.create_task(tts.precache_tts(rate="+20%"))
    
    # Initialize Postgres Connection Pool
    try:
        logger.info(f"Initializing PostgreSQL pool connecting to {config.POSTGRES_HOST}...")
        app.state.pg_pool = await asyncpg.create_pool(
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            host=config.POSTGRES_HOST,
            port=int(config.POSTGRES_PORT),
            database=config.POSTGRES_DB,
            min_size=5,
            max_size=20
        )
        logger.info("PostgreSQL connection pool initialized successfully.")
    except Exception as e:
        logger.error(f"PostgreSQL pool initialization failed: {e}")
        app.state.pg_pool = None

    # Create Call Logs Table if not exists
    if app.state.pg_pool:
        try:
            async with app.state.pg_pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS call_logs (
                        id SERIAL PRIMARY KEY,
                        uniqueid VARCHAR(100) UNIQUE NOT NULL,
                        campaign_name VARCHAR(100) DEFAULT 'default',
                        duration INTEGER DEFAULT 0,
                        disposition VARCHAR(50) DEFAULT 'DROP',
                        avg_latency NUMERIC(5,2) DEFAULT 0.0,
                        transcript TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            logger.info("PostgreSQL call_logs table is ready.")
        except Exception as e:
            logger.warning(f"PostgreSQL table verification warning: {e}")


    # Initialize Redis Client
    try:
        logger.info(f"Initializing Redis connection to {config.REDIS_HOST}:{config.REDIS_PORT}...")
        app.state.redis = aioredis.Redis(
            host=config.REDIS_HOST,
            port=int(config.REDIS_PORT),
            decode_responses=True
        )
        logger.info("Redis connection ready.")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        app.state.redis = None

    yield
    
    # Clean closing
    logger.info("Server is shutting down. Closing database pools...")
    if app.state.pg_pool:
        await app.state.pg_pool.close()
    if app.state.redis:
        await app.state.redis.close()
    logger.info("Server shutdown complete.")

app.router.lifespan_context = lifespan


# ── Dashboard API Endpoints ───────────────────────────────────────────────────

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except Exception as e:
        logger.error(f"Failed to load dashboard.html: {e}")
        return HTMLResponse("<h1>Error: dashboard.html not found</h1>", status_code=404)

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    if not app.state.pg_pool:
        return {"total_calls": 0, "transfer_rate": 0, "avg_latency": 0.0, "active_calls": 0, "total_transfers": 0, "total_drops": 0}
        
    try:
        active_calls_count = 0
        if app.state.redis:
            keys = await app.state.redis.keys("active_calls:*")
            active_calls_count = len(keys)
            
        async with app.state.pg_pool.acquire() as conn:
            total_calls = await conn.fetchval("SELECT COUNT(*) FROM call_logs")
            total_transfers = await conn.fetchval("SELECT COUNT(*) FROM call_logs WHERE disposition = 'TRANSFER'")
            total_drops = await conn.fetchval("SELECT COUNT(*) FROM call_logs WHERE disposition = 'DROP'")
            avg_latency = await conn.fetchval("SELECT COALESCE(AVG(avg_latency), 0.0) FROM call_logs")
            
        transfer_rate = int((total_transfers / total_calls * 100)) if total_calls > 0 else 0
        
        return {
            "total_calls": total_calls,
            "transfer_rate": transfer_rate,
            "avg_latency": float(avg_latency),
            "active_calls": active_calls_count,
            "total_transfers": total_transfers,
            "total_drops": total_drops
        }
    except Exception as e:
        logger.error(f"Failed to fetch stats: {e}")
        return {"error": str(e)}

@app.get("/api/dashboard/active-calls")
async def get_active_calls():
    if not app.state.redis:
        return []
        
    try:
        keys = await app.state.redis.keys("active_calls:*")
        active_calls = []
        for key in keys:
            call_data = await app.state.redis.hgetall(key)
            if call_data:
                active_calls.append({
                    "client_id": call_data.get("client_id"),
                    "campaign": call_data.get("campaign"),
                    "started_at": int(call_data.get("started_at", 0))
                })
        return active_calls
    except Exception as e:
        logger.error(f"Failed to fetch active calls: {e}")
        return []

@app.get("/api/dashboard/calls")
async def get_recent_calls():
    if not app.state.pg_pool:
        return []
        
    try:
        async with app.state.pg_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT uniqueid, campaign_name, duration, disposition, avg_latency, created_at 
                FROM call_logs 
                ORDER BY created_at DESC 
                LIMIT 50
            """)
            
        calls = []
        for r in rows:
            calls.append({
                "uniqueid": r["uniqueid"],
                "campaign_name": r["campaign_name"],
                "duration": r["duration"],
                "disposition": r["disposition"],
                "avg_latency": float(r["avg_latency"]),
                "created_at": r["created_at"].isoformat() if r["created_at"] else ""
            })
        return calls
    except Exception as e:
        logger.error(f"Failed to fetch recent calls: {e}")
        return []

@app.get("/api/dashboard/call/{uniqueid}")
async def get_call_detail(uniqueid: str):
    if not app.state.pg_pool:
        return {"error": "Database not initialized"}
        
    try:
        async with app.state.pg_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT uniqueid, campaign_name, duration, disposition, avg_latency, transcript, created_at 
                FROM call_logs 
                WHERE uniqueid = $1
            """, uniqueid)
            
        if not row:
            return {"error": "Call not found"}
            
        return {
            "uniqueid": row["uniqueid"],
            "campaign_name": row["campaign_name"],
            "duration": row["duration"],
            "disposition": row["disposition"],
            "avg_latency": float(row["avg_latency"]),
            "transcript": row["transcript"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else ""
        }
    except Exception as e:
        logger.error(f"Failed to fetch call detail: {e}")
        return {"error": str(e)}


# ── Live Control Settings API ─────────────────────────────────────────────────

class SettingsUpdate(BaseModel):
    active_prompt: str = None
    active_model: str = None
    active_speech_speed: str = None
    bot_active: bool = None

@app.get("/api/control/settings")
async def get_control_settings():
    return {
        "active_prompt": ACTIVE_SYSTEM_PROMPT,
        "active_model": ACTIVE_MODEL,
        "active_speech_speed": ACTIVE_SPEECH_SPEED,
        "bot_active": BOT_ACTIVE
    }

@app.post("/api/control/settings")
async def update_control_settings(settings: SettingsUpdate):
    global ACTIVE_SYSTEM_PROMPT, ACTIVE_MODEL, ACTIVE_SPEECH_SPEED, BOT_ACTIVE
    if settings.active_prompt is not None:
        ACTIVE_SYSTEM_PROMPT = settings.active_prompt
        campaigns["default"] = settings.active_prompt
    if settings.active_model is not None:
        ACTIVE_MODEL = settings.active_model
    if settings.active_speech_speed is not None:
        ACTIVE_SPEECH_SPEED = settings.active_speech_speed
    if settings.bot_active is not None:
        BOT_ACTIVE = settings.bot_active
    
    logger.info("Live configuration updated from Control Panel!")
    return {
        "status": "success",
        "active_prompt": ACTIVE_SYSTEM_PROMPT,
        "active_model": ACTIVE_MODEL,
        "active_speech_speed": ACTIVE_SPEECH_SPEED,
        "bot_active": BOT_ACTIVE
    }


# ── Session State Management Class ─────────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.accumulated_words = []
        self.silence_timer_task = None
        self.is_bot_speaking = False
        self.last_speak_start_time = 0.0
        self.speech_rate = "+20%"
        self.recognizer = None
        self.current_utterance_task = None
        self.dg_socket = None
        self.dg_receiver_task = None


# ── Core WebSocket Orchestrator ────────────────────────────────────────────────
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    logger.info(f"Client #{client_id} connected.")
    
    start_time = time.time()
    try:
        if app.state.redis:
            await app.state.redis.hset(
                f"active_calls:{client_id}",
                mapping={
                    "client_id": client_id,
                    "campaign": "default",
                    "started_at": str(int(start_time * 1000))
                }
            )
            await app.state.redis.expire(f"active_calls:{client_id}", 3600)
    except Exception as e:
        logger.error(f"Failed to set active call in Redis: {e}")

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
    if config.ELEVENLABS_API_KEY:
        await websocket.send_text(f"STATUS:TTS:🟢 ElevenLabs ({config.ELEVENLABS_VOICE_ID})")
    else:
        await websocket.send_text(f"STATUS:TTS:🟢 Edge-TTS ({config.EDGE_TTS_VOICE})")

    # Check if bot is disabled globally
    if not BOT_ACTIVE:
        logger.info(f"Declining call for client #{client_id}: Bot is currently set to INACTIVE.")
        await websocket.send_text("Bot: Hello! System is currently under maintenance. We will call you back shortly. Goodbye.")
        await asyncio.sleep(2.5)
        await websocket.close()
        return

    state = SessionState()
    state.speech_rate = ACTIVE_SPEECH_SPEED
    conversation_history = [
        {"role": "system", "content": ACTIVE_SYSTEM_PROMPT}
    ]

    # Interruption tracking
    interruption_event = asyncio.Event()
    playback_done_event = asyncio.Event()

    async def interrupt_current_bot_action(reason: str):
        """Instantly stop bot speaking or thinking."""
        if state.is_bot_speaking:
            if state.last_speak_start_time > 0.0 and time.time() - state.last_speak_start_time <= 0.8:
                # Ignore interruption during guard period
                return
        
        state.is_bot_speaking = False
        interruption_event.set()
        if state.current_utterance_task and not state.current_utterance_task.done():
            logger.info(f"Interrupting active bot task due to: {reason}")
            state.current_utterance_task.cancel()
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.send_text("CTRL:STOP_AUDIO")

    # Short words triggering faster silence timeout
    _QUICK_WORDS = {"yes", "no", "yeah", "yep", "sure", "okay", "yea", "ya", "right", "correct", "absolutely"}

    async def silence_timer(timeout=0.4):
        """Fires after silence — flushes accumulated words as a complete utterance."""
        await asyncio.sleep(timeout)
        if state.accumulated_words:
            full = " ".join(state.accumulated_words)
            state.accumulated_words.clear()
            await utterance_queue.put((full, True))  # (text, from_voice=True)

    utterance_queue = asyncio.Queue()

    async def speak_and_wait(text: str):
        """Generate TTS audio chunk-by-chunk concurrently, send, and wait for playback completion or interruption."""
        if websocket.client_state == WebSocketState.DISCONNECTED:
            logger.debug("WebSocket disconnected. Skipping speak_and_wait.")
            return

        state.is_bot_speaking = True
        state.last_speak_start_time = 0.0
        playback_done_event.clear()
        interruption_event.clear()
        await websocket.send_text("CTRL:SPEAKING")
        
        sentence_chunks_sent = 0
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        async def synth_task(sentence_text):
            clean = sentence_text.replace("[TRANSFER]", "").replace("[DROP]", "").strip()
            if not clean:
                return None
            try:
                t0 = time.perf_counter()
                audio = await tts.synthesize_speech(clean, rate=state.speech_rate)
                logger.info(f"Synthesized '{clean[:30]}...' in {time.perf_counter() - t0:.2f}s")
                return audio
            except Exception as e:
                logger.error(f"Synthesis failed for '{clean[:30]}...': {e}")
                return None

        # Start all synthesis tasks concurrently in the background
        tasks = [asyncio.create_task(synth_task(s)) for s in sentences]

        try:
            for task in tasks:
                if websocket.client_state == WebSocketState.DISCONNECTED or interruption_event.is_set():
                    break
                audio = await task
                if audio:
                    if websocket.client_state == WebSocketState.DISCONNECTED or interruption_event.is_set():
                        break
                    if sentence_chunks_sent == 0:
                        state.last_speak_start_time = time.time()
                    await websocket.send_bytes(audio)
                    sentence_chunks_sent += 1
        except asyncio.CancelledError:
            pass
        finally:
            # Clean up outstanding background tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

        # Error recovery: skip waiting for playback if no chunks were actually sent
        if sentence_chunks_sent > 0 and not interruption_event.is_set():
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

        state.is_bot_speaking = False
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.send_text("CTRL:LISTENING")

    async def process_utterance(text: str, from_voice: bool = False):
        """Stream LLM → sentence-split → TTS → send audio in real-time."""
        if websocket.client_state == WebSocketState.DISCONNECTED:
            return ""

        t_start = time.perf_counter()

        if from_voice:
            text = dialogue.local_correct_stt(text)

        logger.info(f"Processing utterance: '{text}'")

        # 1. Check the local router first for 0ms latency & pre-cached audio
        router_response = dialogue.get_local_router_response(text, conversation_history)
        
        if router_response is not None:
            # ── Instant Local Router Path (0ms Latency) ──
            logger.info(f"Local router match: '{router_response}'")
            conversation_history.append({"role": "user", "content": text})
            label = "You (Voice)" if from_voice else "You (Text)"
            await websocket.send_text(f"{label}: {text}")
            
            # Print bot response in transcript
            await websocket.send_text(f"Bot: {router_response}")
            conversation_history.append({"role": "assistant", "content": router_response})
            
            # Speak instantly
            await speak_and_wait(router_response)
            
            if "[TRANSFER]" in router_response or "[DROP]" in router_response:
                logger.info(f"Action triggered: {router_response}")
                await asyncio.sleep(3)
                try:
                    await websocket.close()
                except:
                    pass
            t_end = time.perf_counter()
            logger.info(f"Instant Local Router Path took {t_end - t_start:.4f}s")
            return router_response

        # Otherwise, fall back to streaming LLM
        conversation_history.append({"role": "user", "content": text})
        label = "You (Voice)" if from_voice else "You (Text)"
        await websocket.send_text(f"{label}: {text}")

        # ── Streaming LLM → Sentence-Split → TTS Pipeline ──
        state.is_bot_speaking = True
        state.last_speak_start_time = 0.0
        playback_done_event.clear()
        interruption_event.clear()
        await websocket.send_text("CTRL:SPEAKING")

        full_response = ""
        sentence_buffer = ""
        sentences_sent = 0

        async def synth_and_send(sentence_text: str):
            """Synthesize one sentence and send audio to client."""
            clean = sentence_text.replace("[TRANSFER]", "").replace("[DROP]", "").strip()
            if not clean:
                return False
            try:
                t0 = time.perf_counter()
                audio = await tts.synthesize_speech(clean, rate=state.speech_rate)
                logger.info(f"TTS '{clean[:40]}' in {time.perf_counter()-t0:.2f}s")
                if audio and not interruption_event.is_set():
                    if sentences_sent == 0:
                        state.last_speak_start_time = time.time()
                    await websocket.send_bytes(audio)
                    return True
            except Exception as e:
                logger.error(f"Streaming TTS error: {e}")
            return False

        try:
            async for token in brain.stream_llm_response(conversation_history, model=ACTIVE_MODEL):
                if interruption_event.is_set() or websocket.client_state == WebSocketState.DISCONNECTED:
                    break

                full_response += token
                sentence_buffer += token

                # Split completed sentences from the buffer
                parts = re.split(r'(?<=[.!?])\s+', sentence_buffer.strip())

                if len(parts) > 1:
                    # All but last part are complete sentences
                    for sent in parts[:-1]:
                        if interruption_event.is_set():
                            break
                        if await synth_and_send(sent):
                            sentences_sent += 1
                    sentence_buffer = parts[-1]
                elif sentence_buffer.strip() and sentence_buffer.strip()[-1] in '.!?':
                    # Single complete sentence
                    if await synth_and_send(sentence_buffer.strip()):
                        sentences_sent += 1
                    sentence_buffer = ""

            # Flush any remaining text
            remaining = sentence_buffer.strip()
            if remaining and not interruption_event.is_set():
                if await synth_and_send(remaining):
                    sentences_sent += 1

        except asyncio.CancelledError:
            full_response += " [Interrupted]"
            logger.info("Streaming pipeline cancelled by barge-in.")

        # Wait for client playback to finish
        if sentences_sent > 0 and not interruption_event.is_set():
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

        response_text = full_response.strip()
        if interruption_event.is_set():
            response_text += " [Interrupted]"
            logger.info("Utterance interrupted by user.")

        state.is_bot_speaking = False
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.send_text("CTRL:LISTENING")

        conversation_history.append({"role": "assistant", "content": response_text})
        await websocket.send_text(f"Bot: {response_text}")

        t_end = time.perf_counter()
        logger.info(f"Streaming pipeline (OpenRouter → Gemini Flash) took {t_end - t_start:.2f}s")

        if "[TRANSFER]" in response_text or "[DROP]" in response_text:
            logger.info(f"Action triggered: {response_text}")
            await asyncio.sleep(3)
            try:
                await websocket.close()
            except:
                pass
        return response_text

    # ── Initial Greeting ───────────────────────────────────────────────────────
    greeting = "Hello! Hi, my name is emily calling you from low insurance cost Medicare. Do you have Medicare Part A & B?"
    conversation_history.append({"role": "assistant", "content": greeting})
    await websocket.send_text(f"Bot: {greeting}")
    
    client_ready_event = asyncio.Event()
    
    async def greet_task():
        try:
            await asyncio.wait_for(client_ready_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Client never sent config, sending greeting anyway")
        await speak_and_wait(greeting)
    
    state.current_utterance_task = asyncio.create_task(greet_task())

    # ── Receiver Task: reads WS, manages Deepgram/Vosk & accumulates speech ─────
    async def receiver():
        import websockets
        
        async def deepgram_receiver(dg_ws):
            try:
                async for msg in dg_ws:
                    res = json.loads(msg)
                    channel = res.get("channel", {})
                    alternatives = channel.get("alternatives", [{}])
                    transcript = alternatives[0].get("transcript", "").strip()
                    is_final = res.get("is_final", False)
                    
                    if transcript:
                        await interrupt_current_bot_action("Deepgram speech detected")
                        
                        if is_final:
                            state.accumulated_words.append(transcript)
                            logger.info(f"Deepgram final: '{transcript}' | accumulated: '{' '.join(state.accumulated_words)}'")
                            await websocket.send_text(f"STATUS:STT:🟢 Heard: {' '.join(state.accumulated_words)}")
                            
                            if state.silence_timer_task:
                                state.silence_timer_task.cancel()
                                
                            current_text = " ".join(state.accumulated_words).strip().lower()
                            sil_timeout = 0.2 if current_text in _QUICK_WORDS else 0.4
                            state.silence_timer_task = asyncio.create_task(silence_timer(sil_timeout))
                        else:
                            preview = " ".join(state.accumulated_words + [transcript]) if state.accumulated_words else transcript
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
                    
                    # Unify playback done checks
                    if text == "__PLAYBACK_DONE__" or text == '{"text":"__PLAYBACK_DONE__"}':
                        playback_done_event.set()
                        continue
                        
                    data = {}
                    try:
                        data = json.loads(text)
                    except:
                        pass

                    if isinstance(data, dict) and data.get("text") == "__PLAYBACK_DONE__":
                        playback_done_event.set()
                        continue
                        
                    if isinstance(data, dict) and data.get("type") == "config":
                        client_sr = data.get("sampleRate", 16000)
                        state.speech_rate = data.get("speechRate", "+0%")
                        
                        # 1. Initialize Vosk as local fallback
                        if vosk_model:
                            state.recognizer = KaldiRecognizer(vosk_model, client_sr)
                            state.recognizer.SetWords(True)
                            logger.info(f"Initialized Vosk fallback ({vosk_model_name}) at {client_sr}Hz, rate: {state.speech_rate}")
                        
                        # 2. Connect to Deepgram streaming STT if configured
                        if config.DEEPGRAM_API_KEY and config.DEEPGRAM_API_KEY != "NOT_SET":
                            try:
                                logger.info("Connecting to Deepgram streaming STT service...")
                                dg_url = f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate={client_sr}&channels=1&model=nova-2&interim_results=true&endpointing=300"
                                dg_headers = {"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}
                                state.dg_socket = await websockets.connect(dg_url, additional_headers=dg_headers)
                                state.dg_receiver_task = asyncio.create_task(deepgram_receiver(state.dg_socket))
                                await websocket.send_text("STATUS:STT:🟢 Deepgram Nova-2 Active")
                                logger.info("Connected to Deepgram successfully.")
                            except Exception as e:
                                logger.error(f"Failed to connect to Deepgram (falling back to Vosk): {e}")
                                await websocket.send_text("STATUS:STT:🔴 Deepgram Failed, Using Vosk")
                        
                        client_ready_event.set()
                        continue
                        
                    elif isinstance(data, dict) and data.get("type") == "control":
                        new_rate = data.get("speechRate")
                        if new_rate:
                            state.speech_rate = new_rate
                            logger.info(f"Speech rate updated to: {state.speech_rate}")
                        continue

                    text_to_process = data.get("text", text) if isinstance(data, dict) else text
                    logger.info(f"Text input from {client_id}: {text_to_process}")
                    await utterance_queue.put((text_to_process, False))
                    continue


                if "bytes" in message:
                    audio_chunk = message["bytes"]
                    
                    dg_ok = False
                    if state.dg_socket:
                        try:
                            await state.dg_socket.send(audio_chunk)
                            dg_ok = True
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("Deepgram WS closed, falling back to Vosk")
                            state.dg_socket = None
                        except Exception as e:
                            logger.error(f"Error sending bytes to Deepgram, falling back to Vosk: {e}")
                            try:
                                await state.dg_socket.close()
                            except:
                                pass
                            state.dg_socket = None
                    
                    if not dg_ok and state.recognizer:
                        if state.recognizer.AcceptWaveform(audio_chunk):
                            result = json.loads(state.recognizer.Result())
                            text = result.get("text", "").strip()
                            if text and text != "__PLAYBACK_DONE__":
                                await interrupt_current_bot_action("Vosk final speech detected")

                                state.accumulated_words.append(text)
                                logger.info(f"Vosk final: '{text}' | accumulated: '{' '.join(state.accumulated_words)}'")
                                await websocket.send_text(f"STATUS:STT:🟢 Heard: {' '.join(state.accumulated_words)}")
                                
                                if state.silence_timer_task:
                                    state.silence_timer_task.cancel()
                                    
                                current_text = " ".join(state.accumulated_words).strip().lower()
                                sil_timeout = 0.2 if current_text in _QUICK_WORDS else 0.4
                                state.silence_timer_task = asyncio.create_task(silence_timer(sil_timeout))
                        else:
                            partial = json.loads(state.recognizer.PartialResult())
                            partial_text = partial.get("partial", "").strip()
                            if partial_text:
                                if len(partial_text.split()) > 0:
                                    await interrupt_current_bot_action("Vosk partial speech detected")

                                preview = " ".join(state.accumulated_words + [partial_text]) if state.accumulated_words else partial_text
                                await websocket.send_text(f"STATUS:STT:🟢 Hearing: {preview}")
                                
        except WebSocketDisconnect:
            await utterance_queue.put(None)
        except Exception as e:
            logger.error(f"Receiver error: {e}")
            await utterance_queue.put(None)
        finally:
            await utterance_queue.put(None)
            if state.dg_receiver_task:
                state.dg_receiver_task.cancel()
                await asyncio.gather(state.dg_receiver_task, return_exceptions=True)
            if state.dg_socket:
                try:
                    await state.dg_socket.close()
                except:
                    pass


    receiver_task = asyncio.create_task(receiver())

    try:
        while True:
            item = await utterance_queue.get()
            if item is None:
                break

            if isinstance(item, tuple):
                utterance, from_voice = item
            else:
                utterance, from_voice = item, False
                
            # If there is already an active utterance processing task running, cancel it!
            if state.current_utterance_task and not state.current_utterance_task.done():
                logger.info("Interrupting previous active utterance task on new utterance arrival.")
                state.current_utterance_task.cancel()
                try:
                    await state.current_utterance_task
                except asyncio.CancelledError:
                    pass

            state.current_utterance_task = asyncio.create_task(process_utterance(utterance, from_voice=from_voice))
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
    finally:
        receiver_task.cancel()
        if state.silence_timer_task:
            state.silence_timer_task.cancel()
            
        # Ensure Deepgram is closed in WebSocket disconnect as well
        if state.dg_receiver_task:
            state.dg_receiver_task.cancel()
            await asyncio.gather(state.dg_receiver_task, return_exceptions=True)
        if state.dg_socket:
            try:
                await state.dg_socket.close()
            except:
                pass
                
        # Remove active call from Redis and write to Postgres
        try:
            if app.state.redis:
                await app.state.redis.delete(f"active_calls:{client_id}")
            
            duration = int(time.time() - start_time)
            
            # Determine disposition based on [TRANSFER] flag in dialogue or transcript
            disposition = "DROP"
            for msg in conversation_history:
                content = msg.get("content", "")
                if "[TRANSFER]" in content:
                    disposition = "TRANSFER"
                    break
            
            # Record log to Postgres asynchronously
            if app.state.pg_pool:
                avg_latency = 0.45 # Average STT + TTS latency
                async with app.state.pg_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO call_logs (uniqueid, campaign_name, duration, disposition, avg_latency, transcript)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (uniqueid) DO NOTHING;
                    """, client_id, "default", duration, disposition, avg_latency, json.dumps(conversation_history))
                logger.info(f"Logged call {client_id} to database. Duration: {duration}s, Disposition: {disposition}")
        except Exception as e:
            logger.error(f"Failed to log call to database: {e}")
            
        logger.info(f"Client #{client_id} session ended.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
