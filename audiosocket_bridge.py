import asyncio
import httpx
import websockets
import json
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("audiosocket_bridge")

# Configuration
TCP_HOST = "0.0.0.0"
TCP_PORT = 9092
WS_URL_TEMPLATE = "ws://127.0.0.1:8080/ws/{uuid}"
SETTINGS_API_URL = "http://127.0.0.1:8080/api/control/settings"

# 20ms audio chunk size for signed-linear 16-bit PCM at 8000Hz mono
# 8000 samples/sec * 2 bytes/sample * 0.02 sec = 320 bytes
CHUNK_SIZE = 320
PACING_INTERVAL = 0.02 # 20 milliseconds

async def transcode_mp3_to_pcm(mp3_bytes: bytes, target_sr: int = 8000) -> bytes:
    """Decode MP3 bytes into raw pcm_s16le 8000Hz mono audio using FFmpeg."""
    if not mp3_bytes:
        return b""
    try:
        process = await asyncio.create_subprocess_exec(
            'ffmpeg', '-i', 'pipe:0', 
            '-f', 's16le', '-acodec', 'pcm_s16le', 
            '-ar', str(target_sr), '-ac', '1', 'pipe:1',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )
        stdout, _ = await process.communicate(input=mp3_bytes)
        return stdout
    except Exception as e:
        logger.error(f"FFmpeg transcoding failed: {e}")
        return b""

async def fetch_speech_rate() -> str:
    """Fetch current active speech rate from FastAPI controls to keep TTS synchronized."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(SETTINGS_API_URL)
            if response.status_code == 200:
                data = response.json()
                return data.get("active_speech_speed", "+20%")
    except Exception as e:
        logger.warning(f"Failed to fetch active speech rate, using default +20%: {e}")
    return "+20%"

async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info('peername')
    logger.info(f"Incoming AudioSocket connection from {peer}")
    
    ws = None
    tcp_to_ws_task = None
    ws_to_tcp_task = None
    
    try:
        # 1. Parse AudioSocket Header: 3 bytes
        # Byte 0: Type (0x01 = UUID, 0x10 = Audio, 0x02 = Hangup)
        # Bytes 1-2: Big-endian unsigned 16-bit Length
        header = await reader.readexactly(3)
        msg_type = header[0]
        length = (header[1] << 8) | header[2]
        
        if msg_type != 0x01:
            logger.error(f"First packet must be UUID (type 0x01), got {msg_type}. Closing connection.")
            writer.close()
            await writer.wait_closed()
            return
            
        # 2. Extract UUID payload
        uuid_bytes = await reader.readexactly(length)
        uuid_str = uuid_bytes.decode('utf-8', errors='replace').strip()
        logger.info(f"Parsed call UUID: '{uuid_str}'")
        
        # 3. Dynamic Configuration settings
        speech_rate = await fetch_speech_rate()
        logger.info(f"Using speech rate config: '{speech_rate}'")
        
        # 4. Connect to Voicebot WebSocket service
        ws_url = WS_URL_TEMPLATE.format(uuid=uuid_str)
        logger.info(f"Establishing WebSocket tunnel to {ws_url}...")
        ws = await websockets.connect(ws_url)
        
        # Send initial config to bot
        config_payload = {
            "type": "config",
            "sampleRate": 8000, # AudioSocket is 8000Hz mono standard
            "speechRate": speech_rate
        }
        await ws.send(json.dumps(config_payload))
        logger.info("Config payload successfully tunneled to bot.")
        
        # Stop flag to signal cooperative exit of streaming loops
        stop_event = asyncio.Event()

        # Audio playback cancellation queue
        playback_queue = asyncio.Queue()
        
        # Task A: TCP Socket (Asterisk Audio) -> WebSocket (FastAPI STT)
        async def tcp_to_ws():
            try:
                while not stop_event.is_set():
                    # Read 3-byte AudioSocket header
                    hdr = await reader.readexactly(3)
                    t = hdr[0]
                    l = (hdr[1] << 8) | hdr[2]
                    
                    if t == 0x02: # Hangup frame
                        logger.info(f"[{uuid_str}] Hangup frame received from Asterisk.")
                        stop_event.set()
                        break
                        
                    payload = await reader.readexactly(l)
                    
                    if t == 0x10: # Audio frame
                        # Send binary audio bytes directly to WebSocket
                        await ws.send(payload)
                    elif t == 0x03: # DTMF keypress
                        dtmf_digit = payload.decode('utf-8', errors='replace')
                        logger.info(f"[{uuid_str}] Caller pressed DTMF key: '{dtmf_digit}'")
            except asyncio.IncompleteReadError:
                logger.info(f"[{uuid_str}] Asterisk disconnected (TCP EOF).")
                stop_event.set()
            except Exception as e:
                logger.error(f"[{uuid_str}] Error in TCP-to-WS: {e}")
                stop_event.set()
                
        # Task B: WebSocket (Bot TTS MP3) -> TCP Socket (Asterisk Audio PCM)
        async def ws_to_tcp():
            try:
                async for msg in ws:
                    if stop_event.is_set():
                        break
                        
                    if isinstance(msg, str):
                        # Text control messages
                        if msg == "CTRL:STOP_AUDIO":
                            logger.info(f"[{uuid_str}] Interruption barge-in detected. Flushing playback buffer.")
                            # Clear the playback queue instantly to stop playing sound
                            while not playback_queue.empty():
                                playback_queue.get_nowait()
                        continue
                        
                    elif isinstance(msg, bytes):
                        # Audio MP3 buffer from TTS engine
                        # 1. Transcode MP3 -> signed 16-bit PCM 8000Hz mono
                        pcm_audio = await transcode_mp3_to_pcm(msg, target_sr=8000)
                        if pcm_audio:
                            # Feed the PCM bytes into our pacing queue
                            await playback_queue.put(pcm_audio)
                            
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"[{uuid_str}] Bot WebSocket disconnected.")
                stop_event.set()
            except Exception as e:
                logger.error(f"[{uuid_str}] Error in WS-to-TCP: {e}")
                stop_event.set()

        # Task C: Paced playback worker to prevent buffer overflows in Asterisk
        async def pcm_pacing_worker():
            try:
                while not stop_event.is_set():
                    pcm_data = await playback_queue.get()
                    
                    # Split raw PCM data into 20ms chunks (320 bytes)
                    idx = 0
                    while idx < len(pcm_data) and not stop_event.is_set():
                        # If the playback queue was cleared during this loop (interruption), break early!
                        if playback_queue.empty() and idx > 0 and len(pcm_data) - idx > CHUNK_SIZE * 5:
                            # Quick protection check to stop playing if we were interrupted
                            pass
                            
                        chunk = pcm_data[idx:idx+CHUNK_SIZE]
                        idx += CHUNK_SIZE
                        
                        # Pad trailing chunk if smaller than CHUNK_SIZE
                        if len(chunk) < CHUNK_SIZE:
                            chunk = chunk + (b'\x00' * (CHUNK_SIZE - len(chunk)))
                            
                        # Wrap inside AudioSocket frame (Type 0x10, Length 320 (0x0140))
                        # Length 320 -> High byte: 0x01, Low byte: 0x40
                        header = bytes([0x10, 0x01, 0x40])
                        writer.write(header + chunk)
                        await writer.drain()
                        
                        # Sleep exactly 20ms to match real-time playback speed
                        await asyncio.sleep(PACING_INTERVAL)
                        
            except Exception as e:
                logger.error(f"[{uuid_str}] Error in PCM Pacing Worker: {e}")
                stop_event.set()

        # Spawn concurrent transport futures
        tcp_to_ws_task = asyncio.create_task(tcp_to_ws())
        ws_to_tcp_task = asyncio.create_task(ws_to_tcp())
        pacing_task = asyncio.create_task(pcm_pacing_worker())
        
        # Wait until stop_event is fired by any side
        await stop_event.wait()
        
        # Clean shutdown of tasks
        tcp_to_ws_task.cancel()
        ws_to_tcp_task.cancel()
        pacing_task.cancel()
        
        await asyncio.gather(tcp_to_ws_task, ws_to_tcp_task, pacing_task, return_exceptions=True)
        
    except Exception as e:
        logger.error(f"Handler general exception: {e}")
    finally:
        # Tear down connection states safely
        logger.info(f"Tearing down bridge connection for peer {peer}")
        if ws:
            try:
                await ws.close()
            except:
                pass
        writer.close()
        try:
            await writer.wait_closed()
        except:
            pass
        logger.info(f"Bridge connection clean-up complete for peer {peer}")

async def main():
    logger.info("==================================================")
    logger.info("Starting AudioSocket-to-WebSocket Bridge Service...")
    logger.info(f"Binding TCP server to {TCP_HOST}:{TCP_PORT}")
    logger.info(f"Routing websocket traffic to {WS_URL_TEMPLATE}")
    logger.info("==================================================")
    
    server = await asyncio.start_server(handle_connection, TCP_HOST, TCP_PORT)
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bridge server stopped by user interrupt.")
