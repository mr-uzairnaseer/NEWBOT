import asyncio
import websockets
import time

async def test():
    url = "ws://127.0.0.1:8080/ws/browser_test"
    print(f"Connecting to {url}...")
    t0 = time.time()
    try:
        async with websockets.connect(url) as ws:
            t_connect = time.time() - t0
            print(f"Connected in {t_connect:.3f} seconds!")
            
            # Wait to receive the greeting text and audio
            audio_count = 0
            text_received = False
            t_start = time.time()
            
            while True:
                msg = await ws.recv()
                t_elapsed = time.time() - t_start
                if isinstance(msg, bytes):
                    audio_count += 1
                    print(f"[{t_elapsed:.3f}s] Received audio chunk #{audio_count} ({len(msg)} bytes)")
                    # We only care about the first few messages
                    if audio_count >= 1:
                        break
                else:
                    print(f"[{t_elapsed:.3f}s] Received text: {msg}")
                    
    except Exception as e:
        print(f"Connection failed: {e}")

asyncio.run(test())
