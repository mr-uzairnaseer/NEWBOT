import asyncio
import numpy as np
import sounddevice as sd
import websockets
import av
import io
import json

WS_URL = "ws://localhost:8080/ws/mock_dialer_01"
SAMPLE_RATE = 16000
CHANNELS = 1

def decode_mp3_to_pcm16(mp3_bytes: bytes, target_rate: int = 24000) -> bytes:
    """Decode MP3 audio bytes to raw PCM16 bytes using PyAV."""
    try:
        input_file = io.BytesIO(mp3_bytes)
        container = av.open(input_file)
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format='s16', layout='mono', rate=target_rate)
        
        pcm_chunks = []
        for packet in container.demux(stream):
            for frame in packet.decode():
                resampled = resampler.resample(frame)
                for f in resampled:
                    pcm_chunks.append(f.to_ndarray().tobytes())
        return b"".join(pcm_chunks)
    except Exception as e:
        print(f"Error decoding MP3 stream: {e}")
        return b""

async def run():
    print(f"Connecting to {WS_URL}...")
    async with websockets.connect(WS_URL) as ws:
        print("Connected. Sending configuration...")
        await ws.send(json.dumps({
            "type": "config",
            "sampleRate": 16000,
            "speechRate": "+20%"
        }))
        print("Connected. Speak now.")
        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue()

        def callback(indata, outdata, frames, time, status):
            # Send mic audio as int16
            data = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            asyncio.run_coroutine_threadsafe(audio_queue.put(data), loop)

        stream = sd.Stream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback)
        with stream:
            async def sender():
                while True:
                    await ws.send(await audio_queue.get())

            async def receiver():
                while True:
                    try:
                        msg = await ws.recv()
                        if isinstance(msg, bytes):
                            # Decode incoming MP3 stream to raw PCM16 bytes
                            pcm_data = decode_mp3_to_pcm16(msg, target_rate=24000)
                            if pcm_data:
                                audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                                audio = audio.reshape(-1, 1)
                                sd.play(audio, samplerate=24000)
                                sd.wait()
                                await ws.send("__PLAYBACK_DONE__")
                        else:
                            print(msg)
                            if "[TRANSFER]" in msg or "[DROP]" in msg:
                                print("Call ended.")
                                break
                    except websockets.ConnectionClosed:
                        break

            t1 = asyncio.create_task(sender())
            t2 = asyncio.create_task(receiver())
            done, pending = await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()

if __name__ == "__main__":
    asyncio.run(run())