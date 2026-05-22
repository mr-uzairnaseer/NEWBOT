import asyncio
import os
import httpx

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
        print(".env loaded successfully")
    except Exception as e:
        print(f"Could not load .env: {e}")

async def main():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
    
    print(f"ELEVENLABS_API_KEY: {api_key[:10]}...")
    print(f"ELEVENLABS_VOICE_ID: {voice_id}")
    print(f"ELEVENLABS_MODEL_ID: {model_id}")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "text": "Hello, my name is emily. How are you doing today?",
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, json=data, headers=headers)
        if response.status_code == 200:
            print("Synthesis succeeded! Size:", len(response.content), "bytes")
            with open("test_elevenlabs.mp3", "wb") as f:
                f.write(response.content)
            print("Saved to test_elevenlabs.mp3")
        else:
            print(f"Synthesis failed with status {response.status_code}: {response.text}")

if __name__ == "__main__":
    asyncio.run(main())
