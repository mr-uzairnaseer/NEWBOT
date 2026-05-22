import asyncio
import edge_tts

async def main():
    # Print the available formats or options
    communicate = edge_tts.Communicate("Hello", "en-US-GuyNeural")
    print("Communicate class attributes:", dir(communicate))
    
if __name__ == "__main__":
    asyncio.run(main())
