import av
import io
import os

def test_decode():
    mp3_path = "test_elevenlabs.mp3"
    if not os.path.exists(mp3_path):
        print(f"File {mp3_path} does not exist. Skipping.")
        return
        
    with open(mp3_path, "rb") as f:
        mp3_bytes = f.read()
        
    print(f"MP3 size: {len(mp3_bytes)} bytes")
    
    input_file = io.BytesIO(mp3_bytes)
    container = av.open(input_file)
    stream = container.streams.audio[0]
    
    resampler = av.AudioResampler(
        format='s16',
        layout='mono',
        rate=16000
    )
    
    pcm_bytes = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            resampled = resampler.resample(frame)
            for f in resampled:
                pcm_bytes.append(f.to_ndarray().tobytes())
                
    full_pcm = b"".join(pcm_bytes)
    print(f"Decoded PCM size: {len(full_pcm)} bytes")
    
if __name__ == "__main__":
    test_decode()
