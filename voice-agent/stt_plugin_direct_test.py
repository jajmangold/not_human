"""Test the EXACT STT plugin instantiation agent.py uses, directly against
clean pre-loaded audio -- isolates "does openai.STT() correctly talk to
crispasr-stt" from "does my synthetic LiveKit room audio feed introduce
frame-timing/VAD artifacts", which the e2e room test could not distinguish."""
import asyncio
import os

from dotenv import load_dotenv
from livekit import rtc
from livekit.plugins import openai

load_dotenv(dotenv_path="../.env")

WHISPER_STT_URL = os.environ.get("WHISPER_STT_URL", "http://127.0.0.1:18097")
WAV = "corpus/spk260_2.wav"
EXPECTED = "OH WON'T SHE BE SAVAGE IF I'VE KEPT HER WAITING"


async def main():
    stt = openai.STT(
        base_url=f"{WHISPER_STT_URL}/v1",
        api_key="not-required-on-tailnet",
        model="whisper-1",
        language="en",
    )

    import wave
    with wave.open(WAV, "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())

    frame = rtc.AudioFrame(data=raw, sample_rate=sr, num_channels=1, samples_per_channel=len(raw) // 2)
    print(f"expected: {EXPECTED!r}")
    result = await stt.recognize(buffer=frame)
    print(f"crispasr-stt (via the exact same openai.STT plugin agent.py uses) returned: {result.alternatives[0].text!r}")


asyncio.run(main())
