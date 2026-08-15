"""
Speech-to-Text (Sarvam AI) Independent Test & Latency Benchmark Script
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Generates sample PCM audio header, sends payload to Sarvam STT service,
and benchmarks STT response latency (Target: < 100 ms).
"""

import os
import sys
import wave
import io

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.services.stt_service import SarvamSTTService

def create_sample_wav_bytes() -> bytes:
    """
    Generates a valid 1-second 16kHz mono WAV audio file in-memory.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)     # Mono
        wav_file.setsampwidth(2)     # 16-bit
        wav_file.setframerate(16000) # 16kHz
        # Write 16000 zero-amplitude silent sample frames
        wav_file.writeframes(b'\x00\x00' * 16000)
    return buffer.getvalue()

def test_stt_pipeline():
    print("=" * 75)
    print("SARVAM AI SPEECH-TO-TEXT (STT) BENCHMARK")
    print("=" * 75)

    audio_bytes = create_sample_wav_bytes()
    print(f"[*] Generated Sample Audio Buffer Size: {len(audio_bytes)} bytes (16kHz Mono WAV)")

    stt_service = SarvamSTTService()
    print(f"[*] Executing Speech-to-Text Transcription via {stt_service.api_url}...")

    response = stt_service.transcribe_audio_bytes(audio_bytes, filename="test_audio.wav")

    print("\n" + "-" * 75)
    print("STT TRANSCRIPTION RESULT:")
    print("-" * 75)
    print(f"  • STT Provider       : {response.provider}")
    print(f"  • Detected Language  : {response.language_code}")
    print(f"  • Transcript Output  : \"{response.transcript}\"")
    print(f"  • STT Turnaround Time: {response.latency_ms:.2f} ms (Target: < 100 ms)")
    print(f"  • Confidence Score   : {response.confidence:.2f}")
    print("=" * 75)

if __name__ == "__main__":
    test_stt_pipeline()
