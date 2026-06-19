from __future__ import annotations

import base64
import io
import subprocess
import wave
from pathlib import Path
from typing import Any

import cv2
import librosa
import numpy as np


def encode_jpeg_rgb(frame: np.ndarray, quality: int, max_size: int = 0) -> bytes:
    if max_size and max(frame.shape[0], frame.shape[1]) > max_size:
        height, width = frame.shape[:2]
        scale = max_size / max(height, width)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def decode_omni_audio_delta(audio_b64: str) -> np.ndarray:
    audio_bytes = base64.b64decode(audio_b64)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return (audio_int16.astype(np.float32) / 32768.0).astype(np.float32)


def save_pcm16_wav(audio_chunks: list[bytes], path: Path, sample_rate: int = 24000) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = b"".join(audio_chunks)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    samples = len(pcm) // 2
    return {
        "path": str(path),
        "bytes": len(pcm),
        "samples": samples,
        "duration_s": samples / float(sample_rate),
        "sample_rate": sample_rate,
    }


def pcm16_silence_bytes(duration_ms: int, sample_rate: int = 24000) -> bytes:
    samples = max(0, int(sample_rate * duration_ms / 1000))
    return np.zeros((samples,), dtype=np.int16).tobytes()


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = np.where(audio < 0, audio * 32768.0, audio * 32767.0).astype(np.int16)
    return pcm.tobytes()


def float32_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(float32_to_pcm16_bytes(audio))
    return bio.getvalue()


def resample_float32(audio: np.ndarray, orig_sr: int, target_sr: int = 24000) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if int(orig_sr) == int(target_sr):
        return audio
    return librosa.resample(audio, orig_sr=int(orig_sr), target_sr=int(target_sr)).astype(np.float32)


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {result.stderr.strip() or result.stdout.strip()}")
    return {"path": str(output_path), "bytes": output_path.stat().st_size if output_path.exists() else 0}
