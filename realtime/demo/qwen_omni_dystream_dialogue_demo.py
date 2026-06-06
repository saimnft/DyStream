"""Browser voice -> Qwen-Omni -> DyStream realtime talking-head demo.

This file is intentionally separate from websocket_mic_demo.py so the existing
browser-mic -> DyStream realtime video path is not affected.

Run example:
    export DASHSCOPE_API_KEY=sk-xxx
    python realtime/demo/qwen_omni_dystream_dialogue_demo.py \
        --ref-image img_files/11_resize.png \
        --ref-motion img_files/11.npz \
        --no-preprocess-image \
        --lookahead-frames 20 \
        --denoising-steps 1 \
        --guidance-mode all_only \
        --render-mode batch \
        --async-render \
        --port 7863

Then open http://127.0.0.1:7863.
"""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
import time
import traceback
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "FastAPI/uvicorn are required for this demo. Install with:\n"
        "  pip install fastapi uvicorn\n"
        f"Original import error: {exc}"
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtime.demo.qwen_omni_browser_voice_demo import _iter_dashscope_omni_chunks  # noqa: E402
from realtime.dystream.stateful_stream_engine import StatefulStreamDyStreamEngine  # noqa: E402


INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Qwen-Omni DyStream Dialogue Demo</title>
  <style>
    body { font-family: sans-serif; margin: 24px; background: #111; color: #eee; }
    button { font-size: 16px; padding: 8px 16px; margin-right: 8px; }
    textarea { width: 720px; height: 80px; background: #1c1c1c; color: #eee; }
    canvas { margin-top: 16px; background: #222; width: 512px; height: 512px; }
    pre { white-space: pre-wrap; background: #1c1c1c; padding: 12px; border-radius: 6px; width: 720px; min-height: 220px; }
    .hint { color: #aaa; }
  </style>
</head>
<body>
  <h2>Qwen-Omni → DyStream Realtime Dialogue Demo</h2>
  <p class="hint">Record one user utterance per turn. The WebSocket can stay open for multiple half-duplex dialogue turns.</p>
  <div>
    <button id="startBtn">Start Recording</button>
    <button id="stopBtn" disabled>Stop & Send</button>
  </div>
  <p>Optional prompt:</p>
  <textarea id="prompt">Please answer briefly in Chinese.</textarea>
  <div><canvas id="canvas" width="512" height="512"></canvas></div>
  <pre id="log"></pre>

<script>
let ws = null;
let audioCtx = null;
let source = null;
let processor = null;
let mediaStream = null;
let chunks = [];
let sampleRate = 0;
let recording = false;
let recvFrames = 0;
let turnId = 0;

const logEl = document.getElementById('log');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function floatTo16BitPCM(view, offset, input) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
}

function encodeWav(samples, sr) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);
  floatTo16BitPCM(view, 44, samples);
  return buffer;
}

function mergeChunks(chunks) {
  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

async function drawJpegBlob(blob) {
  const bmp = await createImageBitmap(blob);
  ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
  bmp.close();
}

async function start() {
  if (recording) return;
  if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
    logEl.textContent = '';
    recvFrames = 0;
    turnId = 0;
    const wsProtocol = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${wsProtocol}://${location.host}/ws/dialogue`);
    ws.binaryType = 'blob';
    ws.onopen = () => log('WebSocket opened. Recording...');
    ws.onmessage = async (event) => {
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'text_delta') log(`text chunk: ${msg.data}`);
          else if (msg.type === 'timing_event') {
            log(`timing turn=${msg.turn_id} ${msg.event}: +${msg.elapsed_s.toFixed(3)}s`);
          }
          else if (msg.type === 'timing_summary') {
            log(`timing summary turn=${msg.turn_id}: ${JSON.stringify(msg.timings_s)}`);
          }
          else if (msg.type === 'frame_status') log(`turn=${msg.turn_id ?? '-'} source=${msg.source ?? '-'} preview_total=${msg.sent_frames}, generated=${msg.frames}, saved_video=${msg.saved_video_frames ?? 0}`);
          else if (msg.type === 'turn_done') {
            log(`turn ${msg.turn_id} done. sent_frames=${msg.sent_frames}, rendered=${msg.rendered_frames_seen}, saved_video=${msg.saved_video_frames}`);
            if (msg.assistant_audio_path) log(`assistant audio: ${msg.assistant_audio_path}`);
            if (msg.assistant_audio_padded_path) log(`assistant padded audio: ${msg.assistant_audio_padded_path}`);
            if (msg.av_output_path) log(`AV output: ${msg.av_output_path}`);
            if (msg.mux_error) log(`AV mux error: ${msg.mux_error}`);
            document.getElementById('startBtn').disabled = false;
          }
          else if (msg.type === 'done') {
            log(`done. sent_frames=${msg.sent_frames}`);
            if (msg.assistant_audio_path) log(`assistant audio: ${msg.assistant_audio_path}`);
            if (msg.assistant_audio_padded_path) log(`assistant padded audio: ${msg.assistant_audio_padded_path}`);
            if (msg.av_output_path) log(`AV output: ${msg.av_output_path}`);
            document.getElementById('startBtn').disabled = false;
          }
          else if (msg.type === 'error') {
            log(`ERROR: ${msg.message}`);
            document.getElementById('startBtn').disabled = false;
          }
          else log(event.data);
        } catch (_) {
          log(event.data);
        }
        return;
      }
      recvFrames += 1;
      await drawJpegBlob(event.data);
      if (recvFrames % 25 === 0) log(`received video frames=${recvFrames}`);
    };
    ws.onclose = () => {
      log('WebSocket closed.');
      document.getElementById('startBtn').disabled = false;
      document.getElementById('stopBtn').disabled = true;
    };
    ws.onerror = (err) => { log('WebSocket error. See browser console.'); console.error(err); };
  }

  chunks = [];
  recording = true;
  turnId += 1;
  document.getElementById('startBtn').disabled = true;
  document.getElementById('stopBtn').disabled = false;

  if (ws.readyState !== WebSocket.OPEN) {
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', resolve, { once: true });
      ws.addEventListener('error', reject, { once: true });
    });
  }

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  sampleRate = audioCtx.sampleRate;
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  source = audioCtx.createMediaStreamSource(mediaStream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (event) => {
    if (!recording) return;
    const input = event.inputBuffer.getChannelData(0);
    const copy = new Float32Array(input.length);
    copy.set(input);
    chunks.push(copy);
  };
  source.connect(processor);
  processor.connect(audioCtx.destination);
  log(`Mic started. sample_rate=${sampleRate}`);
}

async function stopAndSend() {
  recording = false;
  document.getElementById('stopBtn').disabled = true;
  if (processor) processor.disconnect();
  if (source) source.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  if (audioCtx) await audioCtx.close();

  const samples = mergeChunks(chunks);
  const wav = encodeWav(samples, sampleRate);
  log(`Sending WAV: samples=${samples.length}, sample_rate=${sampleRate}, bytes=${wav.byteLength}`);
  ws.send(JSON.stringify({
    type: 'audio',
    turn_id: turnId,
    mime_type: 'audio/wav',
    sample_rate: sampleRate,
    prompt: document.getElementById('prompt').value || ''
  }));
  ws.send(wav);
  log(`Turn ${turnId} sent; waiting for response...`);
}

document.getElementById('startBtn').onclick = start;
document.getElementById('stopBtn').onclick = stopAndSend;
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Omni audio chunks to DyStream video dialogue demo.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--model", default=os.getenv("QWEN_OMNI_MODEL", "qwen3.5-omni-plus"))
    parser.add_argument("--voice", default=os.getenv("QWEN_OMNI_VOICE", "Tina"))
    parser.add_argument("--system-prompt", default="你是一个端到端数字人对话助手，请回答简洁自然。")

    parser.add_argument("--ref-image", default="img_files/11_resize.png")
    parser.add_argument("--ref-motion", default="img_files/11.npz")
    parser.add_argument("--lookahead-frames", type=int, default=20)
    parser.add_argument("--audio-encode-left-context-frames", type=int, default=96)
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--guidance-mode", choices=["full", "uncond_all", "all_only"], default="all_only")
    parser.add_argument("--render-mode", choices=["per_frame", "batch", "none"], default="batch")
    parser.add_argument("--render-frame-stride", type=int, default=1)
    parser.add_argument("--async-render", action="store_true")
    parser.add_argument("--no-preprocess-image", action="store_true")
    parser.add_argument("--preprocess-output-dir", default="realtime/outputs/preprocess")
    parser.add_argument("--encode-listener-audio", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-audio", action="store_true")
    parser.add_argument("--amp-motion", action="store_true")
    parser.add_argument("--amp-render", action="store_true")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--jpeg-max-size", type=int, default=0)
    parser.add_argument("--send-frame-stride", type=int, default=1, help="Send only every Nth rendered frame to the browser. 1 sends all frames; 0 disables video downlink while still rendering/verifying frames server-side.")
    parser.add_argument("--verify-output-dir", default="", help="If set, save server-side verification JPEG frames here without relying on browser downlink.")
    parser.add_argument("--verify-save-every-frames", type=int, default=25, help="Save every Nth rendered frame to --verify-output-dir. 0 disables saving.")
    parser.add_argument("--verify-video-path", default="", help="If set, stream all rendered frames into this server-side MP4/AVI file without relying on browser downlink.")
    parser.add_argument("--verify-video-fps", type=float, default=25.0, help="FPS metadata for --verify-video-path.")
    parser.add_argument("--tail-silence-ms", type=int, default=1000, help="Append silence after Omni finishes so lookahead-based DyStream can render the tail.")
    parser.add_argument("--multi-turn", action="store_true", help="Keep the WebSocket open for multiple half-duplex dialogue turns and maintain lightweight text history.")
    parser.add_argument("--max-history-turns", type=int, default=3, help="Number of recent assistant responses to include in the next turn prompt.")
    parser.add_argument("--session-output-dir", default="", help="If set, create a timestamped session directory and save each dialogue turn under turn_XXX/.")
    parser.add_argument("--warmup-engine", action="store_true", help="Build the DyStream engine as soon as the WebSocket connects and reuse it across turns with reset_stream_state().")
    parser.add_argument("--startup-warmup", action="store_true", help="Warm up DyStream once before starting uvicorn so first browser interaction does not pay model loading cost.")
    return parser.parse_args()


def encode_jpeg_rgb(frame: np.ndarray, quality: int, max_size: int = 0) -> bytes:
    if max_size and max(frame.shape[0], frame.shape[1]) > max_size:
        h, w = frame.shape[:2]
        scale = max_size / max(h, w)
        frame = cv2.resize(frame, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
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


def build_engine(args: argparse.Namespace) -> StatefulStreamDyStreamEngine:
    return StatefulStreamDyStreamEngine(
        ref_image_path=args.ref_image,
        ref_motion_path=args.ref_motion,
        denoising_steps=args.denoising_steps,
        preprocess_image=not args.no_preprocess_image,
        preprocess_output_dir=args.preprocess_output_dir,
        lookahead_frames=args.lookahead_frames,
        audio_encode_left_context_frames=args.audio_encode_left_context_frames,
        encode_listener_audio=args.encode_listener_audio,
        guidance_mode=args.guidance_mode,
        render_mode=args.render_mode,
        render_frame_stride=args.render_frame_stride,
        async_render=args.async_render,
        use_ema=not args.no_ema,
        amp=args.amp,
        amp_audio=args.amp_audio,
        amp_motion=args.amp_motion,
        amp_render=args.amp_render,
        amp_dtype=args.amp_dtype,
        profile=False,
    )


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse(INDEX_HTML)

    @app.websocket("/ws/dialogue")
    async def websocket_dialogue(ws: WebSocket):
        await ws.accept()
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            await ws.send_json({"type": "error", "message": f"Missing env var: {args.api_key_env}"})
            await ws.close()
            return
        try:
            api_key.encode("ascii")
        except UnicodeEncodeError:
            await ws.send_json({
                "type": "error",
                "message": (
                    f"Env var {args.api_key_env} contains non-ASCII characters. "
                    "Please export the real DashScope API key, not a placeholder like '你的 DashScope API Key'."
                ),
            })
            await ws.close()
            return

        engine: Optional[StatefulStreamDyStreamEngine] = None
        sent_frames = 0
        rendered_frames_seen = 0
        saved_verify_frames = 0
        saved_video_frames = 0
        last_frame_mean = 0.0
        last_frame_std = 0.0
        last_frame_checksum = 0
        verify_video_writer: Optional[Any] = None
        verify_video_backend = ""
        verify_output_dir: Optional[Path] = None
        verify_video_path: Optional[Path] = None
        assistant_audio_path: Optional[Path] = None
        assistant_audio_padded_path: Optional[Path] = None
        av_output_path: Optional[Path] = None
        assistant_history: list[str] = []
        turn_id = 0
        session_dir: Optional[Path] = None
        if args.multi_turn or args.session_output_dir:
            session_base_dir = Path(args.session_output_dir or "realtime/outputs/omni_dialogue_sessions")
            session_dir = session_base_dir / datetime.now().strftime("session_%Y%m%d_%H%M%S")
            session_dir.mkdir(parents=True, exist_ok=True)
            print(f"[QwenOmniDyStreamDialogueDemo] session output dir: {session_dir}")

        def build_turn_prompt(prompt: str) -> str:
            prompt = prompt or "Please answer the latest voice input briefly in Chinese."
            recent_history = assistant_history[-max(0, int(args.max_history_turns)):]
            if not recent_history:
                return prompt
            history_text = "\n".join(
                f"Assistant previous turn {idx + 1}: {text}"
                for idx, text in enumerate(recent_history)
            )
            return (
                "This is an ongoing multi-turn voice conversation.\n"
                "The exact previous user audio transcripts are unavailable, but your previous answers are listed below.\n"
                f"{history_text}\n\n"
                "Now answer the user's latest voice input. Follow the system prompt's language and style requirements.\n"
                f"Current turn instruction: {prompt}"
            )

        def setup_turn_outputs(current_turn_id: int) -> None:
            nonlocal verify_output_dir, verify_video_path, assistant_audio_path, assistant_audio_padded_path, av_output_path
            if session_dir is not None:
                turn_dir = session_dir / f"turn_{current_turn_id:03d}"
                turn_dir.mkdir(parents=True, exist_ok=True)
                verify_output_dir = turn_dir
                verify_video_path = turn_dir / "dialogue_video.mp4"
                assistant_audio_path = turn_dir / "assistant_audio.wav"
                assistant_audio_padded_path = turn_dir / "assistant_audio_padded.wav"
                av_output_path = turn_dir / "dialogue_av.mp4"
                return
            verify_output_dir = Path(args.verify_output_dir) if args.verify_output_dir else None
            if verify_output_dir is not None:
                verify_output_dir.mkdir(parents=True, exist_ok=True)
            verify_video_path = Path(args.verify_video_path) if args.verify_video_path else None
            if verify_video_path is not None:
                verify_video_path.parent.mkdir(parents=True, exist_ok=True)
                assistant_audio_path = verify_video_path.with_name(f"{verify_video_path.stem}_assistant_audio.wav")
                assistant_audio_padded_path = verify_video_path.with_name(f"{verify_video_path.stem}_assistant_audio_padded.wav")
                av_output_path = verify_video_path.with_name(f"{verify_video_path.stem}_av.mp4")
            elif verify_output_dir is not None:
                assistant_audio_path = verify_output_dir / "assistant_audio.wav"
                assistant_audio_padded_path = verify_output_dir / "assistant_audio_padded.wav"
                av_output_path = None
            else:
                assistant_audio_path = None
                assistant_audio_padded_path = None
                av_output_path = None

        def reset_turn_counters() -> None:
            nonlocal sent_frames, rendered_frames_seen, saved_verify_frames, saved_video_frames
            nonlocal last_frame_mean, last_frame_std, last_frame_checksum
            nonlocal verify_video_writer, verify_video_backend
            sent_frames = 0
            rendered_frames_seen = 0
            saved_verify_frames = 0
            saved_video_frames = 0
            last_frame_mean = 0.0
            last_frame_std = 0.0
            last_frame_checksum = 0
            verify_video_writer = None
            verify_video_backend = ""

        def close_verify_video_writer() -> None:
            nonlocal verify_video_writer
            if verify_video_writer is None:
                return
            if verify_video_backend == "imageio":
                verify_video_writer.close()
            else:
                verify_video_writer.release()
            print(
                f"[QwenOmniDyStreamDialogueDemo] verify video saved to {verify_video_path} "
                f"({saved_video_frames} frames, backend={verify_video_backend})"
            )
            verify_video_writer = None

        def write_verify_video_frame(frame: np.ndarray) -> None:
            nonlocal verify_video_writer, verify_video_backend, saved_video_frames
            if verify_video_path is None:
                return
            h, w = frame.shape[:2]
            if verify_video_writer is None:
                try:
                    import imageio.v2 as imageio

                    verify_video_writer = imageio.get_writer(
                        str(verify_video_path),
                        fps=float(args.verify_video_fps),
                        codec="libx264",
                        pixelformat="yuv420p",
                        macro_block_size=1,
                    )
                    verify_video_backend = "imageio"
                    print(f"[QwenOmniDyStreamDialogueDemo] verify video writer opened with imageio/libx264: {verify_video_path}")
                except Exception as exc:
                    print(f"[QwenOmniDyStreamDialogueDemo] imageio video writer failed ({exc}); falling back to OpenCV mp4v")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    verify_video_writer = cv2.VideoWriter(str(verify_video_path), fourcc, float(args.verify_video_fps), (w, h))
                    if not verify_video_writer.isOpened():
                        raise RuntimeError(f"Failed to open verify video writer: {verify_video_path}")
                    verify_video_backend = "opencv"
            if verify_video_backend == "imageio":
                verify_video_writer.append_data(frame)
            else:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                verify_video_writer.write(bgr)
            saved_video_frames += 1

        async def handle_rendered_frame(frame: np.ndarray) -> bool:
            """Verify/save a rendered frame and optionally downlink it to browser."""
            nonlocal rendered_frames_seen, saved_verify_frames, sent_frames
            nonlocal last_frame_mean, last_frame_std, last_frame_checksum
            rendered_frames_seen += 1
            last_frame_mean = float(np.mean(frame))
            last_frame_std = float(np.std(frame))
            last_frame_checksum = int(np.asarray(frame, dtype=np.uint64).sum() % 1_000_000_007)
            write_verify_video_frame(frame)

            if (
                verify_output_dir is not None
                and args.verify_save_every_frames > 0
                and rendered_frames_seen % args.verify_save_every_frames == 0
            ):
                verify_path = verify_output_dir / f"frame_{rendered_frames_seen:06d}.jpg"
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(verify_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
                saved_verify_frames += 1

            if args.send_frame_stride <= 0:
                return False
            if rendered_frames_seen % args.send_frame_stride != 0:
                return False
            await ws.send_bytes(encode_jpeg_rgb(frame, args.jpeg_quality, args.jpeg_max_size))
            sent_frames += 1
            return True

        async def handle_rendered_frames(frames: list[np.ndarray] | tuple[np.ndarray, ...], source: str, extra: Optional[dict[str, Any]] = None) -> int:
            preview_frames = 0
            handled_frames = 0
            for frame in frames:
                if args.render_mode == "none":
                    continue
                handled_frames += 1
                if await handle_rendered_frame(frame):
                    preview_frames += 1
            payload = {
                "type": "frame_status",
                "source": source,
                "frames": len(frames),
                "preview_frames": preview_frames,
                "sent_frames": sent_frames,
                "rendered_frames_seen": rendered_frames_seen,
                "saved_verify_frames": saved_verify_frames,
                "saved_video_frames": saved_video_frames,
                "last_frame_mean": last_frame_mean,
                "last_frame_std": last_frame_std,
                "last_frame_checksum": last_frame_checksum,
                "send_frame_stride": args.send_frame_stride,
            }
            if extra:
                payload.update(extra)
            await ws.send_json(payload)
            return handled_frames

        try:
            if args.warmup_engine:
                await ws.send_json({"type": "status", "message": "Warming up DyStream engine for this WebSocket session..."})
                engine = build_engine(args)
                await ws.send_json({"type": "status", "message": "DyStream engine warmup complete. You can start sending dialogue turns."})

            while True:
                meta = await ws.receive_json()
                if meta.get("type") == "stop":
                    break
                if meta.get("type") != "audio":
                    await ws.send_json({"type": "error", "message": "Expected JSON metadata with type='audio' first."})
                    if not args.multi_turn:
                        await ws.close()
                        return
                    continue
                message = await ws.receive()
                wav_bytes: Optional[bytes] = message.get("bytes")
                if not wav_bytes:
                    await ws.send_json({"type": "error", "message": "Expected WAV binary payload after metadata."})
                    if not args.multi_turn:
                        await ws.close()
                        return
                    continue

                turn_id += 1
                client_turn_id = int(meta.get("turn_id") or turn_id)
                reset_turn_counters()
                setup_turn_outputs(client_turn_id)
                assistant_text_parts: list[str] = []
                assistant_audio_chunks: list[bytes] = []
                turn_receive_s = time.perf_counter()
                timings_s: dict[str, float] = {"server_received_wav": 0.0}

                async def mark_timing(event: str) -> None:
                    elapsed_s = time.perf_counter() - turn_receive_s
                    timings_s.setdefault(event, elapsed_s)
                    print(f"[QwenOmniDyStreamDialogueTiming] turn={client_turn_id} {event}=+{elapsed_s:.3f}s")
                    await ws.send_json({
                        "type": "timing_event",
                        "turn_id": client_turn_id,
                        "event": event,
                        "elapsed_s": elapsed_s,
                    })

                await ws.send_json({
                    "type": "turn_start",
                    "turn_id": client_turn_id,
                    "session_dir": str(session_dir) if session_dir is not None else "",
                    "verify_video_path": str(verify_video_path) if verify_video_path is not None else "",
                    "assistant_audio_path": str(assistant_audio_path) if assistant_audio_path is not None else "",
                    "assistant_audio_padded_path": str(assistant_audio_padded_path) if assistant_audio_padded_path is not None else "",
                    "av_output_path": str(av_output_path) if av_output_path is not None else "",
                })
                if engine is None:
                    await ws.send_json({"type": "status", "message": "Loading DyStream engine..."})
                    engine_load_start_s = time.perf_counter()
                    engine = build_engine(args)
                    timings_s["engine_load"] = time.perf_counter() - engine_load_start_s
                    await mark_timing("engine_ready")
                else:
                    await ws.send_json({"type": "status", "message": "Resetting warm DyStream engine state..."})
                    engine_reset_start_s = time.perf_counter()
                    engine.reset_stream_state()
                    timings_s["engine_reset"] = time.perf_counter() - engine_reset_start_s
                    await mark_timing("engine_ready")
                await ws.send_json({"type": "status", "message": f"Calling {args.model}; streaming Omni audio into DyStream..."})

                prompt = build_turn_prompt(str(meta.get("prompt") or ""))
                await mark_timing("omni_call_start")
                for chunk in _iter_dashscope_omni_chunks(
                    api_key=api_key,
                    model=args.model,
                    wav_bytes=wav_bytes,
                    prompt=prompt,
                    system_prompt=args.system_prompt,
                    voice=args.voice,
                ):
                    chunk_type = chunk.get("type")
                    if chunk_type == "text_delta":
                        if "first_text_delta" not in timings_s:
                            await mark_timing("first_text_delta")
                        assistant_text_parts.append(str(chunk.get("data") or ""))
                        await ws.send_json({**chunk, "turn_id": client_turn_id})
                        continue
                    if chunk_type != "audio_delta":
                        continue

                    if "first_audio_delta" not in timings_s:
                        await mark_timing("first_audio_delta")
                    audio_b64 = str(chunk.get("base64") or "")
                    audio_bytes = base64.b64decode(audio_b64)
                    assistant_audio_chunks.append(audio_bytes)
                    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    omni_audio = (audio_int16.astype(np.float32) / 32768.0).astype(np.float32)
                    engine.push_audio_chunk(omni_audio, sample_rate=24000)
                    step_start_s = time.perf_counter()
                    frames = engine.step()
                    timings_s["dystream_step_total"] = timings_s.get("dystream_step_total", 0.0) + (time.perf_counter() - step_start_s)
                    handled_frames = await handle_rendered_frames(
                        frames,
                        source="omni_audio_delta",
                        extra={"turn_id": client_turn_id, "audio_bytes": int(chunk.get("bytes") or 0)},
                    )
                    if handled_frames > 0 and "first_dystream_frame" not in timings_s:
                        await mark_timing("first_dystream_frame")

                await mark_timing("omni_stream_end")

                if args.tail_silence_ms > 0:
                    silence_samples = int(24000 * args.tail_silence_ms / 1000)
                    engine.push_audio_chunk(np.zeros((silence_samples,), dtype=np.float32), sample_rate=24000)
                    step_start_s = time.perf_counter()
                    frames = engine.step()
                    timings_s["dystream_step_total"] = timings_s.get("dystream_step_total", 0.0) + (time.perf_counter() - step_start_s)
                    handled_frames = await handle_rendered_frames(frames, source="tail_silence", extra={"turn_id": client_turn_id})
                    if handled_frames > 0 and "first_dystream_frame" not in timings_s:
                        await mark_timing("first_dystream_frame")

                flush_start_s = time.perf_counter()
                flush_frames = list(engine.flush())
                timings_s["dystream_flush"] = time.perf_counter() - flush_start_s
                handled_frames = await handle_rendered_frames(flush_frames, source="flush", extra={"turn_id": client_turn_id})
                if handled_frames > 0 and "first_dystream_frame" not in timings_s:
                    await mark_timing("first_dystream_frame")
                close_video_start_s = time.perf_counter()
                close_verify_video_writer()
                timings_s["video_writer_close"] = time.perf_counter() - close_video_start_s

                audio_info: dict[str, Any] = {}
                padded_audio_info: dict[str, Any] = {}
                av_info: dict[str, Any] = {}
                mux_error = ""
                if assistant_audio_chunks and assistant_audio_path is not None:
                    audio_save_start_s = time.perf_counter()
                    audio_info = save_pcm16_wav(assistant_audio_chunks, assistant_audio_path, sample_rate=24000)
                    timings_s["assistant_audio_save"] = time.perf_counter() - audio_save_start_s
                if assistant_audio_chunks and assistant_audio_padded_path is not None:
                    padded_audio_save_start_s = time.perf_counter()
                    padded_chunks = list(assistant_audio_chunks)
                    if args.tail_silence_ms > 0:
                        padded_chunks.append(pcm16_silence_bytes(args.tail_silence_ms, sample_rate=24000))
                    padded_audio_info = save_pcm16_wav(padded_chunks, assistant_audio_padded_path, sample_rate=24000)
                    timings_s["assistant_audio_padded_save"] = time.perf_counter() - padded_audio_save_start_s
                mux_audio_path = assistant_audio_padded_path if assistant_audio_padded_path is not None and assistant_audio_padded_path.exists() else assistant_audio_path
                if verify_video_path is not None and av_output_path is not None and mux_audio_path is not None and mux_audio_path.exists() and verify_video_path.exists():
                    mux_start_s = time.perf_counter()
                    try:
                        av_info = mux_video_audio(verify_video_path, mux_audio_path, av_output_path)
                    except Exception as exc:
                        mux_error = str(exc)
                        print(f"[QwenOmniDyStreamDialogueDemo] AV mux failed: {mux_error}")
                    timings_s["av_mux"] = time.perf_counter() - mux_start_s
                if saved_video_frames > 0:
                    timings_s["video_duration_s"] = saved_video_frames / float(args.verify_video_fps)
                if audio_info:
                    timings_s["assistant_audio_duration_s"] = float(audio_info.get("duration_s") or 0.0)
                if padded_audio_info:
                    timings_s["assistant_audio_padded_duration_s"] = float(padded_audio_info.get("duration_s") or 0.0)

                if not args.warmup_engine:
                    engine.close()
                    engine = None

                assistant_text = "".join(assistant_text_parts).strip()
                if assistant_text:
                    assistant_history.append(assistant_text)
                timings_s["turn_total"] = time.perf_counter() - turn_receive_s
                print(f"[QwenOmniDyStreamDialogueTiming] turn={client_turn_id} summary={timings_s}")
                await ws.send_json({
                    "type": "timing_summary",
                    "turn_id": client_turn_id,
                    "timings_s": timings_s,
                })
                await ws.send_json({
                    "type": "turn_done" if args.multi_turn else "done",
                    "turn_id": client_turn_id,
                    "sent_frames": sent_frames,
                    "rendered_frames_seen": rendered_frames_seen,
                    "saved_verify_frames": saved_verify_frames,
                    "saved_video_frames": saved_video_frames,
                    "last_frame_checksum": last_frame_checksum,
                    "assistant_text": assistant_text,
                    "history_turns": len(assistant_history),
                    "verify_video_path": str(verify_video_path) if verify_video_path is not None else "",
                    "assistant_audio_path": str(assistant_audio_path) if assistant_audio_path is not None else "",
                    "assistant_audio_padded_path": str(assistant_audio_padded_path) if assistant_audio_padded_path is not None else "",
                    "av_output_path": str(av_output_path) if av_output_path is not None else "",
                    "audio_info": audio_info,
                    "padded_audio_info": padded_audio_info,
                    "av_info": av_info,
                    "mux_error": mux_error,
                })
                if not args.multi_turn:
                    await ws.close()
                    return

        except WebSocketDisconnect:
            return
        except Exception as exc:
            traceback.print_exc()
            await ws.send_json({"type": "error", "message": str(exc)})
            await ws.close()
        finally:
            close_verify_video_writer()
            if engine is not None:
                engine.close()

    return app


def main() -> None:
    args = parse_args()
    if args.startup_warmup:
        print("[QwenOmniDyStreamDialogueDemo] startup warmup: building DyStream engine once before uvicorn...")
        warmup_engine = build_engine(args)
        warmup_engine.close()
        print("[QwenOmniDyStreamDialogueDemo] startup warmup complete.")
    app = build_app(args)
    print(f"[QwenOmniDyStreamDialogueDemo] Open http://127.0.0.1:{args.port}")
    print(
        f"[QwenOmniDyStreamDialogueDemo] model={args.model}, ref_image={args.ref_image}, "
        f"lookahead_frames={args.lookahead_frames}, render_mode={args.render_mode}"
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
