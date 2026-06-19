"""Session-continuous browser voice -> Qwen-Omni -> DyStream stream demo.

This demo keeps one DyStream state, one video writer, and one audio timeline for
an entire WebSocket session. User speech is streamed as listener audio
(speaker=silence); Qwen-Omni response audio is streamed as speaker audio
(listener=silence). Press End Session to mux one session-level AV MP4.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError as exc:
    raise SystemExit(f"Install fastapi uvicorn first. Original error: {exc}")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtime.engine_utils import build_engine  # noqa: E402
from realtime.media_utils import (  # noqa: E402
    encode_jpeg_rgb,
    float32_to_pcm16_bytes,
    float32_to_wav_bytes,
    mux_video_audio,
    pcm16_silence_bytes,
    resample_float32,
    save_pcm16_wav,
)
from realtime.omni_client import iter_dashscope_omni_chunks  # noqa: E402
from realtime.dystream.stateful_stream_engine import StatefulStreamDyStreamEngine  # noqa: E402


USER_VOICE_RMS_THRESHOLD = 0.01


INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Session Stream Qwen-Omni DyStream Demo</title>
  <style>
    body { font-family: sans-serif; margin: 24px; background: #111; color: #eee; }
    button { font-size: 16px; padding: 8px 16px; margin-right: 8px; }
    textarea { width: 720px; height: 80px; background: #1c1c1c; color: #eee; }
    canvas { margin-top: 16px; background: #222; width: 512px; height: 512px; }
    pre { white-space: pre-wrap; background: #1c1c1c; padding: 12px; border-radius: 6px; width: 720px; min-height: 260px; }
    .hint { color: #aaa; }
  </style>
</head>
<body>
  <h2>Session-continuous Qwen-Omni → DyStream Demo</h2>
  <p class="hint">Begin streams user audio as listener audio. Stop calls Omni. End Session writes one long session AV MP4.</p>
  <div>
    <button id="startBtn">Begin</button>
    <button id="stopBtn" disabled>Stop User & Ask Omni</button>
    <button id="endBtn" disabled>End Session</button>
  </div>
  <p>Prompt:</p>
  <textarea id="prompt">Please answer with one short natural sentence. Use the same language as the user unless explicitly requested otherwise.</textarea>
  <div><canvas id="canvas" width="512" height="512"></canvas></div>
  <pre id="log"></pre>

<script>
let ws = null;
let audioCtx = null;
let source = null;
let processor = null;
let mediaStream = null;
let sampleRate = 0;
let recording = false;
let turnId = 0;
let recvFrames = 0;

const logEl = document.getElementById('log');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

async function drawJpegBlob(blob) {
  const bmp = await createImageBitmap(blob);
  ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
  bmp.close();
}

async function ensureWs() {
  if (ws && ws.readyState === WebSocket.OPEN) return;
  logEl.textContent = '';
  recvFrames = 0;
  turnId = 0;
  const wsProtocol = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${wsProtocol}://${location.host}/ws/session_stream`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    log('WebSocket opened.');
    document.getElementById('endBtn').disabled = false;
  };
  ws.onmessage = async (event) => {
    if (typeof event.data === 'string') {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'text_delta') log(`text chunk: ${msg.data}`);
        else if (msg.type === 'timing_event') log(`timing ${msg.event}: +${msg.elapsed_s.toFixed(3)}s`);
        else if (msg.type === 'turn_summary') {
          log(`turn summary ${msg.turn_id}: ${JSON.stringify(msg.timings_s)}`);
          document.getElementById('startBtn').disabled = false;
          document.getElementById('stopBtn').disabled = true;
        }
        else if (msg.type === 'session_done') {
          log(`SESSION DONE: video=${msg.video_path}`);
          log(`SESSION DONE: audio=${msg.audio_path}`);
          log(`SESSION DONE: av=${msg.av_path}`);
          log(`session summary: ${JSON.stringify(msg.timings_s)}`);
          document.getElementById('startBtn').disabled = false;
          document.getElementById('stopBtn').disabled = true;
          document.getElementById('endBtn').disabled = true;
        }
        else if (msg.type === 'frame_status') log(`frame source=${msg.source}, generated=${msg.frames}, saved=${msg.saved_video_frames}, sent=${msg.sent_frames}`);
        else if (msg.type === 'status') log(`status: ${msg.message}`);
        else if (msg.type === 'error') {
          log(`ERROR: ${msg.message}`);
          document.getElementById('startBtn').disabled = false;
          document.getElementById('stopBtn').disabled = true;
        }
        else log(event.data);
      } catch (_) { log(event.data); }
      return;
    }
    recvFrames += 1;
    await drawJpegBlob(new Blob([event.data], {type: 'image/jpeg'}));
  };
  ws.onclose = () => {
    log('WebSocket closed.');
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('endBtn').disabled = true;
  };
  ws.onerror = (err) => { log('WebSocket error. See console.'); console.error(err); };
  if (ws.readyState !== WebSocket.OPEN) {
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', resolve, { once: true });
      ws.addEventListener('error', reject, { once: true });
    });
  }
}

async function begin() {
  if (recording) return;
  await ensureWs();
  turnId += 1;
  recording = true;
  document.getElementById('startBtn').disabled = true;
  document.getElementById('stopBtn').disabled = false;

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
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(copy.buffer);
  };
  source.connect(processor);
  processor.connect(audioCtx.destination);
  ws.send(JSON.stringify({
    type: 'stream_start',
    turn_id: turnId,
    sample_rate: sampleRate,
    prompt: document.getElementById('prompt').value || ''
  }));
  log(`Begin turn=${turnId}, sample_rate=${sampleRate}`);
}

async function stopUser() {
  if (!recording) return;
  recording = false;
  document.getElementById('stopBtn').disabled = true;
  if (processor) processor.disconnect();
  if (source) source.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  if (audioCtx) await audioCtx.close();
  ws.send(JSON.stringify({ type: 'stream_stop', turn_id: turnId, prompt: document.getElementById('prompt').value || '' }));
  log(`Stop user turn=${turnId}; waiting for Omni...`);
}

function endSession() {
  if (recording) stopUser();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'end_session' }));
    log('End Session requested.');
  }
}

document.getElementById('startBtn').onclick = begin;
document.getElementById('stopBtn').onclick = stopUser;
document.getElementById('endBtn').onclick = endSession;
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Session-continuous Qwen-Omni audio/listener DyStream demo.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--model", default=os.getenv("QWEN_OMNI_MODEL", "qwen3.5-omni-plus"))
    parser.add_argument("--voice", default=os.getenv("QWEN_OMNI_VOICE", "Tina"))
    parser.add_argument("--system-prompt", default="You are a concise real-time digital human assistant. Reply naturally with one short spoken sentence. Use the same language as the user unless explicitly requested otherwise.")
    parser.add_argument("--ref-image", default="img_files/11_resize.png")
    parser.add_argument("--ref-motion", default="img_files/11.npz")
    parser.add_argument("--lookahead-frames", type=int, default=10)
    parser.add_argument("--audio-encode-left-context-frames", type=int, default=64)
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--guidance-mode", choices=["full", "uncond_all", "all_only"], default="all_only")
    parser.add_argument("--render-mode", choices=["per_frame", "batch", "none"], default="batch")
    parser.add_argument("--render-frame-stride", type=int, default=1)
    parser.add_argument("--async-render", action="store_true")
    parser.add_argument("--no-preprocess-image", action="store_true")
    parser.add_argument("--preprocess-output-dir", default="realtime/outputs/preprocess")
    parser.add_argument("--encode-listener-audio", action="store_true", default=True)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-audio", action="store_true")
    parser.add_argument("--amp-motion", action="store_true")
    parser.add_argument("--amp-render", action="store_true")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--jpeg-max-size", type=int, default=0)
    parser.add_argument("--send-frame-stride", type=int, default=0)
    parser.add_argument("--session-output-dir", default="realtime/outputs/omni_session_stream")
    parser.add_argument("--verify-video-fps", type=float, default=25.0)
    parser.add_argument("--tail-silence-ms", type=int, default=1000)
    parser.add_argument("--session-idle-during-omni-wait", action="store_true", help="Generate silence-driven idle frames/audio while waiting for Omni chunks.")
    parser.add_argument("--omni-wait-idle-chunk-ms", type=int, default=200, help="Silence chunk duration used for idle generation while waiting for Omni.")
    parser.add_argument("--warmup-engine", action="store_true")
    parser.add_argument("--startup-warmup", action="store_true")
    parser.add_argument("--max-history-turns", type=int, default=3)
    return parser.parse_args()


def build_app(args: argparse.Namespace) -> FastAPI: 
    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse(INDEX_HTML)

    @app.websocket("/ws/session_stream")
    async def websocket_session_stream(ws: WebSocket):
        await ws.accept()
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            await ws.send_json({"type": "error", "message": f"Missing env var: {args.api_key_env}"})
            await ws.close()
            return

        session_start_s = time.perf_counter()
        session_dir = Path(args.session_output_dir) / datetime.now().strftime("session_stream_%Y%m%d_%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)
        video_path = session_dir / "session_dialogue_video.mp4"
        audio_path = session_dir / "session_dialogue_audio.wav"
        av_path = session_dir / "session_dialogue_av.mp4"
        turns_log_path = session_dir / "turns.jsonl"
        print(f"[QwenOmniDyStreamSessionStream] session dir: {session_dir}")

        engine: Optional[StatefulStreamDyStreamEngine] = None
        video_writer: Optional[Any] = None
        video_backend = ""
        saved_video_frames = 0
        sent_frames = 0
        rendered_frames_seen = 0
        session_audio_chunks: list[bytes] = []
        assistant_history: list[str] = []
        session_storage_started = False
        session_timings: dict[str, float] = {
            "session_start": 0.0,
            "listener_audio_duration_s": 0.0,
            "speaker_audio_duration_s": 0.0,
            "listener_dystream_step_total": 0.0,
            "speaker_dystream_step_total": 0.0,
            "omni_wait_idle_duration_s": 0.0,
            "omni_wait_idle_step_total": 0.0,
            "pre_voice_dropped_frames": 0.0,
        }

        def elapsed() -> float:
            return time.perf_counter() - session_start_s

        async def mark(event: str, turn_id: int = 0) -> None:
            t = elapsed()
            session_timings.setdefault(event, t)
            print(f"[QwenOmniDyStreamSessionTiming] turn={turn_id} {event}=+{t:.3f}s")
            await ws.send_json({"type": "timing_event", "turn_id": turn_id, "event": event, "elapsed_s": t})

        def build_turn_prompt(prompt: str) -> str: # 存储历史回答，提供给后续轮次参考
            prompt = prompt or "Please answer with one short natural sentence. Use the same language as the user unless explicitly requested otherwise."
            recent_history = assistant_history[-max(0, int(args.max_history_turns)):]
            if not recent_history:
                return prompt
            history_text = "\n".join(f"Assistant previous turn {idx + 1}: {text}" for idx, text in enumerate(recent_history))
            return (
                "This is an ongoing multi-turn voice conversation.\n"
                "The exact previous user audio transcripts are unavailable, but your previous answers are listed below.\n"
                f"{history_text}\n\n"
                "Now answer the user's latest voice input. Follow the system prompt's language and style requirements.\n"
                f"Current turn instruction: {prompt}"
            )

        def open_video_writer_if_needed(frame: np.ndarray) -> None: # 打开MP4 writer
            nonlocal video_writer, video_backend
            if video_writer is not None:
                return
            h, w = frame.shape[:2]
            try:
                import imageio.v2 as imageio

                video_writer = imageio.get_writer(
                    str(video_path),
                    fps=float(args.verify_video_fps),
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=1,
                )
                video_backend = "imageio"
            except Exception as exc:
                print(f"[QwenOmniDyStreamSessionStream] imageio writer failed ({exc}); fallback opencv mp4v")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(video_path), fourcc, float(args.verify_video_fps), (w, h))
                if not video_writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {video_path}")
                video_backend = "opencv"
            print(f"[QwenOmniDyStreamSessionStream] video writer opened: {video_path} backend={video_backend}")

        def close_video_writer() -> None:
            nonlocal video_writer
            if video_writer is None:
                return
            if video_backend == "imageio":
                video_writer.close()
            else:
                video_writer.release()
            print(f"[QwenOmniDyStreamSessionStream] video saved: {video_path} frames={saved_video_frames}")
            video_writer = None

        # 处理生成帧：保存视频、发送预览、统计数据
        async def handle_frames(frames: list[np.ndarray] | tuple[np.ndarray, ...], source: str, turn_id: int = 0) -> int:
            nonlocal saved_video_frames, sent_frames, rendered_frames_seen
            preview_frames = 0
            for frame in frames:
                if args.render_mode == "none":
                    continue
                rendered_frames_seen += 1
                open_video_writer_if_needed(frame)
                if video_backend == "imageio":
                    video_writer.append_data(frame)
                else:
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                saved_video_frames += 1
                if args.send_frame_stride > 0 and rendered_frames_seen % args.send_frame_stride == 0:
                    await ws.send_bytes(encode_jpeg_rgb(frame, args.jpeg_quality, args.jpeg_max_size))
                    sent_frames += 1
                    preview_frames += 1
            await ws.send_json({
                "type": "frame_status",
                "turn_id": turn_id,
                "source": source,
                "frames": len(frames),
                "saved_video_frames": saved_video_frames,
                "sent_frames": sent_frames,
                "preview_frames": preview_frames,
            })
            return len(frames)

        async def finalize_session() -> None:
            if engine is not None and args.tail_silence_ms > 0:
                silence_samples = int(24000 * args.tail_silence_ms / 1000)
                engine.push_audio_chunk(np.zeros((silence_samples,), dtype=np.float32), sample_rate=24000)
                session_audio_chunks.append(pcm16_silence_bytes(args.tail_silence_ms, sample_rate=24000))
                step_start = time.perf_counter()
                frames = engine.step()
                session_timings["speaker_dystream_step_total"] += time.perf_counter() - step_start
                await handle_frames(frames, "final_tail_silence")
                flush_start = time.perf_counter()
                flush_frames = list(engine.flush())
                session_timings["dystream_flush"] = time.perf_counter() - flush_start
                await handle_frames(flush_frames, "final_flush")
            close_start = time.perf_counter()
            close_video_writer()
            session_timings["video_writer_close"] = time.perf_counter() - close_start
            audio_info = save_pcm16_wav(session_audio_chunks, audio_path, sample_rate=24000) if session_audio_chunks else {}
            mux_error = ""
            av_info: dict[str, Any] = {}
            if video_path.exists() and audio_path.exists():
                mux_start = time.perf_counter()
                try:
                    av_info = mux_video_audio(video_path, audio_path, av_path)
                except Exception as exc:
                    mux_error = str(exc)
                    print(f"[QwenOmniDyStreamSessionStream] AV mux failed: {mux_error}")
                session_timings["av_mux"] = time.perf_counter() - mux_start
            session_timings["session_total"] = elapsed()
            session_timings["video_duration_s"] = saved_video_frames / float(args.verify_video_fps) if saved_video_frames else 0.0
            session_timings["audio_duration_s"] = float(audio_info.get("duration_s") or 0.0)
            session_timings["overall_step_total"] = session_timings["listener_dystream_step_total"] + session_timings["speaker_dystream_step_total"] + session_timings["omni_wait_idle_step_total"]
            session_timings["overall_generation_ratio"] = session_timings["overall_step_total"] / max(session_timings["video_duration_s"], 1e-6)
            await ws.send_json({
                "type": "session_done",
                "session_dir": str(session_dir),
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "av_path": str(av_path),
                "audio_info": audio_info,
                "av_info": av_info,
                "mux_error": mux_error,
                "saved_video_frames": saved_video_frames,
                "sent_frames": sent_frames,
                "timings_s": session_timings,
            })

        try:
            await ws.send_json({"type": "status", "message": f"Session output dir: {session_dir}"})
            if args.warmup_engine:
                await ws.send_json({"type": "status", "message": "Warming up DyStream engine..."})
                load_start = time.perf_counter()
                engine = build_engine(args)
                session_timings["engine_load"] = time.perf_counter() - load_start
                await mark("engine_ready")

            while True:
                meta = await ws.receive_json()
                meta_type = meta.get("type")
                if meta_type in {"end_session", "stop"}:
                    await mark("end_session")
                    await finalize_session()
                    await ws.close()
                    return
                if meta_type != "stream_start":
                    await ws.send_json({"type": "error", "message": "Expected stream_start or end_session."})
                    continue

                turn_id = int(meta.get("turn_id") or 0)
                input_sr = int(meta.get("sample_rate") or 48000)
                prompt = build_turn_prompt(str(meta.get("prompt") or ""))
                turn_start = elapsed()
                turn_timings: dict[str, float] = {"turn_start": turn_start}
                user_chunks: list[np.ndarray] = []
                user_duration_s = 0.0
                listener_step_s = 0.0
                pre_voice_dropped_frames = 0
                await mark("listener_stream_start", turn_id)

                if engine is None:
                    load_start = time.perf_counter()
                    engine = build_engine(args)
                    session_timings["engine_load"] = time.perf_counter() - load_start
                    await mark("engine_ready", turn_id)

                while True:
                    message = await ws.receive()
                    if message.get("bytes") is not None:
                        chunk = np.frombuffer(message["bytes"], dtype=np.float32).copy()
                        if chunk.size == 0:
                            continue
                        user_chunks.append(chunk)
                        user_duration_s += chunk.size / float(input_sr)
                        chunk_rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float32))))
                        if not session_storage_started and chunk_rms >= USER_VOICE_RMS_THRESHOLD:
                            session_storage_started = True
                            await mark("first_saved_user_voice", turn_id)
                        user_24k = resample_float32(chunk, orig_sr=input_sr, target_sr=24000)
                        if session_storage_started:
                            session_audio_chunks.append(float32_to_pcm16_bytes(user_24k))
                        engine.push_audio_chunk(np.zeros_like(chunk, dtype=np.float32), sample_rate=input_sr, audio_other_chunk=chunk)
                        step_start = time.perf_counter()
                        frames = engine.step()
                        step_elapsed = time.perf_counter() - step_start
                        listener_step_s += step_elapsed
                        session_timings["listener_dystream_step_total"] += step_elapsed
                        if frames and "first_listener_frame" not in session_timings:
                            await mark("first_listener_frame", turn_id)
                        if session_storage_started:
                            await handle_frames(frames, "listener_audio", turn_id)
                        else:
                            pre_voice_dropped_frames += len(frames)
                            session_timings["pre_voice_dropped_frames"] += float(len(frames))
                        continue
                    text_data = message.get("text")
                    if not text_data:
                        continue
                    control = json.loads(text_data)
                    if control.get("type") == "stream_stop":
                        break

                session_timings["listener_audio_duration_s"] += user_duration_s
                turn_timings["listener_audio_duration_s"] = user_duration_s
                turn_timings["listener_dystream_step_total"] = listener_step_s
                turn_timings["listener_realtime_ratio"] = listener_step_s / max(user_duration_s, 1e-6)
                turn_timings["pre_voice_dropped_frames"] = float(pre_voice_dropped_frames)
                await mark("listener_stream_stop", turn_id)

                user_audio = np.concatenate(user_chunks).astype(np.float32) if user_chunks else np.zeros((0,), dtype=np.float32)
                wav_bytes = float32_to_wav_bytes(user_audio, input_sr)
                assistant_text_parts: list[str] = []
                await mark("omni_call_start", turn_id)
                speaker_step_s = 0.0
                speaker_audio_s = 0.0
                wait_idle_duration_s = 0.0
                wait_idle_step_s = 0.0
                omni_queue: queue.Queue[dict[str, Any]] = queue.Queue()

                def run_omni_worker() -> None:
                    try:
                        for omni_chunk in iter_dashscope_omni_chunks(
                            api_key=api_key,
                            model=args.model,
                            wav_bytes=wav_bytes,
                            prompt=prompt,
                            system_prompt=args.system_prompt,
                            voice=args.voice,
                        ):
                            omni_queue.put({"kind": "chunk", "data": omni_chunk})
                    except Exception as exc:  # forwarded to websocket loop
                        omni_queue.put({"kind": "error", "message": str(exc)})
                    finally:
                        omni_queue.put({"kind": "done"})

                omni_thread = threading.Thread(target=run_omni_worker, daemon=True)
                omni_thread.start()
                omni_done = False
                idle_chunk_ms = max(20, int(args.omni_wait_idle_chunk_ms))
                idle_timeout_s = idle_chunk_ms / 1000.0
                idle_samples = int(24000 * idle_chunk_ms / 1000)
                while not omni_done:
                    try:
                        item = omni_queue.get(timeout=idle_timeout_s if args.session_idle_during_omni_wait else None)
                    except queue.Empty:
                        silence = np.zeros((idle_samples,), dtype=np.float32)
                        session_audio_chunks.append(pcm16_silence_bytes(idle_chunk_ms, sample_rate=24000))
                        wait_idle_duration_s += idle_samples / 24000.0
                        session_timings["omni_wait_idle_duration_s"] += idle_samples / 24000.0
                        engine.push_audio_chunk(silence, sample_rate=24000)
                        step_start = time.perf_counter()
                        frames = engine.step()
                        step_elapsed = time.perf_counter() - step_start
                        wait_idle_step_s += step_elapsed
                        session_timings["omni_wait_idle_step_total"] += step_elapsed
                        if frames and "first_omni_wait_idle_frame" not in turn_timings:
                            turn_timings["first_omni_wait_idle_frame"] = elapsed()
                            await mark("first_omni_wait_idle_frame", turn_id)
                        await handle_frames(frames, "omni_wait_silence", turn_id)
                        continue

                    if item.get("kind") == "done":
                        omni_done = True
                        continue
                    if item.get("kind") == "error":
                        raise RuntimeError(str(item.get("message") or "Omni worker failed"))
                    chunk = item.get("data") or {}
                    chunk_type = chunk.get("type")
                    if chunk_type == "text_delta":
                        if "first_text_delta" not in turn_timings:
                            turn_timings["first_text_delta"] = elapsed()
                            await mark("first_text_delta", turn_id)
                        assistant_text_parts.append(str(chunk.get("data") or ""))
                        await ws.send_json({**chunk, "turn_id": turn_id})
                        continue
                    if chunk_type != "audio_delta":
                        continue
                    if "first_audio_delta" not in turn_timings:
                        turn_timings["first_audio_delta"] = elapsed()
                        await mark("first_audio_delta", turn_id)
                    audio_bytes = base64.b64decode(str(chunk.get("base64") or ""))
                    session_audio_chunks.append(audio_bytes)
                    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    speaker_audio_s += len(audio_int16) / 24000.0
                    omni_audio = (audio_int16.astype(np.float32) / 32768.0).astype(np.float32)
                    engine.push_audio_chunk(omni_audio, sample_rate=24000)
                    step_start = time.perf_counter()
                    frames = engine.step()
                    step_elapsed = time.perf_counter() - step_start
                    speaker_step_s += step_elapsed
                    session_timings["speaker_dystream_step_total"] += step_elapsed
                    if frames and "first_speaker_frame" not in turn_timings:
                        turn_timings["first_speaker_frame"] = elapsed()
                        await mark("first_speaker_frame", turn_id)
                    await handle_frames(frames, "speaker_audio", turn_id)
                omni_thread.join(timeout=0.1)

                await mark("omni_stream_end", turn_id)
                session_timings["speaker_audio_duration_s"] += speaker_audio_s
                assistant_text = "".join(assistant_text_parts).strip()
                if assistant_text:
                    assistant_history.append(assistant_text)
                turn_timings.update({
                    "turn_end": elapsed(),
                    "speaker_audio_duration_s": speaker_audio_s,
                    "speaker_dystream_step_total": speaker_step_s,
                    "speaker_realtime_ratio": speaker_step_s / max(speaker_audio_s, 1e-6),
                    "omni_wait_idle_duration_s": wait_idle_duration_s,
                    "omni_wait_idle_step_total": wait_idle_step_s,
                    "saved_video_frames_total": float(saved_video_frames),
                    "assistant_text": assistant_text,
                })
                with turns_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"turn_id": turn_id, "timings_s": turn_timings}, ensure_ascii=False) + "\n")
                await ws.send_json({"type": "turn_summary", "turn_id": turn_id, "timings_s": turn_timings})
                await ws.send_json({"type": "status", "message": f"Turn {turn_id} complete. You can Begin next turn or End Session."})

        except WebSocketDisconnect:
            return
        except Exception as exc:
            traceback.print_exc()
            await ws.send_json({"type": "error", "message": str(exc)})
            await ws.close()
        finally:
            close_video_writer()
            if engine is not None:
                engine.close()

    return app


def main() -> None:
    args = parse_args()
    if args.startup_warmup:
        print("[QwenOmniDyStreamSessionStream] startup warmup...")
        warmup_engine = build_engine(args)
        warmup_engine.close()
        print("[QwenOmniDyStreamSessionStream] startup warmup complete.")
    app = build_app(args)
    print(f"[QwenOmniDyStreamSessionStream] Open http://127.0.0.1:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
