"""Minimal WebSocket microphone demo for the stateful DyStream engine.

This is a lightweight backend + browser page for testing true audio-chunk input:

    python realtime/demo/websocket_mic_demo.py \
        --ref-image img_files/11_resize.png \
        --ref-motion img_files/11.npz \
        --chunk-ms 200 \
        --lookahead-frames 20 \
        --denoising-steps 1 \
        --guidance-mode all_only \
        --render-mode batch \
        --async-render \
        --no-preprocess-image

Then open http://127.0.0.1:7861 in a browser that can access your microphone.

Notes:
- This is a single-client demo intended for development, not production.
- The browser sends Float32 PCM microphone chunks over WebSocket.
- The backend sends JPEG frames back over the same WebSocket.
- Audio playback/synchronization is not implemented yet; this only displays video.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
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

from realtime.dystream.stateful_stream_engine import StatefulStreamDyStreamEngine  # noqa: E402


INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>DyStream WebSocket Mic Demo</title>
  <style>
    body { font-family: sans-serif; margin: 24px; background: #111; color: #eee; }
    button { font-size: 16px; padding: 8px 16px; margin-right: 8px; }
    canvas { margin-top: 16px; background: #222; width: 512px; height: 512px; }
    pre { white-space: pre-wrap; background: #1c1c1c; padding: 12px; border-radius: 6px; }
  </style>
</head>
<body>
  <h2>DyStream WebSocket Mic Demo</h2>
  <button id="startBtn">Start mic stream</button>
  <button id="stopBtn" disabled>Stop</button>
  <div><canvas id="canvas" width="512" height="512"></canvas></div>
  <pre id="log"></pre>

<script>
const chunkMs = __CHUNK_MS__;
let ws = null;
let audioCtx = null;
let source = null;
let processor = null;
let mediaStream = null;
let pending = [];
let pendingSamples = 0;
let sentChunks = 0;
let recvFrames = 0;
let micStarted = false;
let backendReady = false;
let serverReceivedChunks = 0;
let pendingFrameMetas = [];

const logEl = document.getElementById('log');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function concatFloat32(chunks, totalSamples) {
  const out = new Float32Array(totalSamples);
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

async function startMicCapture() {
  if (micStarted) return;
  micStarted = true;
  await audioCtx.resume();
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  source = audioCtx.createMediaStreamSource(mediaStream);

  // Deprecated but widely available and enough for a development demo.
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  const sampleRate = audioCtx.sampleRate;
  const chunkSamples = Math.max(1, Math.floor(sampleRate * chunkMs / 1000));
  log(`Microphone capture started. sample_rate=${sampleRate}, chunk_samples=${chunkSamples}`);

  processor.onaudioprocess = (event) => {
    if (!ws || ws.readyState !== WebSocket.OPEN || !backendReady) return;

    const input = event.inputBuffer.getChannelData(0);
    const copy = new Float32Array(input.length);
    copy.set(input);
    pending.push(copy);
    pendingSamples += copy.length;

    while (pendingSamples >= chunkSamples) {
      const merged = concatFloat32(pending, pendingSamples);
      const sendChunk = merged.slice(0, chunkSamples);
      const rest = merged.slice(chunkSamples);
      pending = rest.length ? [rest] : [];
      pendingSamples = rest.length;
      ws.send(sendChunk.buffer);
      sentChunks += 1;
      if (sentChunks % 5 === 0) {
        log(`client-status sent_chunks=${sentChunks}, server_recv_chunks=${serverReceivedChunks}, client_queue_chunks=${sentChunks - serverReceivedChunks}, ws_buffered=${ws.bufferedAmount}`);
      }

    }
  };

  source.connect(processor);
  processor.connect(audioCtx.destination);
}

async function start() {
  document.getElementById('startBtn').disabled = true;
  document.getElementById('stopBtn').disabled = false;
  logEl.textContent = '';
  sentChunks = 0;
  recvFrames = 0;
  micStarted = false;
  backendReady = false;
  serverReceivedChunks = 0;

  const wsProtocol = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${wsProtocol}://${location.host}/ws/mic`);
  ws.binaryType = 'blob';
  ws.onopen = async () => {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const sampleRate = audioCtx.sampleRate;
    ws.send(JSON.stringify({ type: 'start', sample_rate: sampleRate, chunk_ms: chunkMs }));
    log(`WebSocket opened. Browser sample_rate=${sampleRate}, chunk_ms=${chunkMs}`);
    log('Starting microphone capture now, but dropping audio until backend engine is ready...');
    await startMicCapture();
  };

  ws.onmessage = async (event) => {
    if (typeof event.data === 'string') {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'status') {
          serverReceivedChunks = msg.recv_chunks;
          log(`server-status recv_chunks=${msg.recv_chunks}, sent_chunks=${sentChunks}, client_queue_chunks=${sentChunks - msg.recv_chunks}, process=${msg.process_s.toFixed(3)}s, send=${(msg.send_s ?? 0).toFixed(3)}s, frames=${msg.frames}, sent_kb=${((msg.sent_chunk_bytes ?? 0) / 1024).toFixed(1)}, input=${msg.input_s.toFixed(2)}s, generated=${msg.generated_s.toFixed(2)}s, engine_lag=${msg.engine_lag_s.toFixed(2)}s`);
          return;
        }
        if (msg.type === 'frame_meta') {
          pendingFrameMetas.push(msg);
          return;
        }
      } catch (_) {}
      log(event.data);
      if (event.data.includes('Engine ready')) {
        backendReady = true;
        log('Backend ready; now sending microphone chunks.');
      }
      return;
    }
    recvFrames += 1;
    const meta = pendingFrameMetas.length ? pendingFrameMetas.shift() : null;
    const receivePerfMs = performance.now();
    const receiveEpochMs = Date.now();
    const drawStartMs = performance.now();
    await drawJpegBlob(event.data);
    const drawEndMs = performance.now();
    if (meta) {
      const approxNetworkMs = receiveEpochMs - meta.server_send_start_epoch_ms;
      log(`frame-trace seq=${meta.seq}, q_before=${meta.video_queue_before}, approx_server_send_to_browser_recv=${approxNetworkMs.toFixed(1)}ms, browser_recv_to_draw_start=${(drawStartMs - receivePerfMs).toFixed(1)}ms, browser_draw=${(drawEndMs - drawStartMs).toFixed(1)}ms`);
    }
    if (recvFrames % 25 === 0) {
      log(`received frames=${recvFrames}, sent_chunks=${sentChunks}`);
    }
  };

  ws.onclose = () => {
    log('WebSocket closed.');
    cleanupAudio();
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
  };

  ws.onerror = (err) => {
    log('WebSocket error. See browser console.');
    console.error(err);
  };
}

function cleanupAudio() {
  if (processor) processor.disconnect();
  if (source) source.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  if (audioCtx) audioCtx.close();
  processor = null;
  source = null;
  mediaStream = null;
  audioCtx = null;
  micStarted = false;
  backendReady = false;
  serverReceivedChunks = 0;
  pendingFrameMetas = [];
  pending = [];
  pendingSamples = 0;
}

function stop() {
  cleanupAudio();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send('stop');
    ws.close();
  }
}

document.getElementById('startBtn').onclick = start;
document.getElementById('stopBtn').onclick = stop;
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DyStream WebSocket microphone demo.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--ref-image", default="img_files/11_resize.png")
    parser.add_argument("--ref-motion", default="img_files/11.npz")
    parser.add_argument("--chunk-ms", type=int, default=200)
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
    parser.add_argument("--jpeg-max-size", type=int, default=0, help="Resize outgoing JPEG frames so the longer side is at most this size. 0 keeps original size.")
    parser.add_argument("--video-queue-size", type=int, default=60, help="Maximum outgoing JPEG frames to buffer. Old video frames are dropped when the browser is slow.")
    parser.add_argument("--send-fps", type=float, default=25.0, help="Pace outgoing JPEG frames to this FPS.")
    parser.add_argument("--send-frame-stride", type=int, default=1, help="Send only every Nth rendered frame to the browser. 1 sends all frames; 0 disables video downlink while still rendering/verifying frames server-side.")
    parser.add_argument("--verify-output-dir", default="", help="If set, save server-side verification JPEG frames here without relying on browser downlink.")
    parser.add_argument("--verify-save-every-frames", type=int, default=25, help="Save every Nth rendered frame to --verify-output-dir. 0 disables saving.")
    parser.add_argument("--verify-video-path", default="", help="If set, stream all rendered frames into this server-side MP4/AVI file without relying on browser downlink.")
    parser.add_argument("--verify-video-fps", type=float, default=25.0, help="FPS metadata for --verify-video-path.")
    parser.add_argument("--trace-every-frames", type=int, default=0, help="If >0, emit timing metadata for every Nth outgoing video frame.")
    parser.add_argument("--trace-audio-receive", action="store_true", help="Print when raw audio binary chunks arrive at the WebSocket server.")
    parser.add_argument("--status-every-chunks", type=int, default=5, help="Send/log backlog status every N received audio chunks.")
    return parser.parse_args()


def encode_jpeg_rgb(frame: np.ndarray, quality: int, max_size: int = 0) -> bytes:
    if max_size and max(frame.shape[0], frame.shape[1]) > max_size:
        h, w = frame.shape[:2]
        scale = max_size / max(h, w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse(INDEX_HTML.replace("__CHUNK_MS__", str(args.chunk_ms)))

    @app.websocket("/ws/mic")
    async def websocket_mic(ws: WebSocket):
        await ws.accept()
        engine: Optional[StatefulStreamDyStreamEngine] = None
        sample_rate: Optional[int] = None
        sent_frames = 0
        received_chunks = 0
        processed_chunks = 0
        dropped_video_frames = 0
        rendered_frames_seen = 0
        saved_verify_frames = 0
        last_frame_mean = 0.0
        last_frame_std = 0.0
        last_frame_checksum = 0
        verify_video_writer: Optional[Any] = None
        verify_video_backend = ""
        saved_video_frames = 0
        audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        video_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max(1, int(args.video_queue_size)))
        status_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=10)
        stop_event = asyncio.Event()

        def put_latest_status(status: dict) -> None:
            if status_queue.full():
                try:
                    status_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            status_queue.put_nowait(status)

        def put_video_frame(payload: bytes) -> None:
            nonlocal dropped_video_frames
            if video_queue.full():
                try:
                    video_queue.get_nowait()
                    dropped_video_frames += 1
                except asyncio.QueueEmpty:
                    pass
            video_queue.put_nowait(payload)

        verify_output_dir = Path(args.verify_output_dir) if args.verify_output_dir else None
        if verify_output_dir is not None:
            verify_output_dir.mkdir(parents=True, exist_ok=True)
        verify_video_path = Path(args.verify_video_path) if args.verify_video_path else None
        if verify_video_path is not None:
            verify_video_path.parent.mkdir(parents=True, exist_ok=True)

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
                    print(f"[WebSocketDemo] verify video writer opened with imageio/libx264: {verify_video_path}")
                except Exception as exc:
                    print(f"[WebSocketDemo] imageio video writer failed ({exc}); falling back to OpenCV mp4v")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    verify_video_writer = cv2.VideoWriter(
                        str(verify_video_path),
                        fourcc,
                        float(args.verify_video_fps),
                        (w, h),
                    )
                    if not verify_video_writer.isOpened():
                        raise RuntimeError(f"Failed to open verify video writer: {verify_video_path}")
                    verify_video_backend = "opencv"
            if verify_video_backend == "imageio":
                verify_video_writer.append_data(frame)
            else:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                verify_video_writer.write(bgr)
            saved_video_frames += 1

        def handle_rendered_frame(frame: np.ndarray) -> tuple[bool, int]:
            """Verify/save a rendered frame and optionally enqueue it for browser preview.

            Returns (queued_for_browser, queued_payload_bytes).
            """
            nonlocal rendered_frames_seen, saved_verify_frames, last_frame_mean, last_frame_std, last_frame_checksum
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
                return False, 0
            if rendered_frames_seen % args.send_frame_stride != 0:
                return False, 0
            payload = encode_jpeg_rgb(frame, args.jpeg_quality, args.jpeg_max_size)
            put_video_frame(payload)
            return True, len(payload)

        async def receive_audio_loop() -> None:
            nonlocal received_chunks
            while not stop_event.is_set():
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if "text" in message and message["text"]:
                    if message["text"] == "stop":
                        break
                    continue
                data = message.get("bytes")
                if not data:
                    continue
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                received_chunks += 1
                if args.trace_audio_receive:
                    print(
                        "[WebSocketDemoAudioReceive] "
                        f"received_chunks={received_chunks}, bytes={len(data)}, "
                        f"samples={audio_chunk.size}, audio_queue_before={audio_queue.qsize()}"
                    )
                await audio_queue.put(audio_chunk)
            stop_event.set()

        async def inference_loop() -> None:
            nonlocal processed_chunks
            while not stop_event.is_set():
                try:
                    audio_chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                processed_chunks += 1
                process_start = asyncio.get_running_loop().time()
                engine.push_audio_chunk(audio_chunk, sample_rate=sample_rate)
                frames = engine.step()
                process_s = asyncio.get_running_loop().time() - process_start
                input_s = len(engine.audio_buffer) / engine.audio_sr
                generated_s = engine.generated_frames / engine.pose_fps
                lookahead_s = args.lookahead_frames / engine.pose_fps
                engine_lag_s = max(0.0, input_s - lookahead_s - generated_s)

                encode_start = asyncio.get_running_loop().time()
                sent_chunk_frames = 0
                sent_chunk_bytes = 0
                for frame in frames:
                    if args.render_mode == "none":
                        continue
                    queued, payload_bytes = handle_rendered_frame(frame)
                    if queued:
                        sent_chunk_bytes += payload_bytes
                        sent_chunk_frames += 1
                encode_s = asyncio.get_running_loop().time() - encode_start

                if args.status_every_chunks > 0 and processed_chunks % args.status_every_chunks == 0:
                    status = {
                        "type": "status",
                        "recv_chunks": processed_chunks,
                        "received_chunks": received_chunks,
                        "audio_queue": audio_queue.qsize(),
                        "process_s": process_s,
                        "send_s": encode_s,
                        "frames": len(frames),
                        "sent_chunk_frames": sent_chunk_frames,
                        "sent_chunk_bytes": sent_chunk_bytes,
                        "rendered_frames_seen": rendered_frames_seen,
                        "saved_verify_frames": saved_verify_frames,
                        "saved_video_frames": saved_video_frames,
                        "last_frame_mean": last_frame_mean,
                        "last_frame_std": last_frame_std,
                        "last_frame_checksum": last_frame_checksum,
                        "send_frame_stride": args.send_frame_stride,
                        "video_queue": video_queue.qsize(),
                        "dropped_video_frames": dropped_video_frames,
                        "input_s": input_s,
                        "generated_s": generated_s,
                        "engine_lag_s": engine_lag_s,
                    }
                    print(
                        "[WebSocketDemo] "
                        f"recv_chunks={processed_chunks}, received_chunks={received_chunks}, "
                        f"audio_queue={audio_queue.qsize()}, process={process_s:.3f}s, "
                        f"encode={encode_s:.3f}s, video_frames={len(frames)}, preview_frames={sent_chunk_frames}, "
                        f"sent_kb={sent_chunk_bytes / 1024:.1f}, video_queue={video_queue.qsize()}, "
                        f"dropped_video={dropped_video_frames}, rendered_total={rendered_frames_seen}, "
                        f"saved_verify={saved_verify_frames}, saved_video={saved_video_frames}, "
                        f"last_checksum={last_frame_checksum}, "
                        f"input={input_s:.2f}s, generated={generated_s:.2f}s, engine_lag={engine_lag_s:.2f}s"
                    )
                    put_latest_status(status)

        async def send_loop() -> None:
            nonlocal sent_frames
            frame_interval_s = 1.0 / max(1e-6, float(args.send_fps))
            next_frame_at = asyncio.get_running_loop().time()
            while not stop_event.is_set():
                try:
                    status = status_queue.get_nowait()
                    await ws.send_json(status)
                    continue
                except asyncio.QueueEmpty:
                    pass
                try:
                    payload = await asyncio.wait_for(video_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                now = asyncio.get_running_loop().time()
                delay_s = next_frame_at - now
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
                seq = sent_frames + 1
                trace_this = args.trace_every_frames > 0 and seq % args.trace_every_frames == 0
                send_start_loop_s = asyncio.get_running_loop().time()
                if trace_this:
                    await ws.send_json({
                        "type": "frame_meta",
                        "seq": seq,
                        "server_send_start_epoch_ms": time.time() * 1000.0,
                        "video_queue_before": video_queue.qsize(),
                    })
                await ws.send_bytes(payload)
                send_elapsed_ms = (asyncio.get_running_loop().time() - send_start_loop_s) * 1000.0
                sent_frames += 1
                if trace_this:
                    print(
                        "[WebSocketDemoFrameTrace] "
                        f"seq={seq}, ws_send_elapsed_ms={send_elapsed_ms:.1f}, "
                        f"video_queue_after={video_queue.qsize()}"
                    )
                next_frame_at = max(next_frame_at + frame_interval_s, asyncio.get_running_loop().time())

        tasks: list[asyncio.Task] = []
        try:
            start_msg = await ws.receive_json()
            if start_msg.get("type") != "start":
                await ws.send_text("Expected start message first.")
                await ws.close()
                return
            sample_rate = int(start_msg["sample_rate"])
            await ws.send_text("Loading DyStream engine...")
            engine = StatefulStreamDyStreamEngine(
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
            await ws.send_text("Engine ready. Start speaking.")

            tasks = [
                asyncio.create_task(receive_audio_loop()),
                asyncio.create_task(inference_loop()),
                asyncio.create_task(send_loop()),
            ]
            await tasks[0]
            stop_event.set()

        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            stop_event.set()
            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            if engine is not None:
                try:
                    for frame in engine.flush():
                        if args.render_mode == "none":
                            continue
                        queued, _ = handle_rendered_frame(frame)
                        if queued:
                            sent_frames += 1
                except Exception:
                    pass
                if verify_video_writer is not None:
                    if verify_video_backend == "imageio":
                        verify_video_writer.close()
                    else:
                        verify_video_writer.release()
                    print(f"[WebSocketDemo] verify video saved to {verify_video_path} ({saved_video_frames} frames, backend={verify_video_backend})")
                engine.close()
            print(
                f"[WebSocketDemo] client disconnected, sent_frames={sent_frames}, "
                f"rendered_frames_seen={rendered_frames_seen}, saved_verify_frames={saved_verify_frames}, "
                f"saved_video_frames={saved_video_frames}, last_frame_checksum={last_frame_checksum}"
            )

    return app


def main() -> None:
    args = parse_args()
    app = build_app(args)
    print(f"[WebSocketDemo] Open http://127.0.0.1:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
