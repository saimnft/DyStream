"""Browser voice -> Qwen-Omni API streaming chunk demo.

Run:
    set DASHSCOPE_API_KEY=sk-xxx
    python realtime/demo/qwen_omni_browser_voice_demo.py --port 7862

Then open http://127.0.0.1:7862, record one utterance, and watch streamed
text/audio chunks returned by Qwen-Omni.

This demo intentionally only verifies the browser-audio -> Omni -> chunk-output
path. It does not connect Omni audio output to DyStream video yet.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "fastapi, uvicorn and openai are required for this demo. Install with:\n"
        "  pip install fastapi uvicorn openai\n"
        f"Original import error: {exc}"
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Qwen-Omni Browser Voice Chunk Demo</title>
  <style>
    body { font-family: sans-serif; margin: 24px; background: #111; color: #eee; }
    button { font-size: 16px; padding: 8px 16px; margin-right: 8px; }
    textarea { width: 720px; height: 80px; background: #1c1c1c; color: #eee; }
    pre { white-space: pre-wrap; background: #1c1c1c; padding: 12px; border-radius: 6px; width: 720px; min-height: 260px; }
    .hint { color: #aaa; }
  </style>
</head>
<body>
  <h2>Qwen-Omni Browser Voice Chunk Demo</h2>
  <p class="hint">Press Start, speak, press Stop & Send. The browser encodes mic PCM as WAV and sends it to the backend WebSocket.</p>
  <div>
    <button id="startBtn">Start Recording</button>
    <button id="stopBtn" disabled>Stop & Send</button>
  </div>
  <p>Optional prompt:</p>
  <textarea id="prompt">请用中文简短回答我的问题。</textarea>
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

const logEl = document.getElementById('log');
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

async function start() {
  logEl.textContent = '';
  chunks = [];
  recording = true;
  document.getElementById('startBtn').disabled = true;
  document.getElementById('stopBtn').disabled = false;

  const wsProtocol = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${wsProtocol}://${location.host}/ws/omni`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => log('WebSocket opened. Recording...');
  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'text_delta') log(`text chunk: ${msg.data}`);
      else if (msg.type === 'audio_delta') log(`audio chunk: ${msg.bytes} bytes base64=${msg.base64_chars} chars`);
      else if (msg.type === 'usage') log(`usage: ${JSON.stringify(msg.data)}`);
      else if (msg.type === 'done') log('done.');
      else if (msg.type === 'error') log(`ERROR: ${msg.message}`);
      else log(event.data);
    } catch (_) {
      log(event.data);
    }
  };
  ws.onclose = () => log('WebSocket closed.');
  ws.onerror = (err) => { log('WebSocket error. See browser console.'); console.error(err); };

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

  const meta = {
    type: 'audio',
    mime_type: 'audio/wav',
    sample_rate: sampleRate,
    prompt: document.getElementById('prompt').value || ''
  };
  ws.send(JSON.stringify(meta));
  ws.send(wav);
  document.getElementById('startBtn').disabled = false;
}

document.getElementById('startBtn').onclick = start;
document.getElementById('stopBtn').onclick = stopAndSend;
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser voice to Qwen-Omni streaming chunk demo.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument(
        "--api-mode",
        choices=["dashscope", "openai"],
        default=os.getenv("QWEN_OMNI_API_MODE", "dashscope"),
        help="dashscope uses the native multimodal API and is recommended for audio input; openai is kept for compatible-mode experiments.",
    )
    parser.add_argument("--base-url", default=os.getenv("QWEN_OMNI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--model", default=os.getenv("QWEN_OMNI_MODEL", "qwen3.5-omni-plus"))
    parser.add_argument("--voice", default=os.getenv("QWEN_OMNI_VOICE", "Tina"))
    parser.add_argument(
        "--audio-message-format",
        choices=["audio_url", "input_audio"],
        default=os.getenv("QWEN_OMNI_AUDIO_MESSAGE_FORMAT", "input_audio"),
        help="How to put browser WAV into OpenAI-compatible messages. Use audio_url only if your model endpoint requires it.",
    )
    parser.add_argument("--system-prompt", default="你是一个端到端数字人对话助手，请回答简洁自然。")
    return parser.parse_args()


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _delta_to_dict(delta: Any) -> dict[str, Any]:
    model_dump = _safe_getattr(delta, "model_dump")
    if callable(model_dump):
        return model_dump(exclude_none=True)
    if isinstance(delta, dict):
        return delta
    return _safe_getattr(delta, "__dict__", {}) or {}


def _extract_text_delta(delta: Any) -> str:
    content = _safe_getattr(delta, "content")
    if content:
        return str(content)
    data = _delta_to_dict(delta)
    return str(data.get("content") or "")


def _extract_audio_base64(delta: Any) -> str:
    audio = _safe_getattr(delta, "audio")
    if isinstance(audio, dict):
        return str(audio.get("data") or "")
    data = _delta_to_dict(delta)
    audio = data.get("audio") or {}
    if isinstance(audio, dict):
        return str(audio.get("data") or "")
    return ""


def _build_audio_message(wav_bytes: bytes, prompt: str, system_prompt: str, audio_message_format: str) -> list[dict[str, Any]]:
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    if audio_message_format == "input_audio":
        user_content = [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
        ]
    else:
        audio_url = f"data:audio/wav;base64,{audio_b64}"
        user_content = [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
        ]
    if prompt:
        user_content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _to_builtin(obj: Any) -> Any:
    model_dump = _safe_getattr(obj, "model_dump")
    if callable(model_dump):
        return model_dump(exclude_none=True)
    to_dict = _safe_getattr(obj, "to_dict")
    if callable(to_dict):
        return to_dict()
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    obj_dict = _safe_getattr(obj, "__dict__")
    if isinstance(obj_dict, dict):
        return {k: _to_builtin(v) for k, v in obj_dict.items() if not k.startswith("_")}
    return obj


def _extract_content_items(payload: Any) -> list[Any]:
    payload = _to_builtin(payload)
    output = payload.get("output", {}) if isinstance(payload, dict) else {}
    choices = output.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", [])
    return content if isinstance(content, list) else [content]


def _iter_dashscope_omni_chunks(
    api_key: str,
    model: str,
    wav_bytes: bytes,
    prompt: str,
    system_prompt: str,
    voice: str,
):
    try:
        from dashscope import MultiModalConversation
    except ImportError as exc:
        raise RuntimeError("dashscope is required for audio input. Install with: pip install dashscope") from exc

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        audio_uri = Path(tmp_path).resolve().as_uri()
        messages = [
            {"role": "system", "content": [{"text": system_prompt}]},
            {"role": "user", "content": [{"audio": audio_uri}, {"text": prompt or "请回答这段语音中的问题。"}]},
        ]

        responses = MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            stream=True,
            incremental_output=True,
            result_format="message",
            modalities=["text", "audio"],
            audio={"voice": voice, "format": "wav"},
        )

        for response in responses:
            status_code = getattr(response, "status_code", None)
            if status_code is not None and int(status_code) >= 400:
                code = getattr(response, "code", "")
                message = getattr(response, "message", "")
                raise RuntimeError(f"<{status_code}> {code}: {message}")

            payload = _to_builtin(response)
            items = _extract_content_items(payload)
            yielded = False
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("text"):
                    yielded = True
                    yield {"type": "text_delta", "data": item["text"]}
                audio_item = item.get("audio")
                if isinstance(audio_item, dict):
                    audio_b64 = str(audio_item.get("data") or "")
                    if audio_b64:
                        yielded = True
                        try:
                            audio_bytes = len(base64.b64decode(audio_b64))
                        except Exception:
                            audio_bytes = 0
                        yield {
                            "type": "audio_delta",
                            "base64": audio_b64,
                            "base64_chars": len(audio_b64),
                            "bytes": audio_bytes,
                        }
            if not yielded:
                yield {"type": "raw_chunk", "data": payload}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _iter_openai_omni_chunks(
    api_key: str,
    base_url: str,
    model: str,
    wav_bytes: bytes,
    prompt: str,
    system_prompt: str,
    voice: str,
    audio_message_format: str,
):
    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=_build_audio_message(
            wav_bytes=wav_bytes,
            prompt=prompt,
            system_prompt=system_prompt,
            audio_message_format=audio_message_format,
        ),
        modalities=["text", "audio"],
        audio={"voice": voice, "format": "wav"},
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in completion:
        if getattr(chunk, "usage", None):
            usage = chunk.usage.model_dump(exclude_none=True) if hasattr(chunk.usage, "model_dump") else chunk.usage
            yield {"type": "usage", "data": usage}

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = choices[0].delta

        text_delta = _extract_text_delta(delta)
        if text_delta:
            yield {"type": "text_delta", "data": text_delta}

        audio_b64 = _extract_audio_base64(delta)
        if audio_b64:
            try:
                audio_bytes = len(base64.b64decode(audio_b64))
            except Exception:
                audio_bytes = 0
            yield {
                "type": "audio_delta",
                "base64": audio_b64,
                "base64_chars": len(audio_b64),
                "bytes": audio_bytes,
            }


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse(INDEX_HTML)

    @app.websocket("/ws/omni")
    async def ws_omni(ws: WebSocket):
        await ws.accept()
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            await ws.send_json({"type": "error", "message": f"Missing env var: {args.api_key_env}"})
            await ws.close()
            return

        try:
            meta = await ws.receive_json()
            if meta.get("type") != "audio":
                await ws.send_json({"type": "error", "message": "Expected JSON metadata with type='audio' first."})
                await ws.close()
                return

            message = await ws.receive()
            wav_bytes: Optional[bytes] = message.get("bytes")
            if not wav_bytes:
                await ws.send_json({"type": "error", "message": "Expected WAV binary payload after metadata."})
                await ws.close()
                return

            await ws.send_json({
                "type": "status",
                "message": f"received wav bytes={len(wav_bytes)}, calling {args.model} via {args.api_mode}...",
            })

            prompt = str(meta.get("prompt") or "")
            if args.api_mode == "dashscope":
                chunk_iter = _iter_dashscope_omni_chunks(
                    api_key=api_key,
                    model=args.model,
                    wav_bytes=wav_bytes,
                    prompt=prompt,
                    system_prompt=args.system_prompt,
                    voice=args.voice,
                )
            else:
                chunk_iter = _iter_openai_omni_chunks(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    wav_bytes=wav_bytes,
                    prompt=prompt,
                    system_prompt=args.system_prompt,
                    voice=args.voice,
                    audio_message_format=args.audio_message_format,
            )

            for chunk_msg in chunk_iter:
                await ws.send_json(chunk_msg)

            await ws.send_json({"type": "done"})
            await ws.close()

        except WebSocketDisconnect:
            return
        except Exception as exc:
            await ws.send_json({"type": "error", "message": str(exc)})
            await ws.close()

    return app


def main() -> None:
    args = parse_args()
    app = build_app(args)
    print(f"[QwenOmniBrowserVoiceDemo] Open http://127.0.0.1:{args.port}")
    print(
        f"[QwenOmniBrowserVoiceDemo] model={args.model}, api_mode={args.api_mode}, "
        f"base_url={args.base_url}, audio_message_format={args.audio_message_format}"
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
