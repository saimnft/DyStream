from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _to_builtin(obj: Any) -> Any:
    model_dump = _safe_getattr(obj, "model_dump")
    if callable(model_dump):
        return model_dump(exclude_none=True)
    to_dict = _safe_getattr(obj, "to_dict")
    if callable(to_dict):
        return to_dict()
    if isinstance(obj, dict):
        return {key: _to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(value) for value in obj]
    obj_dict = _safe_getattr(obj, "__dict__")
    if isinstance(obj_dict, dict):
        return {key: _to_builtin(value) for key, value in obj_dict.items() if not key.startswith("_")}
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


def iter_dashscope_omni_chunks(
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
