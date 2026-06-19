#!/usr/bin/env bash
set -euo pipefail

# Run the session-continuous Qwen-Omni -> DyStream dialogue stream demo.
# One WebSocket session writes one long session_dialogue_video.mp4,
# session_dialogue_audio.wav, and session_dialogue_av.mp4.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "ERROR: DASHSCOPE_API_KEY is not set." >&2
  echo "Run: export DASHSCOPE_API_KEY='your_api_key'" >&2
  exit 1
fi

# Default settings match the final June 6 streaming-demo run.
# Usage:
#   source .env && bash run_qwen_omni_dystream_session_stream.sh
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"
MODEL="${MODEL:-qwen3.5-omni-plus}"
VOICE="${VOICE:-Tina}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a concise real-time digital human assistant. Reply naturally with one short spoken sentence. Use the same language as the user unless explicitly requested otherwise.}"
REF_IMAGE="${REF_IMAGE:-img_files/11_resize.png}"
REF_MOTION="${REF_MOTION:-img_files/11.npz}"
LOOKAHEAD_FRAMES="${LOOKAHEAD_FRAMES:-20}"
AUDIO_ENCODE_LEFT_CONTEXT_FRAMES="${AUDIO_ENCODE_LEFT_CONTEXT_FRAMES:-96}"
DENOISING_STEPS="${DENOISING_STEPS:-1}"
GUIDANCE_MODE="${GUIDANCE_MODE:-all_only}"
RENDER_MODE="${RENDER_MODE:-batch}"
RENDER_FRAME_STRIDE="${RENDER_FRAME_STRIDE:-1}"
JPEG_QUALITY="${JPEG_QUALITY:-80}"
JPEG_MAX_SIZE="${JPEG_MAX_SIZE:-0}"
SEND_FRAME_STRIDE="${SEND_FRAME_STRIDE:-0}"
SESSION_OUTPUT_DIR="${SESSION_OUTPUT_DIR:-realtime/outputs/omni_session_stream}"
VERIFY_VIDEO_FPS="${VERIFY_VIDEO_FPS:-25}"
TAIL_SILENCE_MS="${TAIL_SILENCE_MS:-1000}"
SESSION_IDLE_DURING_OMNI_WAIT="${SESSION_IDLE_DURING_OMNI_WAIT:-1}"
OMNI_WAIT_IDLE_CHUNK_MS="${OMNI_WAIT_IDLE_CHUNK_MS:-200}"
WARMUP_ENGINE="${WARMUP_ENGINE:-1}"
STARTUP_WARMUP="${STARTUP_WARMUP:-1}"
MAX_HISTORY_TURNS="${MAX_HISTORY_TURNS:-3}"

PY_ARGS=(
  --model "${MODEL}"
  --voice "${VOICE}"
  --system-prompt "${SYSTEM_PROMPT}"
  --ref-image "${REF_IMAGE}"
  --ref-motion "${REF_MOTION}"
  --lookahead-frames "${LOOKAHEAD_FRAMES}"
  --audio-encode-left-context-frames "${AUDIO_ENCODE_LEFT_CONTEXT_FRAMES}"
  --denoising-steps "${DENOISING_STEPS}"
  --guidance-mode "${GUIDANCE_MODE}"
  --render-mode "${RENDER_MODE}"
  --render-frame-stride "${RENDER_FRAME_STRIDE}"
  --async-render
  --no-preprocess-image
  --encode-listener-audio
  --host "${HOST}"
  --port "${PORT}"
  --jpeg-quality "${JPEG_QUALITY}"
  --jpeg-max-size "${JPEG_MAX_SIZE}"
  --send-frame-stride "${SEND_FRAME_STRIDE}"
  --session-output-dir "${SESSION_OUTPUT_DIR}"
  --verify-video-fps "${VERIFY_VIDEO_FPS}"
  --tail-silence-ms "${TAIL_SILENCE_MS}"
  --omni-wait-idle-chunk-ms "${OMNI_WAIT_IDLE_CHUNK_MS}"
  --max-history-turns "${MAX_HISTORY_TURNS}"
)

if [[ "${WARMUP_ENGINE}" == "1" || "${WARMUP_ENGINE,,}" == "true" || "${WARMUP_ENGINE,,}" == "yes" ]]; then
  PY_ARGS+=(--warmup-engine)
fi
if [[ "${STARTUP_WARMUP}" == "1" || "${STARTUP_WARMUP,,}" == "true" || "${STARTUP_WARMUP,,}" == "yes" ]]; then
  PY_ARGS+=(--startup-warmup)
fi
if [[ "${SESSION_IDLE_DURING_OMNI_WAIT}" == "1" || "${SESSION_IDLE_DURING_OMNI_WAIT,,}" == "true" || "${SESSION_IDLE_DURING_OMNI_WAIT,,}" == "yes" ]]; then
  PY_ARGS+=(--session-idle-during-omni-wait)
fi

python realtime/demo/qwen_omni_dystream_session_stream_demo.py "${PY_ARGS[@]}"
