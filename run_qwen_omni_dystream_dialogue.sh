#!/usr/bin/env bash
set -euo pipefail

# Run the Qwen-Omni -> DyStream realtime dialogue demo.
#
# Usage:
#   export DASHSCOPE_API_KEY="sk-xxx"
#   bash run_qwen_omni_dystream_dialogue.sh
#
# Optional overrides:
#   PORT=6007 MODEL=qwen3.5-omni-plus VOICE=Tina bash run_qwen_omni_dystream_dialogue.sh
#   SYSTEM_PROMPT="You are a concise Chinese-speaking digital human assistant." bash run_qwen_omni_dystream_dialogue.sh
#   # Multi-turn dialogue with per-turn server-side MP4 outputs:
#   MULTI_TURN=1 SEND_FRAME_STRIDE=0 SESSION_OUTPUT_DIR=realtime/outputs/omni_dialogue_sessions bash run_qwen_omni_dystream_dialogue.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "ERROR: DASHSCOPE_API_KEY is not set." >&2
  echo "Run: export DASHSCOPE_API_KEY='your_api_key'" >&2
  exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6007}"
MODEL="${MODEL:-qwen3.5-omni-plus}"
VOICE="${VOICE:-Tina}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a concise Chinese-speaking digital human assistant. Please answer naturally and briefly in Chinese.}"
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
SEND_FRAME_STRIDE="${SEND_FRAME_STRIDE:-1}"
VERIFY_OUTPUT_DIR="${VERIFY_OUTPUT_DIR:-}"
VERIFY_SAVE_EVERY_FRAMES="${VERIFY_SAVE_EVERY_FRAMES:-25}"
VERIFY_VIDEO_PATH="${VERIFY_VIDEO_PATH:-}"
VERIFY_VIDEO_FPS="${VERIFY_VIDEO_FPS:-25}"
TAIL_SILENCE_MS="${TAIL_SILENCE_MS:-1000}"
MULTI_TURN="${MULTI_TURN:-0}"
WARMUP_ENGINE="${WARMUP_ENGINE:-1}"
STARTUP_WARMUP="${STARTUP_WARMUP:-0}"
MAX_HISTORY_TURNS="${MAX_HISTORY_TURNS:-3}"
SESSION_OUTPUT_DIR="${SESSION_OUTPUT_DIR:-}"

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
  --host "${HOST}"
  --port "${PORT}"
  --jpeg-quality "${JPEG_QUALITY}"
  --jpeg-max-size "${JPEG_MAX_SIZE}"
  --send-frame-stride "${SEND_FRAME_STRIDE}"
  --verify-output-dir "${VERIFY_OUTPUT_DIR}"
  --verify-save-every-frames "${VERIFY_SAVE_EVERY_FRAMES}"
  --verify-video-path "${VERIFY_VIDEO_PATH}"
  --verify-video-fps "${VERIFY_VIDEO_FPS}"
  --tail-silence-ms "${TAIL_SILENCE_MS}"
  --max-history-turns "${MAX_HISTORY_TURNS}"
  --session-output-dir "${SESSION_OUTPUT_DIR}"
)

if [[ "${MULTI_TURN}" == "1" || "${MULTI_TURN,,}" == "true" || "${MULTI_TURN,,}" == "yes" ]]; then
  PY_ARGS+=(--multi-turn)
fi
if [[ "${WARMUP_ENGINE}" == "1" || "${WARMUP_ENGINE,,}" == "true" || "${WARMUP_ENGINE,,}" == "yes" ]]; then
  PY_ARGS+=(--warmup-engine)
fi
if [[ "${STARTUP_WARMUP}" == "1" || "${STARTUP_WARMUP,,}" == "true" || "${STARTUP_WARMUP,,}" == "yes" ]]; then
  PY_ARGS+=(--startup-warmup)
fi

python realtime/demo/qwen_omni_dystream_dialogue_demo.py "${PY_ARGS[@]}"
