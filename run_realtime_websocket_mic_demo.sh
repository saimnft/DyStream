#!/usr/bin/env bash
set -euo pipefail

# Run the existing browser-mic -> DyStream realtime video demo.
# This script is for the previous realtime work and does not involve Qwen-Omni.
#
# Usage:
#   bash run_realtime_websocket_mic_demo.sh
#
# Optional overrides:
#   PORT=6006 CHUNK_MS=800 LOOKAHEAD_FRAMES=20 bash run_realtime_websocket_mic_demo.sh
#   # Verify server-side realtime rendering without video downlink:
#   SEND_FRAME_STRIDE=0 VERIFY_OUTPUT_DIR=realtime/outputs/verify bash run_realtime_websocket_mic_demo.sh
#   SEND_FRAME_STRIDE=0 VERIFY_VIDEO_PATH=realtime/outputs/verify/realtime_verify.mp4 bash run_realtime_websocket_mic_demo.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"
REF_IMAGE="${REF_IMAGE:-img_files/11_resize.png}"
REF_MOTION="${REF_MOTION:-img_files/11.npz}"
CHUNK_MS="${CHUNK_MS:-800}"
LOOKAHEAD_FRAMES="${LOOKAHEAD_FRAMES:-20}"
AUDIO_ENCODE_LEFT_CONTEXT_FRAMES="${AUDIO_ENCODE_LEFT_CONTEXT_FRAMES:-96}"
DENOISING_STEPS="${DENOISING_STEPS:-1}"
GUIDANCE_MODE="${GUIDANCE_MODE:-all_only}"
RENDER_MODE="${RENDER_MODE:-batch}"
RENDER_FRAME_STRIDE="${RENDER_FRAME_STRIDE:-1}"
STATUS_EVERY_CHUNKS="${STATUS_EVERY_CHUNKS:-1}"
VIDEO_QUEUE_SIZE="${VIDEO_QUEUE_SIZE:-60}"
SEND_FPS="${SEND_FPS:-25}"
SEND_FRAME_STRIDE="${SEND_FRAME_STRIDE:-1}"
VERIFY_OUTPUT_DIR="${VERIFY_OUTPUT_DIR:-}"
VERIFY_SAVE_EVERY_FRAMES="${VERIFY_SAVE_EVERY_FRAMES:-25}"
VERIFY_VIDEO_PATH="${VERIFY_VIDEO_PATH:-}"
VERIFY_VIDEO_FPS="${VERIFY_VIDEO_FPS:-25}"
TRACE_EVERY_FRAMES="${TRACE_EVERY_FRAMES:-25}"
JPEG_QUALITY="${JPEG_QUALITY:-80}"
JPEG_MAX_SIZE="${JPEG_MAX_SIZE:-0}"

python realtime/demo/websocket_mic_demo.py \
  --ref-image "${REF_IMAGE}" \
  --ref-motion "${REF_MOTION}" \
  --chunk-ms "${CHUNK_MS}" \
  --lookahead-frames "${LOOKAHEAD_FRAMES}" \
  --audio-encode-left-context-frames "${AUDIO_ENCODE_LEFT_CONTEXT_FRAMES}" \
  --denoising-steps "${DENOISING_STEPS}" \
  --guidance-mode "${GUIDANCE_MODE}" \
  --render-mode "${RENDER_MODE}" \
  --render-frame-stride "${RENDER_FRAME_STRIDE}" \
  --async-render \
  --no-preprocess-image \
  --host "${HOST}" \
  --port "${PORT}" \
  --status-every-chunks "${STATUS_EVERY_CHUNKS}" \
  --video-queue-size "${VIDEO_QUEUE_SIZE}" \
  --send-fps "${SEND_FPS}" \
  --send-frame-stride "${SEND_FRAME_STRIDE}" \
  --verify-output-dir "${VERIFY_OUTPUT_DIR}" \
  --verify-save-every-frames "${VERIFY_SAVE_EVERY_FRAMES}" \
  --verify-video-path "${VERIFY_VIDEO_PATH}" \
  --verify-video-fps "${VERIFY_VIDEO_FPS}" \
  --trace-every-frames "${TRACE_EVERY_FRAMES}" \
  --trace-audio-receive \
  --jpeg-quality "${JPEG_QUALITY}" \
  --jpeg-max-size "${JPEG_MAX_SIZE}"
