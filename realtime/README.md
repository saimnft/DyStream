# Experimental Realtime Qwen-Omni + DyStream Demo

This folder contains an experimental realtime digital-human extension built on top of DyStream for assessment and development purposes. The current entrypoint is the session-continuous browser voice -> Qwen-Omni -> DyStream demo launched by `run_qwen_omni_dystream_session_stream.sh` from the repository root.

## Run

Create a local `.env` file in the repository root with your own DashScope API key:

```bash
cat > .env <<'EOF'
export DASHSCOPE_API_KEY="your_dashscope_api_key"
EOF
```

Then start the demo:

```bash
source .env
bash run_qwen_omni_dystream_session_stream.sh
```

By default, the service listens on `http://127.0.0.1:6006`.

## Current Session Flow

```text
run_qwen_omni_dystream_session_stream.sh
  -> demo/qwen_omni_dystream_session_stream_demo.py
  -> omni_client.py / media_utils.py / engine_utils.py
  -> browser WebSocket /ws/session_stream
  -> Qwen-Omni streaming response audio
  -> dystream/stateful_stream_engine.py
  -> session-level video/audio/AV outputs
```

The session demo keeps one DyStream state, one video writer, and one audio timeline for the whole browser session. User speech is streamed as listener audio, Qwen-Omni response audio is streamed as speaker audio, and the final session is saved as one continuous AV result.

## File Guide

Current minimal realtime files:

- `demo/qwen_omni_dystream_session_stream_demo.py`: browser session WebSocket app.
- `omni_client.py`: DashScope Qwen-Omni streaming chunk iterator.
- `media_utils.py`: shared JPEG, PCM/WAV, resampling, and AV muxing helpers.
- `engine_utils.py`: DyStream engine construction helper used by the session app.
- `dystream/stateful_stream_engine.py`: stateful online DyStream inference engine.
- `demo/__init__.py` and `dystream/__init__.py`: package marker files.

Legacy/local test demos were removed from this temporary minimal version to keep only the final session-stream path.

## Outputs

Generated outputs are written under `outputs/`, especially `outputs/omni_session_stream/`. These are runtime artifacts and are not required source files for reproducing the demo.
