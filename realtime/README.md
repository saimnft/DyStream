# Realtime DyStream Demo Scaffold

This folder contains the realtime/streaming integration code for the audio-video conversation demo.

## Goal

Connect realtime audio output from Qwen-Omni to DyStream online inference, then render talking-head video frames incrementally.

```text
Qwen-Omni audio stream
  -> audio chunk receiver / format converter
  -> DyStream online motion generation
  -> latent-to-frame renderer
  -> realtime display / streaming output
```

## Suggested milestones

1. `demo/local_wav_stream_demo.py`: simulate realtime audio by splitting a local wav into chunks.
2. `dystream/engine.py`: wrap existing DyStream model loading, motion generation, and rendering into a stateful engine.
3. `audio/chunk_buffer.py`: maintain audio chunks, sample-rate conversion, and frame alignment.
4. `omni/client.py`: connect Qwen-Omni realtime audio output.
5. `demo/omni_dystream_demo.py`: final end-to-end demo.

## Important principle

First make DyStream stream with a local wav. Only after that, replace the local wav source with Qwen-Omni audio chunks.
