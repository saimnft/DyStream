"""Local wav pseudo-streaming demo for DyStream.

Example:
    python realtime/demo/local_wav_stream_demo.py \
        --ref-image img_files/11.png \
        --audio wav_files/11.wav \
        --chunk-ms 100 \
        --output realtime/outputs/realtime_local_demo.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import imageio
import librosa
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



def iter_audio_chunks(audio: np.ndarray, sr: int, chunk_ms: int):
    chunk_size = int(sr * chunk_ms / 1000)
    for start in range(0, len(audio), chunk_size):
        yield audio[start : start + chunk_size]


def parse_args():
    parser = argparse.ArgumentParser(description="Pseudo-stream DyStream with a local wav file.")
    parser.add_argument("--ref-image", default="img_files/11.png", help="Reference face image.")
    parser.add_argument("--ref-motion", default=None, help="Optional reference motion npz. Used only with --no-preprocess-image.")
    parser.add_argument("--audio", default="wav_files/11.wav", help="Speaker audio wav.")
    parser.add_argument("--audio-other", default=None, help="Optional listener audio wav.")
    parser.add_argument(
        "--encode-listener-audio",
        action="store_true",
        help="Encode --audio-other with the second Wav2Vec2 branch. By default listener encoding is disabled for faster realtime single-speaker generation.",
    )
    parser.add_argument("--chunk-ms", type=int, default=100, help="Simulated audio packet size in ms.")
    parser.add_argument("--denoising-steps", type=int, default=5)
    parser.add_argument("--output", default="realtime/outputs/realtime_local_demo.mp4", help="Output video path.")
    parser.add_argument("--display", action="store_true", help="Display frames with OpenCV.")
    parser.add_argument("--realtime-sleep", action="store_true", help="Sleep chunk_ms between chunks.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional max audio duration for quick test.")
    parser.add_argument("--profile", action="store_true", help="Print coarse per-stage timing for the stateful engine.")
    parser.add_argument(
        "--guidance-mode",
        choices=["full", "uncond_all", "all_only"],
        default="full",
        help="CFG mode for stateful motion inference. full=5 branches, uncond_all=2 branches, all_only=1 branch and fastest.",
    )
    parser.add_argument(
        "--render-mode",
        choices=["per_frame", "batch", "none"],
        default="per_frame",
        help="Video rendering mode. none benchmarks motion-only speed and does not write a playable video.",
    )
    parser.add_argument(
        "--render-frame-stride",
        type=int,
        default=1,
        help="Render every Nth frame and duplicate it for skipped frames. Higher is faster but more stuttery.",
    )
    parser.add_argument(
        "--async-render",
        action="store_true",
        help="Pipeline rendering one chunk behind motion generation. Adds about one chunk of output latency but may improve wall-clock throughput.",
    )
    parser.add_argument("--no-ema", action="store_true", help="Skip EMA weight swapping during streaming inference for speed benchmarking.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for audio/motion/render inference.")
    parser.add_argument("--amp-audio", action="store_true", help="Use CUDA autocast only for Wav2Vec2 audio feature extraction.")
    parser.add_argument("--amp-motion", action="store_true", help="Use CUDA autocast only for motion generation.")
    parser.add_argument("--amp-render", action="store_true", help="Use CUDA autocast only for video rendering. May cause artifacts; disabled unless requested.")
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Autocast dtype when --amp is enabled.",
    )
    parser.add_argument(
        "--lookahead-frames",
        type=int,
        default=2,
        help="Delay stateful output by N frames for audio lookahead. 1 frame = 40ms at 25fps.",
    )
    parser.add_argument(
        "--audio-encode-left-context-frames",
        type=int,
        default=120,
        help="Sliding-window audio encoding left context for stateful engine. 120 frames ~= 4.8s at 25fps.",
    )
    parser.add_argument(
        "--strict-causal-audio-encoder",
        action="store_true",
        help="Use experimental causal/limited-lookahead Wav2Vec2 encoder. Set DYSTREAM_WAV2VEC2_LOOKAHEAD separately.",
    )
    parser.add_argument("--no-mux-audio", action="store_true", help="Do not mux input audio into the saved mp4.")
    parser.add_argument(
        "--engine",
        choices=["stateful", "pseudo"],
        default="stateful",
        help="stateful avoids re-sampling old frames; pseudo re-runs full history.",
    )
    parser.add_argument(
        "--no-preprocess-image",
        action="store_true",
        help="Skip image preprocessing and use --ref-image/--ref-motion directly.",
    )
    parser.add_argument(
        "--preprocess-output-dir",
        default="realtime/outputs/preprocess",
        help="Directory to save resized/masked image and image latent.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    audio, sr = librosa.load(args.audio, sr=None, mono=True)
    if args.max_seconds is not None:
        audio = audio[: int(sr * args.max_seconds)]

    if args.audio_other:
        audio_other, sr_other = librosa.load(args.audio_other, sr=sr, mono=True)
        if args.max_seconds is not None:
            audio_other = audio_other[: int(sr * args.max_seconds)]
        if not args.encode_listener_audio:
            print("[Demo] --audio-other was provided but listener encoding is disabled. Add --encode-listener-audio to use it.")
    else:
        audio_other = None

    if args.strict_causal_audio_encoder:
        os.environ["DYSTREAM_EXPERIMENTAL_CAUSAL_WAV2VEC2"] = "1"

    if args.engine == "stateful":
        from realtime.dystream.stateful_stream_engine import StatefulStreamDyStreamEngine
        engine_cls = StatefulStreamDyStreamEngine
    else:
        from realtime.dystream.pseudo_stream_engine import PseudoStreamDyStreamEngine
        engine_cls = PseudoStreamDyStreamEngine

    engine_kwargs = dict(
        ref_image_path=args.ref_image,
        ref_motion_path=args.ref_motion,
        denoising_steps=args.denoising_steps,
        preprocess_image=not args.no_preprocess_image,
        preprocess_output_dir=args.preprocess_output_dir,
    )
    if args.engine == "stateful":
        engine_kwargs["lookahead_frames"] = args.lookahead_frames
        engine_kwargs["audio_encode_left_context_frames"] = args.audio_encode_left_context_frames
        engine_kwargs["encode_listener_audio"] = args.encode_listener_audio
        engine_kwargs["guidance_mode"] = args.guidance_mode
        engine_kwargs["render_mode"] = args.render_mode
        engine_kwargs["render_frame_stride"] = args.render_frame_stride
        engine_kwargs["async_render"] = args.async_render
        engine_kwargs["use_ema"] = not args.no_ema
        engine_kwargs["amp"] = args.amp
        engine_kwargs["amp_audio"] = args.amp_audio
        engine_kwargs["amp_motion"] = args.amp_motion
        engine_kwargs["amp_render"] = args.amp_render
        engine_kwargs["amp_dtype"] = args.amp_dtype
        engine_kwargs["profile"] = args.profile

    engine = engine_cls(**engine_kwargs)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = None if (args.engine == "stateful" and args.render_mode == "none") else imageio.get_writer(args.output, fps=engine.pose_fps)

    total_frames = 0
    start_time = time.time()

    try:
        for chunk_idx, chunk in enumerate(iter_audio_chunks(audio, sr, args.chunk_ms)):
            start = int(chunk_idx * sr * args.chunk_ms / 1000)
            end = start + len(chunk)
            other_chunk = audio_other[start:end] if (audio_other is not None and args.encode_listener_audio) else None

            engine.push_audio_chunk(chunk, sample_rate=sr, audio_other_chunk=other_chunk)
            frames = engine.step()

            for frame in frames:
                if writer is not None:
                    writer.append_data(frame)
                total_frames += 1

                if args.display:
                    cv2.imshow("DyStream pseudo-stream", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt

            display_total = engine.generated_frames if (args.engine == "stateful" and args.render_mode == "none") else total_frames
            print(
                f"[chunk {chunk_idx:04d}] input={len(engine.audio_buffer) / engine.audio_sr:.2f}s, "
                f"new_frames={len(frames)}, total_frames={display_total}",
                flush=True,
            )
            if args.profile and args.engine == "stateful":
                p = engine.profile_totals
                print(
                    f"[profile] audio={p['audio']:.3f}s, motion={p['motion']:.3f}s, "
                    f"render={p['render']:.3f}s, step_total={p['step']:.3f}s",
                    flush=True,
                )

            if args.realtime_sleep:
                time.sleep(args.chunk_ms / 1000)

    except KeyboardInterrupt:
        print("[Demo] Interrupted by user.")
    finally:
        if args.engine == "stateful" and hasattr(engine, "flush"):
            frames = engine.flush()
            for frame in frames:
                if writer is not None:
                    writer.append_data(frame)
                total_frames += 1

                if args.display:
                    cv2.imshow("DyStream pseudo-stream", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if frames:
                print(f"[Demo] Flushed async render frames: {len(frames)}", flush=True)
        if hasattr(engine, "close"):
            engine.close()
        if writer is not None:
            writer.close()
        if args.display:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time

    if not args.no_mux_audio and writer is not None and total_frames > 0:
        try:
            import moviepy.editor as mpe

            temp_output = args.output.replace(".mp4", "_video_only.mp4")
            os.replace(args.output, temp_output)
            video_clip = mpe.VideoFileClip(temp_output)
            audio_clip = mpe.AudioFileClip(args.audio)
            if args.max_seconds is not None:
                audio_clip = audio_clip.subclip(0, min(args.max_seconds, video_clip.duration))
            elif audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclip(0, video_clip.duration)
            video_clip.set_audio(audio_clip).write_videofile(
                args.output,
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
            video_clip.close()
            audio_clip.close()
            os.remove(temp_output)
            print(f"[Demo] Muxed audio into: {args.output}")
        except Exception as exc:
            print(f"[Demo] Warning: failed to mux audio: {exc}")

    generated_frames = engine.generated_frames if args.engine == "stateful" else total_frames
    if writer is not None:
        print(f"[Demo] Saved video to: {args.output}")
    else:
        print("[Demo] Render mode is 'none'; no playable video was written.")
    print(f"[Demo] Generated {generated_frames} frames in {elapsed:.2f}s")
    if args.profile and args.engine == "stateful":
        p = engine.profile_totals
        print(
            f"[Demo] Profile totals: audio={p['audio']:.3f}s, motion={p['motion']:.3f}s, "
            f"render={p['render']:.3f}s, step_total={p['step']:.3f}s"
        )


if __name__ == "__main__":
    main()
