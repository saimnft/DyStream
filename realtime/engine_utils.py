from __future__ import annotations

import argparse

from realtime.dystream.stateful_stream_engine import StatefulStreamDyStreamEngine


def build_engine(args: argparse.Namespace) -> StatefulStreamDyStreamEngine:
    return StatefulStreamDyStreamEngine(
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
