"""Pseudo-streaming DyStream engine.

This engine reuses the existing offline inference code, but feeds an accumulated
local audio buffer and only renders newly available frames. It is intended as a
first realtime demo scaffold, not as the final optimized online implementation.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as dystream_app  # noqa: E402


class PseudoStreamDyStreamEngine:
    """A minimal pseudo-streaming wrapper around the current DyStream inference.

    Current behavior:
    - Accumulates speaker audio chunks.
    - Re-runs the existing model inference on the accumulated audio.
    - Emits only frames that have not been emitted before.

    This is computationally wasteful, but is much easier to validate before
    implementing true cached online inference.
    """

    def __init__(
        self,
        ref_image_path: str,
        ref_motion_path: Optional[str] = None,
        denoising_steps: int = 5,
        cfg_audio: float = 0.5,
        cfg_audio_other: float = 0.5,
        cfg_anchor: float = 0.0,
        cfg_all: float = 1.0,
        device: Optional[str] = None,
        preprocess_image: bool = True,
        preprocess_output_dir: Optional[str] = "realtime/outputs/preprocess",
    ) -> None:
        self.device = device or dystream_app.DEVICE
        self.denoising_steps = int(denoising_steps)

        print("[PseudoStream] Loading DyStream and visualization models...")
        dystream_app.load_dystream_model()
        dystream_app.load_visualization_model()

        self.model = dystream_app._dystream_model
        self.cfg = dystream_app._dystream_cfg
        self.ema = dystream_app._dystream_ema
        self.noise_scheduler = dystream_app._noise_scheduler
        self.vis_ctx = dystream_app._vis_ctx

        self.model.cfg_audio = cfg_audio
        self.model.cfg_audio_other = cfg_audio_other
        self.model.cfg_anchor = cfg_anchor
        self.model.cfg_all = cfg_all

        self.audio_sr = int(OmegaConf.select(self.cfg.config, "model.audio_sr", default=16000))
        self.pose_fps = int(OmegaConf.select(self.cfg.config, "model.pose_fps", default=25))
        self.samples_per_frame = int(self.audio_sr / self.pose_fps)
        self.prefix_frames = int(self.model.inpainting_length)

        self.ref_image_path = ref_image_path
        self.ref_motion_path = ref_motion_path
        self.preprocess_image = preprocess_image

        if preprocess_image:
            self.ref_image, self.ref_motion = self._preprocess_ref_image(
                ref_image_path,
                preprocess_output_dir,
            )
        else:
            if ref_motion_path is None:
                raise ValueError("ref_motion_path is required when preprocess_image=False")
            self.ref_image = Image.open(ref_image_path).convert("RGB")
            self.ref_motion = self._load_ref_motion(ref_motion_path)

        self._prepare_renderer_cache()

        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.audio_other_buffer = np.zeros((0,), dtype=np.float32)
        self.emitted_frames = 0

        print(
            f"[PseudoStream] Ready. audio_sr={self.audio_sr}, pose_fps={self.pose_fps}, "
            f"samples_per_frame={self.samples_per_frame}, prefix_frames={self.prefix_frames}"
        )

    def _preprocess_ref_image(
        self,
        image_path: str,
        output_dir: Optional[str],
    ) -> tuple[Image.Image, torch.Tensor]:
        """Run the same in-memory image preprocessing path as app.py custom inference."""
        print(f"[PseudoStream] Preprocessing reference image: {image_path}")
        image_pil = Image.open(image_path).convert("RGB")
        resized_pil, masked_pil, motion_latent_cpu = dystream_app.process_image(image_pil)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            stem = Path(image_path).stem
            resized_path = os.path.join(output_dir, f"{stem}_resize.png")
            masked_path = os.path.join(output_dir, f"{stem}_masked.png")
            latent_path = os.path.join(output_dir, f"{stem}_motion_from_image.npz")
            resized_pil.save(resized_path)
            masked_pil.save(masked_path)
            np.savez(latent_path, motion_latent=motion_latent_cpu.numpy())
            print(f"[PseudoStream] Saved preprocessed image to: {resized_path}")
            print(f"[PseudoStream] Saved masked image to: {masked_path}")
            print(f"[PseudoStream] Saved image motion latent to: {latent_path}")

        motion = motion_latent_cpu.float().to(self.device)
        if motion.dim() == 1:
            motion = motion.unsqueeze(0)
        if motion.dim() == 2:
            motion = motion.unsqueeze(0)
        return resized_pil, motion

    def _load_ref_motion(self, npz_path: str) -> torch.Tensor:
        data = np.load(npz_path, allow_pickle=True)
        if "motion_latent" in data:
            arr = data["motion_latent"]
        elif "random_data" in data:
            arr = data["random_data"]
        else:
            raise KeyError(f"{npz_path} has no 'motion_latent' or 'random_data' key")

        motion = torch.from_numpy(arr).float().to(self.device)
        if motion.dim() == 1:
            motion = motion.unsqueeze(0)
        if motion.dim() == 2:
            motion = motion.unsqueeze(0)
        return motion  # [1, T, 512]

    def _prepare_renderer_cache(self) -> None:
        transform = self.vis_ctx["transform"]
        ref_img_tensor = transform(self.ref_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.ref_face_feat = self.vis_ctx["face_encoder"](ref_img_tensor)
        self.anchor_motion = self.ref_motion[:, 0:1, :].float().to(self.device)

    def push_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        audio_other_chunk: Optional[np.ndarray] = None,
    ) -> None:
        """Append one audio chunk to the internal buffer.

        Args:
            audio_chunk: mono or multi-channel audio array.
            sample_rate: input sample rate.
            audio_other_chunk: optional listener audio chunk.
        """
        audio_chunk = self._to_16k_mono_float(audio_chunk, sample_rate)
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk], axis=0)

        if audio_other_chunk is not None:
            other = self._to_16k_mono_float(audio_other_chunk, sample_rate)
        else:
            other = np.zeros_like(audio_chunk)
        self.audio_other_buffer = np.concatenate([self.audio_other_buffer, other], axis=0)

    def _to_16k_mono_float(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=-1)
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 2.0:
            audio = audio / 32768.0
        if sample_rate != self.audio_sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.audio_sr)
        return audio.astype(np.float32)

    @torch.no_grad()
    def step(self) -> List[np.ndarray]:
        """Generate and render newly available frames.

        Returns:
            List of RGB uint8 frames.
        """
        target_frames = len(self.audio_buffer) // self.samples_per_frame
        if target_frames <= self.emitted_frames:
            return []

        motion_pred = self._infer_motion_for_current_buffer()
        available_frames = min(target_frames, motion_pred.shape[1])
        if available_frames <= self.emitted_frames:
            return []

        new_motion = motion_pred[:, self.emitted_frames:available_frames, :]
        frames = self._render_motion_frames(new_motion)
        self.emitted_frames = available_frames
        return frames

    @torch.no_grad()
    def _infer_motion_for_current_buffer(self) -> torch.Tensor:
        prefix_samples = self.prefix_frames * self.samples_per_frame
        audio = np.concatenate([np.zeros(prefix_samples, dtype=np.float32), self.audio_buffer])
        audio_other = np.concatenate([np.zeros(prefix_samples, dtype=np.float32), self.audio_other_buffer])

        # Make both tracks equal length.
        min_len = min(len(audio), len(audio_other))
        audio = audio[:min_len]
        audio_other = audio_other[:min_len]

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        audio_other_tensor = torch.from_numpy(audio_other).float().unsqueeze(0).to(self.device)

        total_frames_with_prefix = audio_tensor.shape[1] // self.samples_per_frame
        motion_latent_in = self.anchor_motion.repeat(1, total_frames_with_prefix, 1)

        if self.ema is not None:
            self.ema.to(self.device)
            ctx = self.ema.average_parameters(self.model.parameters())
        else:
            ctx = nullcontext()

        with ctx:
            motion_pred = self.model.inference(
                audio_tensor,
                audio_other=audio_other_tensor,
                init_motion=motion_latent_in,
                cond_motion=motion_latent_in,
                anchor_motion=self.anchor_motion,
                noise_scheduler=self.noise_scheduler,
                num_inference_steps=self.denoising_steps,
            )

        return motion_pred[:, self.prefix_frames:, :]

    @torch.no_grad()
    def _render_motion_frames(self, motion_latents: torch.Tensor) -> List[np.ndarray]:
        if motion_latents.dim() == 2:
            motion_latents = motion_latents.unsqueeze(0)
        motion_latents = motion_latents.float().to(self.device)

        flow_estimator = self.vis_ctx["flow_estimator"]
        face_generator = self.vis_ctx["face_generator"]

        frames = []
        for idx in range(motion_latents.shape[1]):
            tgt = flow_estimator(self.anchor_motion.squeeze(0), motion_latents[:, idx, :])
            recon = face_generator(tgt, self.ref_face_feat)
            frame = recon[0].permute(1, 2, 0).detach().cpu().numpy()
            frame = np.clip((frame + 1) / 2 * 255, 0, 255).astype(np.uint8)
            frames.append(frame)
        return frames
