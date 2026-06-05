"""Stateful local-stream DyStream engine.

Unlike pseudo_stream_engine, this wrapper does not re-generate old motion frames.
It keeps `past_motion` and only samples newly available frames, which avoids
chunk-boundary jitter caused by repeatedly re-sampling the whole history.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as dystream_app  # noqa: E402


class StatefulStreamDyStreamEngine:
    """Stateful frame-by-frame engine for local wav streaming tests."""

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
        lookahead_frames: int = 2,
        audio_encode_left_context_frames: int = 120,
        encode_listener_audio: bool = True,
        guidance_mode: str = "full",
        render_mode: str = "per_frame",
        render_frame_stride: int = 1,
        async_render: bool = False,
        use_ema: bool = True,
        amp: bool = False,
        amp_audio: bool = False,
        amp_motion: bool = False,
        amp_render: bool = False,
        amp_dtype: str = "float16",
        profile: bool = False,
    ) -> None:
        self.device = device or dystream_app.DEVICE
        self.denoising_steps = int(denoising_steps)
        self.encode_listener_audio = bool(encode_listener_audio)
        self.guidance_mode = str(guidance_mode)
        if self.guidance_mode not in {"full", "uncond_all", "all_only"}:
            raise ValueError(f"Unsupported guidance_mode: {self.guidance_mode}")
        self.render_mode = str(render_mode)
        if self.render_mode not in {"per_frame", "batch", "none"}:
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")
        self.render_frame_stride = max(1, int(render_frame_stride))
        self.async_render = bool(async_render and self.render_mode != "none")
        self._render_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1) if self.async_render else None
        self._pending_render_future: Optional[Future[tuple[List[np.ndarray], float]]] = None
        self.use_ema = bool(use_ema)
        self.amp_audio = bool(amp or amp_audio)
        self.amp_motion = bool(amp or amp_motion)
        self.amp_render = bool(amp or amp_render)
        self.amp = self.amp_audio or self.amp_motion or self.amp_render
        self.amp_dtype = str(amp_dtype)
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError(f"Unsupported amp_dtype: {self.amp_dtype}")
        self.profile = bool(profile)
        self.profile_totals = {
            "audio": 0.0,
            "motion": 0.0,
            "render": 0.0,
            "step": 0.0,
        }
        self.last_step_profile = self._new_step_profile()

        print("[StatefulStream] Loading DyStream and visualization models...")
        dystream_app.load_dystream_model()
        dystream_app.load_visualization_model()

        self.model = dystream_app._dystream_model
        self.cfg = dystream_app._dystream_cfg
        self.ema = dystream_app._dystream_ema
        self.noise_scheduler = dystream_app._noise_scheduler
        self.vis_ctx = dystream_app._vis_ctx

        self.model.cfg_audio = cfg_audio
        self.model.cfg_audio_other = cfg_audio_other if self.encode_listener_audio else 0.0
        self.model.cfg_anchor = cfg_anchor
        self.model.cfg_all = cfg_all
        self.model.guidance_mode = self.guidance_mode

        self.audio_sr = int(OmegaConf.select(self.cfg.config, "model.audio_sr", default=16000))
        self.pose_fps = int(OmegaConf.select(self.cfg.config, "model.pose_fps", default=25))
        self.samples_per_frame = int(self.audio_sr / self.pose_fps)
        self.audio_per_motion = int(self.cfg.model.audio_fps // self.cfg.model.pose_fps)
        self.window = int(self.cfg.model.cbh_window_length)
        self.prefix_frames = int(self.model.inpainting_length)
        self.lookahead_frames = max(0, int(lookahead_frames))
        # Encode only a sliding audio window instead of the whole accumulated
        # stream. This is the main speed/quality tradeoff knob for streaming.
        # 120 frames at 25fps ~= 4.8s left context.
        self.audio_encode_left_context_frames = max(self.window, int(audio_encode_left_context_frames))

        if preprocess_image:
            self.ref_image, self.ref_motion = self._preprocess_ref_image(ref_image_path, preprocess_output_dir)
        else:
            if ref_motion_path is None:
                raise ValueError("ref_motion_path is required when preprocess_image=False")
            self.ref_image = Image.open(ref_image_path).convert("RGB")
            self.ref_motion = self._load_ref_motion(ref_motion_path)

        self._prepare_renderer_cache()
        self.reset_stream_state()

        print(
            f"[StatefulStream] Ready. audio_sr={self.audio_sr}, pose_fps={self.pose_fps}, "
            f"window={self.window}, prefix_frames={self.prefix_frames}, "
            f"lookahead_frames={self.lookahead_frames}, "
            f"audio_encode_left_context_frames={self.audio_encode_left_context_frames}, "
            f"encode_listener_audio={self.encode_listener_audio}, "
            f"guidance_mode={self.guidance_mode}, "
            f"render_mode={self.render_mode}, render_frame_stride={self.render_frame_stride}, "
            f"async_render={self.async_render}, "
            f"use_ema={self.use_ema}, amp_audio={self.amp_audio}, "
            f"amp_motion={self.amp_motion}, amp_render={self.amp_render}, "
            f"amp_dtype={self.amp_dtype}, "
            f"profile={self.profile}"
        )

    def _new_step_profile(self) -> dict:
        return {
            "audio": 0.0,
            "motion": 0.0,
            "render": 0.0,
            "render_wait": 0.0,
            "step": 0.0,
            "batches": 0,
            "generated_frames": 0,
            "returned_frames": 0,
        }

    def _step_profile_add(self, key: str, value: float) -> None:
        if key in self.last_step_profile:
            self.last_step_profile[key] += value

    def _sync_if_profile(self) -> None:
        if self.profile and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def _profile_add(self, key: str, start_time: float) -> float:
        if not self.profile:
            return 0.0
        self._sync_if_profile()
        duration = time.perf_counter() - start_time
        self.profile_totals[key] += duration
        self._step_profile_add(key, duration)
        return duration

    def _autocast_context(self, module: str):
        enabled = {
            "audio": self.amp_audio,
            "motion": self.amp_motion,
            "render": self.amp_render,
        }.get(module, False)
        if not enabled or not torch.cuda.is_available() or not str(self.device).startswith("cuda"):
            return nullcontext()
        dtype = torch.float16 if self.amp_dtype == "float16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def reset_stream_state(self) -> None:
        if getattr(self, "_pending_render_future", None) is not None:
            self._pending_render_future.result()
            self._pending_render_future = None
        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.audio_other_buffer = np.zeros((0,), dtype=np.float32)
        self.generated_frames = 0
        self.past_motion = self.anchor_motion.repeat(1, self.prefix_frames, 1)

    def _preprocess_ref_image(self, image_path: str, output_dir: Optional[str]) -> tuple[Image.Image, torch.Tensor]:
        print(f"[StatefulStream] Preprocessing reference image: {image_path}")
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
            print(f"[StatefulStream] Saved preprocessed image to: {resized_path}")
            print(f"[StatefulStream] Saved masked image to: {masked_path}")
            print(f"[StatefulStream] Saved image motion latent to: {latent_path}")

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
        return motion

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
        if audio.size > 0 and np.max(np.abs(audio)) > 2.0:
            audio = audio / 32768.0
        if sample_rate != self.audio_sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.audio_sr)
        return audio.astype(np.float32)

    @torch.no_grad()
    def step(self) -> List[np.ndarray]:
        """Generate all newly available frames without re-sampling old frames."""
        target_audio_frames = len(self.audio_buffer) // self.samples_per_frame
        # Delay output by a few frames so the audio encoder can use short future
        # context. With 25 fps, 1 frame = 40 ms; 2 frames ~= 80 ms lookahead.
        max_generatable = max(0, target_audio_frames - self.lookahead_frames)
        # one_clip_only_inference needs a full `window` of audio features and
        # only starts producing after `prefix_frames`; wait until the current
        # frame has enough audio context.
        available_feature_frames = max(0, target_audio_frames + self.prefix_frames)

        frames = []
        self.last_step_profile = self._new_step_profile()
        self._sync_if_profile()
        step_start = time.perf_counter()
        while self.generated_frames < max_generatable:
            remaining = max_generatable - self.generated_frames
            max_feature_batch = max(0, available_feature_frames - self.generated_frames - self.prefix_frames - 1)
            batch_frames = min(remaining, max_feature_batch)
            if batch_frames <= 0:
                break

            motion = self._generate_motion_batch(batch_frames)
            self.last_step_profile["batches"] += 1
            self.last_step_profile["generated_frames"] += int(motion.shape[1])
            self.past_motion = torch.cat([self.past_motion, motion], dim=1)[:, -self.prefix_frames :, :]
            self.generated_frames += motion.shape[1]
            if self.render_mode != "none":
                frames.extend(self._render_motion_frames_pipelined(motion))
        self.last_step_profile["returned_frames"] = len(frames)
        self._profile_add("step", step_start)
        return frames

    @torch.no_grad()
    def _compute_audio_features(self, audio_tensor: torch.Tensor, audio_other_tensor: torch.Tensor):
        audio_list = [item.cpu().numpy() for item in audio_tensor]
        with self._autocast_context("audio"):
            inputs = self.model.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
            audio_fea = self.model.audio_encoder_face(
                torch.cat([inputs.input_values, torch.zeros([1, 80], device=self.device)], dim=-1)
            )["high_level"]
            audio_fea = F.interpolate(
                audio_fea.transpose(1, 2), scale_factor=(self.cfg.model.pose_fps / 50), mode="linear", align_corners=True
            ).transpose(1, 2)

        if not self.encode_listener_audio:
            return audio_fea, torch.zeros_like(audio_fea)

        audio_other_list = [item.cpu().numpy() for item in audio_other_tensor]
        with self._autocast_context("audio"):
            other_inputs = self.model.audio_processor(audio_other_list, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
            audio_other_fea = self.model.audio_encoder_face_other(
                torch.cat([other_inputs.input_values, torch.zeros([1, 80], device=self.device)], dim=-1)
            )["high_level"]
            audio_other_fea = F.interpolate(
                audio_other_fea.transpose(1, 2), scale_factor=(self.cfg.model.pose_fps / 50), mode="linear", align_corners=True
            ).transpose(1, 2)
        return audio_fea, audio_other_fea

    def _build_prefixed_audio_tensors(
        self,
        feature_start: int,
        feature_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a local prefixed-audio slice for sliding-window encoding.

        `feature_start` and `feature_end` are global feature-frame indices in the
        same prefixed coordinate system used by the original inference().
        """
        prefix_samples = self.prefix_frames * self.samples_per_frame
        full_len = prefix_samples + len(self.audio_buffer)
        full_other_len = prefix_samples + len(self.audio_other_buffer)
        start_sample = max(0, feature_start * self.samples_per_frame)
        end_sample = max(start_sample, feature_end * self.samples_per_frame)

        def slice_with_prefix(buffer: np.ndarray, total_len: int) -> np.ndarray:
            out = np.zeros((end_sample - start_sample,), dtype=np.float32)
            src_start = max(start_sample, prefix_samples)
            src_end = min(end_sample, total_len)
            if src_end > src_start:
                dst_start = src_start - start_sample
                buf_start = src_start - prefix_samples
                buf_end = src_end - prefix_samples
                out[dst_start : dst_start + (src_end - src_start)] = buffer[buf_start:buf_end]
            return out

        audio = slice_with_prefix(self.audio_buffer, full_len)
        audio_other = slice_with_prefix(self.audio_other_buffer, full_other_len)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        audio_other_tensor = torch.from_numpy(audio_other).float().unsqueeze(0).to(self.device)
        return audio_tensor, audio_other_tensor

    @torch.no_grad()
    def _generate_motion_batch(self, gen_frames: int) -> torch.Tensor:
        gen_frames = int(gen_frames)
        if gen_frames <= 0:
            raise ValueError("gen_frames must be positive")

        start_idx = self.generated_frames
        feature_len = self.prefix_frames + gen_frames + 1
        end_idx = start_idx + feature_len

        current_audio_idx = start_idx + self.prefix_frames
        desired_feature_start = max(0, current_audio_idx - self.audio_encode_left_context_frames)
        feature_start = min(start_idx, desired_feature_start)

        available_feature_end = self.prefix_frames + (len(self.audio_buffer) // self.samples_per_frame)
        last_current_audio_idx = current_audio_idx + gen_frames - 1
        desired_feature_end = last_current_audio_idx + self.lookahead_frames + 1
        required_feature_end = end_idx
        feature_end = min(max(required_feature_end, desired_feature_end), available_feature_end)

        audio_tensor, audio_other_tensor = self._build_prefixed_audio_tensors(feature_start, feature_end)
        audio_start = time.perf_counter()
        audio_fea, audio_other_fea = self._compute_audio_features(audio_tensor, audio_other_tensor)
        self._profile_add("audio", audio_start)

        local_start = start_idx - feature_start
        local_end = local_start + feature_len

        if audio_fea.shape[1] < local_end:
            pad_len = local_end - audio_fea.shape[1]
            audio_fea = torch.cat([audio_fea, audio_fea[:, -1:].repeat(1, pad_len, 1)], dim=1)
        if audio_other_fea.shape[1] < local_end:
            pad_len = local_end - audio_other_fea.shape[1]
            audio_other_fea = torch.cat([audio_other_fea, audio_other_fea[:, -1:].repeat(1, pad_len, 1)], dim=1)

        audio_slice_len = feature_len * self.audio_per_motion
        audio_slice_start = local_start * self.audio_per_motion
        audio_slice = audio_tensor[:, audio_slice_start : audio_slice_start + audio_slice_len]
        audio_other_slice = audio_other_tensor[:, audio_slice_start : audio_slice_start + audio_slice_len]
        if audio_slice.shape[1] < audio_slice_len:
            pad_len = audio_slice_len - audio_slice.shape[1]
            audio_slice = torch.cat([audio_slice, torch.zeros(audio_slice.shape[0], pad_len, device=self.device)], dim=1)
        if audio_other_slice.shape[1] < audio_slice_len:
            pad_len = audio_slice_len - audio_other_slice.shape[1]
            audio_other_slice = torch.cat([audio_other_slice, torch.zeros(audio_other_slice.shape[0], pad_len, device=self.device)], dim=1)

        if self.ema is not None and self.use_ema:
            self.ema.to(self.device)
            ctx = self.ema.average_parameters(self.model.parameters())
        else:
            ctx = nullcontext()

        self._sync_if_profile()
        motion_start = time.perf_counter()
        with ctx, self._autocast_context("motion"):
            out = self.model.one_clip_only_inference(
                per_compute_audio_feature=audio_fea[:, local_start:local_end],
                per_compute_audio_other_feature=audio_other_fea[:, local_start:local_end],
                past_audio_self=None,
                audio_self=audio_slice,
                past_audio_other=None,
                audio_other=audio_other_slice,
                past_motion=self.past_motion,
                gen_frames=gen_frames,
                anchor_latent=self.anchor_motion,
                noise_scheduler=self.noise_scheduler,
                num_inference_steps=self.denoising_steps,
                guidance_mode=self.guidance_mode,
            )
        self._profile_add("motion", motion_start)
        return out[:, -gen_frames:, :].float()

    @torch.no_grad()
    def _generate_next_motion(self) -> torch.Tensor:
        return self._generate_motion_batch(1)

    def _render_motion_frames_timed(self, motion_latents: torch.Tensor) -> tuple[List[np.ndarray], float]:
        self._sync_if_profile()
        render_start = time.perf_counter()
        frames = self._render_motion_frames(motion_latents)
        if self.profile:
            self._sync_if_profile()
        duration = time.perf_counter() - render_start
        if self.profile:
            self.profile_totals["render"] += duration
        return frames, duration

    def _render_motion_frames_profiled(self, motion_latents: torch.Tensor) -> List[np.ndarray]:
        frames, duration = self._render_motion_frames_timed(motion_latents)
        self._step_profile_add("render", duration)
        return frames

    def _render_motion_frames_pipelined(self, motion_latents: torch.Tensor) -> List[np.ndarray]:
        """Render with one-chunk delay so previous render can overlap current motion.

        In async mode, each call returns the previous render result and submits
        the current motion latents to a single render worker. Call flush() after
        the last audio chunk to retrieve the final submitted render.
        """
        if not self.async_render:
            return self._render_motion_frames_profiled(motion_latents)

        previous_future = self._pending_render_future
        if previous_future is not None:
            wait_start = time.perf_counter()
            completed, previous_render_duration = previous_future.result()
            self._step_profile_add("render_wait", time.perf_counter() - wait_start)
            self._step_profile_add("render", previous_render_duration)
        else:
            completed = []
        self._pending_render_future = self._render_executor.submit(
            self._render_motion_frames_timed,
            motion_latents.detach(),
        )
        return completed

    def flush(self) -> List[np.ndarray]:
        """Return any frames still pending in the async render worker."""
        if self._pending_render_future is None:
            return []
        frames, _ = self._pending_render_future.result()
        self._pending_render_future = None
        return frames

    def close(self) -> None:
        self.flush()
        if self._render_executor is not None:
            self._render_executor.shutdown(wait=True)
            self._render_executor = None

    @torch.no_grad()
    def _render_motion_frames(self, motion_latents: torch.Tensor) -> List[np.ndarray]:
        if self.render_mode == "none":
            return []

        total = motion_latents.shape[1]
        render_indices = list(range(0, total, self.render_frame_stride))
        if not render_indices:
            return []
        selected = motion_latents[:, render_indices, :].float()

        if self.render_mode == "batch":
            try:
                rendered = self._render_motion_batch(selected)
            except Exception as exc:
                print(f"[StatefulStream] Batch render failed, falling back to per-frame render: {exc}")
                rendered = self._render_motion_per_frame(selected)
        else:
            rendered = self._render_motion_per_frame(selected)

        if self.render_frame_stride == 1:
            return rendered

        frames = []
        rendered_pos = 0
        current = rendered[0]
        next_render_idx = render_indices[1] if len(render_indices) > 1 else total + 1
        for idx in range(total):
            if idx == next_render_idx:
                rendered_pos += 1
                current = rendered[rendered_pos]
                next_render_idx = render_indices[rendered_pos + 1] if rendered_pos + 1 < len(render_indices) else total + 1
            frames.append(current)
        return frames

    def _render_motion_per_frame(self, motion_latents: torch.Tensor) -> List[np.ndarray]:
        flow_estimator = self.vis_ctx["flow_estimator"]
        face_generator = self.vis_ctx["face_generator"]

        frames = []
        with self._autocast_context("render"):
            for idx in range(motion_latents.shape[1]):
                tgt = flow_estimator(self.anchor_motion.squeeze(0), motion_latents[:, idx, :])
                recon = face_generator(tgt, self.ref_face_feat)
                frame = recon[0].permute(1, 2, 0).detach().float().cpu().numpy()
                frame = np.clip((frame + 1) / 2 * 255, 0, 255).astype(np.uint8)
                frames.append(frame)
        return frames

    def _render_motion_batch(self, motion_latents: torch.Tensor) -> List[np.ndarray]:
        flow_estimator = self.vis_ctx["flow_estimator"]
        face_generator = self.vis_ctx["face_generator"]

        batch = motion_latents.shape[1]
        anchor = self.anchor_motion.squeeze(0).repeat(batch, 1)
        target = motion_latents.squeeze(0)
        face_feat = self.ref_face_feat
        if isinstance(face_feat, torch.Tensor) and face_feat.shape[0] == 1 and batch > 1:
            repeat_dims = [batch] + [1] * (face_feat.dim() - 1)
            face_feat = face_feat.repeat(*repeat_dims)

        with self._autocast_context("render"):
            tgt = flow_estimator(anchor, target)
            recon = face_generator(tgt, face_feat)
            video = recon.permute(0, 2, 3, 1).detach().float().cpu().numpy()
        video = np.clip((video + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return [frame for frame in video]
