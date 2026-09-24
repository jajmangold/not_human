import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from transformers import WhisperModel

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from scripts.reaction_schedule import (
    BLINK_SLOT,
    HEAD_POSE_SLOTS,
    QUIET_SLOTS,
    REACTION_SLOTS,
    blink_envelope,
    gaze_envelope,
    head_pose_envelope,
    reaction_envelope,
)
from motion_compositor import gaze_region_only, upper_face_only
from pose_warp import backward_pose_flow, warp_with_pose_flow

import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time
import subprocess
import hashlib
from collections import OrderedDict


_gpu_pipeline_graphs = {}
MAX_POSE_FLOW_CACHE = max(1, int(os.environ.get("MUSE_MAX_POSE_FLOW_CACHE", "64")))
MAX_CACHED_AVATARS = max(1, int(os.environ.get("MUSE_MAX_CACHED_AVATARS", "4")))


class JobCancelled(Exception):
    """Expected control-flow exception when a browser session disconnects."""


def run_gpu_pipeline(latent_batch, audio_feature_batch, use_taesd):
    """Run the fixed-shape UNet and TAESD through a CUDA graph when enabled.

    Eager Diffusers launches hundreds of tiny kernels per call. On the CMP fleet
    their CPU dispatch gaps cost substantially more than the kernels themselves.
    A captured graph keeps mutable static inputs and replays the exact same model
    graph without rebuilding the Python/CUDA launch sequence for every microbatch.
    """
    if os.environ.get("MUSE_USE_CUDA_GRAPH", "0") != "1":
        output = unet.model(
            latent_batch, timesteps, encoder_hidden_states=audio_feature_batch
        ).sample
        return output, None

    key = (
        tuple(latent_batch.shape), tuple(audio_feature_batch.shape),
        latent_batch.dtype, use_taesd,
    )
    cached = _gpu_pipeline_graphs.get(key)
    if cached is None:
        static_latent = latent_batch.clone()
        static_audio = audio_feature_batch.clone()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                static_output = unet.model(
                    static_latent, timesteps, encoder_hidden_states=static_audio
                ).sample
                static_rgb = None
                if use_taesd:
                    decoded = taesd.decode(static_output.to(taesd.dtype)).sample
                    static_rgb = (decoded / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            static_output = unet.model(
                static_latent, timesteps, encoder_hidden_states=static_audio
            ).sample
            static_rgb = None
            if use_taesd:
                decoded = taesd.decode(static_output.to(taesd.dtype)).sample
                static_rgb = (decoded / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float()
        cached = (graph, static_latent, static_audio, static_output, static_rgb)
        _gpu_pipeline_graphs[key] = cached
        print(f"[PROFILE] captured GPU pipeline CUDA graph for {key[:2]}", flush=True)

    graph, static_latent, static_audio, static_output, static_rgb = cached
    static_latent.copy_(latent_batch)
    static_audio.copy_(audio_feature_batch)
    graph.replay()
    return static_output, static_rgb


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


def write_stream_frame(path, frame, extension, encode_params=()):
    """Publish a complete encoded frame; directory watchers never see partial JPEGs."""
    ok, encoded = cv2.imencode(extension, frame, list(encode_params))
    if not ok:
        raise RuntimeError(f"could not encode stream frame {path}")
    temporary = f"{path}.tmp-{threading.get_ident()}"
    with open(temporary, "wb") as handle:
        handle.write(encoded.tobytes())
    os.replace(temporary, path)


def write_stream_face_bbox(directory, bbox):
    """Publish the tracked face bounds alongside streamed frames."""
    values = [int(round(float(value))) for value in bbox]
    temporary = os.path.join(directory, f".face_bbox-{threading.get_ident()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(values, handle)
    os.replace(temporary, os.path.join(directory, "face_bbox.json"))


def stable_avatar_id(video_path, bbox_shift):
    """Return an identity that survives separate audio jobs for one source face.

    Include file metadata so replacing a portrait at the same path cannot reuse stale
    latents/masks.  The generated id is deliberately filesystem-safe and does not expose
    the source path in the results directory.
    """
    try:
        stat = os.stat(video_path)
        stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        stamp = "missing"
    material = "|".join((os.path.realpath(video_path), stamp, str(bbox_shift), args.version))
    return "avatar_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def get_audio_feature_fast(audio_processor, wav_path, weight_dtype):
    """Extract Whisper mel features without librosa's slow WAV loader.

    The archived path spends ~27 s on the first ``librosa.load`` call in the
    resident image. MuseTalk jobs are already required to use 16 kHz WAV, so
    soundfile can decode directly and preserve the feature-extractor behavior.
    Non-16-kHz input falls back to the upstream resampling path.
    """
    import soundfile as sf

    samples, sampling_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sampling_rate != 16000:
        # librosa's default resampler is the other large cold-path stall in the
        # archived image. SciPy's polyphase filter is deterministic, high quality
        # for speech, and keeps the same mono/16 kHz contract.
        from scipy.signal import resample_poly

        samples = resample_poly(samples, 16000, sampling_rate).astype("float32", copy=False)
        sampling_rate = 16000
    segment_length = 30 * sampling_rate
    features = []
    for start in range(0, len(samples), segment_length):
        feature = audio_processor.feature_extractor(
            samples[start:start + segment_length],
            return_tensors="pt",
            sampling_rate=sampling_rate,
        ).input_features
        features.append(feature.to(dtype=weight_dtype) if weight_dtype is not None else feature)
    return features, len(samples)


@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation,
                 source_start_frame=0, bank_start_frame=0):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.source_start_frame = source_start_frame
        self.bank_start_frame = bank_start_frame
        # 根据版本设置不同的基础路径
        if args.version == "v15":
            self.base_path = f"./results/{args.version}/avatars/{avatar_id}"
        else:  # v1
            self.base_path = f"./results/avatars/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "source_start_frame": source_start_frame,
            "bank_start_frame": bank_start_frame,
            "version": args.version
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self._warmup_keys = set()
        self._pose_flow_cache = OrderedDict()
        self.init()

    def _blend_source_value(self, values, base_index, low_index, high_index, amount):
        if low_index == high_index:
            target, mix = values[low_index], amount
        elif amount <= 0.5:
            target, mix = values[low_index], amount * 2.0
        else:
            low, high, upper_mix = values[low_index], values[high_index], (amount - 0.5) * 2.0
            if torch.is_tensor(low):
                target = low * (1.0 - upper_mix) + high * upper_mix
            elif isinstance(low, np.ndarray):
                target = cv2.addWeighted(low, 1.0 - upper_mix, high, upper_mix, 0)
            else:
                target = [a * (1.0 - upper_mix) + b * upper_mix for a, b in zip(low, high)]
            mix = 1.0
        base = values[base_index]
        if torch.is_tensor(base):
            return base * (1.0 - mix) + target * mix
        if isinstance(base, np.ndarray):
            return cv2.addWeighted(base, 1.0 - mix, target, mix, 0)
        return [int(round(a * (1.0 - mix) + b * mix)) for a, b in zip(base, target)]

    def _mix_source_value(self, lower, upper, amount):
        if amount <= 0:
            return lower
        if torch.is_tensor(lower):
            return lower * (1.0 - amount) + upper * amount
        if isinstance(lower, np.ndarray):
            return cv2.addWeighted(lower, 1.0 - amount, upper, amount, 0)
        return [int(round(a * (1.0 - amount) + b * amount)) for a, b in zip(lower, upper)]

    def _add_source_delta(self, current, base, target, amount):
        if torch.is_tensor(current):
            return current + (target - base) * amount
        if isinstance(current, np.ndarray):
            value = current.astype(np.float32) + (target.astype(np.float32) - base.astype(np.float32)) * amount
            return np.clip(value, 0, 255).astype(current.dtype) if current.dtype == np.uint8 else value.astype(current.dtype)
        return [int(round(value + (upper - lower) * amount)) for value, lower, upper in zip(current, base, target)]

    def _warp_pose_frame(self, current, frame_cycle, base_index, head_index, amount):
        """Interpolate displaced geometry without cross-dissolving two faces."""
        cache_key = (base_index, head_index)
        flow = self._pose_flow_cache.pop(cache_key, None)
        if flow is None:
            flow = backward_pose_flow(frame_cycle[base_index], frame_cycle[head_index])
        self._pose_flow_cache[cache_key] = flow
        while len(self._pose_flow_cache) > MAX_POSE_FLOW_CACHE:
            self._pose_flow_cache.popitem(last=False)
        return warp_with_pose_flow(current, flow, amount)

    def _render_sources(self, video_num, fps, reaction):
        """Build aligned upper-face inputs; the mouth is still rendered afterward."""
        if reaction and reaction.get("reaction") == "startup":
            ready_count = int(reaction.get("ready_frames", self.bank_start_frame))
            if ready_count < 0:
                raise ValueError("ready_frames must be non-negative")
            if ready_count > self.bank_start_frame or ready_count > video_num:
                raise ValueError("startup audio/source is shorter than ready_frames")
            if ready_count > len(self.frame_list_cycle):
                raise ValueError("ready_frames exceeds the prepared source cycle")
            idle = self._render_sources(video_num - ready_count, fps, {"reaction": "idle"})
            ready_indices = list(range(ready_count))
            ready = {
                "frames": [self.frame_list_cycle[index] for index in ready_indices],
                "base_frames": [self.frame_list_cycle[index] for index in ready_indices],
                "gaze_frames": [self.frame_list_cycle[index] for index in ready_indices],
                "coords": [self.coord_list_cycle[index] for index in ready_indices],
                "latents": [self.input_latent_list_cycle[index] for index in ready_indices],
                "mask_coords": [self.mask_coords_list_cycle[index] for index in ready_indices],
                "masks": [self.mask_list_cycle[index] for index in ready_indices],
                "gaze_amounts": [0.0 for _ in ready_indices],
                "blink_amounts": [0.0 for _ in ready_indices],
            }
            return {key: ready[key] + idle[key] for key in ready}

        offset = min(self.bank_start_frame, len(self.frame_list_cycle))
        frame_cycle = self.frame_list_cycle[offset:]
        coord_cycle = self.coord_list_cycle[offset:]
        latent_cycle = self.input_latent_list_cycle[offset:]
        mask_coord_cycle = self.mask_coords_list_cycle[offset:]
        mask_cycle = self.mask_list_cycle[offset:]
        bank_size = len(frame_cycle)
        if bank_size <= 0:
            raise ValueError("bank_start_frame removes every settled source frame")
        if bank_size < 16:
            indices = [index % bank_size for index in range(video_num)]
        elif reaction and reaction.get("reaction") == "idle":
            quiet_frames = max(1, video_num - 4)
            blink_curve = (0.0, 0.68, 1.0, 0.22)
            sources = {"frames": [], "base_frames": [], "gaze_frames": [], "coords": [], "latents": [], "mask_coords": [], "masks": [], "gaze_amounts": [], "blink_amounts": []}
            for index in range(video_num):
                phase = 0.0
                gaze_amount = 0.0
                if quiet_frames > 1 and index < quiet_frames:
                    rest_t = index / (quiet_frames - 1)
                    # Random access gaze axis: left/up -> camera -> right/down.
                    # The browser owns time and seeks this pre-MuseTalk bank.
                    if rest_t <= 0.5:
                        lower_slot, upper_slot = QUIET_SLOTS[1], QUIET_SLOTS[0]
                        phase = 3.0 * (rest_t * 2.0) ** 2 - 2.0 * (rest_t * 2.0) ** 3
                        gaze_amount = 1.0 - phase
                    else:
                        lower_slot, upper_slot = QUIET_SLOTS[0], QUIET_SLOTS[2]
                        local = (rest_t - 0.5) * 2.0
                        phase = 3.0 * local**2 - 2.0 * local**3
                        gaze_amount = phase
                else:
                    lower_slot = upper_slot = QUIET_SLOTS[0]
                blink = blink_curve[index - quiet_frames] if index >= quiet_frames else 0.0
                quiet_frame = self._mix_source_value(
                    frame_cycle[lower_slot], frame_cycle[upper_slot], phase)
                # MuseTalk always sees the camera-facing latent and landmarks.
                # Gaze is an upper-face overlay; feeding its small full-face
                # residual into the lip model makes the silent mouth twitch.
                sources["base_frames"].append(frame_cycle[QUIET_SLOTS[0]])
                sources["gaze_frames"].append(quiet_frame)
                sources["frames"].append(self._add_source_delta(
                    quiet_frame, frame_cycle[QUIET_SLOTS[0]], frame_cycle[BLINK_SLOT], blink))
                sources["coords"].append(coord_cycle[QUIET_SLOTS[0]])
                sources["latents"].append(latent_cycle[QUIET_SLOTS[0]])
                sources["gaze_amounts"].append(gaze_amount)
                sources["blink_amounts"].append(blink)
                # A stable mouth mask is deliberate: idle upper-face motion and
                # blinking must not move the region MuseTalk owns.
                sources["mask_coords"].append(mask_coord_cycle[QUIET_SLOTS[0]])
                sources["masks"].append(mask_cycle[QUIET_SLOTS[0]])
            return sources
        else:
            indices = None

        if indices is not None:
            return {
                "frames": [frame_cycle[index] for index in indices],
                "base_frames": [frame_cycle[index] for index in indices],
                "gaze_frames": [frame_cycle[index] for index in indices],
                "coords": [coord_cycle[index] for index in indices],
                "latents": [latent_cycle[index] for index in indices],
                "mask_coords": [mask_coord_cycle[index] for index in indices],
                "masks": [mask_cycle[index] for index in indices],
                "gaze_amounts": [0.0 for _ in indices],
                "blink_amounts": [0.0 for _ in indices],
            }

        name = str((reaction or {}).get("reaction", "neutral"))
        low_index, high_index = REACTION_SLOTS.get(name, (0, 0))
        sources = {"frames": [], "base_frames": [], "gaze_frames": [], "coords": [], "latents": [], "mask_coords": [], "masks": [], "gaze_amounts": [], "blink_amounts": []}
        for index in range(video_num):
            base_index = QUIET_SLOTS[0]
            amount = reaction_envelope(index, video_num, fps, reaction)
            blink = blink_envelope(index, fps, reaction)
            semantic_frame = self._blend_source_value(frame_cycle, base_index, low_index, high_index, amount)
            semantic_coord = self._blend_source_value(coord_cycle, base_index, low_index, high_index, amount)
            semantic_latent = self._blend_source_value(latent_cycle, base_index, low_index, high_index, amount)
            for primitive in (reaction or {}).get("primitive_mix", [])[:2]:
                secondary_slots = REACTION_SLOTS.get(str(primitive.get("reaction", "")))
                secondary_mix = min(0.22, max(0.0, float(primitive.get("weight", 0.0))))
                secondary_amount = amount * secondary_mix
                if not secondary_slots or secondary_amount <= 0:
                    continue
                secondary_frame = self._blend_source_value(frame_cycle, base_index, *secondary_slots, 1.0)
                secondary_coord = self._blend_source_value(coord_cycle, base_index, *secondary_slots, 1.0)
                secondary_latent = self._blend_source_value(latent_cycle, base_index, *secondary_slots, 1.0)
                semantic_frame = self._add_source_delta(
                    semantic_frame, frame_cycle[base_index], secondary_frame, secondary_amount)
                semantic_coord = self._add_source_delta(
                    semantic_coord, coord_cycle[base_index], secondary_coord, secondary_amount)
                semantic_latent = self._add_source_delta(
                    semantic_latent, latent_cycle[base_index], secondary_latent, secondary_amount)
            # Build expression, gaze, and blink at the neutral geometry first.
            # The final optical-flow warp moves each complete face once; an
            # RGB alpha blend between neutral and rotated frames creates the
            # doubled shadow/ghost silhouette reported by the user.
            prepose_frame = semantic_frame
            head_axis, head_direction, head_amount = head_pose_envelope(index, fps, reaction)
            head_index = HEAD_POSE_SLOTS.get((head_axis, head_direction))
            if head_index is not None and head_amount > 0:
                # Landmarks and latent remain continuously interpolated so the
                # MuseTalk mouth stays registered to the warped head.
                semantic_coord = self._add_source_delta(
                    semantic_coord, coord_cycle[base_index], coord_cycle[head_index], head_amount)
                semantic_latent = self._add_source_delta(
                    semantic_latent, latent_cycle[base_index], latent_cycle[head_index], head_amount)
                base_frame = self._warp_pose_frame(
                    prepose_frame, frame_cycle, base_index, head_index, head_amount)
            else:
                base_frame = prepose_frame
            base_coord, base_latent = semantic_coord, semantic_latent
            gaze_direction, gaze_amount = gaze_envelope(index, fps, reaction)
            gaze_frame = prepose_frame
            if gaze_amount > 0:
                gaze_amount *= 1.0 - 0.35 * amount
                gaze_index = QUIET_SLOTS[1] if gaze_direction < 0 else QUIET_SLOTS[2]
                gaze_frame = self._add_source_delta(
                    gaze_frame, frame_cycle[base_index], frame_cycle[gaze_index], gaze_amount)
            visible_frame = self._add_source_delta(
                gaze_frame, frame_cycle[base_index], frame_cycle[BLINK_SLOT], blink)
            if head_index is not None and head_amount > 0:
                gaze_frame = self._warp_pose_frame(
                    gaze_frame, frame_cycle, base_index, head_index, head_amount)
                visible_frame = self._warp_pose_frame(
                    visible_frame, frame_cycle, base_index, head_index, head_amount)
            sources["base_frames"].append(base_frame)
            sources["gaze_frames"].append(gaze_frame)
            sources["frames"].append(visible_frame)
            # Eye direction is composited after MuseTalk. Keep gaze landmark
            # and latent deltas out of lip inference so mouth ownership is real.
            sources["coords"].append(base_coord)
            sources["latents"].append(base_latent)
            sources["gaze_amounts"].append(gaze_amount)
            sources["blink_amounts"].append(blink)
            nearest = low_index if amount >= 0.25 else base_index
            if amount >= 0.75:
                nearest = high_index
            if head_index is not None and head_amount > 0:
                # Keep one stable neutral parsing mask while RGB/landmarks/
                # latent move through the partial pose. Switching to the
                # endpoint mask at 40% exposes a one-frame blend-boundary pop.
                nearest = base_index
            sources["mask_coords"].append(mask_coord_cycle[nearest])
            sources["masks"].append(mask_cycle[nearest])
        return sources

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                response = "y"  # resident server: always recreate non-interactively
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.frame_list_cycle = read_imgs(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.mask_list_cycle = read_imgs(input_mask_list)
            else:
                print("*********************************")
                print(f"  creating avator: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        extracted_img_list = input_img_list
        if self.source_start_frame:
            if self.source_start_frame >= len(input_img_list):
                raise ValueError("source_start_frame removes every source frame")
            input_img_list = input_img_list[self.source_start_frame:]

        print("extracting landmarks...")
        _tl = time.time()
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        if self.source_start_frame:
            # The video extractor populated full_imgs with every frame. Rebuild
            # the directory from the settled slice so cached reloads cannot
            # silently restore the one-shot ready gesture. Landmark extraction
            # has already loaded the retained frames into frame_list; _prep below
            # writes that in-memory slice back with fresh zero-based filenames.
            for extracted_path in extracted_img_list:
                os.remove(extracted_path)
        print(f"[PROFILE] landmark+bbox = {time.time()-_tl:.2f}s for {len(frame_list)} frames")
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        _te = time.time()
        # Optional: use TAESD (tiny distilled VAE) for the ENCODE too. Its latent space matches SD-VAE's
        # (verified mean/std ~equal, scaling_factor=1.0), so it drops into the UNet input. ~10x faster
        # encode. Quality is A/B-tested; MUSE_TAESD_ENCODE=1 to enable.
        _use_tae_enc = (os.environ.get("MUSE_TAESD_ENCODE", "0") == "1" or os.path.exists("/io/.taesd_encode")) and (taesd is not None)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if args.version == "v15":
                y2 = y2 + args.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]  # 更新coord_list中的bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            if _use_tae_enc:
                _m = vae.preprocess_img(resized_crop_frame, half_mask=True).half()
                _r = vae.preprocess_img(resized_crop_frame, half_mask=False).half()
                with torch.no_grad():
                    latents = torch.cat([taesd.encode(_m).latents, taesd.encode(_r).latents], dim=1)
            else:
                latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)
        print(f"[PROFILE] VAE latent-encode ({'TAESD' if _use_tae_enc else 'SD-VAE'}) = {time.time()-_te:.2f}s for {len(input_latent_list)} frames")
        _tm = time.time()

        # ping-pong doubling (forward+reverse) only needed when audio is LONGER than the ref video so the
        # loop wraps smoothly. Our pipeline always renders video == audio length, so skip it (halves the
        # prepare/mask cost). Set MUSE_NO_PINGPONG=0 to restore the original behavior.
        if os.environ.get("MUSE_NO_PINGPONG", "1") == "1":
            self.frame_list_cycle = frame_list
            self.coord_list_cycle = coord_list
            self.input_latent_list_cycle = input_latent_list
        else:
            self.frame_list_cycle = frame_list + frame_list[::-1]
            self.coord_list_cycle = coord_list + coord_list[::-1]
            self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        N = len(self.frame_list_cycle)
        self.mask_coords_list_cycle = [None] * N
        self.mask_list_cycle = [None] * N

        # parallelize the prepare loop across threads. FaceParsing is ~310ms/frame but only 8.8ms is the
        # GPU net — the rest is CPU (.cpu() sync, argmax over [19,512,512], cv2 morphology) which releases
        # the GIL and parallelizes. BiSeNet forward is thread-safe in eval mode, so we DON'T lock it
        # (locking the whole call serialized the 300ms CPU part and was the real mask-prepare bottleneck).
        _mode = args.parsing_mode if args.version == "v15" else "raw"
        def _prep(i):
            frame = self.frame_list_cycle[i]
            x1, y1, x2, y2 = self.coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=_mode)
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle[i] = crop_box
            self.mask_list_cycle[i] = mask
        _workers = int(os.environ.get("MUSE_PREP_WORKERS", "10"))
        with ThreadPoolExecutor(max_workers=_workers) as _ex:
            list(tqdm(_ex.map(_prep, range(N)), total=N, desc="prepare"))
        print(f"[PROFILE] mask-prepare = {time.time()-_tm:.2f}s for {N} frames")

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def process_frames(self, res_frame_queue, video_len, skip_save_images, stream_format, sources):
        # Worker: pull (idx, res_frame) tuples until a None sentinel. Index is explicit so multiple
        # workers can run concurrently without scrambling frame order. get_image_blending is pure
        # numpy/cv2 (releases the GIL) so threads parallelize across cores.
        ncyc = len(sources["coords"])
        while True:
            item = res_frame_queue.get(block=True)
            if item is None:
                break
            idx, res_frame = item
            bbox = sources["coords"][idx % ncyc]
            ori_frame = copy.deepcopy(sources["frames"][idx % ncyc])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue
            mask = sources["masks"][idx % ncyc]
            mask_crop_box = sources["mask_coords"][idx % ncyc]
            gaze = sources["gaze_amounts"][idx % ncyc]
            blink = sources["blink_amounts"][idx % ncyc]
            if gaze > 0 or blink > 0:
                base_frame = copy.deepcopy(sources["base_frames"][idx % ncyc])
                base_combine = get_image_blending(base_frame, res_frame, bbox, mask, mask_crop_box)
                if gaze > 0:
                    gaze_frame = copy.deepcopy(sources["gaze_frames"][idx % ncyc])
                    gaze_combine = get_image_blending(gaze_frame, res_frame, bbox, mask, mask_crop_box)
                    # The gaze frame already contains the scheduled amplitude;
                    # apply its upper-face pixels once, never through the mouth.
                    combine_frame = gaze_region_only(base_combine, gaze_combine, bbox)
                else:
                    combine_frame = base_combine
            if blink > 0:
                # Preserve the existing blink response while layering it over
                # either the neutral or gaze-isolated upper face.
                blink_combine = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                combine_frame = upper_face_only(combine_frame, blink_combine, bbox, blink)
            if gaze <= 0 and blink <= 0:
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
            if skip_save_images is False:
                output_dir = getattr(self, "frame_output_dir", f"{self.avatar_path}/tmp")
                if stream_format in {"jpg", "jpeg"}:
                    write_stream_frame(
                        f"{output_dir}/{str(idx).zfill(8)}.jpg", combine_frame,
                        ".jpg", (cv2.IMWRITE_JPEG_QUALITY, 82))
                else:
                    write_stream_frame(
                        f"{output_dir}/{str(idx).zfill(8)}.png", combine_frame, ".png")

    @torch.no_grad()
    def inference(self, audio_path, out_vid_name, fps, skip_save_images, stream_dir=None,
                  stream_format="png", reaction=None, cancel_path=None):
        # A stream chunk writes frames directly to the caller's /io directory. It
        # does not invoke ffmpeg; consumers can display each frame as it arrives.
        self.frame_output_dir = stream_dir or (self.avatar_path + '/tmp')
        os.makedirs(self.frame_output_dir, exist_ok=True)
        print("start inference")
        if cancel_path and os.path.exists(cancel_path):
            raise JobCancelled()
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        # Extract audio features
        whisper_input_features, librosa_length = get_audio_feature_fast(
            audio_processor, audio_path, weight_dtype
        )
        t_af = time.time()
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        print(f"[PROFILE] get_audio_feature={t_af-start_time:.2f}s  get_whisper_chunk={time.time()-t_af:.2f}s")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)
        render_sources = self._render_sources(video_num, fps, reaction)
        if stream_dir and render_sources["coords"]:
            metadata_index = min(
                int((reaction or {}).get("ready_frames", 0)),
                len(render_sources["coords"]) - 1,
            )
            write_stream_face_bbox(stream_dir, render_sources["coords"][metadata_index])
        res_frame_queue = queue.Queue()
        self.idx = 0
        # Spawn N blend workers (pure-CPU get_image_blending parallelizes across cores)
        n_blend = int(os.environ.get("MUSE_BLEND_WORKERS", "10"))
        process_threads = [threading.Thread(target=self.process_frames,
                                            args=(res_frame_queue, video_num, skip_save_images,
                                                  stream_format, render_sources))
                           for _ in range(n_blend)]
        for _t in process_threads:
            _t.start()

        gen = list(datagen(whisper_chunks,
                     render_sources["latents"],
                     self.batch_size))
        use_taesd = os.environ.get("USE_TAESD", "0") == "1"
        # Warm up once per real input shape. Repeating this for every short live
        # chunk adds needless latency after the resident has already paid the
        # torch.compile/cuDNN setup cost.
        wb, lb = gen[0]
        warmup_key = (tuple(wb.shape), tuple(lb.shape), use_taesd)
        if (os.environ.get("MUSE_USE_CUDA_GRAPH", "0") != "1"
                and warmup_key not in self._warmup_keys):
            with torch.no_grad():
                af = pe(wb.to(device)); lb = lb.to(device=device, dtype=unet.model.dtype)
                pl = unet.model(lb, timesteps, encoder_hidden_states=af).sample
                if use_taesd:
                    _ = taesd.decode(pl.to(taesd.dtype)).sample
                else:
                    _ = vae.decode_latents(pl.to(vae.vae.dtype))
            torch.cuda.synchronize()
            self._warmup_keys.add(warmup_key)
        start_time = time.time()
        res_frame_list = []
        put_idx = 0

        t_gpu_pipeline = 0.0; t_output_transfer = 0.0
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            if cancel_path and os.path.exists(cancel_path):
                for _ in process_threads:
                    res_frame_queue.put(None)
                for _t in process_threads:
                    _t.join()
                raise JobCancelled()
            valid_frames = latent_batch.shape[0]
            if valid_frames < self.batch_size:
                padding = self.batch_size - valid_frames
                whisper_batch = torch.cat(
                    (whisper_batch, whisper_batch[-1:].repeat(padding, 1, 1)), dim=0
                )
                latent_batch = torch.cat(
                    (latent_batch, latent_batch[-1:].repeat(padding, 1, 1, 1)), dim=0
                )
            torch.cuda.synchronize(); _a = time.time()
            with torch.cuda.nvtx.range("musetalk/input_projection"):
                audio_feature_batch = pe(whisper_batch.to(device))
                latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

            with torch.cuda.nvtx.range("musetalk/gpu_pipeline"):
                pred_latents, captured_rgb = run_gpu_pipeline(
                    latent_batch, audio_feature_batch, use_taesd
                )
            torch.cuda.synchronize(); _b = time.time(); t_gpu_pipeline += _b - _a
            with torch.cuda.nvtx.range("musetalk/output_transfer"):
                if captured_rgb is not None:
                    img = captured_rgb.detach().cpu().numpy()
                    recon = (img * 255).round().astype("uint8")[..., ::-1]
                elif use_taesd:
                    img = taesd.decode(pred_latents.to(taesd.dtype)).sample      # [-1,1], scaled-latent convention
                    img = (img / 2 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).float().numpy()
                    recon = (img * 255).round().astype("uint8")[..., ::-1]       # RGB->BGR
                else:
                    pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
                    recon = vae.decode_latents(pred_latents)
            recon = recon[:valid_frames]
            torch.cuda.synchronize(); _c = time.time(); t_output_transfer += _c - _b
            if i == 0:
                cv2.imwrite(f"{self.avatar_path}/decode_sample_{'taesd' if use_taesd else 'sdvae'}.png", recon[0])
            for res_frame in recon:
                res_frame_queue.put((put_idx, res_frame)); put_idx += 1
        gen_done = time.time()
        print(f"[PROFILE] GPU gen loop = {gen_done - start_time:.2f}s for {video_num} frames "
              f"({video_num/(gen_done-start_time):.1f} fps)  |  "
              f"gpu_pipeline={t_gpu_pipeline:.2f}s output_transfer={t_output_transfer:.2f}s")
        # Signal workers to stop (one sentinel each), then drain
        for _ in process_threads:
            res_frame_queue.put(None)
        for _t in process_threads:
            _t.join()
        print(f"[PROFILE] after blend-drain join = {time.time()-start_time:.2f}s "
              f"(blend backlog wait = {time.time()-gen_done:.2f}s)")

        if args.skip_save_images is True:
            print('Total process time of {} frames without saving images = {}s'.format(
                video_num,
                time.time() - start_time))
        else:
            print('Total process time of {} frames including saving images = {}s'.format(
                video_num,
                time.time() - start_time))

        if out_vid_name is not None and args.skip_save_images is False and stream_dir is None:
            # optional
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
            print(cmd_img2video)
            os.system(cmd_img2video)

            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")  # on
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
            print(cmd_combine_audio)
            os.system(cmd_combine_audio)

            os.remove(f"{self.avatar_path}/temp.mp4")
            shutil.rmtree(f"{self.avatar_path}/tmp")
            print(f"result is save to {output_vid}")
        print("\n")


if __name__ == "__main__":
    sys.argv = ["muse_server",
                "--version", "v15", "--gpu_id", "0",
                "--unet_model_path", "/opt/MuseTalk/models/musetalkV15/unet.pth",
                "--unet_config", "/opt/MuseTalk/models/musetalkV15/musetalk.json",
                "--whisper_dir", "/opt/MuseTalk/models/whisper",
                "--ffmpeg_path", "/usr/bin"]
    '''
    This script is used to simulate online chatting and applies necessary pre-processing such as face detection and face parsing in advance. During online chatting, only UNet and the VAE decoder are involved, which makes MuseTalk real-time.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Version of MuseTalk: v1 or v15")
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/musetalk.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalk/pytorch_model.bin", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--inference_config", type=str, default="configs/inference/realtime.yaml")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--result_dir", default='./results', help="Directory for output results")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int,
                        default=int(os.environ.get("MUSE_BATCH_SIZE", "20")),
                        help="Batch size for inference")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')
    parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--skip_save_images",
                       action="store_true",
                       help="Whether skip saving images for better generation speed calculation",
                       )

    args = parser.parse_args()

    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        # Choose path separator based on operating system
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    # Set computing device
    torch.backends.cudnn.benchmark = True                       # autotune fixed-shape convs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)

    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)

    taesd = None
    if os.environ.get("USE_TAESD", "0") == "1":
        from diffusers import AutoencoderTiny
        taesd = AutoencoderTiny.from_pretrained(os.environ.get("TAESD_DIR", "/opt/MuseTalk/models/taesd"))
        taesd = taesd.half().to(device).eval()
        print("[PROFILE] using TAESD decoder")

    if os.environ.get("USE_COMPILE", "0") == "1":
        try:
            unet.model = torch.compile(unet.model)
            print("[PROFILE] torch.compile(unet) enabled")
        except Exception as e:
            print(f"[PROFILE] torch.compile failed: {e}")

    if os.environ.get("USE_XFORMERS", "0") == "1":
        try:
            unet.model.enable_xformers_memory_efficient_attention()
            print("[PROFILE] xformers memory_efficient_attention enabled")
        except Exception as e:
            print(f"[PROFILE] xformers enable failed ({type(e).__name__}): {e}")
            import xformers.ops as xops
            print("[PROFILE] xformers import ok; will need manual attn processor")
    unet.model = unet.model.half().to(device)

    # Initialize audio processor and Whisper model
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    if os.environ.get("MUSE_AUDIO_WARMUP", "1") == "1":
        # WhisperFeatureExtractor's first FFT call may load CPU code. Pay that
        # once during resident startup so the first live utterance is not penalized.
        _warm_start = time.time()
        audio_processor.feature_extractor(
            np.zeros(16000, dtype=np.float32), return_tensors="pt", sampling_rate=16000
        )
        print(f"[PROFILE] audio feature warmup = {time.time() - _warm_start:.2f}s", flush=True)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Initialize face parser with configurable parameters based on version
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:  # v1
        fp = FaceParsing()

    # ---- resident job loop: poll JOBS dir for {video,audio,out} ----------------
    import glob as _glob, shutil as _shutil, time as _time
    JOBS = os.environ.get("MUSE_JOBS", "/io/muse_jobs")
    cache_avatars = os.environ.get("MUSE_CACHE_AVATARS", "1") == "1"
    avatar_cache = OrderedDict()
    os.makedirs(JOBS, exist_ok=True)
    ready_marker = os.path.join(os.path.dirname(JOBS), "MUSE_SERVER_READY")
    try:
        os.remove(ready_marker)
    except FileNotFoundError:
        pass
    print("MUSE_SERVER_READY", flush=True)
    with open(ready_marker, "w", encoding="utf-8") as marker:
        marker.write("ready\n")
    while True:
        for jf in sorted(_glob.glob(os.path.join(JOBS, "*.json"))):
            name = os.path.basename(jf)[:-5]
            try:
                spec = json.load(open(jf)); os.remove(jf)
                cancel_path = spec.get("cancel_path")
                if cancel_path and os.path.exists(cancel_path):
                    open(os.path.join(JOBS, name + ".done"), "w").write("cancelled")
                    continue
                video = spec["video"]
                prepare_only = bool(spec.get("prepare_only", False))
                audio = spec.get("audio")
                outp = spec.get("out")
                stream_dir = spec.get("stream_dir")
                stream_format = str(spec.get("stream_format", "png")).lower()
                if stream_format not in {"png", "jpg", "jpeg"}:
                    raise ValueError("stream_format must be png or jpg")
                if not prepare_only and not audio:
                    raise ValueError("render job requires audio")
                if not prepare_only and not outp and not stream_dir:
                    raise ValueError("job requires either 'out' or 'stream_dir'")
                bbox_shift = int(spec.get("bbox_shift", 0))
                source_start_frame = int(spec.get("source_start_frame", 0))
                bank_start_frame = int(spec.get("bank_start_frame", 0))
                if source_start_frame < 0:
                    raise ValueError("source_start_frame must be non-negative")
                if bank_start_frame < 0:
                    raise ValueError("bank_start_frame must be non-negative")
                # A stable avatar is the key to the actual MuseTalk realtime path:
                # preparation runs once, while later jobs only do audio/UNet/VAE work.
                cache_key = (spec.get("avatar_id") or stable_avatar_id(video, bbox_shift),
                             os.path.realpath(video), bbox_shift, source_start_frame, bank_start_frame)
                avatar = avatar_cache.get(cache_key) if cache_avatars else None
                if avatar is not None:
                    avatar_cache.move_to_end(cache_key)
                if avatar is None:
                    aid = cache_key[0]
                    avatar_root = os.path.join(
                        "./results", args.version, "avatars", aid
                    ) if args.version == "v15" else os.path.join("./results", "avatars", aid)
                    info_path = os.path.join(avatar_root, "avator_info.json")
                    ready = False
                    if os.path.isfile(info_path):
                        try:
                            with open(info_path, "r", encoding="utf-8") as handle:
                                info = json.load(handle)
                            ready = (os.path.realpath(info.get("video_path", "")) ==
                                     os.path.realpath(video)
                                     and int(info.get("bbox_shift", 0)) == bbox_shift
                                     and int(info.get("source_start_frame", 0)) == source_start_frame
                                     and int(info.get("bank_start_frame", 0)) == bank_start_frame
                                     and os.path.exists(os.path.join(avatar_root, "latents.pt")))
                        except (OSError, ValueError, TypeError, json.JSONDecodeError):
                            ready = False
                    avatar = Avatar(avatar_id=aid, video_path=video,
                                    bbox_shift=bbox_shift,
                                    batch_size=args.batch_size,
                                    preparation=not ready,
                                    source_start_frame=source_start_frame,
                                    bank_start_frame=bank_start_frame)
                    if cache_avatars:
                        avatar_cache[cache_key] = avatar
                        avatar_cache.move_to_end(cache_key)
                        while len(avatar_cache) > MAX_CACHED_AVATARS:
                            avatar_cache.popitem(last=False)
                t0 = _time.time()
                if not prepare_only:
                    requested_batch = int(spec.get("batch_size", args.batch_size))
                    if requested_batch < 1:
                        raise ValueError("batch_size must be positive")
                    previous_batch = avatar.batch_size
                    avatar.batch_size = requested_batch
                    try:
                        avatar.inference(audio, "out" if outp else None,
                                         int(spec.get("fps", args.fps)), False,
                                         stream_dir=stream_dir, stream_format=stream_format,
                                         reaction=spec.get("reaction"), cancel_path=cancel_path)
                    finally:
                        avatar.batch_size = previous_batch
                    if outp:
                        produced = os.path.join(avatar.video_out_path, "out.mp4")
                        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
                        _shutil.copy(produced, outp)
                if not cache_avatars:
                    _shutil.rmtree(avatar.avatar_path, ignore_errors=True)
                dt = round(_time.time() - t0, 1)
                target = avatar.avatar_path if prepare_only else (outp or stream_dir)
                open(os.path.join(JOBS, name + ".done"), "w").write(f"ok {dt}s -> {target}")
                print(f"MUSE JOB DONE {name} {dt}s -> {target}", flush=True)
            except JobCancelled:
                open(os.path.join(JOBS, name + ".done"), "w").write("cancelled")
                print(f"MUSE JOB CANCELLED {name}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                open(os.path.join(JOBS, name + ".err"), "w").write(str(e))
        # Live phrases arrive as separate jobs. A one-second poll interval creates
        # an audible/video gap after every phrase even when inference is faster
        # than the playback clock, so keep the resident dispatch loop responsive.
        _time.sleep(float(os.environ.get("MUSE_JOB_POLL_INTERVAL", "0.01")))
