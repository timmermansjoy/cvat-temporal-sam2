# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import base64
import collections
import contextlib
import io
import json
import math
import os
import pickle
import time
import uuid
from dataclasses import dataclass

import cv2
import numpy as np
import redis
import torch
import torchvision.transforms
from PIL import Image, UnidentifiedImageError
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import fill_holes_in_mask_scores


MODEL_ID = "facebook/sam2.1-hiera-tiny"
STATE_TTL_SECONDS = 8 * 60 * 60
LOCK_TTL_SECONDS = 10 * 60
LOCK_WAIT_SECONDS = 30
MAX_IMAGE_PIXELS = 100_000_000
MAX_BATCH_SIZE = 10
STATE_KEY_PREFIX = "cvat:sam2:state:"


class RequestError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class PreprocessedImage:
    width: int
    height: int
    vision_feats: list[torch.Tensor]
    vision_pos_embeds: list[torch.Tensor]
    feat_sizes: list[tuple[int, int]]


@dataclass
class TrackingState:
    frame_idx: int
    predictor_outputs: dict


def _is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _as_int(value, field):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise RequestError(f"{field} must be an integer")


def _decode_mask(points, width, height):
    if not isinstance(points, list) or len(points) < 5:
        raise RequestError("mask points must contain run lengths and a bounding box")

    values = [_as_int(value, "mask point") for value in points]
    run_lengths, x1, y1, x2, y2 = (
        values[:-4],
        values[-4],
        values[-3],
        values[-2],
        values[-1],
    )
    x2 += 1
    y2 += 1
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise RequestError("mask bounding box is outside the image")
    if any(length < 0 for length in run_lengths):
        raise RequestError("mask run lengths must not be negative")
    area = (x2 - x1) * (y2 - y1)
    if len(run_lengths) > area + 1 or sum(run_lengths) != area:
        raise RequestError("mask run lengths do not match its bounding box")

    mask = np.repeat((np.arange(len(run_lengths)) & 1) != 0, run_lengths)
    result = np.zeros((height, width), dtype=bool)
    result[y1:y2, x1:x2] = mask.reshape((y2 - y1, x2 - x1))
    return result


def _encode_mask(mask):
    nonzero_y, nonzero_x = mask.nonzero()
    if nonzero_x.size == 0:
        x1 = y1 = 0
        x2 = y2 = 1
    else:
        x1, y1 = (np.min(axis).item() for axis in (nonzero_x, nonzero_y))
        x2, y2 = (np.max(axis).item() + 1 for axis in (nonzero_x, nonzero_y))

    flat = mask[y1:y2, x1:x2].ravel()
    (run_indices,) = np.diff(
        flat, prepend=[not flat[0]], append=[not flat[-1]]
    ).nonzero()
    if flat[0]:
        run_lengths = np.diff(run_indices, prepend=[0])
    else:
        run_lengths = np.diff(run_indices)
    return run_lengths.tolist() + [x1, y1, x2 - 1, y2 - 1]


def _validate_shape(shape, width, height):
    if not isinstance(shape, dict):
        raise RequestError("shape must be an object")
    shape_type = shape.get("type")
    points = shape.get("points")
    if shape_type not in {"mask", "polygon"}:
        raise RequestError("shape type must be mask or polygon")
    if not isinstance(points, list):
        raise RequestError("shape points must be an array")

    if shape_type == "mask":
        _decode_mask(points, width, height)
    else:
        if len(points) < 6 or len(points) % 2:
            raise RequestError(
                "polygon points must contain at least three coordinate pairs"
            )
        if any(not _is_number(value) for value in points):
            raise RequestError("polygon coordinates must be finite numbers")
        if any(
            x < 0 or x > width or y < 0 or y > height
            for x, y in zip(points[::2], points[1::2])
        ):
            raise RequestError("polygon coordinates are outside the image")

    return {"type": shape_type, "points": points}


def _validate_token(token):
    if not isinstance(token, str) or len(token) != 32:
        raise RequestError("invalid tracker state")
    try:
        int(token, 16)
    except ValueError as error:
        raise RequestError("invalid tracker state") from error
    return token


def _validate_prompt_points(points, field, width, height):
    if points is None:
        return []
    if not isinstance(points, list):
        raise RequestError(f"{field} must be an array")
    result = []
    for point in points:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(not _is_number(value) for value in point)
        ):
            raise RequestError(f"{field} must contain coordinate pairs")
        if not (0 <= point[0] <= width and 0 <= point[1] <= height):
            raise RequestError(f"{field} contains a point outside the image")
        result.append(point)
    return result


def _validate_prompt_box(boxes, width, height):
    if boxes in (None, []):
        return None
    if (
        not isinstance(boxes, list)
        or len(boxes) != 2
        or any(
            not isinstance(point, list)
            or len(point) != 2
            or any(not _is_number(value) for value in point)
            for point in boxes
        )
    ):
        raise RequestError("obj_bbox must contain one box")
    box = boxes[0] + boxes[1]
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise RequestError("obj_bbox is outside the image")
    return np.asarray(box, dtype=np.float32)


def _move_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_cpu(item) for item in value)
    return value


def _cuda_timing_events(device):
    if torch.device(device).type != "cuda":
        return None
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _collect_cuda_timing(timing, key, event_groups):
    event_groups = [events for events in event_groups if events]
    if event_groups and all(events[1].query() for events in event_groups):
        timing[key] = round(
            sum(events[0].elapsed_time(events[1]) for events in event_groups), 3
        )


def _elapsed_ms(started_at):
    return round((time.perf_counter() - started_at) * 1000, 3)


class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.set_autocast_enabled(True)
            torch.set_autocast_gpu_dtype(torch.bfloat16)
            if torch.cuda.get_device_properties(self.device).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self.predictor = SAM2VideoPredictor.from_pretrained(
            MODEL_ID,
            device=self.device,
            vos_optimized=self.device.type == "cuda",
        )
        if self.device.type == "cuda":
            # SAM2ImagePredictor lacks the VOS path's clone between compiled modules.
            self.predictor.sam_prompt_encoder.register_forward_hook(
                lambda _module, _inputs, outputs: tuple(
                    output.clone() for output in outputs
                )
            )
        self.image_predictor = SAM2ImagePredictor(self.predictor)
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (self.predictor.image_size, self.predictor.image_size)
                ),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    @torch.inference_mode()
    def preprocess_image(self, image, timing, timing_events=None):
        started_at = time.perf_counter()
        image = image.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)
        timing["preprocess_cpu_ms"] = round(
            timing.get("preprocess_cpu_ms", 0) + _elapsed_ms(started_at), 3
        )
        if timing_events:
            timing_events[0].record()
        image_tensor = image_tensor.to(device=self.device)
        backbone_out = self.predictor.forward_image(image_tensor)
        if timing_events:
            timing_events[1].record()
        vision_feats = backbone_out["backbone_fpn"][
            -self.predictor.num_feature_levels :
        ]
        vision_pos_embeds = backbone_out["vision_pos_enc"][
            -self.predictor.num_feature_levels :
        ]
        return PreprocessedImage(
            width=image.width,
            height=image.height,
            vision_feats=[
                feature.flatten(2).permute(2, 0, 1) for feature in vision_feats
            ],
            vision_pos_embeds=[
                embedding.flatten(2).permute(2, 0, 1) for embedding in vision_pos_embeds
            ],
            feat_sizes=[
                (embedding.shape[-2], embedding.shape[-1])
                for embedding in vision_pos_embeds
            ],
        )

    @torch.inference_mode()
    def interact(self, image, pos_points, neg_points, box):
        point_coords = np.asarray(pos_points + neg_points, dtype=np.float32)
        point_labels = np.asarray(
            [1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32
        )
        if not len(point_coords) and box is None:
            return {"shapes": []}

        self.image_predictor.set_image(image)
        masks, scores, _logits = self.image_predictor.predict(
            point_coords=point_coords if len(point_coords) else None,
            point_labels=point_labels if len(point_labels) else None,
            box=box,
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return {
            "shapes": [
                {
                    "type": "mask",
                    "points": _encode_mask(masks[best].astype(bool)),
                    "attributes": [{"spec_id": 0, "value": str(float(scores[best]))}],
                }
            ]
        }

    def _call_predictor(self, *, image, frame_idx, **kwargs):
        output = self.predictor.track_step(
            current_vision_feats=image.vision_feats,
            current_vision_pos_embeds=image.vision_pos_embeds,
            feat_sizes=image.feat_sizes,
            point_inputs=None,
            frame_idx=frame_idx,
            num_frames=frame_idx + 1,
            **kwargs,
        )
        return {
            "maskmem_features": output["maskmem_features"],
            "maskmem_pos_enc": output["maskmem_pos_enc"][-1:],
            "pred_masks": fill_holes_in_mask_scores(
                output["pred_masks"], self.predictor.fill_hole_area
            ),
            "obj_ptr": output["obj_ptr"],
        }

    @torch.inference_mode()
    def init_state(self, image, shape, timing_events=None):
        mask = (
            _decode_mask(shape["points"], image.width, image.height)
            if shape["type"] == "mask"
            else self._polygon_to_mask(shape["points"], image.width, image.height)
        )
        resized_mask = torch.nn.functional.interpolate(
            torch.from_numpy(mask).float()[None, None],
            (self.predictor.image_size, self.predictor.image_size),
            mode="bilinear",
            align_corners=False,
        )
        if timing_events:
            timing_events[0].record()
        resized_mask = (resized_mask >= 0.5).float().to(device=self.device)
        output = self._call_predictor(
            image=image,
            frame_idx=0,
            is_init_cond_frame=True,
            mask_inputs=resized_mask,
            output_dict={},
        )
        if timing_events:
            timing_events[1].record()
        return TrackingState(
            frame_idx=0,
            predictor_outputs={
                "cond_frame_outputs": {0: output},
                "non_cond_frame_outputs": collections.OrderedDict(),
            },
        )

    @staticmethod
    def _polygon_to_mask(points, width, height):
        mask = np.zeros((height, width), dtype=np.uint8)
        contour = np.asarray(points, dtype=np.int32).reshape((-1, 2))
        contour[:, 0] = np.clip(contour[:, 0], 0, width - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [contour], 1)
        return mask.astype(bool)

    @torch.inference_mode()
    def track(self, image, state, shape_type, timing_events=None):
        state.frame_idx += 1
        if timing_events:
            timing_events[0].record()
        output = self._call_predictor(
            image=image,
            frame_idx=state.frame_idx,
            is_init_cond_frame=False,
            mask_inputs=None,
            output_dict=state.predictor_outputs,
        )
        non_cond_outputs = state.predictor_outputs["non_cond_frame_outputs"]
        non_cond_outputs[state.frame_idx] = output
        while len(non_cond_outputs) > self.predictor.num_maskmem:
            non_cond_outputs.popitem(last=False)

        mask = (
            torch.nn.functional.interpolate(
                output["pred_masks"],
                size=(image.height, image.width),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )[0, 0]
            > 0
        )
        if timing_events:
            timing_events[1].record()
        if not mask.any():
            return None
        mask = mask.cpu().numpy()
        if shape_type == "mask":
            return {"type": "mask", "points": _encode_mask(mask)}

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        contour = cv2.approxPolyDP(
            max(contours, key=cv2.contourArea), epsilon=1.0, closed=True
        )
        if contour.shape[0] < 3:
            return None
        return {"type": "polygon", "points": contour.flatten().tolist()}


class StateStore:
    def __init__(self):
        password = os.getenv("CVAT_FUNCTIONS_REDIS_PASSWORD")
        self.redis = redis.Redis(
            host=os.getenv("CVAT_FUNCTIONS_REDIS_HOST", "localhost"),
            port=int(os.getenv("CVAT_FUNCTIONS_REDIS_PORT", "6379")),
            password=password or None,
            socket_connect_timeout=5,
            socket_timeout=15,
        )
        self.redis.ping()

    @staticmethod
    def _key(token):
        return f"{STATE_KEY_PREFIX}{token}"

    @staticmethod
    def _serialize(*, image, shape_type, state):
        payload = {
            "version": 1,
            "width": image.width,
            "height": image.height,
            "shape_type": shape_type,
            "frame_idx": state.frame_idx,
            "predictor_outputs": _move_to_cpu(state.predictor_outputs),
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return buffer.getvalue()

    def create_many(self, states):
        tokens = [uuid.uuid4().hex for _state in states]
        serialized = [
            (
                self._key(token),
                self._serialize(image=image, shape_type=shape_type, state=state),
            )
            for token, (image, shape_type, state) in zip(tokens, states)
        ]
        with self.redis.pipeline(transaction=True) as pipeline:
            for key, value in serialized:
                pipeline.set(key, value, ex=STATE_TTL_SECONDS)
            pipeline.execute()
        return tokens, sum(len(value) for _key, value in serialized)

    def load(self, token, image, device):
        value = self.redis.get(self._key(token))
        if value is None:
            raise RequestError("tracker state not found or expired", status_code=404)
        try:
            payload = torch.load(
                io.BytesIO(value), map_location=device, weights_only=True
            )
        except (pickle.UnpicklingError, RuntimeError, ValueError, EOFError) as error:
            raise RequestError("invalid tracker state") from error
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise RequestError("invalid tracker state")
        width, height = payload.get("width"), payload.get("height")
        shape_type = payload.get("shape_type")
        frame_idx = payload.get("frame_idx")
        outputs = payload.get("predictor_outputs")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or not isinstance(height, int)
            or isinstance(height, bool)
            or width != image.width
            or height != image.height
            or shape_type not in {"mask", "polygon"}
            or not isinstance(frame_idx, int)
            or isinstance(frame_idx, bool)
            or frame_idx < 0
            or not isinstance(outputs, dict)
            or not isinstance(outputs.get("cond_frame_outputs"), dict)
            or not isinstance(outputs.get("non_cond_frame_outputs"), dict)
            or any(
                not isinstance(frame, int) or isinstance(frame, bool) or frame < 0
                for frame in outputs["cond_frame_outputs"]
            )
            or any(
                not isinstance(frame, int) or isinstance(frame, bool) or frame < 0
                for frame in outputs["non_cond_frame_outputs"]
            )
        ):
            raise RequestError("tracker state is incompatible with this image")
        outputs["non_cond_frame_outputs"] = collections.OrderedDict(
            outputs["non_cond_frame_outputs"].items()
        )
        return TrackingState(frame_idx=frame_idx, predictor_outputs=outputs), shape_type

    @contextlib.contextmanager
    def lock(self, tokens):
        locks = []
        try:
            for token in sorted(set(tokens)):
                lock = self.redis.lock(
                    f"{self._key(token)}:lock",
                    timeout=LOCK_TTL_SECONDS,
                    blocking_timeout=LOCK_WAIT_SECONDS,
                )
                if not lock.acquire():
                    raise RequestError("tracker state is busy", status_code=409)
                locks.append(lock)
            yield
        finally:
            for lock in reversed(locks):
                with contextlib.suppress(redis.exceptions.LockError):
                    lock.release()


def _decode_image(encoded):
    if not isinstance(encoded, str):
        raise RequestError("image must be a base64 string")
    try:
        image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
    except (ValueError, UnidentifiedImageError, OSError) as error:
        raise RequestError("image is not valid base64-encoded image data") from error
    if (
        image.width <= 0
        or image.height <= 0
        or image.width * image.height > MAX_IMAGE_PIXELS
    ):
        raise RequestError("image dimensions are invalid")
    try:
        image.load()
    except OSError as error:
        raise RequestError("image is not valid image data") from error
    return image.convert("RGB")


def _decode_images(data):
    if "images" not in data:
        return [_decode_image(data.get("image"))]

    encoded_images = data["images"]
    if (
        not isinstance(encoded_images, list)
        or not encoded_images
        or len(encoded_images) > MAX_BATCH_SIZE
    ):
        raise RequestError(
            f"images must contain between 1 and {MAX_BATCH_SIZE} images"
        )
    images = []
    total_pixels = 0
    for encoded in encoded_images:
        image = _decode_image(encoded)
        total_pixels += image.width * image.height
        if total_pixels > MAX_IMAGE_PIXELS:
            raise RequestError("tracking batch image dimensions are too large")
        images.append(image)
    dimensions = {(image.width, image.height) for image in images}
    if len(dimensions) != 1:
        raise RequestError("all images in a tracking batch must have the same dimensions")
    return images


def _finish_timing(timing, request_started_at, batch_size):
    timing["server_total_ms"] = _elapsed_ms(request_started_at)
    if batch_size > 1:
        for key, value in timing.items():
            if key.endswith("_ms"):
                timing[key] = round(value / batch_size, 3)
    timing["batch_size"] = batch_size


def _response(context, body, status_code=200):
    return context.Response(
        body=json.dumps(body),
        headers={},
        content_type="application/json",
        status_code=status_code,
    )


def init_context(context):
    context.logger.info("Initializing SAM2 tracker")
    context.user_data.model = ModelHandler()
    context.user_data.state_store = StateStore()
    context.logger.info("SAM2 tracker initialized")


def handler(context, event):
    try:
        request_started_at = time.perf_counter()
        timing = {}
        data = event.body
        if not isinstance(data, dict):
            raise RequestError("request body must be an object")
        phase_started_at = time.perf_counter()
        raw_images = _decode_images(data)
        batch_size = len(raw_images)
        image = raw_images[0]
        timing["decode_ms"] = _elapsed_ms(phase_started_at)
        if "pos_points" in data or "neg_points" in data or "obj_bbox" in data:
            if batch_size != 1:
                raise RequestError("interaction requests require exactly one image")
            return _response(
                context,
                context.user_data.model.interact(
                    image,
                    _validate_prompt_points(
                        data.get("pos_points"), "pos_points", image.width, image.height
                    ),
                    _validate_prompt_points(
                        data.get("neg_points"), "neg_points", image.width, image.height
                    ),
                    _validate_prompt_box(
                        data.get("obj_bbox"), image.width, image.height
                    ),
                ),
            )

        shapes = data.get("shapes")
        states = data.get("states")
        if not isinstance(shapes, list) or not isinstance(states, list):
            raise RequestError("shapes and states must be arrays")

        if not states:
            if batch_size != 1:
                raise RequestError("tracking initialization requires exactly one image")
            encoder_events = _cuda_timing_events(context.user_data.model.device)
            image = context.user_data.model.preprocess_image(
                image, timing, encoder_events
            )
            tracker_events = []
            phase_started_at = time.perf_counter()
            results = {
                "shapes": [],
                "device": str(context.user_data.model.device),
                "model_id": MODEL_ID,
            }
            initialized_states = []
            for shape in shapes:
                shape = _validate_shape(shape, image.width, image.height)
                events = _cuda_timing_events(context.user_data.model.device)
                state = context.user_data.model.init_state(image, shape, events)
                tracker_events.append(events)
                results["shapes"].append(shape)
                initialized_states.append((image, shape["type"], state))
            timing["tracker_wall_ms"] = _elapsed_ms(phase_started_at)
            phase_started_at = time.perf_counter()
            (
                results["states"],
                timing["state_bytes"],
            ) = context.user_data.state_store.create_many(initialized_states)
            timing["state_save_ms"] = _elapsed_ms(phase_started_at)
            _collect_cuda_timing(timing, "encoder_gpu_ms", [encoder_events])
            _collect_cuda_timing(timing, "tracker_gpu_ms", tracker_events)
            _finish_timing(timing, request_started_at, batch_size)
            results["timing"] = timing
            return _response(context, results)

        if len(shapes) != len(states) or any(shape is not None for shape in shapes):
            raise RequestError(
                "tracking requests require one null shape for each state"
            )
        tokens = [_validate_token(state) for state in states]
        if len(tokens) != len(set(tokens)):
            raise RequestError("tracker states must be unique")

        results = {
            "shapes": [],
            "device": str(context.user_data.model.device),
            "model_id": MODEL_ID,
        }
        with context.user_data.state_store.lock(tokens):
            updated_states = []
            state_load_ms = 0
            tracker_wall_ms = 0
            tracker_events = []
            loaded_states = []
            for token in tokens:
                phase_started_at = time.perf_counter()
                state, shape_type = context.user_data.state_store.load(
                    token, image, context.user_data.model.device
                )
                state_load_ms += _elapsed_ms(phase_started_at)
                loaded_states.append((state, shape_type))

            encoder_events = []
            frame_results = []
            for raw_image in raw_images:
                events = _cuda_timing_events(context.user_data.model.device)
                image = context.user_data.model.preprocess_image(
                    raw_image, timing, events
                )
                encoder_events.append(events)
                frame_shapes = []
                for state, shape_type in loaded_states:
                    phase_started_at = time.perf_counter()
                    events = _cuda_timing_events(context.user_data.model.device)
                    frame_shapes.append(
                        context.user_data.model.track(
                            image, state, shape_type, events
                        )
                    )
                    tracker_events.append(events)
                    tracker_wall_ms += _elapsed_ms(phase_started_at)
                frame_results.append(frame_shapes)

            results["shapes"] = frame_results[-1]
            if batch_size > 1:
                results["frame_results"] = frame_results
            updated_states.extend(
                (image, shape_type, state) for state, shape_type in loaded_states
            )
            timing["state_load_ms"] = round(state_load_ms, 3)
            timing["tracker_wall_ms"] = round(tracker_wall_ms, 3)
            phase_started_at = time.perf_counter()
            (
                results["states"],
                timing["state_bytes"],
            ) = context.user_data.state_store.create_many(updated_states)
            timing["state_save_ms"] = _elapsed_ms(phase_started_at)
        _collect_cuda_timing(timing, "encoder_gpu_ms", encoder_events)
        _collect_cuda_timing(timing, "tracker_gpu_ms", tracker_events)
        _finish_timing(timing, request_started_at, batch_size)
        results["timing"] = timing
        return _response(context, results)
    except RequestError as error:
        context.logger.error("SAM2 tracker request rejected: %s", error)
        return _response(context, {"error": str(error)}, error.status_code)
    except redis.RedisError:
        context.logger.error("SAM2 tracker Redis operation failed")
        return _response(
            context, {"error": "tracker state storage is unavailable"}, 503
        )
