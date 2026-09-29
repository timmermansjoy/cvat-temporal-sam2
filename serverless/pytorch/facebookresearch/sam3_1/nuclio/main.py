# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""CVAT text/point interaction and batched tracking with SAM 3.1 Object Multiplex."""

import base64
import io
import json
import math
import os
import pickle
import time
import uuid
from collections import defaultdict

import cv2
import numpy as np
import redis
import torch
from PIL import Image, UnidentifiedImageError

MODEL_ID = "facebook/sam3.1"
MAX_BATCH_SIZE = 10
MAX_OBJECTS = 64
MAX_IMAGE_PIXELS = 100_000_000
STATE_TTL_SECONDS = 8 * 60 * 60
STATE_KEY_PREFIX = "cvat:sam3.1:state:"


class RequestError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def decode_images(data):
    encoded_images = data.get("images", [data.get("image")])
    if (
        not isinstance(encoded_images, list)
        or not 1 <= len(encoded_images) <= MAX_BATCH_SIZE
    ):
        raise RequestError(f"Provide between 1 and {MAX_BATCH_SIZE} images")
    images = []
    total_pixels = 0
    for encoded in encoded_images:
        if not isinstance(encoded, str):
            raise RequestError("image must be a base64 string")
        try:
            image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
            total_pixels += image.width * image.height
            if total_pixels > MAX_IMAGE_PIXELS:
                raise RequestError("tracking batch image dimensions are too large")
            images.append(image.convert("RGB"))
        except (
            ValueError,
            OSError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
        ) as error:
            raise RequestError("invalid base64-encoded image") from error
    if len({image.size for image in images}) != 1:
        raise RequestError("all images must have the same dimensions")
    return images


def shape_to_mask(shape, width, height):
    if not isinstance(shape, dict) or shape.get("type") not in {"mask", "polygon"}:
        raise RequestError("shape type must be mask or polygon")
    points = shape.get("points")
    if not isinstance(points, list) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in points
    ):
        raise RequestError("shape points must be finite numbers")
    mask = np.zeros((height, width), dtype=np.uint8)
    if shape["type"] == "polygon":
        if (
            len(points) < 6
            or len(points) % 2
            or any(
                not (0 <= x <= width and 0 <= y <= height)
                for x, y in zip(points[::2], points[1::2])
            )
        ):
            raise RequestError("invalid polygon coordinates")
        contour = np.asarray(points, dtype=np.int32).reshape(-1, 2)
        contour[:, 0] = contour[:, 0].clip(0, width - 1)
        contour[:, 1] = contour[:, 1].clip(0, height - 1)
        cv2.fillPoly(mask, [contour], 1)
    else:
        if len(points) < 5 or any(int(value) != value for value in points):
            raise RequestError(
                "mask points must be integer run lengths and a bounding box"
            )
        *runs, x1, y1, x2, y2 = map(int, points)
        if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
            raise RequestError("mask bounding box is outside the image")
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if len(runs) > area + 1 or any(run < 0 for run in runs) or sum(runs) != area:
            raise RequestError("mask run lengths do not match the bounding box")
        mask[y1 : y2 + 1, x1 : x2 + 1] = np.repeat(
            np.arange(len(runs)) % 2, runs
        ).reshape(y2 - y1 + 1, x2 - x1 + 1)
    if not mask.any():
        raise RequestError("tracking requires a non-empty seed mask")
    return mask.astype(bool)


def mask_to_shape(mask, shape_type):
    if not mask.any():
        return None
    if shape_type == "polygon":
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True)
        return (
            {"type": "polygon", "points": contour.flatten().tolist()}
            if len(contour) >= 3
            else None
        )
    ys, xs = mask.nonzero()
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    flat = mask[y1 : y2 + 1, x1 : x2 + 1].ravel()
    indices = np.flatnonzero(np.diff(flat, prepend=not flat[0], append=not flat[-1]))
    runs = np.diff(indices, prepend=0) if flat[0] else np.diff(indices)
    return {"type": "mask", "points": runs.tolist() + [x1, y1, x2, y2]}


def validate_state_ref(ref):
    if not isinstance(ref, dict) or set(ref) != {"session", "object_id"}:
        raise RequestError("invalid SAM3.1 tracker state")
    token, object_id = ref["session"], ref["object_id"]
    if (
        not isinstance(token, str)
        or len(token) != 32
        or any(character not in "0123456789abcdef" for character in token)
        or type(object_id) is not int
        or not 0 <= object_id < MAX_OBJECTS
    ):
        raise RequestError("invalid SAM3.1 tracker state")
    return token, object_id


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(item) for item in value)
    return value


def validate_prompts(data, image):
    texts = data.get("text_prompts", [])
    if (
        not isinstance(texts, list)
        or len(texts) > 1
        or any(
            not isinstance(text, str) or not 1 <= len(text.strip()) <= 256
            for text in texts
        )
    ):
        raise RequestError(
            "text_prompts must contain one non-empty phrase of at most 256 characters"
        )
    prompts = []
    for field in ("pos_points", "neg_points", "obj_bbox"):
        points = data.get(field, [])
        if points is None and field == "obj_bbox":
            points = []
        if not isinstance(points, list) or len(points) > 256:
            raise RequestError(f"{field} must be an array of at most 256 points")
        for point in points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(type(value) not in (int, float) for value in point)
                or not (0 <= point[0] <= image.width and 0 <= point[1] <= image.height)
            ):
                raise RequestError(f"invalid {field} coordinates")
        prompts.append(points)
    positive, negative, box = prompts
    if box and (len(box) != 2 or box[0][0] >= box[1][0] or box[0][1] >= box[1][1]):
        raise RequestError(
            "obj_bbox must contain the top-left and bottom-right corners"
        )
    if negative and not positive:
        raise RequestError("Add a positive point on the object before negative points")
    return texts[0].strip() if texts else None, positive, negative, box


class ModelHandler:
    def __init__(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("SAM3.1 requires an NVIDIA CUDA GPU with BF16 support")
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        self.device = torch.device("cuda")
        checkpoint_path = os.environ.get(
            "SAM31_CHECKPOINT", "/opt/nuclio/models/sam3.1_multiplex.pt"
        )
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path,
            max_num_objects=MAX_OBJECTS,
            use_fa3=False,
            # The checkpoint stores complex RoPE; real-valued compile buffers are generated, not trained.
            use_rope_real=False,
            compile=False,
        )
        # The upstream wrapper enters a persistent autocast context. Use scoped contexts below.
        predictor.bf16_context.__exit__(None, None, None)
        predictor.model.tracker.bf16_context.__exit__(None, None, None)
        self.prompt_model = predictor.model
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Fail on missing weights rather than serving partially initialized predictions.
        self.prompt_model.load_state_dict(
            checkpoint.get("model", checkpoint), strict=True
        )
        del checkpoint
        # Reuse the same tracker and vision encoder for CVAT's existing mask propagation protocol.
        self.predictor = self.prompt_model.tracker.model
        self.predictor.backbone = self.prompt_model.detector.backbone
        self.predictor.non_overlap_masks_for_output = False
        self.prompt_model.score_threshold_detection = 0.2
        self.prompt_model.new_det_thresh = 0.2
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @torch.inference_mode()
    def interact(self, image, text, positive, negative, box):
        if not text and not positive and not box:
            return {"shapes": []}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # ponytail: recompute one frame per edit; cache image features if click latency warrants it.
            state = self.prompt_model.init_state(
                resource_path=[image], offload_video_to_cpu=True
            )
            output = None
            if text or box:
                boxes = None
                if box:
                    (x1, y1), (x2, y2) = box
                    boxes = [
                        [
                            x1 / image.width,
                            y1 / image.height,
                            (x2 - x1) / image.width,
                            (y2 - y1) / image.height,
                        ]
                    ]
                _, output = self.prompt_model.add_prompt(
                    state,
                    frame_idx=0,
                    text_str=text,
                    boxes_xywh=boxes,
                    box_labels=[1] if boxes else None,
                )
            if positive:
                object_id = 1
                if output is not None:
                    ids = output["out_obj_ids"]
                    object_id = int(max(ids, default=0)) + 1
                    x, y = positive[0]
                    candidates = np.flatnonzero(
                        output["out_binary_masks"][
                            :,
                            min(int(y), image.height - 1),
                            min(int(x), image.width - 1),
                        ]
                    )
                    if len(candidates):
                        best = candidates[np.argmax(output["out_probs"][candidates])]
                        object_id = int(ids[best])
                    elif len(ids) >= MAX_OBJECTS:
                        raise RequestError(
                            "Object limit reached; accept these masks before adding another object"
                        )
                _, output = self.prompt_model.add_prompt(
                    state,
                    frame_idx=0,
                    obj_id=object_id,
                    points=torch.tensor(
                        [
                            [x / image.width, y / image.height]
                            for x, y in positive + negative
                        ],
                        dtype=torch.float32,
                    ),
                    point_labels=torch.tensor(
                        [1] * len(positive) + [0] * len(negative), dtype=torch.int32
                    ),
                    rel_coordinates=True,
                )
            if output is None:
                raise RuntimeError("SAM3.1 did not return interaction results")
            shapes = []
            for mask, score in zip(output["out_binary_masks"], output["out_probs"]):
                shape = mask_to_shape(mask, "mask")
                if shape is not None:
                    shape["attributes"] = [{"spec_id": 0, "value": str(float(score))}]
                    shapes.append(shape)
            return {"shapes": shapes}

    def prepare_image(self, image):
        resized = image.resize(
            (self.predictor.image_size, self.predictor.image_size),
            Image.Resampling.BILINEAR,
        )
        return (
            torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float()
            / 127.5
            - 1
        )

    @torch.inference_mode()
    def initialize(self, image, masks):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.predictor.init_state(
                video_height=image.height,
                video_width=image.width,
                num_frames=1,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
            state["images"] = {0: self.prepare_image(image)}
            self.predictor.add_new_masks(
                state,
                frame_idx=0,
                obj_ids=list(range(len(masks))),
                masks=torch.from_numpy(np.stack(masks)),
            )
            self.predictor.propagate_in_video_preflight(state)
        return state

    @torch.inference_mode()
    def track(self, state, image, frame):
        state["images"] = {frame: self.prepare_image(image)}
        state["num_frames"] = frame + 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = list(
                self.predictor.propagate_in_video(
                    state,
                    start_frame_idx=frame,
                    max_frame_num_to_track=0,
                    reverse=False,
                    tqdm_disable=True,
                )
            )
        if len(outputs) != 1 or outputs[0][0] != frame:
            raise RuntimeError("SAM3.1 returned an unexpected tracking frame")
        _, object_ids, _, masks, _ = outputs[0]
        masks = (masks[:, 0] > 0).cpu().numpy()
        keep_from = (
            frame
            - max(self.predictor.num_maskmem, self.predictor.max_obj_ptrs_in_encoder)
            + 1
        )
        for output in [state["output_dict"], *state["output_dict_per_obj"].values()]:
            for old_frame in list(output["non_cond_frame_outputs"]):
                if old_frame < keep_from:
                    del output["non_cond_frame_outputs"][old_frame]
        state["frames_already_tracked"] = {
            index: info
            for index, info in state["frames_already_tracked"].items()
            if index >= keep_from
        }
        state["cached_features"].clear()
        state.pop("images", None)
        return dict(zip(object_ids, masks))


class StateStore:
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv("CVAT_FUNCTIONS_REDIS_HOST", "localhost"),
            port=int(os.getenv("CVAT_FUNCTIONS_REDIS_PORT", "6379")),
            password=os.getenv("CVAT_FUNCTIONS_REDIS_PASSWORD") or None,
            socket_connect_timeout=5,
            socket_timeout=30,
        )
        self.redis.ping()

    @staticmethod
    def serialize(payload):
        state = dict(payload["state"])
        multiplex = state.pop("multiplex_state")
        state.pop("images", None)
        state["cached_features"] = {}
        encoded = {
            **payload,
            "version": 1,
            "state": to_cpu(state),
            "multiplex": {
                "assignments": multiplex.assignments,
                "object_ids": multiplex.object_ids,
                "allowed_bucket_capacity": multiplex.allowed_bucket_capacity,
            },
        }
        buffer = io.BytesIO()
        torch.save(encoded, buffer)
        return buffer.getvalue()

    def load(self, token, image, device):
        from sam3.model.multiplex_utils import MultiplexState

        value = self.redis.get(STATE_KEY_PREFIX + token)
        if value is None:
            raise RequestError(
                "SAM3.1 state expired; start propagation again from a mask", 404
            )
        try:
            payload = torch.load(
                io.BytesIO(value), map_location="cpu", weights_only=True
            )
            if (
                payload["version"] != 1
                or payload["width"] != image.width
                or payload["height"] != image.height
                or type(payload["frame"]) is not int
                or payload["frame"] < 0
                or not 1 <= len(payload["shape_types"]) <= MAX_OBJECTS
                or any(
                    kind not in {"mask", "polygon"} for kind in payload["shape_types"]
                )
            ):
                raise ValueError("incompatible state")
            state = payload["state"]
            state["device"] = torch.device(device)
            state["storage_device"] = torch.device("cpu")
            state["multiplex_state"] = MultiplexState(
                **payload["multiplex"],
                device=torch.device(device),
                dtype=torch.float32,
            )
            return payload
        except (
            KeyError,
            TypeError,
            ValueError,
            AssertionError,
            RuntimeError,
            EOFError,
            pickle.UnpicklingError,
        ) as error:
            raise RequestError(
                "SAM3.1 tracker state is invalid or incompatible with this image"
            ) from error

    def save_many(self, payloads):
        # Immutable snapshots allow retries and cancellation without advancing old states.
        tokens = [uuid.uuid4().hex for _ in payloads]
        values = [self.serialize(payload) for payload in payloads]
        with self.redis.pipeline(transaction=True) as pipeline:
            for token, value in zip(tokens, values):
                pipeline.set(STATE_KEY_PREFIX + token, value, ex=STATE_TTL_SECONDS)
            pipeline.execute()
        return tokens, sum(map(len, values))


def init_context(context):
    context.logger.info("Initializing SAM3.1 detector and multiplex tracker")
    context.user_data.model = ModelHandler()
    context.user_data.state_store = StateStore()
    context.logger.info("SAM3.1 detector and multiplex tracker initialized")


def response(context, body, status_code=200):
    return context.Response(
        body=json.dumps(body),
        headers={},
        content_type="application/json",
        status_code=status_code,
    )


def handler(context, event):
    started = time.perf_counter()
    try:
        data = event.body
        if not isinstance(data, dict):
            raise RequestError("request body must be an object")
        if any(
            key in data
            for key in ("text_prompts", "pos_points", "neg_points", "obj_bbox")
        ):
            images = decode_images(data)
            if len(images) != 1:
                raise RequestError("interaction requires exactly one image")
            image = images[0]
            prompts = validate_prompts(data, image)
            return response(context, context.user_data.model.interact(image, *prompts))
        shapes, refs = data.get("shapes"), data.get("states")
        if (
            not isinstance(shapes, list)
            or not isinstance(refs, list)
            or not 1 <= len(shapes) <= MAX_OBJECTS
        ):
            raise RequestError(
                f"shapes and states must be arrays with 1–{MAX_OBJECTS} objects"
            )
        images = decode_images(data)
        image = images[0]
        model, store = context.user_data.model, context.user_data.state_store
        timing = {"decode_ms": (time.perf_counter() - started) * 1000}
        phase = time.perf_counter()
        if not refs:
            if len(images) != 1:
                raise RequestError("initialization requires exactly one image")
            masks = [
                shape_to_mask(shape, image.width, image.height) for shape in shapes
            ]
            payloads = [
                {
                    "width": image.width,
                    "height": image.height,
                    "frame": 0,
                    "shape_types": [shape["type"] for shape in shapes],
                    "state": model.initialize(image, masks),
                }
            ]
            frame_results = [shapes]
            positions = [(0, object_id) for object_id in range(len(shapes))]
        else:
            if len(refs) != len(shapes) or any(shape is not None for shape in shapes):
                raise RequestError("continuation requires one null shape per state")
            parsed = [validate_state_ref(ref) for ref in refs]
            if len(set(parsed)) != len(parsed):
                raise RequestError("tracker states must be unique")
            groups = defaultdict(list)
            for position, (token, object_id) in enumerate(parsed):
                groups[token].append((position, object_id))
            payloads = [store.load(token, image, model.device) for token in groups]
            positions = [None] * len(refs)
            for group_index, (members, payload) in enumerate(
                zip(groups.values(), payloads)
            ):
                for position, object_id in members:
                    if object_id not in payload["state"]["obj_ids"] or object_id >= len(
                        payload["shape_types"]
                    ):
                        raise RequestError("object is not present in the tracker state")
                    positions[position] = (group_index, object_id)
            timing["state_load_ms"] = (time.perf_counter() - phase) * 1000
            phase = time.perf_counter()
            frame_results = []
            for image in images:
                masks_by_group = []
                for payload in payloads:
                    payload["frame"] += 1
                    masks_by_group.append(
                        model.track(payload["state"], image, payload["frame"])
                    )
                frame_results.append(
                    [
                        mask_to_shape(
                            masks_by_group[group][object_id],
                            payloads[group]["shape_types"][object_id],
                        )
                        for group, object_id in positions
                    ]
                )
        timing["tracker_wall_ms"] = (time.perf_counter() - phase) * 1000
        phase = time.perf_counter()
        tokens, state_bytes = store.save_many(payloads)
        timing["state_save_ms"] = (time.perf_counter() - phase) * 1000
        timing["server_total_ms"] = (time.perf_counter() - started) * 1000
        timing = {key: round(value / len(images), 3) for key, value in timing.items()}
        return response(
            context,
            {
                "shapes": frame_results[-1],
                "frame_results": frame_results,
                "states": [
                    {"session": tokens[group], "object_id": object_id}
                    for group, object_id in positions
                ],
                "device": str(model.device),
                "model_id": MODEL_ID,
                "timing": {
                    **timing,
                    "state_bytes": state_bytes,
                    "batch_size": len(images),
                },
            },
        )
    except RequestError as error:
        return response(context, {"error": str(error)}, error.status_code)
    except redis.RedisError:
        context.logger.exception("SAM3.1 state storage failed")
        return response(context, {"error": "tracker state storage is unavailable"}, 503)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return response(
            context,
            {
                "error": "SAM3.1 GPU memory exhausted; track fewer objects or use SAM2 Tiny"
            },
            503,
        )
