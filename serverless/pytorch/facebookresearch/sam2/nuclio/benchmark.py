#!/usr/bin/env python3
# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""Benchmark a deployed temporal tracker against prompt or provided reference masks."""

import argparse
import base64
import concurrent.futures
import json
import math
import os
import statistics
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest import mock


def post(endpoint, payload):
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"SAM2 returned {error.code}: {error.read().decode()}") from error
    return body, (time.perf_counter() - started) * 1000


def get(url, token):
    authorization = token if " " in token else f"Token {token}"
    request = urllib.request.Request(url, headers={"Authorization": authorization})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"CVAT returned {error.code}: {error.read().decode()}") from error


def foreground_rows(points):
    runs, x1, y1, x2, _y2 = points[:-4], *points[-4:]
    width = x2 - x1 + 1
    rows = {}
    offset = 0
    foreground = False
    for length in runs:
        remaining = length
        while foreground and remaining:
            row, column = divmod(offset, width)
            span = min(remaining, width - column)
            rows.setdefault(y1 + row, []).append((x1 + column, x1 + column + span))
            offset += span
            remaining -= span
        if not foreground:
            offset += remaining
        foreground = not foreground
    return rows


def mask_iou(left, right):
    if left is None or right is None:
        return 0.0
    left_rows = foreground_rows(left)
    right_rows = foreground_rows(right)
    left_area = sum(stop - start for intervals in left_rows.values() for start, stop in intervals)
    right_area = sum(stop - start for intervals in right_rows.values() for start, stop in intervals)
    intersection = 0
    for row in left_rows.keys() & right_rows.keys():
        left_intervals = left_rows[row]
        right_intervals = right_rows[row]
        left_index = right_index = 0
        while left_index < len(left_intervals) and right_index < len(right_intervals):
            left_start, left_stop = left_intervals[left_index]
            right_start, right_stop = right_intervals[right_index]
            intersection += max(0, min(left_stop, right_stop) - max(left_start, right_start))
            if left_stop <= right_stop:
                left_index += 1
            else:
                right_index += 1
    union = left_area + right_area - intersection
    return intersection / union if union else 1.0


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(direction, init_ms, frame_ms, ious, visible):
    return {
        "direction": direction,
        "init_ms": round(init_ms, 1),
        "mean_frame_ms": round(statistics.mean(frame_ms), 1),
        "p95_frame_ms": round(percentile(frame_ms, 0.95), 1),
        "mean_iou": round(statistics.mean(ious), 4),
        "min_iou": round(min(ious), 4),
        "visible_predictions": visible,
        "predictions": len(ious),
    }


def run_chain(endpoint, images, references, direction):
    if direction == "reverse":
        images = list(reversed(images))
        references = list(reversed(references))

    initialized, init_ms = post(endpoint, {
        "image": images[0],
        "shapes": [{"type": "mask", "points": references[0]}],
        "states": [],
    })
    states = initialized["states"]
    frame_ms = []
    ious = []
    visible = 0
    for image, reference in zip(images[1:], references[1:]):
        tracked, elapsed = post(endpoint, {
            "image": image,
            "shapes": [None],
            "states": states,
        })
        states = tracked["states"]
        prediction = tracked["shapes"][0]
        points = prediction["points"] if prediction else None
        visible += prediction is not None
        frame_ms.append(elapsed)
        ious.append(mask_iou(points, reference))
    return summarize(direction, init_ms, frame_ms, ious, visible)


def parse_point(value):
    try:
        point = [int(coordinate) for coordinate in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("point must be X,Y") from error
    if len(point) != 2:
        raise argparse.ArgumentTypeError("point must be X,Y")
    return point


def parse_box(value):
    try:
        box = [int(coordinate) for coordinate in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("box must be X1,Y1,X2,Y2") from error
    if len(box) != 4:
        raise argparse.ArgumentTypeError("box must be X1,Y1,X2,Y2")
    return [box[:2], box[2:]]


def self_test():
    full = [0, 4, 0, 0, 1, 1]
    half = [0, 2, 2, 0, 0, 1, 1]
    assert mask_iou(full, full) == 1
    assert mask_iou(full, half) == 0.5
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "references.json")
        path.write_text(json.dumps({"one.jpg": full, "two.jpg": half}))
        assert load_references(path, [Path("one.jpg"), Path("two.jpg")]) == [full, half]
    annotations = {
        "tracks": [{
            "id": 7,
            "shapes": [
                {"frame": 2, "type": "mask", "outside": False, "source": "manual", "points": full},
                {"frame": 3, "type": "mask", "outside": False, "source": "semi-auto", "points": half},
                {"frame": 4, "type": "mask", "outside": False, "source": "auto", "points": half},
            ],
        }],
    }
    assert extract_cvat_references(annotations, 7, None, None, None) == ([2, 3], [full, half])
    grouped = {
        "tracks": [
            {
                "id": 8,
                "group": 4,
                "source": "manual",
                "shapes": [{"frame": 2, "type": "mask", "points": full}],
            },
            {
                "id": 9,
                "group": 4,
                "source": "semi-auto",
                "shapes": [{"frame": 3, "type": "mask", "points": half}],
            },
        ],
    }
    assert extract_cvat_references(grouped, None, 4, None, None) == ([2, 3], [full, half])
    with mock.patch(
        f"{__name__}.get",
        side_effect=[json.dumps(annotations).encode(), b"frame two", b"frame three"],
    ) as mocked_get:
        images, references = load_cvat_sequence(
            "https://cvat.example/", "secret", 1, 7, None, 2, 3
        )
        assert images == [
            base64.b64encode(b"frame two").decode(),
            base64.b64encode(b"frame three").decode(),
        ]
        assert references == [full, half]
        assert mocked_get.call_args_list[0] == mock.call(
            "https://cvat.example/api/tasks/1/annotations/",
            "secret",
        )


def load_references(path, images):
    loaded = json.loads(path.read_text())
    if isinstance(loaded, list):
        references = loaded
    elif isinstance(loaded, dict):
        try:
            references = [loaded[image.name] for image in images]
        except KeyError as error:
            raise ValueError(f"missing reference for {error.args[0]}") from error
    else:
        raise ValueError("references must be an array or an object keyed by image filename")

    if len(references) != len(images) or any(
        not isinstance(reference, list) for reference in references
    ):
        raise ValueError("references must contain one RLE mask per image")
    return references


def extract_cvat_references(annotations, track_id, group_id, start, stop):
    if (track_id is None) == (group_id is None):
        raise ValueError("select exactly one CVAT track or linked group")
    tracks = [
        track for track in annotations.get("tracks", [])
        if (
            track.get("id") == track_id if track_id is not None
            else track.get("group") == group_id
        )
    ]
    if not tracks:
        identity = f"track {track_id}" if track_id is not None else f"linked group {group_id}"
        raise ValueError(f"CVAT {identity} was not found")

    accepted_sources = {"manual", "semi-auto", "semi_auto"}
    selected = {}
    for track in tracks:
        for shape in track.get("shapes", []):
            if (
                shape.get("type") == "mask"
                and not shape.get("outside", False)
                and (shape.get("source") or track.get("source") or "manual") in accepted_sources
                and (start is None or shape["frame"] >= start)
                and (stop is None or shape["frame"] <= stop)
            ):
                if shape["frame"] in selected:
                    raise ValueError(f"multiple reference masks exist on frame {shape['frame']}")
                selected[shape["frame"]] = shape
    if len(selected) < 2:
        raise ValueError(
            "the selected CVAT object needs at least two visible manual or corrected mask keyframes"
        )
    frames = sorted(selected)
    if frames != list(range(frames[0], frames[-1] + 1)):
        raise ValueError("CVAT reference masks must cover every frame in the selected range")
    return frames, [[int(value) for value in selected[frame]["points"]] for frame in frames]


def load_cvat_sequence(url, token, task_id, track_id, group_id, start, stop):
    base_url = url.rstrip("/")
    annotations = json.loads(get(f"{base_url}/api/tasks/{task_id}/annotations/", token))
    frames, references = extract_cvat_references(
        annotations, track_id, group_id, start, stop
    )
    images = []
    for frame in frames:
        query = urllib.parse.urlencode({
            "number": frame,
            "type": "frame",
            "quality": "original",
        })
        images.append(base64.b64encode(get(
            f"{base_url}/api/tasks/{task_id}/data/?{query}",
            token,
        )).decode())
    return images, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--endpoint", default="http://localhost:32768")
    parser.add_argument("--positive", required=False, type=parse_point)
    parser.add_argument("--negative", action="append", default=[], type=parse_point)
    parser.add_argument("--box", type=parse_box)
    parser.add_argument(
        "--references-json",
        type=Path,
        help="RLE masks as an array or an object keyed by image filename",
    )
    parser.add_argument("--cvat-url", help="CVAT base URL, for example http://localhost:8081")
    parser.add_argument("--cvat-task", type=int, help="CVAT task ID containing reference masks")
    parser.add_argument("--cvat-track", type=int, help="CVAT track ID to evaluate")
    parser.add_argument("--cvat-group", type=int, help="visible linked object ID to evaluate")
    parser.add_argument("--cvat-start", type=int, help="first reference frame")
    parser.add_argument("--cvat-stop", type=int, help="last reference frame")
    parser.add_argument(
        "--cvat-token-env",
        default="CVAT_TOKEN",
        help="environment variable containing the CVAT API token",
    )
    parser.add_argument("--require-mean-iou", type=float)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    confidences = []
    cvat_options = (args.cvat_url, args.cvat_task, args.cvat_track, args.cvat_group)
    if any(option is not None for option in cvat_options):
        if args.cvat_url is None or args.cvat_task is None:
            parser.error("--cvat-url and --cvat-task must be used together")
        if (args.cvat_track is None) == (args.cvat_group is None):
            parser.error("select exactly one of --cvat-track or --cvat-group")
        if args.images or args.references_json or args.positive:
            parser.error("CVAT reference mode cannot be combined with images or prompt references")
        token = os.environ.get(args.cvat_token_env)
        if not token:
            parser.error(f"{args.cvat_token_env} must contain a CVAT API token")
        images, references = load_cvat_sequence(
            args.cvat_url,
            token,
            args.cvat_task,
            args.cvat_track,
            args.cvat_group,
            args.cvat_start,
            args.cvat_stop,
        )
        identity = (
            f"track_{args.cvat_track}" if args.cvat_track is not None
            else f"group_{args.cvat_group}"
        )
        reference_source = f"cvat_task_{args.cvat_task}_{identity}"
    else:
        if len(args.images) < 2:
            parser.error("at least two images are required")
        if args.references_json is None and args.positive is None:
            parser.error("--positive X,Y is required without --references-json")
        images = [base64.b64encode(path.read_bytes()).decode() for path in args.images]
    if args.references_json:
        references = load_references(args.references_json, args.images)
        reference_source = "provided"
    elif not any(option is not None for option in cvat_options):
        references = []
        reference_source = "fresh_prompt"
        for image in images:
            result, _elapsed = post(args.endpoint, {
                "image": image,
                "pos_points": [args.positive],
                "neg_points": args.negative,
                "obj_bbox": args.box,
            })
            if not result["shapes"]:
                raise RuntimeError("the reference prompt produced no mask")
            references.append(result["shapes"][0]["points"])
            confidences.append(float(result["shapes"][0]["attributes"][0]["value"]))

    sequential_started = time.perf_counter()
    sequential = [
        run_chain(args.endpoint, images, references, "forward"),
        run_chain(args.endpoint, images, references, "reverse"),
    ]
    sequential_ms = (time.perf_counter() - sequential_started) * 1000

    concurrent_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        concurrent_results = list(executor.map(
            lambda direction: run_chain(args.endpoint, images, references, direction),
            ("forward", "reverse"),
        ))
    concurrent_ms = (time.perf_counter() - concurrent_started) * 1000

    report = {
        "images": len(images),
        "reference_source": reference_source,
        "reference_confidence": {
            "mean": round(statistics.mean(confidences), 4),
            "min": round(min(confidences), 4),
        } if confidences else None,
        "sequential": sequential,
        "concurrent": concurrent_results,
        "sequential_wall_ms": round(sequential_ms, 1),
        "concurrent_wall_ms": round(concurrent_ms, 1),
        "concurrency_speedup": round(sequential_ms / concurrent_ms, 3),
    }
    print(json.dumps(report, indent=2))

    if args.require_mean_iou is not None and any(
        result["mean_iou"] < args.require_mean_iou for result in sequential
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
