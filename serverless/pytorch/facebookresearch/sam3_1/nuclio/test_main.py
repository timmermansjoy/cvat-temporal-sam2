# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import base64
import contextlib
import copy
import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import main
import numpy as np
import torch
from PIL import Image


class MultiplexState:
    def __init__(self, assignments, object_ids, allowed_bucket_capacity, **_kwargs):
        self.assignments = assignments
        self.object_ids = object_ids
        self.allowed_bucket_capacity = allowed_bucket_capacity


class MemoryRedis:
    def __init__(self):
        self.values = {}
        self.pending = []

    def pipeline(self, transaction):
        assert transaction
        return self

    def __enter__(self):
        self.pending = []
        return self

    def __exit__(self, *_args):
        self.pending = []

    def set(self, key, value, ex):
        assert ex == main.STATE_TTL_SECONDS
        self.pending.append((key, value))

    def execute(self):
        self.values.update(self.pending)

    def get(self, key):
        return self.values.get(key)


class Model:
    device = "cpu"

    def __init__(self):
        self.calls = []
        self.fail_on_frame = None

    def initialize(self, _image, masks):
        object_ids = list(range(len(masks)))
        return {
            "obj_ids": object_ids,
            "masks": torch.from_numpy(np.stack(masks)),
            "multiplex_state": MultiplexState([object_ids], object_ids, 16),
            "images": {0: torch.zeros(3, 16, 16)},
            "cached_features": {0: torch.zeros(100)},
        }

    def track(self, state, _image, frame):
        self.calls.append((list(state["obj_ids"]), frame))
        state["last_frame"] = frame
        if frame == self.fail_on_frame:
            raise RuntimeError("inference failed")
        return {
            object_id: state["masks"][object_id].numpy()
            for object_id in state["obj_ids"]
        }


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.store = main.StateStore.__new__(main.StateStore)
        self.store.redis = MemoryRedis()
        self.context = SimpleNamespace(
            user_data=SimpleNamespace(model=self.model, state_store=self.store),
            logger=mock.Mock(),
            Response=lambda **kwargs: SimpleNamespace(
                **{**kwargs, "body": json.loads(kwargs["body"])}
            ),
        )
        patcher = mock.patch.dict(
            "sys.modules",
            {
                "sam3.model.multiplex_utils": SimpleNamespace(
                    MultiplexState=MultiplexState
                ),
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.image = self.encode_image(16, 16)
        self.mask = np.zeros((16, 16), dtype=bool)
        self.mask[2:6, 2:6] = True
        self.shapes = [
            main.mask_to_shape(self.mask, "mask"),
            {"type": "polygon", "points": [9, 9, 13, 9, 13, 13, 9, 13]},
        ]

    @staticmethod
    def encode_image(width, height):
        buffer = io.BytesIO()
        Image.new("RGB", (width, height)).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def request(self, **data):
        return main.handler(self.context, SimpleNamespace(body=data))

    def initialize(self, shapes=None):
        response = self.request(
            image=self.image, shapes=shapes or self.shapes, states=[]
        )
        self.assertEqual(response.status_code, 200)
        return response.body["states"]

    def test_joint_batches_preserve_order_and_original_state_for_retries(self):
        refs = self.initialize()
        self.assertEqual(refs[0]["session"], refs[1]["session"])
        self.assertNotEqual(refs[0]["object_id"], refs[1]["object_id"])
        original_values = copy.deepcopy(self.store.redis.values)
        response = self.request(
            images=[self.image] * 3, shapes=[None, None], states=refs[::-1]
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.model.calls, [([0, 1], 1), ([0, 1], 2), ([0, 1], 3)])
        self.assertEqual(len(response.body["frame_results"]), 3)
        self.assertEqual(
            [shape["type"] for shape in response.body["shapes"]], ["polygon", "mask"]
        )
        self.assertNotEqual(response.body["states"][0]["session"], refs[0]["session"])
        for key, value in original_values.items():
            self.assertEqual(self.store.redis.values[key], value)
        retried = self.request(image=self.image, shapes=[None, None], states=refs)
        self.assertEqual(retried.status_code, 200)
        self.assertEqual(
            self.model.calls[-1][1], 1, "A retry must use the original temporal state"
        )

    def test_individual_corrections_can_join_unchanged_object_states(self):
        refs = self.initialize()
        corrected = self.initialize([self.shapes[0]])
        result = self.request(
            image=self.image, shapes=[None, None], states=[refs[1], corrected[0]]
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(
            [shape["type"] for shape in result.body["shapes"]], ["polygon", "mask"]
        )
        self.assertEqual(
            len(self.model.calls), 2, "Each independent seed group runs jointly"
        )

    def test_failure_does_not_publish_partial_states(self):
        refs = self.initialize()
        original = copy.deepcopy(self.store.redis.values)
        self.model.fail_on_frame = 2
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.request(images=[self.image] * 3, shapes=[None, None], states=refs)
        self.assertEqual(self.store.redis.values, original)

    def test_invalid_inputs_and_expired_states_do_not_run_inference(self):
        refs = self.initialize()
        invalid = [
            {"states": refs * 2, "shapes": [None] * 4},
            {"states": [{"session": "../unsafe", "object_id": 0}], "shapes": [None]},
            {"states": [{**refs[0], "object_id": True}], "shapes": [None]},
            {"states": [{**refs[0], "object_id": 10}], "shapes": [None]},
            {"states": refs, "shapes": [None]},
            {"states": refs, "shapes": [None, None], "images": [self.image] * 11},
            {
                "states": refs,
                "shapes": [None, None],
                "images": [self.image, self.encode_image(8, 8)],
            },
            {"states": refs, "shapes": [None, None], "image": self.encode_image(8, 8)},
            {"states": [], "shapes": [{"type": "mask", "points": [100, 0, 0, 1, 1]}]},
            {
                "states": [],
                "shapes": [{"type": "polygon", "points": [float("nan")] * 6}],
            },
        ]
        for data in invalid:
            with self.subTest(data=data):
                self.assertEqual(
                    self.request(**{"image": self.image, **data}).status_code, 400
                )
        self.store.redis.values.clear()
        self.assertEqual(
            self.request(
                image=self.image, states=refs, shapes=[None, None]
            ).status_code,
            404,
        )
        self.assertEqual(self.model.calls, [])

    def test_state_serialization_excludes_images_and_feature_cache(self):
        refs = self.initialize()
        payload = self.store.load(refs[0]["session"], Image.new("RGB", (16, 16)), "cpu")
        self.assertNotIn("images", payload["state"])
        self.assertEqual(payload["state"]["cached_features"], {})
        self.assertIsInstance(payload["state"]["multiplex_state"], MultiplexState)
        self.assertEqual(payload["state"]["multiplex_state"].object_ids, [0, 1])

    def test_mask_round_trip_preserves_holes_and_disconnected_regions(self):
        mask = self.mask.copy()
        mask[3, 3] = False
        mask[10:12, 10:12] = True
        encoded = main.mask_to_shape(mask, "mask")
        np.testing.assert_array_equal(main.shape_to_mask(encoded, 16, 16), mask)
        self.assertIsNone(main.mask_to_shape(np.zeros_like(mask), "mask"))

    def test_full_checkpoint_is_checked_and_tracker_shares_detector_backbone(self):
        predictor = mock.MagicMock()
        builder = mock.Mock(return_value=predictor)
        weights = {"detector.language.weight": torch.zeros(1)}
        with (
            mock.patch.dict(
                "sys.modules",
                {
                    "sam3.model_builder": SimpleNamespace(
                        build_sam3_multiplex_video_predictor=builder,
                    )
                },
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True),
            mock.patch.object(torch, "load", return_value={"model": weights}),
        ):
            adapter = main.ModelHandler()
        predictor.model.load_state_dict.assert_called_once_with(weights, strict=True)
        self.assertIs(adapter.predictor, predictor.model.tracker.model)
        self.assertIs(adapter.predictor.backbone, predictor.model.detector.backbone)
        self.assertEqual(builder.call_args.kwargs["max_num_objects"], main.MAX_OBJECTS)
        self.assertFalse(builder.call_args.kwargs["use_rope_real"])
        predictor.bf16_context.__exit__.assert_called_once()
        predictor.model.tracker.bf16_context.__exit__.assert_called_once()

    def test_text_points_and_box_interaction_preserves_other_masks(self):
        adapter = main.ModelHandler.__new__(main.ModelHandler)
        other = np.roll(self.mask, 7, axis=1)
        detected = {
            "out_obj_ids": np.array([4, 9]),
            "out_probs": np.array([0.75, 0.9]),
            "out_binary_masks": np.stack([self.mask, other]),
        }
        refined = {
            **detected,
            "out_binary_masks": np.stack([np.roll(self.mask, 1, axis=0), other]),
        }
        adapter.prompt_model = SimpleNamespace(
            init_state=mock.Mock(return_value={}),
            add_prompt=mock.Mock(side_effect=[(0, detected), (0, refined)]),
        )
        self.context.user_data.model = adapter
        with mock.patch.object(
            torch,
            "autocast",
            side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
        ):
            result = self.request(
                image=self.image,
                text_prompts=["  red car  "],
                pos_points=[[3, 3]],
                neg_points=[[1, 1]],
                obj_bbox=[[0, 0], [8, 8]],
            )
            self.assertEqual(result.status_code, 200)
            self.assertEqual(len(result.body["shapes"]), 2)
            np.testing.assert_array_equal(
                main.shape_to_mask(result.body["shapes"][1], 16, 16), other
            )
            self.assertEqual(
                result.body["shapes"][1]["attributes"], [{"spec_id": 0, "value": "0.9"}]
            )
            text_call, point_call = adapter.prompt_model.add_prompt.call_args_list
            self.assertEqual(text_call.kwargs["text_str"], "red car")
            self.assertEqual(text_call.kwargs["boxes_xywh"], [[0, 0, 0.5, 0.5]])
            self.assertEqual(point_call.kwargs["obj_id"], 4)
            self.assertTrue(point_call.kwargs["rel_coordinates"])
            torch.testing.assert_close(
                point_call.kwargs["points"],
                torch.tensor([[3 / 16, 3 / 16], [1 / 16, 1 / 16]]),
            )
            self.assertEqual(point_call.kwargs["point_labels"].tolist(), [1, 0])

            adapter.prompt_model.add_prompt.reset_mock(side_effect=True)
            adapter.prompt_model.add_prompt.return_value = (0, detected)
            self.assertEqual(
                self.request(image=self.image, text_prompts=["car"]).status_code, 200
            )
            self.assertEqual(
                adapter.prompt_model.add_prompt.call_count,
                1,
                "Text alone must find every mask",
            )
            adapter.prompt_model.add_prompt.reset_mock()
            self.assertEqual(
                self.request(
                    image=self.image, pos_points=[[4, 4]], neg_points=[]
                ).status_code,
                200,
            )
            self.assertNotIn(
                "text_str", adapter.prompt_model.add_prompt.call_args.kwargs
            )
            self.assertEqual(
                adapter.prompt_model.add_prompt.call_args.kwargs["obj_id"], 1
            )
            adapter.prompt_model.add_prompt.reset_mock()
            self.assertEqual(
                self.request(image=self.image, pos_points=[], neg_points=[]).body,
                {"shapes": []},
            )
            adapter.prompt_model.add_prompt.assert_not_called()
        self.assertEqual(
            self.store.redis.values, {}, "Preview must not advance tracking sessions"
        )

    def test_invalid_interaction_prompts_never_reach_model(self):
        self.model.interact = mock.Mock()
        for data in [
            {"text_prompts": "car"},
            {"text_prompts": ["   "]},
            {"text_prompts": ["a", "b"]},
            {"text_prompts": ["x" * 257]},
            {"pos_points": [[float("nan"), 1]]},
            {"pos_points": [[True, 2]]},
            {"pos_points": [[17, 0]]},
            {"pos_points": [[0, 0]] * 257},
            {"pos_points": {}},
            {"pos_points": None},
            {"neg_points": [[1, 1]]},
            {"obj_bbox": [[8, 8], [1, 1]]},
            {"obj_bbox": [[1, 1]]},
            {"text_prompts": ["car"], "images": [self.image, self.image]},
        ]:
            with self.subTest(data=data):
                self.assertEqual(
                    self.request(**{"image": self.image, **data}).status_code, 400
                )
        self.model.interact.assert_not_called()

    def test_model_adapter_seeds_all_masks_together_and_bounds_temporal_memory(self):
        adapter = main.ModelHandler.__new__(main.ModelHandler)
        state = {
            "cached_features": {},
            "frames_already_tracked": {},
            "output_dict": {
                "cond_frame_outputs": {0: {}},
                "non_cond_frame_outputs": {},
            },
            "output_dict_per_obj": {
                0: {"cond_frame_outputs": {0: {}}, "non_cond_frame_outputs": {}}
            },
        }

        def propagate(
            current, start_frame_idx, max_frame_num_to_track, reverse, tqdm_disable
        ):
            self.assertEqual(max_frame_num_to_track, 0, "Request exactly one new frame")
            self.assertFalse(reverse)
            self.assertTrue(tqdm_disable)
            self.assertEqual(current["num_frames"], start_frame_idx + 1)
            current["output_dict"]["non_cond_frame_outputs"][start_frame_idx] = {}
            current["output_dict_per_obj"][0]["non_cond_frame_outputs"][
                start_frame_idx
            ] = {}
            current["frames_already_tracked"][start_frame_idx] = {}
            yield start_frame_idx, [0, 1], None, torch.ones(2, 1, 16, 16), None

        adapter.predictor = SimpleNamespace(
            image_size=16,
            num_maskmem=2,
            max_obj_ptrs_in_encoder=3,
            init_state=mock.Mock(return_value=state),
            add_new_masks=mock.Mock(),
            propagate_in_video_preflight=mock.Mock(),
            propagate_in_video=propagate,
        )
        with mock.patch.object(
            torch,
            "autocast",
            side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
        ):
            initialized = adapter.initialize(
                Image.new("RGB", (16, 16)), [self.mask, self.mask]
            )
            seeded = adapter.predictor.add_new_masks.call_args.kwargs
            self.assertEqual(seeded["obj_ids"], [0, 1])
            self.assertEqual(tuple(seeded["masks"].shape), (2, 16, 16))
            for frame in range(1, 6):
                result = adapter.track(initialized, Image.new("RGB", (16, 16)), frame)
                self.assertEqual(set(result), {0, 1})
        self.assertEqual(set(state["output_dict"]["non_cond_frame_outputs"]), {3, 4, 5})
        self.assertEqual(
            set(state["output_dict_per_obj"][0]["non_cond_frame_outputs"]), {3, 4, 5}
        )
        self.assertEqual(set(state["output_dict"]["cond_frame_outputs"]), {0})
        self.assertEqual(state["cached_features"], {})
        self.assertNotIn("images", state)


if __name__ == "__main__":
    unittest.main()
