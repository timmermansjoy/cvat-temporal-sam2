# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import contextlib
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import main
import redis


class _Response:
    def __init__(self, *, body, status_code, **_kwargs):
        self.body = json.loads(body)
        self.status_code = status_code


class _StateStore:
    def __init__(self):
        self.created = []

    @contextlib.contextmanager
    def lock(self, _tokens):
        yield

    def load(self, token, _image, _device):
        return {"token": token}, "mask"

    def create_many(self, states):
        self.created.append(states)
        return [f"{index + 10:032x}" for index in range(len(states))]


class HandlerTest(unittest.TestCase):
    def setUp(self):
        self.store = _StateStore()
        self.model = SimpleNamespace(
            device="cpu",
            preprocess_image=lambda image: SimpleNamespace(width=image.width, height=image.height),
        )
        self.context = SimpleNamespace(
            Response=_Response,
            logger=mock.Mock(),
            user_data=SimpleNamespace(model=self.model, state_store=self.store),
        )
        self.event = SimpleNamespace(
            body={
                "image": "unused in this test",
                "shapes": [None, None],
                "states": ["1" * 32, "2" * 32],
            }
        )
        self.image = SimpleNamespace(width=16, height=16)
        self.decode_image = mock.patch.object(main, "_decode_image", return_value=self.image)
        self.decode_image.start()
        self.addCleanup(self.decode_image.stop)

    def test_tracking_saves_all_states_after_all_predictions_succeed(self):
        self.model.track = lambda _image, state, _shape_type: {"points": [state["token"]]}

        response = main.handler(self.context, self.event)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.body["shapes"]), 2)
        self.assertEqual(len(self.store.created), 1)
        self.assertNotEqual(response.body["states"], self.event.body["states"])

    def test_tracking_saves_nothing_when_any_prediction_fails(self):
        calls = 0

        def track(_image, _state, _shape_type):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("prediction failed")
            return {"points": []}

        self.model.track = track

        with self.assertRaisesRegex(RuntimeError, "prediction failed"):
            main.handler(self.context, self.event)

        self.assertEqual(self.store.created, [])

    def test_tracking_returns_service_unavailable_when_redis_write_fails(self):
        self.model.track = lambda _image, _state, _shape_type: {"points": []}
        self.store.create_many = mock.Mock(side_effect=redis.RedisError("unavailable"))

        response = main.handler(self.context, self.event)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.body, {"error": "tracker state storage is unavailable"})

    def test_tracking_returns_not_found_for_expired_state_without_saving(self):
        self.store.load = mock.Mock(
            side_effect=main.RequestError("tracker state not found or expired", status_code=404)
        )

        response = main.handler(self.context, self.event)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.body, {"error": "tracker state not found or expired"})
        self.assertEqual(self.store.created, [])

    def test_interaction_returns_a_live_mask_without_changing_tracking_state(self):
        self.event.body = {
            "image": "unused in this test",
            "pos_points": [[4, 5]],
            "neg_points": [[1, 2]],
            "obj_bbox": [[1, 2], [10, 12]],
        }
        self.model.interact = mock.Mock(
            return_value={"shapes": [{"type": "mask", "points": [0, 256, 0, 0, 15, 15]}]}
        )

        response = main.handler(self.context, self.event)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body["shapes"][0]["type"], "mask")
        self.model.interact.assert_called_once()
        self.assertEqual(self.model.interact.call_args.args[3].tolist(), [1, 2, 10, 12])
        self.assertEqual(self.store.created, [])


if __name__ == "__main__":
    unittest.main()
