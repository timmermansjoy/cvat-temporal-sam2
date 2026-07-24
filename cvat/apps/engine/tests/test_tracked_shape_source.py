# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import unittest

from cvat.apps.engine.serializers import TrackedShapeSerializer


class TestTrackedShapeSource(unittest.TestCase):
    def test_source_is_optional_and_validated(self):
        shape = {
            "type": "mask",
            "frame": 1,
            "points": [1, 1, 4, 4, 1],
            "attributes": [],
        }

        serializer = TrackedShapeSerializer(data={**shape, "source": "auto"})
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data["source"], "auto")

        serializer = TrackedShapeSerializer(data=shape)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertIsNone(serializer.validated_data["source"])

        serializer = TrackedShapeSerializer(data={**shape, "source": "prediction"})
        self.assertFalse(serializer.is_valid())
