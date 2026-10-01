import tempfile
import unittest
from pathlib import Path

import numpy as np

from fire_data import BANDS, FireDataset


class FireDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Every time, channel, and pixel has a distinct value to expose axis errors.
        self.timeline = np.arange(16 * 27 * 2 * 3, dtype=np.float32).reshape(16, 27, 2, 3)
        self.windows = np.stack([
            self.timeline[start:start + 10].transpose(1, 0, 2, 3)
            for start in (0, 3, 6)
        ])
        self.labels = np.arange(18, dtype=np.float32).reshape(3, 2, 3) % 2
        self.save("p1")
        self.dataset = FireDataset(self.root)

    def save(self, name, data=None, labels=None):
        np.savez_compressed(self.root / f"{name}.npz",
                            data=self.windows if data is None else data,
                            labels=self.labels if labels is None else labels)

    def test_reconstruction_and_every_selection(self):
        result = self.dataset.get_fire("p1")
        np.testing.assert_array_equal(result, self.timeline)
        self.assertEqual(result.dtype, self.timeline.dtype)
        for day in range(1, 17):
            np.testing.assert_array_equal(self.dataset.get_day("p1", day), self.timeline[day - 1])
        for index, band in enumerate(BANDS):
            np.testing.assert_array_equal(self.dataset.get_band("p1", f" {band.lower()} "), self.timeline[:, index])

    def test_labels_and_metadata(self):
        result = self.dataset.get_labels("p1")
        np.testing.assert_array_equal(result["masks"], self.labels)
        np.testing.assert_array_equal(result["target_days"], [11, 14, 17])
        info = self.dataset.describe_fire("p1")
        self.assertEqual(info["day_count"], 16)
        self.assertEqual(info["target_days"], [11, 14, 17])
        self.assertEqual(info["data_shape"], (16, 27, 2, 3))

    def test_single_window(self):
        self.save("single", self.windows[:1], self.labels[:1])
        np.testing.assert_array_equal(self.dataset.get_fire("single"), self.timeline[:10])
        np.testing.assert_array_equal(self.dataset.get_labels("single")["target_days"], [11])

    def test_invalid_requests(self):
        for day in (0, -1, 17, 1.5, "1", True, np.bool_(True)):
            with self.subTest(day=day), self.assertRaises(ValueError):
                self.dataset.get_day("p1", day)
        np.testing.assert_array_equal(self.dataset.get_day("p1", np.int64(1)), self.timeline[0])
        with self.assertRaisesRegex(KeyError, "list_fires"):
            self.dataset.get_fire("../p1")
        with self.assertRaisesRegex(ValueError, "list_bands"):
            self.dataset.get_band("p1", "invalid")
        with self.assertRaises(FileNotFoundError):
            FireDataset(self.root / "missing")

    def test_overlap_conflict(self):
        broken = self.windows.copy()
        broken[1, 0, 0, 0, 0] += 1
        self.save("broken", broken)
        with self.assertRaisesRegex(ValueError, "windows 1 and 2"):
            self.dataset.get_fire("broken")

    def test_malformed_files(self):
        for name, data, labels in (
            ("channels", self.windows[:, :26], self.labels),
            ("length", self.windows[:, :, :9], self.labels),
            ("empty", self.windows[:0], self.labels[:0]),
            ("labels", self.windows, self.labels[:2]),
        ):
            self.save(name, data, labels)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.dataset.get_fire(name)
        np.savez(self.root / "numbered.npz", data_000=self.windows[0])
        with self.assertRaisesRegex(ValueError, "numbered-window"):
            self.dataset.get_fire("numbered")
        broken = self.windows.copy()
        broken[0, 0, 0, 0, 0] = np.nan
        self.save("nan", broken)
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.dataset.get_fire("nan")

    def test_independent_results_and_cache_replacement(self):
        for result in (self.dataset.get_fire("p1"), self.dataset.get_day("p1", 1),
                       self.dataset.get_band("p1", "I1")):
            result[:] = -100
        labels = self.dataset.get_labels("p1")
        labels["masks"][:] = -1
        labels["target_days"][:] = -1
        np.testing.assert_array_equal(self.dataset.get_fire("p1"), self.timeline)
        np.testing.assert_array_equal(self.dataset.get_labels("p1")["masks"], self.labels)
        np.testing.assert_array_equal(self.dataset.get_labels("p1")["target_days"], [11, 14, 17])
        self.save("p2", self.windows + 1)
        np.testing.assert_array_equal(self.dataset.get_fire("p2"), self.timeline + 1)
        np.testing.assert_array_equal(self.dataset.get_fire("p1"), self.timeline)

    def test_discovery(self):
        self.save("a")
        (self.root / "not_a_file.npz").mkdir()
        self.assertEqual(self.dataset.list_fires(), ["a", "p1"])
        names = self.dataset.list_bands()
        names.clear()
        self.assertEqual(len(self.dataset.list_bands()), 27)


if __name__ == "__main__":
    unittest.main()
