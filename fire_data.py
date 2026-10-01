"""Access processed TS-SatFire observations with 1-based day numbers.

Expected archive layout: data (N, 27, 10, H, W), labels (N, H, W).
Windows start three observations apart. Dates are not stored in this format.
"""

from numbers import Integral
from pathlib import Path

import numpy as np


# Published channel order, with spelling corrected for display:
# https://www.nature.com/articles/s41597-025-06271-3/tables/1
BANDS = (
    "I1", "I2", "I3", "I4", "I5", "M11", "I4 Night", "I5 Night",
    "NDVI", "EVI", "Total Precipitation", "Wind Speed", "Wind Direction",
    "Min Temperature", "Max Temperature", "Energy Release Component",
    "Specific Humidity", "Slope", "Aspect", "Elevation", "PDSI", "Land Cover",
    "Total Precipitation Surface", "Forecast Wind Speed",
    "Forecast Wind Direction", "Forecast Temperature", "Forecast Specific Humidity",
)
WINDOW_LENGTH = 10
WINDOW_STRIDE = 3


class FireDataset:
    """Read one local split directory of processed fire archives.

    Only the most recently requested fire is cached. Public array results are
    independent copies. Stored values are preserved; no normalization is done.
    Compressed NPZ members must be decompressed in full, even for a single day.
    """

    def __init__(self, directory):
        self.directory = Path(directory).expanduser()
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Fire directory does not exist: {self.directory}")
        self._cached_id = None
        self._data = None
        self._labels = None

    def list_fires(self):
        """Return sorted fire IDs (filename stems) in this split."""
        return sorted(p.stem for p in self.directory.glob("*.npz") if p.is_file())

    def list_bands(self):
        """Return channel names in the same order as get_fire/get_day."""
        return list(BANDS)

    def _load(self, fire_id):
        if not isinstance(fire_id, str) or fire_id not in self.list_fires():
            raise KeyError(f"Unknown fire {fire_id!r}. Use list_fires() for valid IDs.")
        if fire_id == self._cached_id:
            return self._data, self._labels

        # Release the previous fire before decompressing another one.
        self._cached_id = self._data = self._labels = None
        path = self.directory / f"{fire_id}.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                if not {"data", "labels"}.issubset(archive.files):
                    raise ValueError(
                        "expected 'data' and 'labels' members; numbered-window "
                        "archives are not supported"
                    )
                data = archive["data"]
                labels = archive["labels"]
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError(f"Cannot load fire {fire_id!r}: {exc}") from exc

        if (data.ndim != 5 or data.shape[0] < 1
                or data.shape[1:3] != (len(BANDS), WINDOW_LENGTH)
                or any(size < 1 for size in data.shape[3:])):
            raise ValueError(
                f"Fire {fire_id!r}: expected data shape (N, 27, 10, H, W) "
                f"with positive N, H, W; found {data.shape}."
            )
        expected_labels = (data.shape[0], *data.shape[3:])
        if labels.shape != expected_labels:
            raise ValueError(
                f"Fire {fire_id!r}: expected labels shape {expected_labels}; "
                f"found {labels.shape}."
            )
        for name, array in (("data", data), ("labels", labels)):
            if array.dtype.kind not in "buif":
                raise ValueError(f"Fire {fire_id!r}: {name} must contain real numeric values.")
            if not np.isfinite(array).all():
                raise ValueError(f"Fire {fire_id!r}: {name} contains NaN or infinity.")

        for index in range(1, len(data)):
            if not np.array_equal(
                data[index - 1, :, WINDOW_STRIDE:],
                data[index, :, :WINDOW_LENGTH - WINDOW_STRIDE],
            ):
                raise ValueError(
                    f"Fire {fire_id!r}: conflicting overlap between windows "
                    f"{index} and {index + 1} (1-based). Cannot reconstruct a "
                    "consistent timeline. Verify the preprocessing version and "
                    "stride, or regenerate the file with consistent overlaps."
                )

        day_count = WINDOW_LENGTH + WINDOW_STRIDE * (len(data) - 1)
        timeline = np.empty((day_count, len(BANDS), *data.shape[3:]), dtype=data.dtype)
        timeline[:WINDOW_LENGTH] = data[0].transpose(1, 0, 2, 3)
        for index in range(1, len(data)):
            start = WINDOW_LENGTH + WINDOW_STRIDE * (index - 1)
            timeline[start:start + WINDOW_STRIDE] = data[index, :, -WINDOW_STRIDE:].transpose(1, 0, 2, 3)

        self._cached_id, self._data, self._labels = fire_id, timeline, labels
        return timeline, labels

    def get_fire(self, fire_id):
        """Return all retained observations as (days, bands, height, width)."""
        data, _ = self._load(fire_id)
        return data.copy()

    def get_day(self, fire_id, day):
        """Return (bands, height, width) for a 1-based observation day."""
        if isinstance(day, (bool, np.bool_)) or not isinstance(day, Integral):
            raise ValueError("day must be an integer starting at 1.")
        data, _ = self._load(fire_id)
        if not 1 <= day <= len(data):
            raise ValueError(f"Fire {fire_id!r}: day must be between 1 and {len(data)}; got {day}.")
        return data[day - 1].copy()

    def get_band(self, fire_id, band):
        """Return (days, height, width) for a band name, ignoring case/edge spaces."""
        lookup = {name.casefold(): i for i, name in enumerate(BANDS)}
        key = band.strip().casefold() if isinstance(band, str) else None
        if key not in lookup:
            raise ValueError(f"Unknown band {band!r}. Use list_bands() for valid names.")
        data, _ = self._load(fire_id)
        return data[:, lookup[key]].copy()

    def get_labels(self, fire_id):
        """Return stored newly burned-area targets and their observation positions.

        These are one target per window, not daily burned-area masks. Target
        positions are 11, 14, 17, ...; the last exceeds the input timeline.
        """
        _, labels = self._load(fire_id)
        return {
            "masks": labels.copy(),
            "target_days": WINDOW_LENGTH + 1 + WINDOW_STRIDE * np.arange(len(labels)),
        }

    def describe_fire(self, fire_id):
        """Return shape and alignment metadata without copying the input arrays."""
        data, labels = self._load(fire_id)
        return {
            "fire_id": fire_id,
            "data_shape": data.shape,
            "labels_shape": labels.shape,
            "day_count": len(data),
            "window_count": len(labels),
            "window_length": WINDOW_LENGTH,
            "stride": WINDOW_STRIDE,
            "target_days": list(range(WINDOW_LENGTH + 1, len(data) + 2, WINDOW_STRIDE)),
        }
