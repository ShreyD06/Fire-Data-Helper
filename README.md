# Fire data helpers

Simple NumPy access to one local train, validation, or test split of
`SamuelWu318/ts-satfire-processed`.

Install the example dependencies with `python -m pip install -r requirements.txt`.
Keep `fire_data.py` alongside your notebook (upload both files to Colab if needed).
Download and extract your data separately, then point the helper at a split:

```python
from fire_data import FireDataset

dataset = FireDataset("./content/data/train")
print(dataset.list_fires())
print(dataset.list_bands())

fire_id = dataset.list_fires()[0]
fire = dataset.get_fire(fire_id)          # (days, 27, height, width)
day = dataset.get_day(fire_id, day=1)     # (27, height, width)
band = dataset.get_band(fire_id, "I4")    # (days, height, width)
targets = dataset.get_labels(fire_id)    # masks and target_days
print(dataset.describe_fire(fire_id))
```

See `FireDataExamples.ipynb` for a walkthrough with plots. The original
`DataAnalysisDataset.ipynb` is unchanged.

## What a day means

Day numbers start at **1** and index the retained input observations. They are
not calendar dates, and the archives do not include every original observation.
The expected input shape is `(windows, 27, 10, height, width)`, with windows
starting three observations apart. The first ten frames plus the last three
frames of every later window give `10 + 3 * (windows - 1)` distinct observations.
Seven-frame overlaps must agree exactly; conflicting files are rejected rather
than silently combined. Verify their preprocessing version/stride or regenerate
them before using these helpers.

Labels have shape `(windows, height, width)` and are **newly burned-area
prediction targets**, not daily burned-area masks. Their target observation
positions are 11, 14, 17, and so on. The final target falls one observation after
the last retained input. Missing daily targets are not filled or inferred.

Stored values and dtypes are preserved. Thermal channels may already contain
cumulative maxima, and other preprocessing has already been applied. The
helpers do not recover raw measurements or reinterpret zeros as missing data.

## Supported files and memory

Only archives with `data` and `labels` members are supported. The separate
`ts-satfire-processed-window` format (`data_000`, `label_000`, etc.) is not.
Filenames without `.npz`, such as `p20777152`, are the fire IDs.

The reader decompresses a complete fire and caches only the most recently used
one. Reading a day or band initially still requires loading that fire. Returned
arrays are independent copies; full-fire results require additional memory.
Create a new `FireDataset` to reload files modified during a session.

## Validation

Run `python -m unittest discover -s tests -v` for synthetic reconstruction,
selection, label alignment, malformed archive, and result independence tests.
Before research use, validate an actual archive with
`dataset.describe_fire(fire_id)` and compare a selected band with the original
archive. The loader checks shapes, finite numeric values, and every overlap.
These checks cannot establish the historical provenance of an older archive.

## Sources

- [Published channel order](https://www.nature.com/articles/s41597-025-06271-3/tables/1)
- [Supplied preprocessing notebook](https://colab.research.google.com/drive/1XLhx0HFi1Qgn8JNNN-fcfyh7WjdVZCEs)

The preprocessing notebook specifies length 10 and stride 3. Its current version
also includes fixes to accumulation and label processing; older stored archives
are not assumed to have received those fixes.
