"""Consolidated clean dataset (data/clean) and its manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def apply_exclusions(df: pd.DataFrame, excl: pd.DataFrame) -> pd.DataFrame | None:
    """Apply one ticker's row-removing actions. ``exclude_span`` rows are kept (masked later).

    Returns None if the ticker is dropped.
    """
    if (excl["action"] == "drop_ticker").any():
        return None
    keep = pd.Series(True, index=df.index)
    for r in excl.itertuples():
        if r.action == "truncate_before":
            keep &= df["date"] >= r.start
        elif r.action == "drop_rows":
            keep &= ~df["date"].between(r.start, r.end)
    return df[keep].reset_index(drop=True)


# exclusion-span reasons after which all earlier adjusted prices are off by a constant factor
LEVEL_SHIFT_REASONS = ("split_not_applied", "bad_dividend")


def context_guards(dates, excl: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Guards for model context windows of one ticker (``dates`` = its rows in data/clean).

    Exclusion spans keep artifacts out of samples, features and labels, but a long context
    window (up to 1024 rows) ending after a span can still reach back across the artifact.

    Returns per row:
    - ``floor``: first row index a window ending at that row may use. After a level shift
      (``LEVEL_SHIFT_REASONS``) the earlier prices are mis-scaled, so the window is cut at the
      event row; rows before it are treated as missing.
    - ``bad``: rows to mask as missing (the single bad row of a ``bad_print``).
    """
    dates = pd.DatetimeIndex(dates)
    n = len(dates)
    floor = np.zeros(n, dtype=np.int64)
    bad = np.zeros(n, dtype=bool)
    spans = excl[(excl["action"] == "exclude_span") & excl["event_date"].notna()]
    for r in spans.itertuples():
        i = dates.searchsorted(r.event_date)
        if i >= n or dates[i] != r.event_date:
            continue  # event row not in the data (e.g. removed by truncation)
        if r.reason in LEVEL_SHIFT_REASONS:
            floor[i:] = np.maximum(floor[i:], i)
        elif r.reason == "bad_print":
            bad[i] = True
    return floor, bad


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class ParquetAppender:
    """Write many frames with one schema into a single Parquet file, in large row groups."""

    def __init__(self, path: Path, rows_per_group: int = 1_000_000):
        self.path = Path(path)
        self.rows_per_group = rows_per_group
        self.buf: list[pd.DataFrame] = []
        self.n_buf = 0
        self.rows = 0
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.tmp = self.path.with_suffix(".tmp")

    def append(self, df: pd.DataFrame) -> None:
        self.buf.append(df)
        self.n_buf += len(df)
        if self.n_buf >= self.rows_per_group:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        table = pa.Table.from_pandas(pd.concat(self.buf, ignore_index=True), preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.tmp, self.schema, compression="zstd")
        self.writer.write_table(table.cast(self.schema), row_group_size=self.rows_per_group)
        self.rows += table.num_rows
        self.buf, self.n_buf = [], 0

    def close(self) -> int:
        self._flush()
        if self.writer is not None:
            self.writer.close()
            self.tmp.replace(self.path)
        return self.rows
