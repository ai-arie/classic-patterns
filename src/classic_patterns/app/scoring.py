"""Scoring of detected setups with the production bundle (``models/production``), without PyTorch.

Used by the screener app and by the bundle builder (calibration and the parity sample),
so both score setups the same way.

Two models, both for the textbook trade plan (stop / target of the pattern, 60 trading days):
- LightGBM on the hand-crafted features (``features.tabular.tabular_features``);
- the distilled Chronos-2-Small student (ONNX int8): context window of the stock's channels +
  the barrier spec (direction, stop and target distance in ATR units, horizon).
Both give calibrated P(target first), P(stop first), P(timeout) and an expected R. At screening
time the entry (next open) is unknown: the spec distances are measured from the decision close.

The numbers are estimates for research / education, not trade recommendations.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from classic_patterns.features.arrays import FeatureStore
from classic_patterns.features.tabular import tabular_features

CLASSES = ("target", "stop", "timeout")
EPS = 1e-9


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def temperature(p: np.ndarray, t: float) -> np.ndarray:
    return softmax(np.log(np.clip(p, EPS, 1.0)) / t)


def expected_r(p: np.ndarray, rr: np.ndarray, cr: dict) -> np.ndarray:
    """Sum of P(class) x the training trades' mean net R of that class (target scaled by the
    trade's planned reward / risk), as ``baseline.lgbm.ClassR``."""
    rt = np.where(np.isfinite(rr) & (rr > 0), cr["k_target"] * rr, cr["r_target"])
    return p[:, 0] * rt + p[:, 1] * cr["r_stop"] + p[:, 2] * cr["r_timeout"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_checksums(root: Path) -> None:
    for line in (root / "checksums.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if sha256(root / name) != digest:
            raise ValueError(f"checksum mismatch: {name}")


@dataclass
class Bundle:
    root: Path
    lgbm: dict
    student: dict
    manifest: dict

    @classmethod
    def load(cls, root: str | Path, verify: bool = True) -> Bundle:
        root = Path(root)
        if verify:
            verify_checksums(root)

        def rd(name):
            return json.loads((root / name).read_text())

        return cls(root, rd("lgbm.json"), rd("student.json"), rd("manifest.json"))

    def booster(self):
        import lightgbm as lgb

        return lgb.Booster(model_file=str(self.root / self.lgbm["model_file"]))


def close_atr(store: FeatureStore, prices: str | Path, g: np.ndarray,
              n: int = 14) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Adjusted close and ATR(n) at global rows ``g`` (prices aligned with the feature store, as
    built by the data pipeline)."""
    from classic_patterns.features import indicators as ind

    px = pd.read_parquet(prices, columns=["date", "high", "low", "close"])
    if len(px) != len(store.rows) or px["date"].iloc[-1] != store.rows["date"].iloc[-1]:
        raise ValueError("prices and feature store are not aligned (rebuild the features)")
    h, lo, c = (px[k].to_numpy(dtype=np.float64) for k in ("high", "low", "close"))
    g = np.asarray(g, dtype=np.int64)
    tick = store.tickers.set_index("ticker_id")
    tid = store.rows["ticker_id"].to_numpy()[g]
    atr = np.full(len(g), np.nan)
    for t in np.unique(tid):
        off, m = int(tick.loc[t, "offset"]), int(tick.loc[t, "n_rows"])
        a = ind.atr(h[off : off + m], lo[off : off + m], c[off : off + m], n)
        k = tid == t
        atr[k] = a[g[k] - off]
    return c[g], atr


def spec_inputs(direction, close, atr, stop, target, horizon: int) -> np.ndarray:
    """Barrier spec of the textbook plan as the student saw it in training: direction, stop and
    target distance in ATR units (from the decision close here), horizon / 60."""
    d = np.asarray(direction, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = d * (np.asarray(close) - np.asarray(stop)) / np.asarray(atr)
        t = d * (np.asarray(target) - np.asarray(close)) / np.asarray(atr)
    f = np.stack([d, s, t, np.full(len(d), horizon / 60.0)], axis=1)
    return np.nan_to_num(f, nan=0.0).astype(np.float32)


_W: dict = {}


def _onnx_chunk(sl: slice) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = _W["threads"], 1
    sess = ort.InferenceSession(_W["path"], so, providers=["CPUExecutionProvider"])
    w, s = _W["windows"][sl], _W["spec"][sl]
    bar = np.empty((len(w), 3), np.float32)
    conv = np.empty(len(w), np.float32)
    for i in range(len(w)):  # one sample per call: int8 activation scales depend on the batch
        out = sess.run(["barrier", "conv"], {"windows": w[i : i + 1], "spec": s[i : i + 1]})
        bar[i] = out[0][0]
        conv[i] = np.ravel(out[1])[0]
    return bar, conv


def student_logits(path: Path, windows: np.ndarray, spec: np.ndarray, jobs: int = 1,
                   threads: int = 2) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Barrier logits [n, 3] and conversion logit [n] from the ONNX student."""
    _W.update(path=str(path), windows=windows, spec=spec, threads=threads)
    n = len(windows)
    if jobs <= 1 or n < 64:
        return _onnx_chunk(slice(0, n))
    import multiprocessing as mp

    step = -(-n // jobs)
    parts = [slice(i, min(i + step, n)) for i in range(0, n, step)]
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(_onnx_chunk, parts))
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


def score_events(bundle: Bundle, events: pd.DataFrame, store: FeatureStore, close: np.ndarray,
                 atr: np.ndarray, jobs: int = 1, raw: bool = False) -> pd.DataFrame:  # fmt: skip
    """Both models for every row of ``events`` (events-table rows: fs_row, pattern, stage,
    direction, stop, target, trigger_price + geometry); ``close`` / ``atr`` at each row.
    ``raw``: also return the uncalibrated probabilities (calibration fitting)."""
    X = tabular_features(events, store, close, atr)
    lg = bundle.lgbm
    p_raw = bundle.booster().predict(X[lg["features"]], num_iteration=lg["best_iter"])
    p_lg = temperature(p_raw, lg["temperature"])
    rr = X["rr_planned"].to_numpy(dtype=np.float64)
    st = bundle.student
    from classic_patterns.models.dataset import WindowBuilder

    wb = WindowBuilder.from_store(store, st["variates"])
    w = wb.windows(events["fs_row"].to_numpy(), st["context_length"])
    spec = spec_inputs(events["direction"], close, atr, events["stop"], events["target"],
                       st["horizon"])  # fmt: skip
    bar, conv = student_logits(bundle.root / st["model_file"], w, spec, jobs)
    s_raw = softmax(bar.astype(np.float64))
    p_st = temperature(s_raw, st["temperature"])
    out = pd.DataFrame(index=events.index)
    out["rr_planned"] = rr
    out["risk_atr"] = X["risk_atr"].to_numpy()
    out["vix"] = X["mkt_vix"].to_numpy()
    out["gspc_above200"] = X["mkt_gspc_dist_sma200_pct"].to_numpy() > 0
    for name, p in (("lgbm", p_lg), ("student", p_st)):
        for k, c in enumerate(CLASSES):
            out[f"{name}_p_{c}"] = p[:, k]
        out[f"{name}_exp_r"] = expected_r(p, rr, bundle.lgbm["class_r"])
    out["student_p_trigger20"] = np.where(events["stage"].to_numpy() == "forming",
                                          1 / (1 + np.exp(-conv)), np.nan)  # fmt: skip
    if raw:
        out[[f"lgbm_raw_{c}" for c in CLASSES]] = p_raw
        out[[f"student_raw_{c}" for c in CLASSES]] = s_raw
    return out
