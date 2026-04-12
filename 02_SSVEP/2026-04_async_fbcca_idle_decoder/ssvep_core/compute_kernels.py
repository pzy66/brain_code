from __future__ import annotations

import time
from typing import Any, Optional, Sequence

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from .compute_backend import ComputeBackend


def build_reference_tensor(
    fs: int,
    win_samples: int,
    freqs: Sequence[float],
    nh: int,
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    t = np.arange(int(win_samples), dtype=np.float64) / float(fs)
    refs: list[np.ndarray] = []
    for freq in freqs:
        cols: list[np.ndarray] = []
        for harmonic in range(1, int(nh) + 1):
            cols.append(np.sin(2.0 * np.pi * harmonic * float(freq) * t))
            cols.append(np.cos(2.0 * np.pi * harmonic * float(freq) * t))
        matrix = np.stack(cols, axis=1)
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
        refs.append(np.asarray(matrix, dtype=dtype))
    return np.stack(refs, axis=0)


def design_sos_lowpass(fs: int, cutoff_hz: float, order: int = 1) -> np.ndarray:
    nyq = float(fs) / 2.0
    return np.asarray(butter(int(order), float(cutoff_hz) / nyq, btype="low", output="sos"), dtype=np.float64)


def design_sos_notch(fs: int, notch_freq: float, q: float) -> Optional[np.ndarray]:
    nyq = float(fs) / 2.0
    if float(notch_freq) >= nyq - 1e-6:
        return None
    b, a = iirnotch(float(notch_freq), float(q), float(fs))
    return np.asarray(tf2sos(b, a), dtype=np.float64)


def design_sos_bandpass(fs: int, low_hz: float, high_hz: float, order: int = 2) -> Optional[np.ndarray]:
    nyq = float(fs) / 2.0
    clipped_high = min(float(high_hz), nyq - 1e-3)
    if float(low_hz) >= clipped_high:
        return None
    sos = butter(int(order), [float(low_hz) / nyq, clipped_high / nyq], btype="band", output="sos")
    return np.asarray(sos, dtype=np.float64)


def _sosfiltfilt_with_backend(
    backend: ComputeBackend,
    sos: np.ndarray,
    values: Any,
    *,
    axis: int,
) -> Any:
    if backend.uses_cuda:
        signal = backend.signal
        if hasattr(signal, "sosfiltfilt"):
            return signal.sosfiltfilt(backend.as_float_array(sos), values, axis=axis)
        host_values, _ = backend.to_host(values)
        filtered = sosfiltfilt(np.asarray(sos, dtype=np.float64), host_values, axis=axis)
        device_values, _ = backend.to_device(filtered)
        return device_values
    return sosfiltfilt(np.asarray(sos, dtype=np.float64), np.asarray(values, dtype=backend.float_dtype), axis=axis)


def preprocess_windows_batch(
    backend: ComputeBackend,
    windows: Any,
    *,
    baseline_sos: np.ndarray,
    notch_sos: Optional[np.ndarray],
) -> Any:
    baseline = _sosfiltfilt_with_backend(backend, baseline_sos, windows, axis=1)
    filtered = windows - baseline
    if notch_sos is not None:
        filtered = _sosfiltfilt_with_backend(backend, notch_sos, filtered, axis=1)
    filtered = filtered - backend.xp.mean(filtered, axis=1, keepdims=True)
    return filtered


def apply_filterbank_batch(
    backend: ComputeBackend,
    windows: Any,
    *,
    subband_sos: Sequence[np.ndarray],
) -> Any:
    filtered = []
    for sos in subband_sos:
        band = _sosfiltfilt_with_backend(backend, sos, windows, axis=1)
        band = band - backend.xp.mean(band, axis=1, keepdims=True)
        filtered.append(band)
    return backend.xp.stack(filtered, axis=1)


def matrix_inv_sqrt_batch(backend: ComputeBackend, matrices: Any, *, reg: float) -> Any:
    xp = backend.xp
    size = int(matrices.shape[-1])
    eye = xp.eye(size, dtype=matrices.dtype)
    stabilized = matrices + float(reg) * eye
    eigvals, eigvecs = xp.linalg.eigh(stabilized)
    eigvals = xp.maximum(eigvals, float(reg))
    inv_sqrt = 1.0 / xp.sqrt(eigvals)
    scaled_vecs = eigvecs * inv_sqrt[..., None, :]
    return xp.matmul(scaled_vecs, xp.swapaxes(eigvecs, -1, -2))


def cca_correlations_batch(
    backend: ComputeBackend,
    x_batch: Any,
    y_refs: Any,
    *,
    reg: float = 1e-6,
) -> Any:
    xp = backend.xp
    x_centered = x_batch - xp.mean(x_batch, axis=1, keepdims=True)
    y_centered = y_refs - xp.mean(y_refs, axis=1, keepdims=True)
    denom = max(int(x_batch.shape[1]) - 1, 1)
    sxx = xp.einsum("btc,btd->bcd", x_centered, x_centered) / float(denom)
    syy = xp.einsum("ftk,ftm->fkm", y_centered, y_centered) / float(denom)
    sxy = xp.einsum("btc,ftm->bfcm", x_centered, y_centered) / float(denom)

    sxx_inv = matrix_inv_sqrt_batch(backend, sxx, reg=reg)
    syy_inv = matrix_inv_sqrt_batch(backend, syy, reg=reg)
    tmat = xp.einsum("bij,bfjm,fmk->bfik", sxx_inv, sxy, syy_inv)
    bf_shape = tmat.shape[:2]
    flattened = tmat.reshape(int(bf_shape[0]) * int(bf_shape[1]), int(tmat.shape[2]), int(tmat.shape[3]))
    singular_values = xp.linalg.svd(flattened, compute_uv=False)
    top = singular_values[:, 0].reshape(bf_shape)
    return top


def fbcca_scores_batch(
    backend: ComputeBackend,
    windows: Any,
    *,
    y_refs: Any,
    baseline_sos: np.ndarray,
    notch_sos: Optional[np.ndarray],
    subband_sos: Sequence[np.ndarray],
    subband_weights: Any,
    reg: float = 1e-6,
) -> tuple[Any, dict[str, float]]:
    timings = {
        "host_to_device_ms": 0.0,
        "preprocess_ms": 0.0,
        "score_ms": 0.0,
        "device_to_host_ms": 0.0,
        "synchronize_ms": 0.0,
    }
    xp = backend.xp
    if backend.is_device_array(windows):
        device_windows = windows
    else:
        device_windows, host_to_device_ms = backend.to_device(windows)
        timings["host_to_device_ms"] = float(host_to_device_ms)

    t0 = time.perf_counter()
    preprocessed = preprocess_windows_batch(
        backend,
        device_windows,
        baseline_sos=np.asarray(baseline_sos, dtype=np.float64),
        notch_sos=None if notch_sos is None else np.asarray(notch_sos, dtype=np.float64),
    )
    subbands = apply_filterbank_batch(
        backend,
        preprocessed,
        subband_sos=[np.asarray(sos, dtype=np.float64) for sos in subband_sos],
    )
    timings["preprocess_ms"] = float((time.perf_counter() - t0) * 1000.0)

    t1 = time.perf_counter()
    scores = xp.zeros((int(device_windows.shape[0]), int(y_refs.shape[0])), dtype=device_windows.dtype)
    weights = backend.as_float_array(subband_weights)
    for subband_index in range(int(subbands.shape[1])):
        rho = cca_correlations_batch(backend, subbands[:, subband_index, :, :], y_refs, reg=reg)
        scores = scores + weights[subband_index] * (rho ** 2)
    timings["score_ms"] = float((time.perf_counter() - t1) * 1000.0)
    timings["synchronize_ms"] = float(backend.synchronize())
    return scores, timings


def fbcca_subband_scores_batch(
    backend: ComputeBackend,
    windows: Any,
    *,
    y_refs: Any,
    baseline_sos: np.ndarray,
    notch_sos: Optional[np.ndarray],
    subband_sos: Sequence[np.ndarray],
    reg: float = 1e-6,
) -> tuple[Any, dict[str, float]]:
    timings = {
        "host_to_device_ms": 0.0,
        "preprocess_ms": 0.0,
        "score_ms": 0.0,
        "device_to_host_ms": 0.0,
        "synchronize_ms": 0.0,
    }
    xp = backend.xp
    if backend.is_device_array(windows):
        device_windows = windows
    else:
        device_windows, host_to_device_ms = backend.to_device(windows)
        timings["host_to_device_ms"] = float(host_to_device_ms)

    t0 = time.perf_counter()
    preprocessed = preprocess_windows_batch(
        backend,
        device_windows,
        baseline_sos=np.asarray(baseline_sos, dtype=np.float64),
        notch_sos=None if notch_sos is None else np.asarray(notch_sos, dtype=np.float64),
    )
    subbands = apply_filterbank_batch(
        backend,
        preprocessed,
        subband_sos=[np.asarray(sos, dtype=np.float64) for sos in subband_sos],
    )
    timings["preprocess_ms"] = float((time.perf_counter() - t0) * 1000.0)

    t1 = time.perf_counter()
    subband_scores = xp.zeros(
        (int(device_windows.shape[0]), int(subbands.shape[1]), int(y_refs.shape[0])),
        dtype=device_windows.dtype,
    )
    for subband_index in range(int(subbands.shape[1])):
        rho = cca_correlations_batch(backend, subbands[:, subband_index, :, :], y_refs, reg=reg)
        subband_scores[:, subband_index, :] = rho ** 2
    timings["score_ms"] = float((time.perf_counter() - t1) * 1000.0)
    timings["synchronize_ms"] = float(backend.synchronize())
    return subband_scores, timings


def benchmark_fbcca_batch_path(
    backend: ComputeBackend,
    windows: Any,
    *,
    y_refs: Any,
    baseline_sos: np.ndarray,
    notch_sos: Optional[np.ndarray],
    subband_sos: Sequence[np.ndarray],
    subband_weights: Any,
    reg: float = 1e-6,
    repeats: int = 3,
) -> dict[str, Any]:
    repeat_count = max(int(repeats), 1)
    sample_windows = np.asarray(windows, dtype=np.float64)
    if sample_windows.ndim != 3:
        raise ValueError("benchmark windows must have shape (batch, samples, channels)")
    sums = {
        "host_to_device_ms": 0.0,
        "preprocess_ms": 0.0,
        "score_ms": 0.0,
        "device_to_host_ms": 0.0,
        "synchronize_ms": 0.0,
    }
    for _ in range(repeat_count):
        scores, timings = fbcca_scores_batch(
            backend,
            sample_windows,
            y_refs=y_refs,
            baseline_sos=np.asarray(baseline_sos, dtype=np.float64),
            notch_sos=None if notch_sos is None else np.asarray(notch_sos, dtype=np.float64),
            subband_sos=[np.asarray(item, dtype=np.float64) for item in subband_sos],
            subband_weights=subband_weights,
            reg=reg,
        )
        _host_scores, device_to_host_ms = backend.to_host(scores)
        timings["device_to_host_ms"] = float(device_to_host_ms)
        for key in sums:
            sums[key] += float(timings.get(key, 0.0))
    return {
        "batch_size": int(sample_windows.shape[0]),
        "win_samples": int(sample_windows.shape[1]),
        "channels": int(sample_windows.shape[2]),
        "repeats": int(repeat_count),
        **{key: float(value / repeat_count) for key, value in sums.items()},
        "total_ms": float(sum(sums.values()) / repeat_count),
    }


def cca_scores_batch(
    backend: ComputeBackend,
    windows: Any,
    *,
    y_refs: Any,
    baseline_sos: np.ndarray,
    notch_sos: Optional[np.ndarray],
    reg: float = 1e-6,
) -> tuple[Any, dict[str, float]]:
    timings = {
        "host_to_device_ms": 0.0,
        "preprocess_ms": 0.0,
        "score_ms": 0.0,
        "device_to_host_ms": 0.0,
        "synchronize_ms": 0.0,
    }
    if backend.is_device_array(windows):
        device_windows = windows
    else:
        device_windows, host_to_device_ms = backend.to_device(windows)
        timings["host_to_device_ms"] = float(host_to_device_ms)

    t0 = time.perf_counter()
    preprocessed = preprocess_windows_batch(
        backend,
        device_windows,
        baseline_sos=np.asarray(baseline_sos, dtype=np.float64),
        notch_sos=None if notch_sos is None else np.asarray(notch_sos, dtype=np.float64),
    )
    timings["preprocess_ms"] = float((time.perf_counter() - t0) * 1000.0)

    t1 = time.perf_counter()
    rho = cca_correlations_batch(backend, preprocessed, y_refs, reg=reg)
    scores = rho ** 2
    timings["score_ms"] = float((time.perf_counter() - t1) * 1000.0)
    timings["synchronize_ms"] = float(backend.synchronize())
    return scores, timings
