from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Optional
import uuid

import numpy as np


_ORIGINAL_MKDTEMP = tempfile.mkdtemp
_ORIGINAL_TEMPORARY_DIRECTORY = tempfile.TemporaryDirectory
_TEMPFILE_FACTORY_PATCHED = False


def _manual_mkdtemp(suffix: Optional[str] = None, prefix: Optional[str] = None, dir: Optional[str] = None) -> str:
    root = Path(dir or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    safe_prefix = "tmp" if prefix is None else str(prefix)
    safe_suffix = "" if suffix is None else str(suffix)
    for _ in range(256):
        candidate = root / f"{safe_prefix}{uuid.uuid4().hex}{safe_suffix}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return str(candidate)
    raise FileExistsError(f"could not create a unique temporary directory under {root}")


class _WritableTemporaryDirectory:
    def __init__(
        self,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[str] = None,
        ignore_cleanup_errors: bool = False,
        **_kwargs: Any,
    ) -> None:
        self.name = _manual_mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        self._ignore_cleanup_errors = bool(ignore_cleanup_errors)

    def __enter__(self) -> str:
        return str(self.name)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.cleanup()

    def __del__(self) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)


def _tempfile_children_are_writable(root: Path) -> bool:
    try:
        with _ORIGINAL_TEMPORARY_DIRECTORY(dir=str(root)) as tmp_name:
            probe = Path(tmp_name) / "write_probe.txt"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _ensure_writable_tempfile_factory(root: Path) -> None:
    global _TEMPFILE_FACTORY_PATCHED
    if _TEMPFILE_FACTORY_PATCHED:
        return
    if _tempfile_children_are_writable(root):
        return
    tempfile.mkdtemp = _manual_mkdtemp  # type: ignore[assignment]
    tempfile.TemporaryDirectory = _WritableTemporaryDirectory  # type: ignore[assignment]
    _TEMPFILE_FACTORY_PATCHED = True


def _candidate_cuda_bin_dirs() -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def _push(path: Path) -> None:
        resolved = str(path).lower()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(path)

    cuda_path_raw = os.environ.get("CUDA_PATH", "").strip()
    if cuda_path_raw:
        root = Path(cuda_path_raw)
        for subdir in (root / "bin", root / "bin" / "win64", root):
            if subdir.exists():
                _push(subdir)

    toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit_root.exists():
        for child in sorted(toolkit_root.glob("v*"), reverse=True):
            for subdir in (child / "bin", child / "bin" / "win64"):
                if subdir.exists():
                    _push(subdir)

    matlab_root = Path(r"C:\Program Files\MATLAB")
    if matlab_root.exists():
        for child in sorted(matlab_root.glob("R*"), reverse=True):
            matlab_cuda_bin = child / "sys" / "cuda" / "win64" / "cuda" / "bin"
            if matlab_cuda_bin.exists():
                _push(matlab_cuda_bin)
            subdir = child / "bin" / "win64"
            if subdir.exists():
                _push(subdir)

    return candidates


def _prepare_cuda_runtime_env() -> None:
    if not os.environ.get("CUDA_PATH", "").strip():
        toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if toolkit_root.exists():
            versions = sorted(toolkit_root.glob("v*"), reverse=True)
            if versions:
                os.environ["CUDA_PATH"] = str(versions[0])
        if not os.environ.get("CUDA_PATH", "").strip():
            matlab_root = Path(r"C:\Program Files\MATLAB")
            if matlab_root.exists():
                for child in sorted(matlab_root.glob("R*"), reverse=True):
                    matlab_cuda_root = child / "sys" / "cuda" / "win64" / "cuda"
                    if matlab_cuda_root.exists():
                        os.environ["CUDA_PATH"] = str(matlab_cuda_root)
                        break

    wanted = ("cublasLt64_12.dll", "cudart64_12.dll")
    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    current_lower = {str(Path(part)).lower() for part in path_parts}
    for bin_dir in _candidate_cuda_bin_dirs():
        if not all((bin_dir / dll_name).exists() for dll_name in wanted):
            continue
        bin_dir_str = str(bin_dir)
        if str(bin_dir).lower() not in current_lower:
            os.environ["PATH"] = bin_dir_str + os.pathsep + os.environ.get("PATH", "")
        if not os.environ.get("CUDA_PATH", "").strip():
            if bin_dir.name.lower() == "win64":
                cuda_root = bin_dir.parent.parent
            elif bin_dir.name.lower() == "bin":
                cuda_root = bin_dir.parent
            else:
                cuda_root = bin_dir
            os.environ["CUDA_PATH"] = str(cuda_root)
        break


def _runtime_root_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "ssvep_gpu_runtime"


def _prepare_cupy_cache_env() -> None:
    cache_dir = _runtime_root_dir() / "cupy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("CUPY_CACHE_DIR", "").strip():
        os.environ["CUPY_CACHE_DIR"] = str(cache_dir)


def _prepare_temp_env() -> None:
    temp_dir = _runtime_root_dir() / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for key in ("TMP", "TEMP", "TMPDIR"):
        os.environ[key] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)
    _ensure_writable_tempfile_factory(temp_dir)


_prepare_cuda_runtime_env()
_prepare_cupy_cache_env()
_prepare_temp_env()

try:  # pragma: no cover - optional dependency
    import cupy as cp
except Exception:  # pragma: no cover - CPU-only environments
    cp = None

cpx_signal = None


DEFAULT_COMPUTE_BACKEND = "auto"
DEFAULT_GPU_DEVICE = 0
DEFAULT_GPU_PRECISION = "float32"
DEFAULT_GPU_WARMUP = True
DEFAULT_GPU_CACHE_POLICY = "full"
ALLOWED_COMPUTE_BACKENDS = ("auto", "cpu", "cuda")
ALLOWED_GPU_PRECISIONS = ("float32", "float64")
ALLOWED_GPU_CACHE_POLICIES = ("windows", "full")


def _host_dtype_for_precision(precision: str, dtype: Optional[Any] = None) -> np.dtype:
    if dtype is not None:
        return np.dtype(dtype)
    return np.dtype(np.float32 if str(precision).strip().lower() == "float32" else np.float64)


def parse_compute_backend_name(raw: Optional[str]) -> str:
    value = str(raw or DEFAULT_COMPUTE_BACKEND).strip().lower()
    if value not in set(ALLOWED_COMPUTE_BACKENDS):
        raise ValueError(f"unsupported compute backend: {raw}")
    return value


def parse_gpu_precision(raw: Optional[str]) -> str:
    value = str(raw or DEFAULT_GPU_PRECISION).strip().lower()
    if value not in set(ALLOWED_GPU_PRECISIONS):
        raise ValueError(f"unsupported gpu precision: {raw}")
    return value


def parse_gpu_cache_policy(raw: Optional[str]) -> str:
    value = str(raw or DEFAULT_GPU_CACHE_POLICY).strip().lower()
    if value not in set(ALLOWED_GPU_CACHE_POLICIES):
        raise ValueError(f"unsupported gpu cache policy: {raw}")
    return value


def cupy_is_available() -> bool:
    return cp is not None


def cuda_runtime_available() -> bool:
    if cp is None:
        return False
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return False
    return count > 0


@dataclass(frozen=True)
class BackendTiming:
    host_to_device_ms: float = 0.0
    device_to_host_ms: float = 0.0
    synchronize_ms: float = 0.0
    warmup_overhead_ms: float = 0.0


class ComputeBackend:
    backend_name = "cpu"
    uses_cuda = False

    def __init__(self, *, precision: str = DEFAULT_GPU_PRECISION, gpu_device: int = DEFAULT_GPU_DEVICE) -> None:
        self.precision = parse_gpu_precision(precision)
        self.gpu_device = int(gpu_device)

    @property
    def float_dtype(self) -> Any:
        return np.float32 if self.precision == "float32" else np.float64

    @property
    def xp(self) -> Any:
        return np

    @property
    def signal(self) -> Any:
        raise NotImplementedError

    def supports_gpu(self) -> bool:
        return False

    def synchronize(self) -> float:
        return 0.0

    def benchmark_warmup(self) -> float:
        return 0.0

    def as_float_array(self, array: Any) -> Any:
        return np.asarray(array, dtype=self.float_dtype)

    def to_device(self, array: Any) -> tuple[Any, float]:
        t0 = time.perf_counter()
        value = np.asarray(array, dtype=self.float_dtype)
        t1 = time.perf_counter()
        return value, float((t1 - t0) * 1000.0)

    def to_host(self, array: Any) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        value = np.asarray(array, dtype=self.float_dtype)
        t1 = time.perf_counter()
        return value, float((t1 - t0) * 1000.0)

    def zeros(self, shape: Any) -> Any:
        return self.xp.zeros(shape, dtype=self.float_dtype)

    def eye(self, size: int) -> Any:
        return self.xp.eye(int(size), dtype=self.float_dtype)

    def is_device_array(self, value: Any) -> bool:
        return False

    def supports_pinned_host_memory(self) -> bool:
        return False

    def alloc_pinned_host_array(self, shape: Any, *, dtype: Optional[Any] = None) -> np.ndarray:
        resolved_dtype = _host_dtype_for_precision(self.precision, dtype)
        return np.empty(shape, dtype=resolved_dtype)

    def microbenchmark_transfer(
        self,
        *,
        sample_shape: tuple[int, ...] = (256, 8),
        repeats: int = 3,
        use_pinned: bool = True,
    ) -> dict[str, Any]:
        shape = tuple(max(int(value), 1) for value in sample_shape)
        repeat_count = max(int(repeats), 1)
        host_dtype = _host_dtype_for_precision(self.precision)
        pinned = bool(use_pinned and self.supports_pinned_host_memory())
        host_array = (
            self.alloc_pinned_host_array(shape, dtype=host_dtype)
            if pinned
            else np.empty(shape, dtype=host_dtype)
        )
        host_array.fill(0.25)
        warmup_ms = float(self.benchmark_warmup())
        host_to_device_ms = 0.0
        device_to_host_ms = 0.0
        synchronize_ms = 0.0
        for _ in range(repeat_count):
            device_value, upload_ms = self.to_device(host_array)
            host_to_device_ms += float(upload_ms)
            _host_value, download_ms = self.to_host(device_value)
            device_to_host_ms += float(download_ms)
            synchronize_ms += float(self.synchronize())
        return {
            "backend_name": str(self.backend_name),
            "sample_shape": [int(value) for value in shape],
            "repeats": int(repeat_count),
            "used_pinned_host_memory": bool(pinned),
            "host_to_device_ms": float(host_to_device_ms / repeat_count),
            "device_to_host_ms": float(device_to_host_ms / repeat_count),
            "synchronize_ms": float(synchronize_ms / repeat_count),
            "warmup_overhead_ms": float(warmup_ms),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "backend_name": str(self.backend_name),
            "uses_cuda": bool(self.uses_cuda),
            "precision": str(self.precision),
            "gpu_device": int(self.gpu_device),
            "cupy_available": bool(cupy_is_available()),
            "cuda_runtime_available": bool(cuda_runtime_available()),
        }


class NumpyBackend(ComputeBackend):
    backend_name = "cpu"
    uses_cuda = False

    @property
    def signal(self) -> Any:
        from scipy import signal as scipy_signal

        return scipy_signal


class CuPyBackend(ComputeBackend):
    backend_name = "cuda"
    uses_cuda = True

    def __init__(self, *, precision: str = DEFAULT_GPU_PRECISION, gpu_device: int = DEFAULT_GPU_DEVICE) -> None:
        global cpx_signal
        if cp is None:
            raise RuntimeError("CuPy backend requested but cupy is not installed")
        if cpx_signal is None:
            import cupyx.scipy.signal as _cpx_signal

            cpx_signal = _cpx_signal
        if not cuda_runtime_available():
            raise RuntimeError("CuPy backend requested but CUDA runtime is not available")
        super().__init__(precision=precision, gpu_device=gpu_device)
        cp.cuda.Device(self.gpu_device).use()

    @property
    def float_dtype(self) -> Any:
        return cp.float32 if self.precision == "float32" else cp.float64

    @property
    def xp(self) -> Any:
        return cp

    @property
    def signal(self) -> Any:
        return cpx_signal

    def supports_gpu(self) -> bool:
        return True

    def synchronize(self) -> float:
        t0 = time.perf_counter()
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        return float((t1 - t0) * 1000.0)

    def benchmark_warmup(self) -> float:
        t0 = time.perf_counter()
        probe = cp.asarray([1.0, 2.0, 3.0], dtype=self.float_dtype)
        _ = cp.linalg.norm(probe)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        return float((t1 - t0) * 1000.0)

    def as_float_array(self, array: Any) -> Any:
        return cp.asarray(array, dtype=self.float_dtype)

    def to_device(self, array: Any) -> tuple[Any, float]:
        t0 = time.perf_counter()
        value = cp.asarray(array, dtype=self.float_dtype)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        return value, float((t1 - t0) * 1000.0)

    def to_host(self, array: Any) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        value = cp.asnumpy(array)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        return np.asarray(value), float((t1 - t0) * 1000.0)

    def is_device_array(self, value: Any) -> bool:
        return isinstance(value, cp.ndarray)

    def supports_pinned_host_memory(self) -> bool:
        return cp is not None

    def alloc_pinned_host_array(self, shape: Any, *, dtype: Optional[Any] = None) -> np.ndarray:
        resolved_dtype = _host_dtype_for_precision(self.precision, dtype)
        shape_tuple = tuple(int(value) for value in np.atleast_1d(shape))
        if len(shape_tuple) == 1 and isinstance(shape, int):
            shape_tuple = (int(shape),)
        element_count = int(np.prod(shape_tuple, dtype=np.int64))
        nbytes = int(element_count * resolved_dtype.itemsize)
        mem = cp.cuda.alloc_pinned_memory(nbytes)
        view = np.frombuffer(mem, dtype=resolved_dtype, count=element_count)
        return view.reshape(shape_tuple)


def resolve_compute_backend(
    requested: Optional[str],
    *,
    gpu_device: int = DEFAULT_GPU_DEVICE,
    precision: str = DEFAULT_GPU_PRECISION,
) -> ComputeBackend:
    backend_name = parse_compute_backend_name(requested)
    precision_name = parse_gpu_precision(precision)
    if backend_name == "cpu":
        return NumpyBackend(precision=precision_name, gpu_device=gpu_device)
    if backend_name == "cuda":
        return CuPyBackend(precision=precision_name, gpu_device=gpu_device)
    if cuda_runtime_available():
        try:
            return CuPyBackend(precision=precision_name, gpu_device=gpu_device)
        except Exception:
            pass
    return NumpyBackend(precision=precision_name, gpu_device=gpu_device)
