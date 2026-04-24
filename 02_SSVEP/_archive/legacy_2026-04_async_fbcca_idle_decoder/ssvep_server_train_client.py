from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import posixpath
import shlex
import stat
import sys
import time
from typing import Any, Callable, Optional, Sequence


THIS_DIR = Path(__file__).resolve().parent
LOCAL_PROFILE_DIR = THIS_DIR / "profiles"
LOCAL_DATASET_ROOT = LOCAL_PROFILE_DIR / "datasets"
LOCAL_SERVER_RUNS_DIR = LOCAL_PROFILE_DIR / "server_runs"
LOCAL_SERVER_PROFILES_DIR = LOCAL_PROFILE_DIR / "server_profiles"
LOCAL_TASK_RECORD_PATH = LOCAL_SERVER_RUNS_DIR / "server_tasks.json"
LOCAL_CODE_ROOT = THIS_DIR

REMOTE_ALLOWED_PREFIX = "/data1/zkx"
REMOTE_ROOT = "/data1/zkx/brain/ssvep"
REMOTE_CODE_DIR = f"{REMOTE_ROOT}/code/2026-04_async_fbcca_idle_decoder"
REMOTE_DATA_DIR = f"{REMOTE_ROOT}/data"
REMOTE_REPORT_ROOT = f"{REMOTE_ROOT}/reports"
REMOTE_PROFILE_ROOT = f"{REMOTE_ROOT}/profiles"
REMOTE_LOG_DIR = f"{REMOTE_ROOT}/logs"
REMOTE_TMP_DIR = f"{REMOTE_ROOT}/tmp"
REMOTE_ENV_PYTHON = "/data1/zkx/miniconda3/envs/brain-ssvep/bin/python"
REMOTE_CODE_SYNC_MANIFEST = f"{REMOTE_CODE_DIR}/.ssvep_code_sync_manifest.json"
DEFAULT_REMOTE_COMPUTE_BACKEND = "cuda"
DEFAULT_REMOTE_GPU_DEVICE = 0
DEFAULT_REMOTE_GPU_PRECISION = "float32"
DEFAULT_REMOTE_GPU_WARMUP = True
DEFAULT_REMOTE_GPU_CACHE_POLICY = "windows"
DEFAULT_REMOTE_WIN_CANDIDATES = "2.5,3.0,3.5,4.0"
DEFAULT_REMOTE_MULTI_SEED_COUNT = 5
DEFAULT_SSH_CONNECT_TIMEOUT_SEC = 12
DEFAULT_SSH_BANNER_TIMEOUT_SEC = 30
DEFAULT_SSH_AUTH_TIMEOUT_SEC = 30
DEFAULT_SSH_CONNECT_RETRIES = 3
CODE_SYNC_EXCLUDED_TOP_LEVEL = {
    "profiles",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
}
CODE_SYNC_EXCLUDED_DIR_PREFIXES = (".tmp", ".pytest", "pytest_temp")
CODE_SYNC_EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo", ".npz", ".zip")


def assert_remote_ssvep_path(path: str) -> str:
    normalized = posixpath.normpath(str(path).replace("\\", "/"))
    if not (normalized == REMOTE_ALLOWED_PREFIX or normalized.startswith(REMOTE_ALLOWED_PREFIX + "/")):
        raise ValueError(f"unsafe remote path outside {REMOTE_ALLOWED_PREFIX}: {path}")
    if not (normalized == REMOTE_ROOT or normalized.startswith(REMOTE_ROOT + "/")):
        raise ValueError(f"unsafe remote path outside {REMOTE_ROOT}: {path}")
    return normalized


def sh_quote(value: str) -> str:
    return shlex.quote(str(value))


def safe_session_id(value: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(value).strip())
    token = token.strip("_")
    return token or datetime.now().strftime("session_%Y%m%d_%H%M%S")


def now_run_id(prefix: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(prefix).strip())
    token = token.strip("_") or "ssvep"
    return f"{token}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _coerce_bool_flag(value: Any) -> bool:
    raw = str(value).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _should_skip_code_dir(rel_dir: Path) -> bool:
    if not rel_dir.parts:
        return False
    name = rel_dir.name.lower()
    if len(rel_dir.parts) == 1 and name in CODE_SYNC_EXCLUDED_TOP_LEVEL:
        return True
    return any(name.startswith(prefix) for prefix in CODE_SYNC_EXCLUDED_DIR_PREFIXES)


def _should_skip_code_file(rel_path: Path) -> bool:
    if not rel_path.parts:
        return False
    if any(part.lower() in CODE_SYNC_EXCLUDED_TOP_LEVEL for part in rel_path.parts[:-1]):
        return True
    if any(part.lower().startswith(prefix) for part in rel_path.parts[:-1] for prefix in CODE_SYNC_EXCLUDED_DIR_PREFIXES):
        return True
    return rel_path.name.lower().endswith(CODE_SYNC_EXCLUDED_FILE_SUFFIXES)


def iter_local_code_files(local_root: Path = LOCAL_CODE_ROOT) -> list[Path]:
    root = Path(local_root).expanduser().resolve()
    files: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda _exc: None):
        current_path = Path(current_root)
        try:
            rel_dir = current_path.relative_to(root)
        except ValueError:
            continue
        filtered_dirs: list[str] = []
        for dirname in dirnames:
            rel_child = rel_dir / dirname if rel_dir.parts else Path(dirname)
            if _should_skip_code_dir(rel_child):
                continue
            filtered_dirs.append(dirname)
        dirnames[:] = filtered_dirs
        for filename in filenames:
            rel_file = rel_dir / filename if rel_dir.parts else Path(filename)
            if _should_skip_code_file(rel_file):
                continue
            local_path = root / rel_file
            if local_path.is_file():
                files.append(local_path)
    files.sort()
    return files


def build_local_code_manifest(local_root: Path = LOCAL_CODE_ROOT) -> dict[str, Any]:
    root = Path(local_root).expanduser().resolve()
    files_meta: dict[str, Any] = {}
    tree_digest = hashlib.sha256()
    for local_path in iter_local_code_files(root):
        rel_path = local_path.relative_to(root).as_posix()
        file_hash = _sha256_file(local_path)
        stat_result = local_path.stat()
        meta = {
            "sha256": file_hash,
            "size": int(stat_result.st_size),
            "mtime_ns": int(stat_result.st_mtime_ns),
        }
        files_meta[rel_path] = meta
        tree_digest.update(rel_path.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(file_hash.encode("ascii"))
        tree_digest.update(b"\0")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "local_root": str(root),
        "tree_hash": tree_digest.hexdigest(),
        "file_count": len(files_meta),
        "files": files_meta,
    }


@dataclass(frozen=True)
class LocalDataset:
    session_id: str
    manifest_path: Path
    npz_path: Path
    trial_count: int
    subject_id: str


@dataclass
class ServerConfig:
    host: str = "10.72.128.221"
    port: int = 22
    username: str = "zhangkexin"
    password: str = ""
    remote_root: str = REMOTE_ROOT
    remote_code_dir: str = REMOTE_CODE_DIR
    remote_python: str = REMOTE_ENV_PYTHON


def discover_local_datasets(dataset_root: Path = LOCAL_DATASET_ROOT) -> list[LocalDataset]:
    root = Path(dataset_root).expanduser().resolve()
    items: list[LocalDataset] = []
    for manifest in sorted(root.glob("*/session_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        npz_path = manifest.parent / "raw_trials.npz"
        if not npz_path.exists():
            continue
        payload = read_json(manifest, {})
        session_id = safe_session_id(str(payload.get("session_id") or manifest.parent.name))
        trials = payload.get("trials", [])
        items.append(
            LocalDataset(
                session_id=session_id,
                manifest_path=manifest,
                npz_path=npz_path,
                trial_count=len(trials) if isinstance(trials, list) else 0,
                subject_id=str(payload.get("subject_id", "")),
            )
        )
    return items


def _import_paramiko():
    try:
        import paramiko  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "paramiko is required for the SSVEP server helper. "
            "Install it in the local brain-vision environment first."
        ) from exc
    return paramiko


class SSHClient:
    def __init__(self, config: ServerConfig, log_fn: Optional[Callable[[str], None]] = None) -> None:
        self.config = config
        self.log_fn = log_fn or (lambda _text: None)
        self._client = None
        self._sftp = None

    def connect(self) -> None:
        paramiko = _import_paramiko()
        last_exc: Optional[Exception] = None
        for attempt in range(1, DEFAULT_SSH_CONNECT_RETRIES + 1):
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    hostname=self.config.host,
                    port=int(self.config.port),
                    username=self.config.username,
                    password=self.config.password,
                    timeout=DEFAULT_SSH_CONNECT_TIMEOUT_SEC,
                    banner_timeout=DEFAULT_SSH_BANNER_TIMEOUT_SEC,
                    auth_timeout=DEFAULT_SSH_AUTH_TIMEOUT_SEC,
                    look_for_keys=False,
                    allow_agent=False,
                )
                transport = client.get_transport()
                if transport is not None:
                    transport.set_keepalive(30)
                self._client = client
                self._sftp = client.open_sftp()
                return
            except Exception as exc:
                last_exc = exc
                try:
                    client.close()
                except Exception:
                    pass
                message = str(exc).strip() or exc.__class__.__name__
                self.log_fn(
                    f"SSH connect attempt {attempt}/{DEFAULT_SSH_CONNECT_RETRIES} failed: {message}"
                )
                if attempt >= DEFAULT_SSH_CONNECT_RETRIES:
                    break
                time.sleep(float(attempt))
        raise RuntimeError(
            "SSH connection failed after "
            f"{DEFAULT_SSH_CONNECT_RETRIES} attempts to {self.config.host}:{int(self.config.port)}. "
            f"Last error: {last_exc}"
        )

    def close(self) -> None:
        if self._sftp is not None:
            self._sftp.close()
        if self._client is not None:
            self._client.close()
        self._sftp = None
        self._client = None

    def exec(self, command: str, *, check: bool = True) -> tuple[int, str, str]:
        if self._client is None:
            raise RuntimeError("SSH is not connected")
        stdin, stdout, stderr = self._client.exec_command(command)
        _ = stdin
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = int(stdout.channel.recv_exit_status())
        if check and code != 0:
            raise RuntimeError(f"remote command failed ({code}): {command}\n{err or out}")
        return code, out, err

    @property
    def sftp(self):
        if self._sftp is None:
            raise RuntimeError("SFTP is not connected")
        return self._sftp

    def mkdir_p(self, remote_dir: str) -> None:
        remote_dir = assert_remote_ssvep_path(remote_dir)
        parts = remote_dir.strip("/").split("/")
        cur = ""
        for part in parts:
            cur += "/" + part
            try:
                self.sftp.stat(cur)
            except IOError:
                self.sftp.mkdir(cur)

    def put_file(self, local_path: Path, remote_path: str) -> None:
        remote_path = assert_remote_ssvep_path(remote_path)
        self.mkdir_p(posixpath.dirname(remote_path))
        self.sftp.put(str(Path(local_path).resolve()), remote_path)

    def get_file(self, remote_path: str, local_path: Path) -> None:
        remote_path = assert_remote_ssvep_path(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.sftp.get(remote_path, str(local_path))

    def read_text(self, remote_path: str, *, encoding: str = "utf-8") -> str:
        remote_path = assert_remote_ssvep_path(remote_path)
        with self.sftp.open(remote_path, "rb") as handle:
            return handle.read().decode(encoding, errors="replace")

    def write_text(self, remote_path: str, text: str, *, encoding: str = "utf-8") -> None:
        remote_path = assert_remote_ssvep_path(remote_path)
        self.mkdir_p(posixpath.dirname(remote_path))
        with self.sftp.open(remote_path, "wb") as handle:
            handle.write(text.encode(encoding))

    def exists(self, remote_path: str) -> bool:
        remote_path = assert_remote_ssvep_path(remote_path)
        try:
            self.sftp.stat(remote_path)
            return True
        except IOError:
            return False

    def is_dir(self, remote_path: str) -> bool:
        remote_path = assert_remote_ssvep_path(remote_path)
        try:
            return stat.S_ISDIR(self.sftp.stat(remote_path).st_mode)
        except IOError:
            return False

    def remove_file(self, remote_path: str) -> None:
        remote_path = assert_remote_ssvep_path(remote_path)
        if self.exists(remote_path) and not self.is_dir(remote_path):
            self.sftp.remove(remote_path)

    def download_dir(self, remote_dir: str, local_dir: Path) -> None:
        remote_dir = assert_remote_ssvep_path(remote_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        for entry in self.sftp.listdir_attr(remote_dir):
            rpath = posixpath.join(remote_dir, entry.filename)
            lpath = local_dir / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                self.download_dir(rpath, lpath)
            else:
                self.get_file(rpath, lpath)

    def tail_file(self, remote_path: str, lines: int = 50) -> str:
        remote_path = assert_remote_ssvep_path(remote_path)
        if not self.exists(remote_path):
            return ""
        command = f"tail -n {int(lines)} {sh_quote(remote_path)}"
        _code, out, _err = self.exec(command, check=False)
        return out


def remote_env_prefix(*, gpu_device: int) -> str:
    env = {
        "CUDA_VISIBLE_DEVICES": str(int(gpu_device)),
        "CUPY_CACHE_DIR": f"{REMOTE_LOG_DIR}/cupy_cache",
        "TMPDIR": REMOTE_TMP_DIR,
        "MPLCONFIGDIR": f"{REMOTE_LOG_DIR}/matplotlib",
        "PYTHONPYCACHEPREFIX": f"{REMOTE_LOG_DIR}/pycache",
    }
    return " ".join(f"{key}={sh_quote(value)}" for key, value in env.items())


def _load_remote_code_manifest(ssh: SSHClient) -> dict[str, Any]:
    if not ssh.exists(REMOTE_CODE_SYNC_MANIFEST):
        return {}
    try:
        payload = json.loads(ssh.read_text(REMOTE_CODE_SYNC_MANIFEST, encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def sync_local_code_tree(
    ssh: SSHClient,
    *,
    local_root: Path = LOCAL_CODE_ROOT,
    remote_code_dir: str = REMOTE_CODE_DIR,
) -> dict[str, Any]:
    root = Path(local_root).expanduser().resolve()
    manifest = build_local_code_manifest(root)
    remote_manifest = _load_remote_code_manifest(ssh)
    remote_files = dict(remote_manifest.get("files") or {})
    local_files = dict(manifest.get("files") or {})
    uploaded: list[str] = []
    removed: list[str] = []

    ssh.mkdir_p(remote_code_dir)
    for rel_path, meta in local_files.items():
        remote_meta = remote_files.get(rel_path)
        if remote_meta == meta:
            continue
        ssh.put_file(root / Path(rel_path), posixpath.join(remote_code_dir, rel_path))
        uploaded.append(rel_path)

    for rel_path in sorted(set(remote_files) - set(local_files)):
        remote_path = posixpath.join(remote_code_dir, rel_path)
        if ssh.exists(remote_path) and not ssh.is_dir(remote_path):
            ssh.remove_file(remote_path)
            removed.append(rel_path)

    sync_payload = {
        **manifest,
        "remote_code_dir": assert_remote_ssvep_path(remote_code_dir),
        "uploaded_count": len(uploaded),
        "removed_count": len(removed),
    }
    ssh.write_text(
        REMOTE_CODE_SYNC_MANIFEST,
        json.dumps(sync_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "local_root": str(root),
        "remote_code_dir": str(remote_code_dir),
        "tree_hash": str(manifest.get("tree_hash", "")),
        "file_count": int(manifest.get("file_count", 0) or 0),
        "uploaded_count": len(uploaded),
        "removed_count": len(removed),
        "uploaded_preview": uploaded[:10],
        "removed_preview": removed[:10],
        "manifest_path": REMOTE_CODE_SYNC_MANIFEST,
    }


def preflight_cuda_or_fail(
    ssh: SSHClient,
    *,
    compute_backend: str,
    gpu_device: int,
) -> dict[str, Any]:
    backend = str(compute_backend or DEFAULT_REMOTE_COMPUTE_BACKEND).strip().lower()
    if backend != "cuda":
        return {"checked": False, "reason": f"compute_backend={backend}"}
    device_id = int(gpu_device)
    check_smi_cmd = "command -v nvidia-smi >/dev/null 2>&1 || { echo CUDA_PREFLIGHT:NO_NVIDIA_SMI; exit 71; }"
    code, out, err = ssh.exec(check_smi_cmd, check=False)
    if code != 0:
        raise RuntimeError(
            "cuda backend requested but GPU preflight failed: "
            f"nvidia-smi unavailable (code={code}, detail={(out or err or '').strip()})"
        )

    # Enumerate indices via `nvidia-smi -L` first; this is usually more tolerant than full query.
    code, out, err = ssh.exec("nvidia-smi -L", check=False)
    lines = [line.strip() for line in str(out or err or "").splitlines() if line.strip()]
    visible_indices: list[int] = []
    for line in lines:
        if not line.startswith("GPU "):
            continue
        head = line.split(":", 1)[0].strip().replace("GPU ", "")
        try:
            visible_indices.append(int(head))
        except Exception:
            continue
    if device_id not in visible_indices:
        raise RuntimeError(
            "cuda backend requested but target GPU device is not visible: "
            f"requested={device_id}, visible={visible_indices}. "
            "Please change GPU Device in UI and retry."
        )

    # Probe only the selected device; avoid failing because of other broken GPUs on the host.
    query_cmd = (
        f"nvidia-smi -i {int(device_id)} --query-gpu=index,name,memory.total "
        "--format=csv,noheader || { echo CUDA_PREFLIGHT:QUERY_FAILED; exit 72; }"
    )
    code, out, err = ssh.exec(query_cmd, check=False)
    detail = (out or err or "").strip()
    if code != 0:
        raise RuntimeError(
            "cuda backend requested but GPU preflight failed: "
            f"selected device query failed (device={device_id}, code={code}, detail={detail}). "
            "Please switch GPU Device and retry."
        )
    rows = [line.strip() for line in str(out).splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(
            "cuda backend requested but selected GPU returned no info. "
            f"device={device_id}"
        )
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) >= 1:
            parsed_rows.append(
                {
                    "index": int(parts[0]) if parts[0].isdigit() else device_id,
                    "name": str(parts[1]) if len(parts) > 1 else "",
                    "memory_total": str(parts[2]) if len(parts) > 2 else "",
                }
            )
    return {
        "checked": True,
        "compute_backend": "cuda",
        "gpu_device": device_id,
        "visible_indices": visible_indices,
        "devices": parsed_rows,
    }


def upload_dataset(ssh: SSHClient, dataset: LocalDataset) -> dict[str, str]:
    remote_dir = assert_remote_ssvep_path(posixpath.join(REMOTE_DATA_DIR, dataset.session_id))
    ssh.mkdir_p(remote_dir)
    manifest_remote = posixpath.join(remote_dir, "session_manifest.json")
    npz_remote = posixpath.join(remote_dir, "raw_trials.npz")
    ssh.put_file(dataset.manifest_path, manifest_remote)
    ssh.put_file(dataset.npz_path, npz_remote)
    return {"remote_dir": remote_dir, "manifest": manifest_remote, "npz": npz_remote}


def upload_profile(ssh: SSHClient, profile_path: Path, *, run_id: str) -> str:
    profile_path = Path(profile_path).expanduser().resolve()
    if not profile_path.exists():
        raise FileNotFoundError(f"profile not found: {profile_path}")
    remote_dir = assert_remote_ssvep_path(posixpath.join(REMOTE_PROFILE_ROOT, run_id))
    remote_profile = posixpath.join(remote_dir, "pretrained_profile.json")
    ssh.mkdir_p(remote_dir)
    ssh.put_file(profile_path, remote_profile)
    return remote_profile


def build_train_command(
    *,
    task: str,
    dataset_manifest_remote: str,
    dataset_manifest_session2_remote: Optional[str] = None,
    run_id: str,
    pretrained_profile_remote: Optional[str] = None,
    profile_eval_mode: str = "fbcca-vs-all",
    compute_backend: str = DEFAULT_REMOTE_COMPUTE_BACKEND,
    gpu_device: int = DEFAULT_REMOTE_GPU_DEVICE,
    gpu_precision: str = DEFAULT_REMOTE_GPU_PRECISION,
    gpu_warmup: bool = DEFAULT_REMOTE_GPU_WARMUP,
    gpu_cache_policy: str = DEFAULT_REMOTE_GPU_CACHE_POLICY,
    win_candidates: str = DEFAULT_REMOTE_WIN_CANDIDATES,
    multi_seed_count: int = DEFAULT_REMOTE_MULTI_SEED_COUNT,
) -> dict[str, str]:
    task = str(task).strip()
    if task not in {
        "fbcca-weights",
        "profile-eval",
        "model-compare",
        "focused-compare",
        "classifier-compare",
        "fbcca-weighted-compare",
    }:
        raise ValueError(f"unsupported remote task: {task}")
    backend = str(compute_backend or DEFAULT_REMOTE_COMPUTE_BACKEND).strip().lower()
    if backend not in {"cuda", "auto", "cpu"}:
        raise ValueError(f"unsupported compute_backend: {compute_backend}")
    gpu_precision_value = str(gpu_precision or DEFAULT_REMOTE_GPU_PRECISION).strip().lower()
    if gpu_precision_value not in {"float32", "float64"}:
        raise ValueError(f"unsupported gpu_precision: {gpu_precision}")
    gpu_cache_policy_value = str(gpu_cache_policy or DEFAULT_REMOTE_GPU_CACHE_POLICY).strip().lower()
    if gpu_cache_policy_value not in {"windows", "full"}:
        raise ValueError(f"unsupported gpu_cache_policy: {gpu_cache_policy}")
    gpu_warmup_bool = bool(_coerce_bool_flag(gpu_warmup))
    # Physical GPU index selected by user on host machine.
    gpu_device_value = int(gpu_device)
    # We pin one GPU via CUDA_VISIBLE_DEVICES, so inside process it is remapped to device 0.
    # Passing the physical id into CLI would trigger invalid ordinal errors.
    runtime_gpu_device_value = 0
    multi_seed_count_value = max(1, int(multi_seed_count))
    win_candidates_value = str(win_candidates).strip()
    dataset_manifest_remote = assert_remote_ssvep_path(dataset_manifest_remote)
    if dataset_manifest_session2_remote:
        dataset_manifest_session2_remote = assert_remote_ssvep_path(dataset_manifest_session2_remote)
    date_dir = datetime.now().strftime("%Y%m%d")
    report_dir = assert_remote_ssvep_path(posixpath.join(REMOTE_REPORT_ROOT, date_dir, run_id))
    profile_dir = assert_remote_ssvep_path(posixpath.join(REMOTE_PROFILE_ROOT, run_id))
    log_path = assert_remote_ssvep_path(posixpath.join(REMOTE_LOG_DIR, f"{run_id}.log"))
    output_profile = assert_remote_ssvep_path(posixpath.join(profile_dir, "default_profile.json"))
    report_json = assert_remote_ssvep_path(posixpath.join(report_dir, "offline_train_eval.json"))
    args = [
        sh_quote(REMOTE_ENV_PYTHON),
        sh_quote(posixpath.join(REMOTE_CODE_DIR, "ssvep_training_evaluation_cli.py")),
        "--task",
        sh_quote(task),
        "--dataset-manifest",
        sh_quote(dataset_manifest_remote),
        "--report-path",
        sh_quote(report_json),
        "--report-root-dir",
        sh_quote(REMOTE_REPORT_ROOT),
        "--organize-report-dir",
        "0",
        "--output-profile",
        sh_quote(output_profile),
        "--compute-backend",
        sh_quote(backend),
        "--gpu-device",
        str(runtime_gpu_device_value),
        "--gpu-precision",
        sh_quote(gpu_precision_value),
        "--gpu-warmup",
        "1" if gpu_warmup_bool else "0",
        "--gpu-cache-policy",
        sh_quote(gpu_cache_policy_value),
        "--win-candidates",
        sh_quote(win_candidates_value),
        "--multi-seed-count",
        str(multi_seed_count_value),
        "--quick-mode",
        "1" if task == "fbcca-weights" else "0",
    ]
    if dataset_manifest_session2_remote:
        args.extend(
            [
                "--dataset-manifest-session2",
                sh_quote(dataset_manifest_session2_remote),
            ]
        )
    if task == "profile-eval":
        if not pretrained_profile_remote:
            raise ValueError("profile-eval requires pretrained_profile_remote")
        pretrained_profile_remote = assert_remote_ssvep_path(pretrained_profile_remote)
        args.extend(
            [
                "--pretrained-profile",
                sh_quote(pretrained_profile_remote),
                "--profile-eval-mode",
                sh_quote(profile_eval_mode),
                "--freeze-profile-weights",
                "1",
            ]
        )
    script = " ".join(args)
    shell_command = (
        f"mkdir -p {sh_quote(report_dir)} {sh_quote(profile_dir)} {sh_quote(REMOTE_LOG_DIR)} {sh_quote(REMOTE_TMP_DIR)} && "
        f"cd {sh_quote(REMOTE_CODE_DIR)} && "
        # NOTE: `nohup VAR=... cmd` is invalid because nohup treats VAR=... as executable.
        # Use `env VAR=... cmd` so environment assignment is applied to the CLI process.
        f"(nohup env {remote_env_prefix(gpu_device=gpu_device_value)} {script} > {sh_quote(log_path)} 2>&1 < /dev/null & echo $!)"
    )
    return {
        "run_id": run_id,
        "task": task,
        "command": shell_command,
        "log_path": log_path,
        "report_dir": report_dir,
        "report_json": report_json,
        "output_profile": output_profile,
        "dataset_manifest_session1": dataset_manifest_remote,
        "dataset_manifest_session2": "" if not dataset_manifest_session2_remote else dataset_manifest_session2_remote,
        "compute_backend": backend,
        "gpu_device": str(gpu_device_value),
        "gpu_device_runtime": str(runtime_gpu_device_value),
        "gpu_precision": gpu_precision_value,
        "gpu_warmup": "1" if gpu_warmup_bool else "0",
        "gpu_cache_policy": gpu_cache_policy_value,
        "win_candidates": win_candidates_value,
        "multi_seed_count": str(multi_seed_count_value),
    }


def start_remote_task(
    ssh: SSHClient,
    command_payload: dict[str, Any],
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    _code, out, _err = ssh.exec(command_payload["command"], check=True)
    pid = str(out.strip().splitlines()[-1]).strip()
    record = dict(command_payload)
    if metadata:
        record.update(dict(metadata))
    record["pid"] = pid
    record["started_at"] = datetime.now().isoformat(timespec="seconds")
    record.setdefault(
        "remote_manifest_paths",
        {
            "session1": str(command_payload.get("dataset_manifest_session1", "")),
            "session2": str(command_payload.get("dataset_manifest_session2", "")),
        },
    )
    record.setdefault(
        "gpu_params",
        {
            "compute_backend": str(command_payload.get("compute_backend", "")),
            "gpu_device": int(command_payload.get("gpu_device", 0) or 0),
            "gpu_device_runtime": int(command_payload.get("gpu_device_runtime", 0) or 0),
            "gpu_precision": str(command_payload.get("gpu_precision", "")),
            "gpu_warmup": bool(_coerce_bool_flag(command_payload.get("gpu_warmup", "0"))),
            "gpu_cache_policy": str(command_payload.get("gpu_cache_policy", "")),
            "win_candidates": str(command_payload.get("win_candidates", "")),
            "multi_seed_count": int(command_payload.get("multi_seed_count", 1) or 1),
        },
    )
    save_task_record(record)
    return record


def save_task_record(record: dict[str, Any]) -> None:
    records = load_task_records()
    records = [item for item in records if str(item.get("run_id")) != str(record.get("run_id"))]
    records.insert(0, dict(record))
    write_json_atomic(LOCAL_TASK_RECORD_PATH, records[:50])


def load_task_records() -> list[dict[str, Any]]:
    payload = read_json(LOCAL_TASK_RECORD_PATH, [])
    return payload if isinstance(payload, list) else []


def latest_task_record() -> Optional[dict[str, Any]]:
    records = load_task_records()
    return dict(records[0]) if records else None


def read_remote_status(ssh: SSHClient, record: dict[str, Any]) -> dict[str, Any]:
    pid = str(record.get("pid", "")).strip()
    log_path = assert_remote_ssvep_path(str(record.get("log_path", "")))
    report_dir = assert_remote_ssvep_path(str(record.get("report_dir", "")))
    progress_path = posixpath.join(report_dir, "progress_snapshot.json")
    progress: dict[str, Any] = {}
    if ssh.exists(progress_path):
        try:
            with ssh.sftp.open(progress_path, "r") as handle:
                progress = json.loads(handle.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            progress = {"error": str(exc)}
    ps_command = f"test -n {sh_quote(pid)} && ps -p {sh_quote(pid)} -o pid=,stat=,etime=,cmd= || true"
    _code, ps_out, _err = ssh.exec(ps_command, check=False)
    gpu_device = int(record.get("gpu_device", 0) or 0)
    fuser_command = f"fuser /dev/nvidia{gpu_device} 2>/dev/null || true"
    _code, fuser_out, _err = ssh.exec(fuser_command, check=False)
    artifacts = {
        "report_json": ssh.exists(posixpath.join(report_dir, "offline_train_eval.json")),
        "report_md": ssh.exists(posixpath.join(report_dir, "offline_train_eval.md")),
        "profile_best_fbcca_weighted": ssh.exists(posixpath.join(report_dir, "profile_best_fbcca_weighted.json")),
        "progress_snapshot": ssh.exists(progress_path),
    }
    return {
        "run_id": str(record.get("run_id", "")),
        "pid": pid,
        "process": ps_out.strip(),
        "gpu_device_fuser": fuser_out.strip(),
        "log_path": log_path,
        "report_dir": report_dir,
        "progress": progress,
        "tail": ssh.tail_file(log_path, lines=50),
        "artifacts": artifacts,
    }


def _derive_metrics_source(record: dict[str, Any], report_payload: dict[str, Any]) -> str:
    remote_s2 = str(dict(record.get("remote_manifest_paths") or {}).get("session2", "")).strip()
    if not remote_s2:
        return "no_session2"
    cross = report_payload.get("chosen_cross_session_metrics")
    if isinstance(cross, dict) and cross:
        return "cross_session"
    return "session1_holdout"


def _normalize_win_candidates(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        return []
    normalized: list[float] = []
    for value in values:
        try:
            normalized.append(round(float(value), 6))
        except Exception:
            continue
    return normalized


def _build_config_consistency(local_run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    run_config_path = local_run_dir / "run_config.json"
    expected = dict(record.get("requested_config") or {})
    if not run_config_path.exists():
        return {
            "checked": False,
            "consistent": False,
            "reason": "run_config_missing",
            "requested_config": expected,
            "actual_config": {},
            "checks": {},
        }
    try:
        actual = json.loads(run_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "checked": False,
            "consistent": False,
            "reason": f"run_config_unreadable:{exc}",
            "requested_config": expected,
            "actual_config": {},
            "checks": {},
        }
    if not isinstance(actual, dict):
        actual = {}
    checks: dict[str, bool] = {}
    if expected:
        checks["task"] = str(actual.get("task", "")).strip() == str(expected.get("task", "")).strip()
        checks["multi_seed_count"] = int(actual.get("multi_seed_count", 0) or 0) == int(
            expected.get("multi_seed_count", 0) or 0
        )
        checks["win_candidates"] = _normalize_win_candidates(actual.get("win_candidates")) == _normalize_win_candidates(
            expected.get("win_candidates")
        )
        checks["compute_backend"] = str(actual.get("compute_backend", "")).strip() == str(
            expected.get("compute_backend", "")
        ).strip()
        expected_models = [str(item) for item in expected.get("model_names", [])]
        actual_models = [str(item) for item in actual.get("model_names", [])]
        if expected_models:
            checks["model_names"] = actual_models == expected_models
    consistent = all(bool(value) for value in checks.values()) if checks else True
    return {
        "checked": True,
        "consistent": bool(consistent),
        "reason": "" if consistent else "requested_config_mismatch",
        "requested_config": expected,
        "actual_config": {
            "task": str(actual.get("task", "")),
            "multi_seed_count": int(actual.get("multi_seed_count", 0) or 0),
            "win_candidates": _normalize_win_candidates(actual.get("win_candidates")),
            "compute_backend": str(actual.get("compute_backend", "")),
            "model_names": [str(item) for item in actual.get("model_names", [])],
        },
        "checks": checks,
    }


def _stamp_report_metadata(local_run_dir: Path, record: dict[str, Any]) -> None:
    report_path = local_run_dir / "offline_train_eval.json"
    if not report_path.exists():
        return
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    config_consistency = _build_config_consistency(local_run_dir, record)
    payload["metrics_source"] = _derive_metrics_source(record, payload)
    payload["server_gpu_params"] = dict(record.get("gpu_params") or {})
    payload["code_sync"] = dict(record.get("code_sync") or {})
    payload["requested_config"] = dict(record.get("requested_config") or {})
    payload["config_consistency"] = config_consistency
    payload["invalid_run"] = not bool(config_consistency.get("consistent", True))
    payload["remote_path_snapshot"] = {
        "remote_root": REMOTE_ROOT,
        "remote_code_dir": REMOTE_CODE_DIR,
        "remote_report_dir": str(record.get("report_dir", "")),
        "remote_report_json": str(record.get("report_json", "")),
        "remote_log_path": str(record.get("log_path", "")),
        "remote_manifest_paths": dict(record.get("remote_manifest_paths") or {}),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_path = local_run_dir / "server_run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "run_id": str(record.get("run_id", "")),
                "task": str(record.get("task", "")),
                "started_at": str(record.get("started_at", "")),
                "metrics_source": str(payload.get("metrics_source", "")),
                "remote_manifest_paths": dict(record.get("remote_manifest_paths") or {}),
                "gpu_params": dict(record.get("gpu_params") or {}),
                "code_sync": dict(record.get("code_sync") or {}),
                "requested_config": dict(record.get("requested_config") or {}),
                "config_consistency": config_consistency,
                "remote_paths": payload.get("remote_path_snapshot", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def download_results(ssh: SSHClient, record: dict[str, Any]) -> dict[str, Any]:
    run_id = safe_session_id(str(record.get("run_id", "")))
    report_dir = assert_remote_ssvep_path(str(record.get("report_dir", "")))
    local_run_dir = LOCAL_SERVER_RUNS_DIR / run_id
    ssh.download_dir(report_dir, local_run_dir)
    _stamp_report_metadata(local_run_dir, record)
    metadata_payload = read_json(local_run_dir / "server_run_metadata.json", {})
    profile_candidates = [
        local_run_dir / "profile_best_fbcca_weighted.json",
        local_run_dir / "profile_best_candidate.json",
    ]
    profile_candidates.extend(sorted(local_run_dir.glob("*profile*.json")))
    downloaded_profile = ""
    for candidate in profile_candidates:
        if candidate.exists():
            LOCAL_SERVER_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
            target = LOCAL_SERVER_PROFILES_DIR / f"{run_id}__{candidate.name}"
            target.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
            downloaded_profile = str(target)
            break
    return {
        "local_run_dir": str(local_run_dir),
        "local_profile": downloaded_profile,
        "invalid_run": bool(metadata_payload.get("config_consistency", {}).get("consistent", True) is False),
        "config_consistency": dict(metadata_payload.get("config_consistency") or {}),
    }


def require_password(config: ServerConfig) -> None:
    if config.password:
        return
    env_password = os.environ.get("SSVEP_SERVER_PASSWORD", "")
    if env_password:
        config.password = env_password
        return
    config.password = getpass.getpass("SSVEP server password: ")


def _find_dataset_by_manifest(manifest: Optional[Path]) -> LocalDataset:
    datasets = discover_local_datasets()
    if manifest is None:
        if not datasets:
            raise FileNotFoundError(f"no local datasets found under {LOCAL_DATASET_ROOT}")
        return datasets[0]
    manifest = Path(manifest).expanduser().resolve()
    npz_path = manifest.parent / "raw_trials.npz"
    if not manifest.exists() or not npz_path.exists():
        raise FileNotFoundError(f"dataset requires session_manifest.json and raw_trials.npz: {manifest.parent}")
    payload = read_json(manifest, {})
    return LocalDataset(
        session_id=safe_session_id(str(payload.get("session_id") or manifest.parent.name)),
        manifest_path=manifest,
        npz_path=npz_path,
        trial_count=len(payload.get("trials", [])) if isinstance(payload.get("trials", []), list) else 0,
        subject_id=str(payload.get("subject_id", "")),
    )


def run_cli(args: argparse.Namespace) -> int:
    if args.action == "scan":
        for item in discover_local_datasets(Path(args.dataset_root)):
            print(f"{item.session_id}\ttrials={item.trial_count}\t{item.manifest_path}")
        return 0

    config = ServerConfig(
        host=str(args.host),
        port=int(args.port),
        username=str(args.username),
        password=str(args.password or ""),
    )
    require_password(config)
    ssh = SSHClient(config, log_fn=print)
    ssh.connect()
    try:
        if args.action == "status":
            record = latest_task_record()
            if record is None:
                raise RuntimeError("no local server task record found")
            print(json.dumps(read_remote_status(ssh, record), ensure_ascii=False, indent=2))
            return 0
        if args.action == "download":
            record = latest_task_record()
            if record is None:
                raise RuntimeError("no local server task record found")
            print(json.dumps(download_results(ssh, record), ensure_ascii=False, indent=2))
            return 0

        dataset = _find_dataset_by_manifest(args.dataset_manifest)
        dataset_session2 = (
            _find_dataset_by_manifest(args.dataset_manifest_session2)
            if args.dataset_manifest_session2 is not None
            else None
        )
        remote_dataset = upload_dataset(ssh, dataset)
        remote_dataset_session2 = None
        if dataset_session2 is not None:
            if dataset_session2.manifest_path.resolve() != dataset.manifest_path.resolve():
                remote_dataset_session2 = upload_dataset(ssh, dataset_session2)
        if args.action == "upload":
            print(
                json.dumps(
                    {
                        "session1": remote_dataset,
                        "session2": remote_dataset_session2,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        run_id = safe_session_id(args.run_id or now_run_id(args.action))
        pretrained_profile_remote = None
        if args.action == "profile-eval":
            if args.pretrained_profile is None:
                raise ValueError("--pretrained-profile is required for profile-eval")
            pretrained_profile_remote = upload_profile(ssh, Path(args.pretrained_profile), run_id=run_id)
        task = {
            "train-weights": "fbcca-weights",
            "fbcca-weighted-compare": "fbcca-weighted-compare",
            "profile-eval": "profile-eval",
            "model-compare": "model-compare",
            "focused-compare": "focused-compare",
            "classifier-compare": "classifier-compare",
        }[args.action]
        gpu_params = {
            "compute_backend": str(args.compute_backend),
            "gpu_device": int(args.gpu_device),
            "gpu_precision": str(args.gpu_precision),
            "gpu_warmup": bool(int(args.gpu_warmup)),
            "gpu_cache_policy": str(args.gpu_cache_policy),
            "win_candidates": str(args.win_candidates),
            "multi_seed_count": int(args.multi_seed_count),
        }
        code_sync = sync_local_code_tree(ssh)
        preflight = preflight_cuda_or_fail(
            ssh,
            compute_backend=str(args.compute_backend),
            gpu_device=int(args.gpu_device),
        )
        payload = build_train_command(
            task=task,
            dataset_manifest_remote=remote_dataset["manifest"],
            dataset_manifest_session2_remote=(
                None if remote_dataset_session2 is None else remote_dataset_session2["manifest"]
            ),
            run_id=run_id,
            pretrained_profile_remote=pretrained_profile_remote,
            profile_eval_mode=str(args.profile_eval_mode),
            compute_backend=str(args.compute_backend),
            gpu_device=int(args.gpu_device),
            gpu_precision=str(args.gpu_precision),
            gpu_warmup=bool(int(args.gpu_warmup)),
            gpu_cache_policy=str(args.gpu_cache_policy),
            win_candidates=str(args.win_candidates),
            multi_seed_count=int(args.multi_seed_count),
        )
        record = start_remote_task(
            ssh,
            payload,
            metadata={
                "session1": str(dataset.manifest_path),
                "session2": "" if dataset_session2 is None else str(dataset_session2.manifest_path),
                "remote_manifest_paths": {
                    "session1": str(remote_dataset.get("manifest", "")),
                    "session2": (
                        ""
                        if remote_dataset_session2 is None
                        else str(remote_dataset_session2.get("manifest", ""))
                    ),
                },
                "gpu_params": gpu_params,
                "gpu_preflight": preflight,
                "code_sync": code_sync,
                "requested_config": {
                    "task": task,
                    "multi_seed_count": int(args.multi_seed_count),
                    "win_candidates": _normalize_win_candidates(args.win_candidates),
                    "compute_backend": str(args.compute_backend),
                },
            },
        )
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    finally:
        ssh.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSVEP server training helper")
    parser.add_argument(
        "--action",
        choices=[
            "gui",
            "scan",
            "upload",
            "train-weights",
            "fbcca-weighted-compare",
            "profile-eval",
            "model-compare",
            "focused-compare",
            "classifier-compare",
            "status",
            "download",
        ],
        default="gui",
    )
    parser.add_argument("--host", default="10.72.128.221")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--username", default="zhangkexin")
    parser.add_argument("--password", default="")
    parser.add_argument("--dataset-root", type=Path, default=LOCAL_DATASET_ROOT)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--dataset-manifest-session2", type=Path, default=None)
    parser.add_argument("--pretrained-profile", type=Path, default=None)
    parser.add_argument("--profile-eval-mode", choices=["fbcca-vs-all", "fbcca-only"], default="fbcca-vs-all")
    parser.add_argument("--compute-backend", default=DEFAULT_REMOTE_COMPUTE_BACKEND, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gpu-device", type=int, default=DEFAULT_REMOTE_GPU_DEVICE)
    parser.add_argument("--gpu-precision", default=DEFAULT_REMOTE_GPU_PRECISION, choices=["float32", "float64"])
    parser.add_argument("--gpu-warmup", type=int, default=1 if DEFAULT_REMOTE_GPU_WARMUP else 0)
    parser.add_argument("--gpu-cache-policy", default=DEFAULT_REMOTE_GPU_CACHE_POLICY, choices=["windows", "full"])
    parser.add_argument("--win-candidates", default=DEFAULT_REMOTE_WIN_CANDIDATES)
    parser.add_argument("--multi-seed-count", type=int, default=DEFAULT_REMOTE_MULTI_SEED_COUNT)
    parser.add_argument("--run-id", default="")
    return parser


def launch_gui() -> int:
    try:
        from PyQt5 import QtCore, QtWidgets
    except Exception as exc:  # pragma: no cover - UI dependency
        raise RuntimeError("PyQt5 is required for the server helper UI") from exc

    class ServerTrainWindow(QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("SSVEP Server Train/Eval Helper")
            self.resize(980, 720)
            self.records: list[dict[str, Any]] = load_task_records()
            self._build_ui()
            self._refresh_datasets()
            self._load_latest_record()

        def _build_ui(self) -> None:
            layout = QtWidgets.QVBoxLayout(self)
            form = QtWidgets.QFormLayout()
            self.host_edit = QtWidgets.QLineEdit("10.72.128.221")
            self.user_edit = QtWidgets.QLineEdit("zhangkexin")
            self.password_edit = QtWidgets.QLineEdit()
            self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)
            self.dataset_combo = QtWidgets.QComboBox()
            self.dataset_session2_combo = QtWidgets.QComboBox()
            self.profile_edit = QtWidgets.QLineEdit()
            self.profile_eval_mode = QtWidgets.QComboBox()
            self.profile_eval_mode.addItems(["fbcca-vs-all", "fbcca-only"])
            self.compute_backend_combo = QtWidgets.QComboBox()
            self.compute_backend_combo.addItems(["cuda", "auto", "cpu"])
            self.compute_backend_combo.setCurrentText(DEFAULT_REMOTE_COMPUTE_BACKEND)
            self.gpu_device_edit = QtWidgets.QLineEdit(str(DEFAULT_REMOTE_GPU_DEVICE))
            self.gpu_precision_combo = QtWidgets.QComboBox()
            self.gpu_precision_combo.addItems(["float32", "float64"])
            self.gpu_precision_combo.setCurrentText(DEFAULT_REMOTE_GPU_PRECISION)
            self.gpu_warmup_check = QtWidgets.QCheckBox("GPU warmup")
            self.gpu_warmup_check.setChecked(bool(DEFAULT_REMOTE_GPU_WARMUP))
            self.gpu_cache_combo = QtWidgets.QComboBox()
            self.gpu_cache_combo.addItems(["windows", "full"])
            self.gpu_cache_combo.setCurrentText(DEFAULT_REMOTE_GPU_CACHE_POLICY)
            self.win_candidates_edit = QtWidgets.QLineEdit(DEFAULT_REMOTE_WIN_CANDIDATES)
            self.multi_seed_spin = QtWidgets.QSpinBox()
            self.multi_seed_spin.setRange(1, 20)
            self.multi_seed_spin.setValue(int(DEFAULT_REMOTE_MULTI_SEED_COUNT))

            form.addRow("Server Host", self.host_edit)
            form.addRow("Username", self.user_edit)
            form.addRow("Password (not saved)", self.password_edit)
            form.addRow("Dataset Session1", self.dataset_combo)
            form.addRow("Dataset Session2 (optional)", self.dataset_session2_combo)
            form.addRow("Pretrained Profile", self.profile_edit)
            form.addRow("Profile Eval Mode", self.profile_eval_mode)
            form.addRow("Compute Backend", self.compute_backend_combo)
            form.addRow("GPU Device", self.gpu_device_edit)
            form.addRow("GPU Precision", self.gpu_precision_combo)
            form.addRow("GPU Cache Policy", self.gpu_cache_combo)
            form.addRow("Win Candidates", self.win_candidates_edit)
            form.addRow("Multi Seed Count", self.multi_seed_spin)
            form.addRow("", self.gpu_warmup_check)
            layout.addLayout(form)

            row = QtWidgets.QHBoxLayout()
            buttons = [
                ("Refresh Datasets", self._refresh_datasets),
                ("Test Connection", self._connect_test),
                ("Upload Dataset(s)", self._upload_only),
                ("Train FBCCA Weights", self._train_weights),
                ("Profile Eval", self._profile_eval),
                ("Focused Compare", self._focused_compare),
                ("Classifier Compare", self._classifier_compare),
                ("Model Compare", self._model_compare),
                ("Refresh Status", self._refresh_status),
                ("Download Results", self._download_results),
            ]
            for label, slot in buttons:
                btn = QtWidgets.QPushButton(label)
                btn.clicked.connect(slot)
                row.addWidget(btn)
            layout.addLayout(row)

            self.status_label = QtWidgets.QLabel("Not connected")
            layout.addWidget(self.status_label)
            self.log_box = QtWidgets.QPlainTextEdit()
            self.log_box.setReadOnly(True)
            layout.addWidget(self.log_box, 1)
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(5000)
            self.timer.timeout.connect(self._refresh_status)
            self.timer.start()

        def _append(self, text: str) -> None:
            self.log_box.appendPlainText(str(text))

        def _refresh_datasets(self) -> None:
            self.dataset_combo.clear()
            self.dataset_session2_combo.clear()
            self.dataset_session2_combo.addItem("(none)", "")
            datasets = discover_local_datasets()
            for item in datasets:
                label = f"{item.session_id} | trials={item.trial_count} | subject={item.subject_id}"
                self.dataset_combo.addItem(label, str(item.manifest_path))
                self.dataset_session2_combo.addItem(label, str(item.manifest_path))
            if self.dataset_combo.count() > 0:
                self.dataset_combo.setCurrentIndex(0)
            self.dataset_session2_combo.setCurrentIndex(0)

        def _selected_dataset(self) -> LocalDataset:
            path = self.dataset_combo.currentData()
            if not path:
                raise RuntimeError("no available dataset")
            return _find_dataset_by_manifest(Path(path))

        def _selected_dataset_session2(self) -> Optional[LocalDataset]:
            path = self.dataset_session2_combo.currentData()
            if not path:
                return None
            return _find_dataset_by_manifest(Path(path))

        def _server_config(self) -> ServerConfig:
            return ServerConfig(
                host=self.host_edit.text().strip(),
                username=self.user_edit.text().strip(),
                password=self.password_edit.text(),
            )

        def _gpu_args(self) -> dict[str, Any]:
            return {
                "compute_backend": str(self.compute_backend_combo.currentText()).strip(),
                "gpu_device": int(self.gpu_device_edit.text().strip() or "0"),
                "gpu_precision": str(self.gpu_precision_combo.currentText()).strip(),
                "gpu_warmup": bool(self.gpu_warmup_check.isChecked()),
                "gpu_cache_policy": str(self.gpu_cache_combo.currentText()).strip(),
                "win_candidates": str(self.win_candidates_edit.text().strip() or DEFAULT_REMOTE_WIN_CANDIDATES),
                "multi_seed_count": int(self.multi_seed_spin.value()),
            }

        def _with_ssh(self, fn: Callable[[SSHClient], Any]) -> Any:
            cfg = self._server_config()
            if not cfg.password:
                raise RuntimeError("please input server password (not persisted)")
            ssh = SSHClient(cfg)
            ssh.connect()
            try:
                return fn(ssh)
            finally:
                ssh.close()

        def _connect_test(self) -> None:
            try:
                def work(ssh: SSHClient) -> str:
                    check_cmd = (
                        f"test -d {sh_quote(REMOTE_ROOT)} && echo ROOT_OK || echo ROOT_MISSING; "
                        "command -v nvidia-smi >/dev/null 2>&1 && "
                        "nvidia-smi -L | head -n 4 || "
                        "echo NVIDIA_SMI_MISSING"
                    )
                    _code, out, _err = ssh.exec(check_cmd, check=False)
                    return out.strip()

                out = self._with_ssh(work)
                self.status_label.setText(f"Connection: {out}")
                self._append(f"Connection check: {out}")
            except Exception as exc:
                self._append(f"Connection failed: {exc}")

        def _upload_only(self) -> None:
            try:
                dataset = self._selected_dataset()
                dataset_session2 = self._selected_dataset_session2()

                def work(ssh: SSHClient) -> dict[str, Any]:
                    payload: dict[str, Any] = {
                        "session1": upload_dataset(ssh, dataset),
                        "session2": None,
                    }
                    if dataset_session2 is not None and dataset_session2.manifest_path.resolve() != dataset.manifest_path.resolve():
                        payload["session2"] = upload_dataset(ssh, dataset_session2)
                    return payload

                result = self._with_ssh(work)
                self._append("Upload done:\n" + json.dumps(result, ensure_ascii=False, indent=2))
            except Exception as exc:
                self._append(f"Upload failed: {exc}")

        def _start_task(self, action: str) -> None:
            try:
                dataset = self._selected_dataset()
                dataset_session2 = self._selected_dataset_session2()
                run_id = now_run_id(action)
                profile_remote = None
                gpu_args = self._gpu_args()

                def work(ssh: SSHClient) -> dict[str, Any]:
                    nonlocal profile_remote
                    code_sync = sync_local_code_tree(ssh)
                    remote_dataset = upload_dataset(ssh, dataset)
                    remote_dataset_session2 = None
                    if dataset_session2 is not None and dataset_session2.manifest_path.resolve() != dataset.manifest_path.resolve():
                        remote_dataset_session2 = upload_dataset(ssh, dataset_session2)
                    preflight = preflight_cuda_or_fail(
                        ssh,
                        compute_backend=str(gpu_args["compute_backend"]),
                        gpu_device=int(gpu_args["gpu_device"]),
                    )
                    if action == "profile-eval":
                        profile_path = Path(self.profile_edit.text().strip()).expanduser()
                        profile_remote = upload_profile(ssh, profile_path, run_id=run_id)
                    task = {
                        "train-weights": "fbcca-weights",
                        "fbcca-weighted-compare": "fbcca-weighted-compare",
                        "profile-eval": "profile-eval",
                        "model-compare": "model-compare",
                        "focused-compare": "focused-compare",
                        "classifier-compare": "classifier-compare",
                    }[action]
                    payload = build_train_command(
                        task=task,
                        dataset_manifest_remote=remote_dataset["manifest"],
                        dataset_manifest_session2_remote=(
                            None if remote_dataset_session2 is None else remote_dataset_session2["manifest"]
                        ),
                        run_id=run_id,
                        pretrained_profile_remote=profile_remote,
                        profile_eval_mode=self.profile_eval_mode.currentText(),
                        compute_backend=str(gpu_args["compute_backend"]),
                        gpu_device=int(gpu_args["gpu_device"]),
                        gpu_precision=str(gpu_args["gpu_precision"]),
                        gpu_warmup=bool(gpu_args["gpu_warmup"]),
                        gpu_cache_policy=str(gpu_args["gpu_cache_policy"]),
                        win_candidates=str(gpu_args["win_candidates"]),
                        multi_seed_count=int(gpu_args["multi_seed_count"]),
                    )
                    return start_remote_task(
                        ssh,
                        payload,
                        metadata={
                            "session1": str(dataset.manifest_path),
                            "session2": "" if dataset_session2 is None else str(dataset_session2.manifest_path),
                            "remote_manifest_paths": {
                                "session1": str(remote_dataset.get("manifest", "")),
                                "session2": (
                                    ""
                                    if remote_dataset_session2 is None
                                    else str(remote_dataset_session2.get("manifest", ""))
                                ),
                            },
                            "gpu_params": dict(gpu_args),
                            "gpu_preflight": preflight,
                            "code_sync": code_sync,
                            "requested_config": {
                                "task": task,
                                "multi_seed_count": int(gpu_args["multi_seed_count"]),
                                "win_candidates": _normalize_win_candidates(gpu_args["win_candidates"]),
                                "compute_backend": str(gpu_args["compute_backend"]),
                            },
                        },
                    )

                record = self._with_ssh(work)
                self.records = load_task_records()
                self._append("Remote task started:\n" + json.dumps(record, ensure_ascii=False, indent=2))
            except Exception as exc:
                self._append(f"Task start failed: {exc}")

        def _train_weights(self) -> None:
            self._start_task("train-weights")

        def _profile_eval(self) -> None:
            self._start_task("profile-eval")

        def _model_compare(self) -> None:
            self._start_task("model-compare")

        def _focused_compare(self) -> None:
            self._start_task("focused-compare")

        def _classifier_compare(self) -> None:
            self._start_task("classifier-compare")

        def _load_latest_record(self) -> None:
            record = latest_task_record()
            if record:
                self.status_label.setText(f"Latest task: {record.get('run_id')} pid={record.get('pid')}")

        def _refresh_status(self) -> None:
            record = latest_task_record()
            if not record or not self.password_edit.text():
                return
            try:
                def work(ssh: SSHClient) -> dict[str, Any]:
                    return read_remote_status(ssh, record)

                status = self._with_ssh(work)
                progress = status.get("progress", {}) if isinstance(status.get("progress"), dict) else {}
                self.status_label.setText(
                    f"run={status.get('run_id')} pid={status.get('pid')} "
                    f"stage={progress.get('stage','')} model={progress.get('model_name','')}"
                )
                self.log_box.setPlainText(
                    json.dumps(
                        {
                            "process": status.get("process", ""),
                            "gpu_device_fuser": status.get("gpu_device_fuser", ""),
                            "progress": progress,
                            "artifacts": status.get("artifacts", {}),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n\nlatest log tail\n"
                    + str(status.get("tail", ""))
                )
            except Exception as exc:
                self._append(f"Refresh status failed: {exc}")

        def _download_results(self) -> None:
            record = latest_task_record()
            if not record:
                self._append("No local task record")
                return
            try:
                def work(ssh: SSHClient) -> dict[str, str]:
                    return download_results(ssh, record)

                result = self._with_ssh(work)
                self._append("Download done:\n" + json.dumps(result, ensure_ascii=False, indent=2))
            except Exception as exc:
                self._append(f"Download failed: {exc}")

    app = QtWidgets.QApplication(sys.argv)
    window = ServerTrainWindow()
    window.show()
    return int(app.exec_())


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "gui":
        return launch_gui()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
