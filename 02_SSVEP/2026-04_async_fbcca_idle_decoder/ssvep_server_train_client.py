from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import getpass
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

REMOTE_ROOT = "/data1/zkx/brain/ssvep"
REMOTE_CODE_DIR = f"{REMOTE_ROOT}/code/2026-04_async_fbcca_idle_decoder"
REMOTE_DATA_DIR = f"{REMOTE_ROOT}/data"
REMOTE_REPORT_ROOT = f"{REMOTE_ROOT}/reports"
REMOTE_PROFILE_ROOT = f"{REMOTE_ROOT}/profiles"
REMOTE_LOG_DIR = f"{REMOTE_ROOT}/logs"
REMOTE_TMP_DIR = f"{REMOTE_ROOT}/tmp"
REMOTE_ENV_PYTHON = "/data1/zkx/miniconda3/envs/brain-ssvep/bin/python"


def assert_remote_ssvep_path(path: str) -> str:
    normalized = posixpath.normpath(str(path).replace("\\", "/"))
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
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.config.host,
            port=int(self.config.port),
            username=self.config.username,
            password=self.config.password,
            timeout=12,
            look_for_keys=False,
            allow_agent=False,
        )
        self._client = client
        self._sftp = client.open_sftp()

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


def remote_env_prefix() -> str:
    env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "CUPY_CACHE_DIR": f"{REMOTE_LOG_DIR}/cupy_cache",
        "TMPDIR": REMOTE_TMP_DIR,
        "MPLCONFIGDIR": f"{REMOTE_LOG_DIR}/matplotlib",
        "PYTHONPYCACHEPREFIX": f"{REMOTE_LOG_DIR}/pycache",
    }
    return " ".join(f"{key}={sh_quote(value)}" for key, value in env.items())


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
    run_id: str,
    pretrained_profile_remote: Optional[str] = None,
    profile_eval_mode: str = "fbcca-vs-all",
) -> dict[str, str]:
    task = str(task).strip()
    if task not in {"fbcca-weights", "profile-eval", "model-compare", "focused-compare", "classifier-compare"}:
        raise ValueError(f"unsupported remote task: {task}")
    dataset_manifest_remote = assert_remote_ssvep_path(dataset_manifest_remote)
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
        "cuda",
        "--gpu-device",
        "0",
        "--gpu-precision",
        "float32",
        "--quick-mode",
        "1" if task == "fbcca-weights" else "0",
    ]
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
        f"(nohup {remote_env_prefix()} {script} > {sh_quote(log_path)} 2>&1 < /dev/null & echo $!)"
    )
    return {
        "run_id": run_id,
        "task": task,
        "command": shell_command,
        "log_path": log_path,
        "report_dir": report_dir,
        "report_json": report_json,
        "output_profile": output_profile,
    }


def start_remote_task(ssh: SSHClient, command_payload: dict[str, str]) -> dict[str, Any]:
    _code, out, _err = ssh.exec(command_payload["command"], check=True)
    pid = str(out.strip().splitlines()[-1]).strip()
    record = dict(command_payload)
    record["pid"] = pid
    record["started_at"] = datetime.now().isoformat(timespec="seconds")
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
    fuser_command = "fuser /dev/nvidia0 2>/dev/null || true"
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


def download_results(ssh: SSHClient, record: dict[str, Any]) -> dict[str, str]:
    run_id = safe_session_id(str(record.get("run_id", "")))
    report_dir = assert_remote_ssvep_path(str(record.get("report_dir", "")))
    local_run_dir = LOCAL_SERVER_RUNS_DIR / run_id
    ssh.download_dir(report_dir, local_run_dir)
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
    return {"local_run_dir": str(local_run_dir), "local_profile": downloaded_profile}


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
        remote_dataset = upload_dataset(ssh, dataset)
        if args.action == "upload":
            print(json.dumps(remote_dataset, ensure_ascii=False, indent=2))
            return 0

        run_id = safe_session_id(args.run_id or now_run_id(args.action))
        pretrained_profile_remote = None
        if args.action == "profile-eval":
            if args.pretrained_profile is None:
                raise ValueError("--pretrained-profile is required for profile-eval")
            pretrained_profile_remote = upload_profile(ssh, Path(args.pretrained_profile), run_id=run_id)
        task = {
            "train-weights": "fbcca-weights",
            "profile-eval": "profile-eval",
            "model-compare": "model-compare",
            "focused-compare": "focused-compare",
            "classifier-compare": "classifier-compare",
        }[args.action]
        payload = build_train_command(
            task=task,
            dataset_manifest_remote=remote_dataset["manifest"],
            run_id=run_id,
            pretrained_profile_remote=pretrained_profile_remote,
            profile_eval_mode=str(args.profile_eval_mode),
        )
        record = start_remote_task(ssh, payload)
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
    parser.add_argument("--pretrained-profile", type=Path, default=None)
    parser.add_argument("--profile-eval-mode", choices=["fbcca-vs-all", "fbcca-only"], default="fbcca-vs-all")
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
            self.setWindowTitle("SSVEP服务器训练助手")
            self.resize(980, 720)
            self.ssh: Optional[SSHClient] = None
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
            self.profile_edit = QtWidgets.QLineEdit()
            self.profile_eval_mode = QtWidgets.QComboBox()
            self.profile_eval_mode.addItems(["fbcca-vs-all", "fbcca-only"])
            form.addRow("服务器", self.host_edit)
            form.addRow("账号", self.user_edit)
            form.addRow("密码(不保存)", self.password_edit)
            form.addRow("本地数据集", self.dataset_combo)
            form.addRow("已训练Profile", self.profile_edit)
            form.addRow("权重评测模式", self.profile_eval_mode)
            layout.addLayout(form)

            row = QtWidgets.QHBoxLayout()
            buttons = [
                ("刷新数据集", self._refresh_datasets),
                ("连接测试", self._connect_test),
                ("上传数据集", self._upload_only),
                ("只训练FBCCA权重", self._train_weights),
                ("读取权重评测", self._profile_eval),
                ("精选模型深度分析", self._focused_compare),
                ("全量分类对比", self._classifier_compare),
                ("全模型端到端对比", self._model_compare),
                ("查看服务器进度", self._refresh_status),
                ("下载结果", self._download_results),
            ]
            for label, slot in buttons:
                btn = QtWidgets.QPushButton(label)
                btn.clicked.connect(slot)
                row.addWidget(btn)
            layout.addLayout(row)
            self.status_label = QtWidgets.QLabel("未连接")
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
            for item in discover_local_datasets():
                self.dataset_combo.addItem(
                    f"{item.session_id} | trials={item.trial_count} | subject={item.subject_id}",
                    str(item.manifest_path),
                )

        def _selected_dataset(self) -> LocalDataset:
            path = self.dataset_combo.currentData()
            if not path:
                raise RuntimeError("没有可用数据集")
            return _find_dataset_by_manifest(Path(path))

        def _server_config(self) -> ServerConfig:
            return ServerConfig(
                host=self.host_edit.text().strip(),
                username=self.user_edit.text().strip(),
                password=self.password_edit.text(),
            )

        def _with_ssh(self, fn: Callable[[SSHClient], Any]) -> Any:
            cfg = self._server_config()
            if not cfg.password:
                raise RuntimeError("请输入服务器密码；密码不会保存")
            ssh = SSHClient(cfg)
            ssh.connect()
            try:
                return fn(ssh)
            finally:
                ssh.close()

        def _connect_test(self) -> None:
            try:
                def work(ssh: SSHClient) -> str:
                    _code, out, _err = ssh.exec(f"test -d {sh_quote(REMOTE_ROOT)} && echo OK || echo MISSING")
                    return out.strip()
                out = self._with_ssh(work)
                self.status_label.setText(f"连接状态: {out}")
                self._append(f"连接测试: {out}")
            except Exception as exc:
                self._append(f"连接失败: {exc}")

        def _upload_only(self) -> None:
            try:
                dataset = self._selected_dataset()
                def work(ssh: SSHClient) -> dict[str, str]:
                    return upload_dataset(ssh, dataset)
                result = self._with_ssh(work)
                self._append("上传完成:\n" + json.dumps(result, ensure_ascii=False, indent=2))
            except Exception as exc:
                self._append(f"上传失败: {exc}")

        def _start_task(self, action: str) -> None:
            try:
                dataset = self._selected_dataset()
                run_id = now_run_id(action)
                profile_remote = None
                def work(ssh: SSHClient) -> dict[str, Any]:
                    nonlocal profile_remote
                    remote_dataset = upload_dataset(ssh, dataset)
                    if action == "profile-eval":
                        profile_path = Path(self.profile_edit.text().strip()).expanduser()
                        profile_remote = upload_profile(ssh, profile_path, run_id=run_id)
                    task = {
                        "train-weights": "fbcca-weights",
                        "profile-eval": "profile-eval",
                        "model-compare": "model-compare",
                        "focused-compare": "focused-compare",
                        "classifier-compare": "classifier-compare",
                    }[action]
                    payload = build_train_command(
                        task=task,
                        dataset_manifest_remote=remote_dataset["manifest"],
                        run_id=run_id,
                        pretrained_profile_remote=profile_remote,
                        profile_eval_mode=self.profile_eval_mode.currentText(),
                    )
                    return start_remote_task(ssh, payload)
                record = self._with_ssh(work)
                self.records = load_task_records()
                self._append("远程任务已启动:\n" + json.dumps(record, ensure_ascii=False, indent=2))
            except Exception as exc:
                self._append(f"任务启动失败: {exc}")

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
                self.status_label.setText(f"最近任务: {record.get('run_id')} pid={record.get('pid')}")

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
                    + "\n\n最近日志:\n"
                    + str(status.get("tail", ""))
                )
            except Exception as exc:
                self._append(f"刷新进度失败: {exc}")

        def _download_results(self) -> None:
            record = latest_task_record()
            if not record:
                self._append("没有本地任务记录")
                return
            try:
                def work(ssh: SSHClient) -> dict[str, str]:
                    return download_results(ssh, record)
                result = self._with_ssh(work)
                self._append("下载完成:\n" + json.dumps(result, ensure_ascii=False, indent=2))
            except Exception as exc:
                self._append(f"下载失败: {exc}")

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
