from __future__ import annotations

import argparse
import json
import os
import posixpath
import shlex
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SSVEP_ROOT = REPO_ROOT / "02_SSVEP"
if str(SSVEP_ROOT) not in sys.path:
    sys.path.insert(0, str(SSVEP_ROOT))

from tools.server_train_client import SSHClient, ServerConfig, assert_remote_ssvep_path


LOCAL_ROOT = REPO_ROOT / "datasets" / "SSVEP" / "external" / "YSU_an"
LOCAL_MANIFEST = LOCAL_ROOT / "_metadata" / "download_manifest.json"
REMOTE_ROOT = "/data1/zkx/brain/ssvep/data/external_sources/ysu_an/raw"
REMOTE_METADATA_DIR = f"{REMOTE_ROOT}/_metadata"


def q(value: str) -> str:
    return shlex.quote(str(value))


def load_manifest() -> list[dict[str, object]]:
    payload = json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"download manifest must be a list: {LOCAL_MANIFEST}")
    return [dict(item) for item in payload]


def remote_file_ok(ssh: SSHClient, remote_path: str, expected_size: int, expected_md5: str) -> bool:
    remote_path = assert_remote_ssvep_path(remote_path)
    command = (
        f"test -f {q(remote_path)} "
        f"&& test $(stat -c %s {q(remote_path)}) -eq {int(expected_size)} "
        f"&& test $(md5sum {q(remote_path)} | awk '{{print $1}}') = {q(expected_md5)}"
    )
    code, _out, _err = ssh.exec(command, check=False)
    return code == 0


def put_with_temp(ssh: SSHClient, local_path: Path, remote_path: str) -> None:
    remote_path = assert_remote_ssvep_path(remote_path)
    tmp_path = assert_remote_ssvep_path(remote_path + ".uploading")
    ssh.mkdir_p(posixpath.dirname(remote_path))
    if ssh.exists(tmp_path):
        ssh.remove_file(tmp_path)
    ssh.put_file(local_path, tmp_path)
    ssh.exec(f"mv -f {q(tmp_path)} {q(remote_path)}")


def upload_json_text(ssh: SSHClient, local_path: Path, remote_path: str) -> None:
    remote_path = assert_remote_ssvep_path(remote_path)
    ssh.write_text(remote_path, local_path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--password", default=os.environ.get("SSVEP_SERVER_PASSWORD", ""))
    parser.add_argument("--host", default="10.72.128.221")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--username", default="zhangkexin")
    args = parser.parse_args()
    if not args.password:
        raise SystemExit("missing --password or SSVEP_SERVER_PASSWORD")

    manifest = load_manifest()
    cfg = ServerConfig(host=args.host, port=args.port, username=args.username, password=args.password)
    ssh = SSHClient(cfg, log_fn=lambda text: print(text, flush=True))
    ssh.connect()
    try:
        ssh.exec(
            "set -e; "
            f"mkdir -p {q(REMOTE_ROOT)} {q(REMOTE_METADATA_DIR)}; "
            f"test -w {q(REMOTE_ROOT)}; "
            f"df -h {q(REMOTE_ROOT)}"
        )

        upload_json_text(
            ssh,
            LOCAL_ROOT / "_metadata" / "figshare_article_24906300_v3.json",
            f"{REMOTE_METADATA_DIR}/figshare_article_24906300_v3.json",
        )
        upload_json_text(ssh, LOCAL_MANIFEST, f"{REMOTE_METADATA_DIR}/download_manifest.json")

        total = len(manifest)
        for index, item in enumerate(manifest, start=1):
            name = str(item["name"])
            local_path = LOCAL_ROOT / name
            remote_path = f"{REMOTE_ROOT}/{name}"
            expected_size = int(item["size_bytes"])
            expected_md5 = str(item["md5"])
            if not local_path.is_file():
                raise FileNotFoundError(local_path)
            if remote_file_ok(ssh, remote_path, expected_size, expected_md5):
                print(f"[{index}/{total}] SKIP remote-ok {name}", flush=True)
                continue
            print(f"[{index}/{total}] UPLOAD {name} ({expected_size} bytes)", flush=True)
            start = time.time()
            put_with_temp(ssh, local_path, remote_path)
            elapsed = max(time.time() - start, 0.001)
            mbps = expected_size / elapsed / 1024 / 1024
            print(f"[{index}/{total}] UPLOADED {name} in {elapsed:.1f}s ({mbps:.2f} MiB/s)", flush=True)
            if not remote_file_ok(ssh, remote_path, expected_size, expected_md5):
                raise RuntimeError(f"remote validation failed after upload: {remote_path}")

        remote_verify = f"{REMOTE_METADATA_DIR}/verify_ysu_an.py"
        verify_code = f"""
import hashlib
import json
from pathlib import Path
root = Path({REMOTE_ROOT!r})
manifest = json.loads((root / "_metadata" / "download_manifest.json").read_text(encoding="utf-8-sig"))
rows = []
ok = True
for item in manifest:
    path = root / item["name"]
    h = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    got = h.hexdigest()
    row = {{
        "name": item["name"],
        "size": path.stat().st_size,
        "expected_size": int(item["size_bytes"]),
        "md5": got,
        "expected_md5": item["md5"],
        "ok": path.stat().st_size == int(item["size_bytes"]) and got == item["md5"],
    }}
    ok = ok and row["ok"]
    rows.append(row)
summary = {{
    "file_count": len(rows),
    "total_size_bytes": sum(row["size"] for row in rows),
    "all_ok": ok,
    "rows": rows,
}}
(root / "_metadata" / "remote_verify_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({{"file_count": summary["file_count"], "total_size_bytes": summary["total_size_bytes"], "all_ok": summary["all_ok"]}}, ensure_ascii=False))
raise SystemExit(0 if ok else 1)
"""
        ssh.write_text(remote_verify, verify_code, encoding="utf-8")
        code, out, err = ssh.exec(f"/data1/zkx/miniconda3/envs/brain-ssvep/bin/python {q(remote_verify)}")
        print(out.strip(), flush=True)
        if err:
            print(err, file=sys.stderr, flush=True)
        return int(code)
    finally:
        ssh.close()


if __name__ == "__main__":
    raise SystemExit(main())
