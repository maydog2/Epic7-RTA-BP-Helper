"""Local Windows patch detection and staged file replacement.

The packaged app checks for ``E7_BP_Helper_patch_*.zip`` beside the executable
before importing TensorFlow. Validation uses only the standard library. File
replacement runs after this process exits, because Windows locks the running
executable and loaded extension modules.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .runtime_paths import is_frozen

PATCH_PROTOCOL = 1
MANIFEST_PROTOCOL = 2
VERSION_FILENAME = "windows_version.json"
APP_MANIFEST_FILENAME = "app_manifest.json"
PATCH_ZIP_PREFIX = "E7_BP_Helper_patch_"
UPDATE_ZIP_PREFIX = "E7_BP_Helper_update_"
PATCH_ZIP_SUFFIX = ".zip"
APPLIED_DIRNAME = "patches_applied"
REJECTED_DIRNAME = "patches_rejected"
STAGING_DIRNAME = ".patch_staging"
BACKUP_DIRNAME = ".patch_backup"
ERROR_LOG_NAME = "windows_patch_error.log"
HISTORY_NAME = "history.json"
EXE_NAME = "E7_BP_Helper.exe"
IGNORED_DIR_NAMES = {
    APPLIED_DIRNAME,
    REJECTED_DIRNAME,
    STAGING_DIRNAME,
    BACKUP_DIRNAME,
}
IGNORED_FILE_NAMES = {
    "recommender_service_port.txt",
    ERROR_LOG_NAME,
    HISTORY_NAME,
    APP_MANIFEST_FILENAME,
}


class PatchError(Exception):
    """A patch cannot be applied safely."""


class PatchNotForThisVersion(PatchError):
    """The archive is valid but its base version is not installed."""


@dataclass(frozen=True)
class AppVersion:
    version: str
    patch_protocol: int

    def to_dict(self) -> dict[str, object]:
        return {"version": self.version, "patch_protocol": self.patch_protocol}


@dataclass(frozen=True)
class PatchManifest:
    from_version: str
    to_version: str
    patch_protocol: int
    files: tuple[tuple[str, str], ...]
    deleted: tuple[str, ...]
    requires_full_update: bool

    @property
    def patch_id(self) -> str:
        return f"{self.from_version}_to_{self.to_version}"


def should_ignore_packaged_path(relative_path: str) -> bool:
    path = safe_relative_path(relative_path)
    if any(part in IGNORED_DIR_NAMES for part in path.parts):
        return True
    name = path.name
    if name in IGNORED_FILE_NAMES:
        return True
    lowered = name.lower()
    if lowered.endswith(".log") or lowered.endswith(".rar"):
        return True
    return lowered.startswith(PATCH_ZIP_PREFIX.lower()) or lowered.startswith(UPDATE_ZIP_PREFIX.lower())


def safe_relative_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PatchError("patch path is empty")
    normalized = value.replace("\\", "/").strip()
    if normalized.startswith("/") or normalized.startswith("./"):
        raise PatchError(f"unsafe patch path: {value}")
    if ":" in normalized:
        raise PatchError(f"unsafe patch path: {value}")
    path = Path(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PatchError(f"unsafe patch path: {value}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_version_file(path: Path) -> AppVersion | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        version = str(payload["version"]).strip()
        protocol = int(payload["patch_protocol"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise PatchError(f"invalid version file: {path}") from exc
    if not version or any(character in version for character in "\\/\0:"):
        raise PatchError(f"invalid version value: {version}")
    return AppVersion(version=version, patch_protocol=protocol)


def installed_version(app_dir: Path, bundle_dir: Path | None = None) -> AppVersion | None:
    candidates = [app_dir / VERSION_FILENAME]
    if bundle_dir is not None:
        candidates.append(bundle_dir / VERSION_FILENAME)
    candidates.append(app_dir / "_internal" / VERSION_FILENAME)
    for candidate in candidates:
        found = read_version_file(candidate)
        if found is not None:
            return found
    return None


def write_version_file(path: Path, version: AppVersion) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(version.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _find_named_zips(app_dir: Path, prefix: str) -> list[Path]:
    matches: list[Path] = []
    if not app_dir.is_dir():
        return matches
    for path in app_dir.iterdir():
        name = path.name
        if path.is_file() and name.startswith(prefix) and name.endswith(PATCH_ZIP_SUFFIX):
            matches.append(path)
    return sorted(matches, key=lambda item: item.name.lower())


def find_patch_zips(app_dir: Path) -> list[Path]:
    return _find_named_zips(app_dir, PATCH_ZIP_PREFIX)


def find_update_zips(app_dir: Path) -> list[Path]:
    return _find_named_zips(app_dir, UPDATE_ZIP_PREFIX)


def _load_manifest(payload: object) -> PatchManifest:
    if not isinstance(payload, dict):
        raise PatchError("patch manifest must be an object")
    try:
        from_version = str(payload["from_version"]).strip()
        to_version = str(payload["to_version"]).strip()
        protocol = int(payload["patch_protocol"])
        requires_full_update = bool(payload.get("requires_full_update", False))
        raw_files = payload.get("files", [])
        raw_deleted = payload.get("deleted", [])
    except (KeyError, TypeError, ValueError) as exc:
        raise PatchError("patch manifest is missing required fields") from exc
    if not from_version or not to_version:
        raise PatchError("patch manifest versions are empty")
    if any(character in f"{from_version}{to_version}" for character in "\\/\0:"):
        raise PatchError("patch manifest version contains an unsafe character")
    if not isinstance(raw_files, list) or not isinstance(raw_deleted, list):
        raise PatchError("patch manifest file lists must be arrays")

    files: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw_files:
        if not isinstance(item, dict):
            raise PatchError("patch file entry must be an object")
        relative = safe_relative_path(str(item.get("path", ""))).as_posix()
        digest = str(item.get("sha256", "")).strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise PatchError(f"invalid sha256 for {relative}")
        if relative in seen:
            raise PatchError(f"duplicate patch path: {relative}")
        seen.add(relative)
        files.append((relative, digest))
    deleted: list[str] = []
    for item in raw_deleted:
        relative = safe_relative_path(str(item)).as_posix()
        if relative in seen:
            raise PatchError(f"path is both updated and deleted: {relative}")
        seen.add(relative)
        deleted.append(relative)
    return PatchManifest(
        from_version=from_version,
        to_version=to_version,
        patch_protocol=protocol,
        files=tuple(files),
        deleted=tuple(deleted),
        requires_full_update=requires_full_update,
    )


def read_manifest_from_zip(zip_path: Path) -> PatchManifest:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                _reject_unsafe_member(info.filename)
            try:
                payload = json.loads(archive.read("patch_manifest.json").decode("utf-8"))
            except KeyError as exc:
                raise PatchError("patch archive is missing patch_manifest.json") from exc
    except zipfile.BadZipFile as exc:
        raise PatchError(f"patch archive is corrupt: {zip_path.name}") from exc
    return _load_manifest(payload)


def _reject_unsafe_member(name: str) -> None:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or ":" in normalized:
        raise PatchError(f"unsafe archive member: {name}")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise PatchError(f"unsafe archive member: {name}")


def validate_patch_zip(zip_path: Path, installed: AppVersion) -> PatchManifest:
    manifest = read_manifest_from_zip(zip_path)
    if manifest.requires_full_update:
        raise PatchError(
            f"{zip_path.name} requires a full Windows download "
            f"({manifest.from_version} -> {manifest.to_version})"
        )
    if manifest.patch_protocol != PATCH_PROTOCOL or installed.patch_protocol != PATCH_PROTOCOL:
        raise PatchError(
            "patch protocol "
            f"{manifest.patch_protocol} does not match installed protocol {installed.patch_protocol}"
        )
    if manifest.from_version != installed.version:
        raise PatchNotForThisVersion(
            f"{zip_path.name} updates {manifest.from_version}, but installed version is {installed.version}"
        )
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for relative, digest in manifest.files:
            member = f"payload/{relative}"
            if member not in names:
                raise PatchError(f"patch payload is missing {relative}")
            actual = hashlib.sha256(archive.read(member)).hexdigest()
            if actual != digest:
                raise PatchError(f"hash mismatch for {relative}")
    return manifest


def _history_path(app_dir: Path) -> Path:
    return app_dir / APPLIED_DIRNAME / HISTORY_NAME


def load_applied_history(app_dir: Path) -> list[dict[str, str]]:
    path = _history_path(app_dir)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        if isinstance(payload, dict):
            payload = [payload]
        else:
            return []
    return [item for item in payload if isinstance(item, dict)]


def _manifest_from_job(job: dict[str, object]) -> PatchManifest:
    version_payload = job.get("version")
    if not isinstance(version_payload, dict):
        raise PatchError("patch job is missing the target version")
    files = tuple(
        (safe_relative_path(str(relative)).as_posix(), "")
        for relative in list(job.get("files") or [])
    )
    deleted = tuple(
        safe_relative_path(str(relative)).as_posix() for relative in list(job.get("deleted") or [])
    )
    return PatchManifest(
        from_version=str(job.get("from_version") or ""),
        to_version=str(version_payload["version"]),
        patch_protocol=int(version_payload["patch_protocol"]),
        files=files,
        deleted=deleted,
        requires_full_update=False,
    )


def remember_applied_patch(app_dir: Path, zip_path: Path, manifest: PatchManifest) -> None:
    history = load_applied_history(app_dir)
    history.append(
        {
            "zip_name": zip_path.name,
            "zip_sha256": sha256_file(zip_path),
            "from_version": manifest.from_version,
            "to_version": manifest.to_version,
        }
    )
    path = _history_path(app_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def patch_already_applied(app_dir: Path, zip_path: Path) -> bool:
    try:
        digest = sha256_file(zip_path)
    except OSError:
        return False
    return any(item.get("zip_sha256") == digest for item in load_applied_history(app_dir))


def quarantine_patch(app_dir: Path, zip_path: Path, reason: str) -> Path:
    rejected = app_dir / REJECTED_DIRNAME
    rejected.mkdir(parents=True, exist_ok=True)
    destination = rejected / zip_path.name
    if destination.exists():
        destination = rejected / f"{zip_path.stem}_{sha256_file(zip_path)[:8]}{zip_path.suffix}"
    shutil.move(str(zip_path), str(destination))
    destination.with_suffix(destination.suffix + ".reason.txt").write_text(
        reason.strip() + "\n",
        encoding="utf-8",
    )
    _append_error_log(app_dir, reason)
    return destination


def _append_error_log(app_dir: Path, message: str) -> None:
    path = app_dir / ERROR_LOG_NAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def stage_patch(app_dir: Path, zip_path: Path, manifest: PatchManifest) -> Path:
    staging = app_dir / STAGING_DIRNAME / manifest.patch_id
    if staging.exists():
        shutil.rmtree(staging)
    payload_root = staging / "payload"
    payload_root.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        for relative, _digest in manifest.files:
            member = f"payload/{relative}"
            destination = payload_root / safe_relative_path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
            actual = sha256_file(destination)
            expected = dict(manifest.files)[relative]
            if actual != expected:
                raise PatchError(f"staged hash mismatch for {relative}")
    return staging


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def commit_patch(job: dict[str, object], *, fail_on: str | None = None) -> None:
    """Replace files in an installed tree and roll back the whole batch on failure."""
    app_dir = Path(str(job["app_dir"]))
    staging = Path(str(job["staging_dir"]))
    backup_root = Path(str(job["backup_dir"]))
    payload_root = staging / "payload"
    if backup_root.exists():
        shutil.rmtree(backup_root)
    backup_root.mkdir(parents=True)
    operations: list[dict[str, object]] = []
    zip_moved_to: Path | None = None
    original_zip = Path(str(job["zip_path"])) if job.get("zip_path") else None
    try:
        for relative in list(job.get("files") or []):
            relative_path = safe_relative_path(str(relative)).as_posix()
            _replace_file(
                app_dir=app_dir,
                payload_root=payload_root,
                backup_root=backup_root,
                relative_path=relative_path,
                operations=operations,
            )
            if fail_on == relative_path:
                raise PatchError(f"injected failure while updating {relative_path}")
        for relative in list(job.get("deleted") or []):
            relative_path = safe_relative_path(str(relative)).as_posix()
            _delete_file(
                app_dir=app_dir,
                backup_root=backup_root,
                relative_path=relative_path,
                operations=operations,
            )
            if fail_on == relative_path:
                raise PatchError(f"injected failure while deleting {relative_path}")
        version_payload = job.get("version")
        if not isinstance(version_payload, dict):
            raise PatchError("patch job is missing the target version")
        version = AppVersion(
            version=str(version_payload["version"]),
            patch_protocol=int(version_payload["patch_protocol"]),
        )
        _write_version_with_backup(app_dir, backup_root, version, operations)
    except Exception:
        _rollback(operations, zip_moved_to, original_zip)
        raise

    try:
        if original_zip is not None and original_zip.is_file():
            applied = app_dir / str(job.get("applied_dir") or APPLIED_DIRNAME)
            applied.mkdir(parents=True, exist_ok=True)
            destination = applied / original_zip.name
            if destination.exists() and not _same_file(destination, original_zip):
                raise PatchError(f"applied patch already exists: {destination.name}")
            if not _same_file(destination, original_zip):
                shutil.move(str(original_zip), str(destination))
            remember_applied_patch(app_dir, destination, _manifest_from_job(job))
        if staging.exists():
            shutil.rmtree(staging)
        if backup_root.exists():
            shutil.rmtree(backup_root)
    except Exception as exc:
        _append_error_log(app_dir, f"Patch files were updated, but cleanup failed: {exc}")


def _replace_file(
    *,
    app_dir: Path,
    payload_root: Path,
    backup_root: Path,
    relative_path: str,
    operations: list[dict[str, object]],
) -> None:
    source = payload_root / relative_path
    if not source.is_file():
        raise PatchError(f"staged payload is missing {relative_path}")
    destination = app_dir / relative_path
    backup = backup_root / relative_path
    existed = destination.exists()
    if existed:
        if destination.is_dir():
            raise PatchError(f"cannot replace a directory: {relative_path}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(destination), str(backup))
    operations.append(
        {
            "kind": "replace",
            "destination": destination,
            "backup": backup if existed else None,
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _delete_file(
    *,
    app_dir: Path,
    backup_root: Path,
    relative_path: str,
    operations: list[dict[str, object]],
) -> None:
    destination = app_dir / relative_path
    if not destination.exists():
        return
    if destination.is_dir():
        raise PatchError(f"cannot delete a directory entry: {relative_path}")
    backup = backup_root / relative_path
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(destination), str(backup))
    operations.append({"kind": "delete", "destination": destination, "backup": backup})


def _write_version_with_backup(
    app_dir: Path,
    backup_root: Path,
    version: AppVersion,
    operations: list[dict[str, object]],
) -> None:
    destination = app_dir / VERSION_FILENAME
    already_backed_up = any(
        isinstance(item.get("destination"), Path) and item["destination"] == destination
        for item in operations
    )
    if not already_backed_up:
        backup = None
        if destination.exists():
            backup = backup_root / VERSION_FILENAME
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(backup))
        operations.append({"kind": "version", "destination": destination, "backup": backup})
    write_version_file(destination, version)


def _rollback(
    operations: list[dict[str, object]],
    zip_moved_to: Path | None,
    original_zip: Path | None,
) -> None:
    for operation in reversed(operations):
        destination = operation.get("destination")
        backup = operation.get("backup")
        if isinstance(destination, Path) and destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if isinstance(backup, Path) and backup.exists():
            destination_path = destination if isinstance(destination, Path) else None
            if destination_path is not None:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(backup), str(destination_path))
    if zip_moved_to is not None and original_zip is not None and zip_moved_to.exists():
        original_zip.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(zip_moved_to), str(original_zip))


def write_patch_job(
    *,
    app_dir: Path,
    zip_path: Path,
    manifest: PatchManifest,
    staging: Path,
    pid: int,
    relaunch: bool,
) -> Path:
    staged_version = staging / "payload" / VERSION_FILENAME
    if staged_version.is_file():
        version_text = staged_version.read_text(encoding="utf-8")
    else:
        version_text = (
            json.dumps(
                AppVersion(manifest.to_version, manifest.patch_protocol).to_dict(),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    job = {
        "wait_pid": pid,
        "app_dir": str(app_dir),
        "staging_dir": str(staging),
        "backup_dir": str(app_dir / BACKUP_DIRNAME / manifest.patch_id),
        "zip_path": str(zip_path),
        "applied_dir": APPLIED_DIRNAME,
        "files": [relative for relative, _digest in manifest.files],
        "deleted": list(manifest.deleted),
        "from_version": manifest.from_version,
        "version": AppVersion(manifest.to_version, manifest.patch_protocol).to_dict(),
        "version_text": version_text,
        "exe_name": EXE_NAME,
        "relaunch": relaunch,
        "error_log": str(app_dir / ERROR_LOG_NAME),
    }
    job_path = staging / "patch_job.json"
    job_path.write_text(json.dumps(job, ensure_ascii=True, indent=2), encoding="utf-8")
    script_path = staging / "apply_patch.ps1"
    script_path.write_text(PATCH_APPLY_SCRIPT, encoding="utf-8")
    return script_path


def spawn_patch_applier(script_path: Path, job_path: Path) -> None:
    if sys.platform != "win32":
        raise PatchError("Windows patch application requires PowerShell")
    log_path = script_path.with_name("apply.log")
    log_handle = log_path.open("a", encoding="utf-8")
    # DETACHED_PROCESS prevents the console PowerShell host from running -File.
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            "-JobPath",
            str(job_path),
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=flags,
        cwd=str(script_path.parent),
    )


@dataclass(frozen=True)
class ManifestUpdate:
    to_version: str
    files: tuple[tuple[str, str], ...]
    deleted: tuple[str, ...]

    @property
    def patch_id(self) -> str:
        return f"manifest_to_{self.to_version}"


def read_update_manifest(zip_path: Path) -> ManifestUpdate:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                _reject_unsafe_member(info.filename)
            try:
                payload = json.loads(archive.read("update_manifest.json").decode("utf-8"))
            except KeyError as exc:
                raise PatchError("update archive is missing update_manifest.json") from exc
    except zipfile.BadZipFile as exc:
        raise PatchError(f"update archive is corrupt: {zip_path.name}") from exc
    if not isinstance(payload, dict) or payload.get("update_type") != "manifest":
        raise PatchError("update archive is not a file-manifest update")
    try:
        to_version = str(payload["to_version"]).strip()
        protocol = int(payload["patch_protocol"])
        raw_files = payload.get("files", [])
        raw_deleted = payload.get("deleted", [])
    except (KeyError, TypeError, ValueError) as exc:
        raise PatchError("update manifest is missing required fields") from exc
    if protocol != MANIFEST_PROTOCOL:
        raise PatchError(f"unsupported manifest protocol: {protocol}")
    if not isinstance(raw_files, list) or not isinstance(raw_deleted, list):
        raise PatchError("update manifest file lists must be arrays")
    files: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw_files:
        if not isinstance(item, dict):
            raise PatchError("update file entry must be an object")
        relative = safe_relative_path(str(item.get("path", ""))).as_posix()
        digest = str(item.get("sha256", "")).strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise PatchError(f"invalid sha256 for {relative}")
        if relative in seen:
            raise PatchError(f"duplicate update path: {relative}")
        seen.add(relative)
        files.append((relative, digest))
    deleted: list[str] = []
    for item in raw_deleted:
        relative = safe_relative_path(str(item)).as_posix()
        if relative in seen:
            raise PatchError(f"path is both updated and deleted: {relative}")
        seen.add(relative)
        deleted.append(relative)
    return ManifestUpdate(to_version=to_version, files=tuple(files), deleted=tuple(deleted))


def _payload_hashes(zip_path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.startswith("payload/") or name.endswith("/"):
                continue
            relative = safe_relative_path(name.removeprefix("payload/")).as_posix()
            hashes[relative] = hashlib.sha256(archive.read(name)).hexdigest()
    return hashes


def plan_manifest_update(
    app_dir: Path,
    update: ManifestUpdate,
    payload_hashes: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Return files to replace and files to delete. Missing payload bytes is an error."""
    desired = dict(update.files)
    replace: list[str] = []
    for relative, digest in update.files:
        local = app_dir / relative
        if local.is_file() and sha256_file(local) == digest:
            continue
        packaged = payload_hashes.get(relative)
        if packaged != digest:
            raise PatchError(
                f"当前安装的 {relative} 与最新版不同，但更新包没有包含它。请下载完整安装包。"
            )
        replace.append(relative)

    local_manifest = _read_app_manifest_paths(app_dir)
    if local_manifest is None:
        deleted = list(update.deleted)
    else:
        deleted = sorted(path for path in local_manifest if path not in desired)
    return replace, deleted


def _read_app_manifest_paths(app_dir: Path) -> set[str] | None:
    path = app_dir / APP_MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        raw_files = payload.get("files", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(raw_files, list):
        return None
    paths: set[str] = set()
    for item in raw_files:
        if isinstance(item, dict) and item.get("path"):
            paths.add(safe_relative_path(str(item["path"])).as_posix())
    return paths


def stage_manifest_files(app_dir: Path, zip_path: Path, update: ManifestUpdate, files: list[str]) -> Path:
    staging = app_dir / STAGING_DIRNAME / update.patch_id
    if staging.exists():
        shutil.rmtree(staging)
    payload_root = staging / "payload"
    payload_root.mkdir(parents=True)
    wanted = set(files)
    with zipfile.ZipFile(zip_path) as archive:
        for relative in files:
            destination = payload_root / safe_relative_path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(f"payload/{relative}"))
            if relative not in wanted:
                raise PatchError(f"unexpected staged file: {relative}")
    return staging


def process_manifest_updates(app_dir: Path, *, pid: int, relaunch: bool) -> int | None:
    zips = find_update_zips(app_dir)
    if not zips:
        return None
    current = installed_version(app_dir)
    if current is None or current.patch_protocol < MANIFEST_PROTOCOL:
        print("文件清单更新需要已安装清单同步版本；这份更新包已保留。")
        return None

    selected: tuple[Path, ManifestUpdate, list[str], list[str]] | None = None
    for zip_path in zips:
        if patch_already_applied(app_dir, zip_path):
            print(f"Windows update already applied, leaving {zip_path.name} in place.")
            continue
        try:
            update = read_update_manifest(zip_path)
            replace, deleted = plan_manifest_update(app_dir, update, _payload_hashes(zip_path))
        except PatchError as exc:
            print(f"Windows update rejected: {exc}")
            quarantine_patch(app_dir, zip_path, str(exc))
            continue
        selected = (zip_path, update, replace, deleted)
        break

    if selected is None:
        return None
    zip_path, update, replace, deleted = selected
    if not replace and not deleted and current.version == update.to_version:
        applied = app_dir / APPLIED_DIRNAME
        applied.mkdir(parents=True, exist_ok=True)
        destination = applied / zip_path.name
        if not destination.exists():
            shutil.move(str(zip_path), str(destination))
        remember_applied_patch(
            app_dir,
            destination,
            PatchManifest(
                from_version=current.version,
                to_version=update.to_version,
                patch_protocol=MANIFEST_PROTOCOL,
                files=(),
                deleted=(),
                requires_full_update=False,
            ),
        )
        print(f"Windows update {zip_path.name} is already installed.")
        return None

    try:
        staging = stage_manifest_files(app_dir, zip_path, update, replace)
        manifest_path = staging / "payload" / APP_MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(
                {
                    "version": update.to_version,
                    "patch_protocol": MANIFEST_PROTOCOL,
                    "files": [
                        {"path": relative, "sha256": digest} for relative, digest in update.files
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest = PatchManifest(
            from_version=current.version,
            to_version=update.to_version,
            patch_protocol=MANIFEST_PROTOCOL,
            files=tuple((relative, dict(update.files)[relative]) for relative in replace)
            + ((APP_MANIFEST_FILENAME, sha256_file(manifest_path)),),
            deleted=tuple(deleted),
            requires_full_update=False,
        )
        script_path = write_patch_job(
            app_dir=app_dir,
            zip_path=zip_path,
            manifest=manifest,
            staging=staging,
            pid=pid,
            relaunch=relaunch,
        )
        spawn_patch_applier(script_path, staging / "patch_job.json")
    except (OSError, PatchError) as exc:
        message = f"Could not stage {zip_path.name}: {exc}"
        print(message)
        _append_error_log(app_dir, message)
        return None

    print(f"Installing Windows update to {update.to_version}. The app will restart.")
    return 0


def process_pending_patches(app_dir: Path, *, pid: int, relaunch: bool = True) -> int | None:
    """Apply one matching patch. Return 0 when the current process must exit."""
    try:
        manifest_result = process_manifest_updates(app_dir, pid=pid, relaunch=relaunch)
    except PatchError as exc:
        _append_error_log(app_dir, str(exc))
        print(f"Windows update skipped: {exc}")
        manifest_result = None
    if manifest_result is not None:
        return manifest_result

    zips = find_patch_zips(app_dir)
    if not zips:
        return None
    try:
        current = installed_version(app_dir)
    except PatchError as exc:
        _append_error_log(app_dir, str(exc))
        print(f"Windows patch skipped: {exc}")
        return None
    if current is None:
        print("Windows patch skipped: installed version metadata is missing.")
        return None

    selected: tuple[Path, PatchManifest] | None = None
    for zip_path in zips:
        if patch_already_applied(app_dir, zip_path):
            print(f"Windows patch already applied, leaving {zip_path.name} in place.")
            continue
        try:
            manifest = validate_patch_zip(zip_path, current)
        except PatchNotForThisVersion as exc:
            print(f"Windows patch not for this version: {exc}")
            continue
        except PatchError as exc:
            print(f"Windows patch rejected: {exc}")
            quarantine_patch(app_dir, zip_path, str(exc))
            continue
        selected = (zip_path, manifest)
        break

    if selected is None:
        return None
    zip_path, manifest = selected
    try:
        staging = stage_patch(app_dir, zip_path, manifest)
        script_path = write_patch_job(
            app_dir=app_dir,
            zip_path=zip_path,
            manifest=manifest,
            staging=staging,
            pid=pid,
            relaunch=relaunch,
        )
        spawn_patch_applier(script_path, staging / "patch_job.json")
    except (OSError, PatchError) as exc:
        message = f"Could not stage {zip_path.name}: {exc}"
        print(message)
        _append_error_log(app_dir, message)
        return None

    print(
        "Installing Windows patch "
        f"{manifest.from_version} -> {manifest.to_version}. The app will restart."
    )
    return 0


def apply_pending_patch_and_maybe_exit() -> int | None:
    if not is_frozen():
        return None
    app_dir = Path(sys.executable).resolve().parent
    return process_pending_patches(app_dir, pid=os.getpid(), relaunch=True)


PATCH_APPLY_SCRIPT = r"""param(
    [Parameter(Mandatory = $true)]
    [string]$JobPath
)

$ErrorActionPreference = "Stop"
$job = Get-Content -LiteralPath $JobPath -Raw -Encoding UTF8 | ConvertFrom-Json
$processId = [int]$job.wait_pid
if ($processId -gt 0) {
    Wait-Process -Id $processId -ErrorAction SilentlyContinue
}

$appDir = [string]$job.app_dir
$payloadRoot = Join-Path ([string]$job.staging_dir) "payload"
$backupRoot = [string]$job.backup_dir
$errorLog = [string]$job.error_log
$operations = New-Object System.Collections.Generic.List[object]
$zipMovedTo = $null

function Add-Operation([string]$Destination, [string]$Backup) {
    $operations.Add([pscustomobject]@{
        Destination = $Destination
        Backup = $Backup
    }) | Out-Null
}

function Restore-Operations {
    for ($index = $operations.Count - 1; $index -ge 0; $index--) {
        $operation = $operations[$index]
        if ($operation.Destination -and (Test-Path -LiteralPath $operation.Destination)) {
            Remove-Item -LiteralPath $operation.Destination -Force
        }
        if ($operation.Backup -and (Test-Path -LiteralPath $operation.Backup)) {
            $parent = Split-Path -Parent $operation.Destination
            if ($parent) {
                New-Item -ItemType Directory -Force -Path $parent | Out-Null
            }
            Move-Item -LiteralPath $operation.Backup -Destination $operation.Destination
        }
    }
    if ($zipMovedTo -and (Test-Path -LiteralPath $zipMovedTo)) {
        Move-Item -LiteralPath $zipMovedTo -Destination ([string]$job.zip_path)
    }
}

try {
    if (Test-Path -LiteralPath $backupRoot) {
        Remove-Item -LiteralPath $backupRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null

    foreach ($relative in @($job.files)) {
        if ([string]::IsNullOrWhiteSpace([string]$relative)) { continue }
        $relativeText = ([string]$relative).Replace("/", "\")
        $source = Join-Path $payloadRoot $relativeText
        $destination = Join-Path $appDir $relativeText
        $backup = Join-Path $backupRoot $relativeText
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "staged payload is missing $relative"
        }
        $backupName = ""
        if (Test-Path -LiteralPath $destination) {
            $backupParent = Split-Path -Parent $backup
            if ($backupParent) {
                New-Item -ItemType Directory -Force -Path $backupParent | Out-Null
            }
            Move-Item -LiteralPath $destination -Destination $backup
            $backupName = $backup
        }
        Add-Operation $destination $backupName
        $destinationParent = Split-Path -Parent $destination
        if ($destinationParent) {
            New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
        }
        Copy-Item -LiteralPath $source -Destination $destination -Force
        if ([string]$job.fail_on -eq [string]$relative) {
            throw "injected failure while updating $relative"
        }
    }

    foreach ($relative in @($job.deleted)) {
        if ([string]::IsNullOrWhiteSpace([string]$relative)) { continue }
        $relativeText = ([string]$relative).Replace("/", "\")
        $destination = Join-Path $appDir $relativeText
        if (-not (Test-Path -LiteralPath $destination)) { continue }
        $backup = Join-Path $backupRoot $relativeText
        $backupParent = Split-Path -Parent $backup
        if ($backupParent) {
            New-Item -ItemType Directory -Force -Path $backupParent | Out-Null
        }
        Move-Item -LiteralPath $destination -Destination $backup
        Add-Operation $destination $backup
        if ([string]$job.fail_on -eq [string]$relative) {
            throw "injected failure while deleting $relative"
        }
    }

    $versionPath = Join-Path $appDir "windows_version.json"
    $alreadyBackedUp = $false
    foreach ($operation in $operations) {
        if ($operation.Destination -eq $versionPath) { $alreadyBackedUp = $true }
    }
    if ((Test-Path -LiteralPath $versionPath) -and -not $alreadyBackedUp) {
        $versionBackup = Join-Path $backupRoot "windows_version.json"
        Move-Item -LiteralPath $versionPath -Destination $versionBackup
        Add-Operation $versionPath $versionBackup
    } elseif (-not $alreadyBackedUp) {
        Add-Operation $versionPath ""
    }
    $versionJson = [string]$job.version_text
    if ([string]::IsNullOrWhiteSpace($versionJson)) {
        $versionJson = ($job.version | ConvertTo-Json -Compress) + "`n"
    }
    [System.IO.File]::WriteAllText($versionPath, $versionJson, (New-Object System.Text.UTF8Encoding $false))
} catch {
    Restore-Operations
    $message = $_.Exception.Message
    if ($errorLog) {
        Add-Content -LiteralPath $errorLog -Value $message -Encoding UTF8
    }
    Write-Error $message
    exit 1
}

try {
    $zipPath = [string]$job.zip_path
    if ($zipPath -and (Test-Path -LiteralPath $zipPath)) {
        $applied = Join-Path $appDir ([string]$job.applied_dir)
        New-Item -ItemType Directory -Force -Path $applied | Out-Null
        $zipMovedTo = Join-Path $applied (Split-Path -Leaf $zipPath)
        Move-Item -LiteralPath $zipPath -Destination $zipMovedTo
        $sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipMovedTo).Hash.ToLower()
        $historyPath = Join-Path $applied "history.json"
        $history = @()
        if (Test-Path -LiteralPath $historyPath) {
            $loaded = Get-Content -LiteralPath $historyPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $history = @($loaded)
        }
        $history += [pscustomobject]@{
            zip_name = (Split-Path -Leaf $zipMovedTo)
            zip_sha256 = $sha
            from_version = [string]$job.from_version
            to_version = [string]$job.version.version
        }
        $history | ConvertTo-Json | Set-Content -LiteralPath $historyPath -Encoding UTF8
    }
    if (Test-Path -LiteralPath ([string]$job.staging_dir)) {
        Remove-Item -LiteralPath ([string]$job.staging_dir) -Recurse -Force
    }
    if (Test-Path -LiteralPath $backupRoot) {
        Remove-Item -LiteralPath $backupRoot -Recurse -Force
    }
} catch {
    if ($errorLog) {
        Add-Content -LiteralPath $errorLog -Value $_.Exception.Message -Encoding UTF8
    }
}

if ($job.relaunch) {
    $executable = Join-Path $appDir ([string]$job.exe_name)
    Start-Process -FilePath $executable -WorkingDirectory $appDir | Out-Null
}
exit 0
"""
