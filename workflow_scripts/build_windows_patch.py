"""Build a file-level Windows patch by comparing two packaged app directories."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.windows_patch import (  # noqa: E402
    EXE_NAME,
    MANIFEST_PROTOCOL,
    PATCH_PROTOCOL,
    APP_MANIFEST_FILENAME,
    AppVersion,
    PatchError,
    installed_version,
    safe_relative_path,
    sha256_file,
    should_ignore_packaged_path,
    write_version_file,
)


def iter_packaged_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if should_ignore_packaged_path(relative):
            continue
        files[relative] = path
    return files


def directory_size(files: dict[str, Path]) -> int:
    return sum(path.stat().st_size for path in files.values())


def compatibility_problems(baseline: Path, target: Path) -> list[str]:
    problems: list[str] = []
    if not (baseline / EXE_NAME).is_file():
        problems.append(f"baseline is missing {EXE_NAME}")
    if not (target / EXE_NAME).is_file():
        problems.append(f"target is missing {EXE_NAME}")
    if not (baseline / "_internal").is_dir() or not (target / "_internal").is_dir():
        problems.append("both trees must contain an _internal directory")
    try:
        old_version = installed_version(baseline)
        new_version = installed_version(target)
    except PatchError as exc:
        return [str(exc)]
    if old_version is None or new_version is None:
        problems.append("both trees must contain windows_version.json")
    elif old_version.patch_protocol != new_version.patch_protocol and not (
        old_version.patch_protocol == PATCH_PROTOCOL
        and new_version.patch_protocol == MANIFEST_PROTOCOL
    ):
        problems.append(
            "patch protocol changed from "
            f"{old_version.patch_protocol} to {new_version.patch_protocol}"
        )
    elif old_version.patch_protocol not in {PATCH_PROTOCOL, MANIFEST_PROTOCOL}:
        problems.append(f"unsupported patch protocol {old_version.patch_protocol}")
    return problems


def build_patch(
    *,
    baseline: Path,
    target: Path,
    output: Path | None = None,
) -> dict[str, object]:
    problems = compatibility_problems(baseline, target)
    old_version = installed_version(baseline)
    new_version = installed_version(target)
    old_files = iter_packaged_files(baseline)
    new_files = iter_packaged_files(target)
    summary: dict[str, object] = {
        "from_version": None if old_version is None else old_version.version,
        "to_version": None if new_version is None else new_version.version,
        "baseline_bytes": directory_size(old_files),
        "target_bytes": directory_size(new_files),
        "patch_bytes": 0,
        "added": [],
        "changed": [],
        "deleted": [],
        "requires_full_update": bool(problems),
        "reasons": problems,
        "output_path": None,
    }
    if problems:
        if output is not None:
            report = output.with_suffix(".full_update_required.json")
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            summary["output_path"] = str(report)
        return summary

    assert old_version is not None
    assert new_version is not None
    added: list[str] = []
    changed: list[str] = []
    deleted = sorted(set(old_files) - set(new_files))
    file_entries: list[dict[str, str]] = []
    for relative in sorted(new_files):
        digest = sha256_file(new_files[relative])
        if relative not in old_files:
            added.append(relative)
            file_entries.append({"path": relative, "sha256": digest})
        elif sha256_file(old_files[relative]) != digest:
            changed.append(relative)
            file_entries.append({"path": relative, "sha256": digest})

    summary["added"] = added
    summary["changed"] = changed
    summary["deleted"] = deleted
    manifest_file = target / APP_MANIFEST_FILENAME
    if (
        manifest_file.is_file()
        and new_version.patch_protocol >= MANIFEST_PROTOCOL
        and all(entry["path"] != APP_MANIFEST_FILENAME for entry in file_entries)
    ):
        file_entries.append({"path": APP_MANIFEST_FILENAME, "sha256": sha256_file(manifest_file)})
        added.append(APP_MANIFEST_FILENAME)
    if not file_entries and not deleted:
        return summary

    if output is None:
        output = (
            target.parent
            / "patches"
            / f"E7_BP_Helper_patch_{old_version.version}_to_{new_version.version}.zip"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "patch_protocol": old_version.patch_protocol,
        "from_version": old_version.version,
        "to_version": new_version.version,
        "requires_full_update": False,
        "files": file_entries,
        "deleted": deleted,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "patch_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        for entry in file_entries:
            relative = safe_relative_path(entry["path"]).as_posix()
            source = new_files.get(relative, target / relative)
            archive.write(source, f"payload/{relative}")
    summary["patch_bytes"] = output.stat().st_size
    summary["output_path"] = str(output)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


APPLY_UPDATE_PS1 = r"""param()
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = Join-Path $here "E7_BP_Helper.exe"
if (-not (Test-Path -LiteralPath $exe)) {
    Write-Host "请把压缩包里的全部文件解压到 E7_BP_Helper.exe 所在文件夹，再双击 安装补丁.bat"
    exit 1
}

$versionPath = Join-Path $here "windows_version.json"
if (-not (Test-Path -LiteralPath $versionPath)) {
    $versionPath = Join-Path $here "_internal\windows_version.json"
}
if (-not (Test-Path -LiteralPath $versionPath)) {
    Write-Host "这个安装包没有补丁功能，请下载完整的新版文件夹。"
    exit 1
}

$installed = (Get-Content -LiteralPath $versionPath -Raw -Encoding UTF8 | ConvertFrom-Json).version
$bundlePath = Join-Path $here "update_bundle.json"
$bundle = Get-Content -LiteralPath $bundlePath -Raw -Encoding UTF8 | ConvertFrom-Json
$target = [string]$bundle.to_version
if ($installed -eq $target) {
    Write-Host "当前已经是 $target"
    exit 0
}

$match = @($bundle.patches) | Where-Object { [string]$_.from_version -eq [string]$installed } | Select-Object -First 1
if (-not $match) {
    Write-Host "没有适用于 $installed 的补丁。这个更新包支持：$($bundle.patches.from_version -join ', ')"
    exit 1
}

$keep = Join-Path $here ([string]$match.file)
if (-not (Test-Path -LiteralPath $keep)) {
    Write-Host "缺少补丁文件：$($match.file)"
    exit 1
}
Get-ChildItem -LiteralPath $here -Filter "E7_BP_Helper_patch_*.zip" |
    Where-Object { $_.Name -ne [string]$match.file } |
    Remove-Item -Force
Start-Process -FilePath $exe -WorkingDirectory $here | Out-Null
Write-Host "正在从 $installed 更新到 $target。程序会自动关闭并重新打开。"
"""

APPLY_UPDATE_BAT = """@echo off\r
chcp 65001 >nul\r
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"%~dp0apply_update.ps1\"\r
if errorlevel 1 pause\r
"""


def build_version_jump_bundle(
    *,
    baselines: list[Path],
    target: Path,
    output: Path,
) -> dict[str, object]:
    """Pack one old-format patch per installed version into a single update zip.

    Released apps accept a patch only when its from_version matches exactly, so
    one manifest cannot update both 1.0.0 and 1.0.1. The bundle carries every
    jump, and apply_update.ps1 keeps only the matching zip before launch.
    """
    if len(baselines) < 1:
        raise ValueError("at least one baseline is required")
    target_version = installed_version(target)
    if target_version is None:
        raise ValueError(f"target has no windows version: {target}")

    staging = output.parent / f".{output.stem}_staging"
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    patches: list[dict[str, str]] = []
    summaries: list[dict[str, object]] = []
    try:
        for baseline in baselines:
            summary = build_patch(baseline=baseline, target=target)
            summaries.append(summary)
            if summary.get("requires_full_update") or not summary.get("output_path"):
                raise ValueError(f"could not build a patch from {baseline}: {summary.get('reasons')}")
            patch_path = Path(str(summary["output_path"]))
            destination = staging / patch_path.name
            destination.write_bytes(patch_path.read_bytes())
            patches.append(
                {
                    "from_version": str(summary["from_version"]),
                    "file": patch_path.name,
                }
            )
        bundle = {
            "to_version": target_version.version,
            "patches": patches,
        }
        (staging / "update_bundle.json").write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "apply_update.ps1").write_text(APPLY_UPDATE_PS1, encoding="utf-8-sig")
        (staging / "安装补丁.bat").write_bytes(APPLY_UPDATE_BAT.encode("ascii"))
        readme = (
            "把压缩包中的所有文件解压到 E7_BP_Helper.exe 所在文件夹。\n"
            "不要只解压其中一个补丁，也不要改文件名。\n"
            "然后双击“安装补丁.bat”。\n"
            "程序会留下与当前版本匹配的补丁，自动关闭并重新打开，完成后即为 "
            f"{target_version.version}。\n"
            "没有 windows_version.json 的旧安装包不能使用这个更新，请下载完整文件夹。\n"
        )
        (staging / "使用说明.txt").write_text(readme, encoding="utf-8-sig")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging).as_posix())
    finally:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)

    return {
        "to_version": target_version.version,
        "output_path": str(output),
        "output_bytes": output.stat().st_size,
        "patches": summaries,
    }


def file_manifest(root: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for relative, path in sorted(iter_packaged_files(root).items()):
        entries.append({"path": relative, "sha256": sha256_file(path)})
    return entries


def write_app_manifest(root: Path) -> Path:
    version = installed_version(root)
    if version is None:
        raise ValueError(f"missing windows version in {root}")
    payload = {
        "version": version.version,
        "patch_protocol": version.patch_protocol,
        "files": file_manifest(root),
    }
    destination = root / APP_MANIFEST_FILENAME
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def build_manifest_update(
    *,
    baselines: list[Path],
    target: Path,
    output: Path,
) -> dict[str, object]:
    """Build one hash-sync update that can move any supplied baseline to target.

    The manifest lists the complete target tree. The payload contains only files
    whose bytes differ from at least one baseline, which keeps TensorFlow out of
    ordinary data and model updates.
    """
    if not baselines:
        raise ValueError("at least one baseline is required")
    target_version = installed_version(target)
    if target_version is None or target_version.patch_protocol < MANIFEST_PROTOCOL:
        raise ValueError("manifest updates require a protocol 2 target")
    for baseline in baselines:
        baseline_version = installed_version(baseline)
        if baseline_version is None or baseline_version.patch_protocol < MANIFEST_PROTOCOL:
            raise ValueError(f"{baseline} does not contain the manifest updater")

    target_files = iter_packaged_files(target)
    baseline_files = [iter_packaged_files(baseline) for baseline in baselines]
    entries: list[dict[str, str]] = []
    payload: list[str] = []
    for relative, path in sorted(target_files.items()):
        digest = sha256_file(path)
        entries.append({"path": relative, "sha256": digest})
        if any(
            relative not in files or sha256_file(files[relative]) != digest for files in baseline_files
        ):
            payload.append(relative)
    deleted = sorted(
        {
            relative
            for files in baseline_files
            for relative in files
            if relative not in target_files
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "update_manifest.json",
            json.dumps(
                {
                    "update_type": "manifest",
                    "patch_protocol": MANIFEST_PROTOCOL,
                    "to_version": target_version.version,
                    "files": entries,
                    "deleted": deleted,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        for relative in payload:
            archive.write(target_files[relative], f"payload/{safe_relative_path(relative).as_posix()}")
    summary = {
        "to_version": target_version.version,
        "output_path": str(output),
        "output_bytes": output.stat().st_size,
        "manifest_files": len(entries),
        "payload_files": payload,
        "deleted": deleted,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def write_build_version(path: Path, version: str) -> None:
    write_version_file(path, AppVersion(version=version, patch_protocol=PATCH_PROTOCOL))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a differential Windows app patch.")
    parser.add_argument(
        "--baseline",
        type=Path,
        action="append",
        help="Previously released app directory. Repeat to build a multi-version update bundle.",
    )
    parser.add_argument("--target", type=Path, help="New packaged app directory.")
    parser.add_argument("--output", type=Path, default=None, help="Patch zip or update-bundle output path.")
    parser.add_argument(
        "--write-app-manifest",
        type=Path,
        help="Write app_manifest.json for an already built app directory and exit.",
    )
    parser.add_argument(
        "--manifest-update",
        action="store_true",
        help="Build a hash-sync update from every --baseline to --target.",
    )
    args = parser.parse_args()
    if args.write_app_manifest is not None:
        destination = write_app_manifest(args.write_app_manifest)
        print(f"Wrote {destination}")
        return 0
    if not args.baseline or args.target is None:
        parser.error("--baseline and --target are required")
    if args.manifest_update:
        if args.output is None:
            parser.error("--output is required for a manifest update")
        summary = build_manifest_update(baselines=args.baseline, target=args.target, output=args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Manifest update size {summary['output_bytes']} bytes.")
        return 0
    if len(args.baseline) > 1:
        if args.output is None:
            raise SystemExit("An --output path is required when building a multi-version update bundle.")
        summary = build_version_jump_bundle(
            baselines=args.baseline,
            target=args.target,
            output=args.output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Update bundle size {summary['output_bytes']} bytes.")
        return 0
    summary = build_patch(baseline=args.baseline[0], target=args.target, output=args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["requires_full_update"]:
        print("Full Windows package required; patch zip was not created.", file=sys.stderr)
        return 3
    if not summary["output_path"]:
        print("No file changes; patch not created.")
    else:
        print(
            "Patch size "
            f"{summary['patch_bytes']} bytes; baseline tree {summary['baseline_bytes']} bytes."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
