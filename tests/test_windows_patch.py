import json
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "workflow_scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "workflow_scripts"))

from backend.windows_patch import (  # noqa: E402
    EXE_NAME,
    PatchError,
    VERSION_FILENAME,
    AppVersion,
    commit_patch,
    load_applied_history,
    process_pending_patches,
    read_manifest_from_zip,
    safe_relative_path,
    sha256_file,
    stage_patch,
    validate_patch_zip,
    write_patch_job,
    write_version_file,
)
from build_windows_patch import (  # noqa: E402
    build_manifest_update,
    build_patch,
    build_version_jump_bundle,
    iter_packaged_files,
    write_app_manifest,
)


def write_tree(root: Path, version: str, *, marker: str, extra: dict[str, str] | None = None) -> None:
    write_version_file(root / VERSION_FILENAME, AppVersion(version, 1))
    internal = root / "_internal"
    (internal / "data").mkdir(parents=True, exist_ok=True)
    (internal / "data" / "hero_details.csv").write_text(marker, encoding="utf-8")
    (internal / "frontend" / "dist").mkdir(parents=True, exist_ok=True)
    (internal / "frontend" / "dist" / "index.html").write_text("<html>base</html>", encoding="utf-8")
    (root / EXE_NAME).write_bytes(b"MZ-fake-executable")
    (root / "recommender_service.log").write_text("old log\n", encoding="utf-8")
    if extra:
        for relative, content in extra.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


def hashes(root: Path) -> dict[str, str]:
    return {relative: sha256_file(path) for relative, path in iter_packaged_files(root).items()}


class WindowsPatchTests(unittest.TestCase):
    def test_safe_path_rejects_traversal_and_absolute_paths(self) -> None:
        for value in ["../outside.txt", "..\\outside.txt", "/etc/passwd", "C:/Windows/explorer.exe", ""]:
            with self.assertRaises(PatchError):
                safe_relative_path(value)

    def test_zip_slip_member_is_rejected(self) -> None:
        with self._temp_dir() as root:
            archive = root / "E7_BP_Helper_patch_bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("patch_manifest.json", "{}")
                handle.writestr("payload/../../outside.txt", b"owned")
            with self.assertRaises(PatchError):
                read_manifest_from_zip(archive)
            self.assertFalse((root / "outside.txt").exists())

    def test_hash_mismatch_and_corrupt_archive_are_rejected(self) -> None:
        with self._temp_dir() as root:
            installed = root / "app"
            write_tree(installed, "1.0.0", marker="old")
            target = root / "new"
            write_tree(target, "1.0.1", marker="new")
            patch = root / "E7_BP_Helper_patch_1.0.0_to_1.0.1.zip"
            build_patch(baseline=installed, target=target, output=patch)
            mismatched = root / "E7_BP_Helper_patch_mismatch.zip"
            with zipfile.ZipFile(patch) as source, zipfile.ZipFile(mismatched, "w") as output:
                manifest = json.loads(source.read("patch_manifest.json"))
                manifest["files"][0]["sha256"] = "0" * 64
                for info in source.infolist():
                    payload = source.read(info)
                    if info.filename == "patch_manifest.json":
                        payload = (json.dumps(manifest) + "\n").encode()
                    output.writestr(info.filename, payload)
            with self.assertRaises(PatchError):
                validate_patch_zip(mismatched, AppVersion("1.0.0", 1))

            corrupt = root / "E7_BP_Helper_patch_corrupt.zip"
            corrupt.write_bytes(b"not a zip")
            with self.assertRaises(PatchError):
                validate_patch_zip(corrupt, AppVersion("1.0.0", 1))

    def test_version_mismatch_is_left_in_place_and_corrupt_patch_is_quarantined(self) -> None:
        with self._temp_dir() as root:
            app = root / "app"
            write_tree(app, "1.0.0", marker="old")
            future = root / "future"
            write_tree(future, "1.0.2", marker="future")
            future_patch = app / "E7_BP_Helper_patch_1.0.1_to_1.0.2.zip"
            build_patch(baseline=self._versioned_copy(root, "1.0.1"), target=future, output=future_patch)
            corrupt = app / "E7_BP_Helper_patch_corrupt.zip"
            corrupt.write_bytes(b"broken")

            result = process_pending_patches(app, pid=0, relaunch=False)

            self.assertIsNone(result)
            self.assertTrue(future_patch.is_file())
            self.assertFalse(corrupt.is_file())
            rejected = list((app / "patches_rejected").glob("*.zip"))
            self.assertEqual(len(rejected), 1)
            self.assertTrue(rejected[0].with_suffix(rejected[0].suffix + ".reason.txt").is_file())

    def test_commit_updates_deletes_and_rolls_back(self) -> None:
        with self._temp_dir() as root:
            app = root / "app"
            write_tree(app, "1.0.0", marker="old", extra={"_internal/data/remove_me.txt": "gone"})
            target = root / "target"
            write_tree(
                target,
                "1.0.1",
                marker="new",
                extra={"_internal/data/added.txt": "added"},
            )
            patch = app / "E7_BP_Helper_patch_1.0.0_to_1.0.1.zip"
            summary = build_patch(baseline=app, target=target, output=patch)
            self.assertIn("_internal/data/added.txt", summary["added"])
            self.assertIn("_internal/data/remove_me.txt", summary["deleted"])
            self.assertNotIn("recommender_service.log", summary["changed"])

            manifest = read_manifest_from_zip(patch)
            staging = stage_patch(app, patch, manifest)
            job = self._job(app, patch, staging, manifest)
            job["fail_on"] = "_internal/data/added.txt"
            with self.assertRaises(PatchError):
                commit_patch(job, fail_on="_internal/data/added.txt")
            self.assertEqual((app / "_internal" / "data" / "hero_details.csv").read_text(encoding="utf-8"), "old")
            self.assertEqual((app / "_internal" / "data" / "remove_me.txt").read_text(encoding="utf-8"), "gone")
            self.assertFalse((app / "_internal" / "data" / "added.txt").exists())
            self.assertTrue(patch.is_file())

            commit_patch(self._job(app, patch, stage_patch(app, patch, manifest), manifest))
            self.assertEqual(hashes(app), hashes(target))
            self.assertEqual((app / "recommender_service.log").read_text(encoding="utf-8"), "old log\n")
            self.assertFalse(patch.is_file())
            self.assertTrue((app / "patches_applied" / patch.name).is_file())
            self.assertEqual(load_applied_history(app)[-1]["to_version"], "1.0.1")

            repeated = app / patch.name
            (app / "patches_applied" / patch.name).replace(repeated)
            self.assertIsNone(process_pending_patches(app, pid=0, relaunch=False))
            self.assertTrue(repeated.is_file())

    def test_patch_round_trip_matches_target_tree(self) -> None:
        with self._temp_dir() as root:
            baseline = root / "baseline"
            target = root / "target"
            installed = root / "installed"
            write_tree(baseline, "1.0.0", marker="alpha", extra={"_internal/data/old.txt": "old"})
            write_tree(target, "1.0.1", marker="beta", extra={"_internal/data/new.txt": "new"})
            (target / EXE_NAME).write_bytes(b"MZ-fake-executable-v2")
            self._copy_tree(baseline, installed)
            patch = root / "E7_BP_Helper_patch_1.0.0_to_1.0.1.zip"
            summary = build_patch(baseline=baseline, target=target, output=patch)
            self.assertGreater(int(summary["patch_bytes"]), 0)
            self.assertNotIn("_internal/frontend/dist/index.html", summary["changed"])
            manifest = read_manifest_from_zip(patch)
            staged_patch = installed / patch.name
            staged_patch.write_bytes(patch.read_bytes())
            commit_patch(
                self._job(
                    installed,
                    staged_patch,
                    stage_patch(installed, staged_patch, manifest),
                    manifest,
                )
            )
            self.assertEqual(hashes(installed), hashes(target))

    def test_version_jump_bundle_contains_one_patch_per_baseline(self) -> None:
        with self._temp_dir() as root:
            first = root / "1.0.0"
            second = root / "1.0.1"
            target = root / "1.0.2"
            write_tree(first, "1.0.0", marker="old")
            write_tree(second, "1.0.1", marker="middle")
            write_tree(target, "1.0.2", marker="new")
            bundle = root / "E7_BP_Helper_update_to_1.0.2.zip"
            summary = build_version_jump_bundle(
                baselines=[first, second],
                target=target,
                output=bundle,
            )
            self.assertEqual(summary["to_version"], "1.0.2")
            with zipfile.ZipFile(bundle) as archive:
                names = set(archive.namelist())
            self.assertIn("E7_BP_Helper_patch_1.0.0_to_1.0.2.zip", names)
            self.assertIn("E7_BP_Helper_patch_1.0.1_to_1.0.2.zip", names)
            self.assertIn("安装补丁.bat", names)
            self.assertIn("apply_update.ps1", names)

    def test_manifest_update_repairs_any_supported_baseline(self) -> None:
        with self._temp_dir() as root:
            first = root / "1.0.3"
            second = root / "1.0.4-old"
            target = root / "1.0.5"
            write_tree(first, "1.0.3", marker="one")
            write_tree(second, "1.0.4", marker="two", extra={"_internal/data/stale.txt": "stale"})
            write_tree(target, "1.0.5", marker="new", extra={"_internal/data/fresh.txt": "fresh"})
            write_version_file(first / "windows_version.json", AppVersion("1.0.3", 2))
            write_version_file(second / "windows_version.json", AppVersion("1.0.4", 2))
            write_version_file(target / "windows_version.json", AppVersion("1.0.5", 2))
            write_app_manifest(first)
            write_app_manifest(second)
            package = root / "E7_BP_Helper_update_to_1.0.5.zip"
            summary = build_manifest_update(baselines=[first, second], target=target, output=package)
            self.assertIn("_internal/data/hero_details.csv", summary["payload_files"])
            self.assertIn("_internal/data/fresh.txt", summary["payload_files"])
            self.assertNotIn("_internal/frontend/dist/index.html", summary["payload_files"])

            from backend.windows_patch import (
                plan_manifest_update,
                read_update_manifest,
                _payload_hashes,
                commit_patch,
                stage_manifest_files,
            )

            update = read_update_manifest(package)
            payload = _payload_hashes(package)
            replace, deleted = plan_manifest_update(first, update, payload)
            self.assertIn("_internal/data/hero_details.csv", replace)
            self.assertIn("_internal/data/fresh.txt", replace)
            commit_patch(
                self._job(
                    first,
                    package,
                    stage_manifest_files(first, package, update, replace),
                    type("M", (), {
                        "patch_id": update.patch_id,
                        "from_version": "1.0.3",
                        "to_version": update.to_version,
                        "patch_protocol": 2,
                        "files": tuple((path, dict(update.files)[path]) for path in replace),
                        "deleted": tuple(deleted),
                    })(),
                )
            )
            self.assertEqual((first / "_internal" / "data" / "hero_details.csv").read_text(encoding="utf-8"), "new")
            self.assertTrue((first / "_internal" / "data" / "fresh.txt").is_file())

            replace_second, deleted_second = plan_manifest_update(second, update, payload)
            self.assertIn("_internal/data/stale.txt", deleted_second)
            self.assertIn("windows_version.json", replace_second)

    def test_protocol_jump_bridge_and_unsupported_change(self) -> None:
        with self._temp_dir() as root:
            baseline = root / "baseline"
            target = root / "target"
            write_tree(baseline, "1.0.2", marker="old")
            write_tree(target, "1.0.3", marker="new")
            write_version_file(target / VERSION_FILENAME, AppVersion("1.0.3", 2))
            write_app_manifest(target)
            summary = build_patch(baseline=baseline, target=target, output=root / "bridge.zip")
            self.assertFalse(summary["requires_full_update"])
            manifest = read_manifest_from_zip(root / "bridge.zip")
            self.assertEqual(manifest.patch_protocol, 1)
            self.assertEqual(manifest.to_version, "1.0.3")
            self.assertIn("app_manifest.json", dict(manifest.files))

            write_version_file(target / VERSION_FILENAME, AppVersion("2.0.0", 3))
            rejected = build_patch(baseline=baseline, target=target, output=root / "patch.zip")
            self.assertTrue(rejected["requires_full_update"])
            self.assertFalse((root / "patch.zip").exists())

    def test_powershell_apply_and_rollback(self) -> None:
        if sys.platform != "win32":
            self.skipTest("PowerShell patch applier is Windows-only")
        with self._temp_dir() as root:
            app = root / "app"
            target = root / "target"
            write_tree(app, "1.0.0", marker="old", extra={"_internal/data/remove_me.txt": "gone"})
            write_tree(target, "1.0.1", marker="new", extra={"_internal/data/added.txt": "added"})
            patch = app / "E7_BP_Helper_patch_1.0.0_to_1.0.1.zip"
            build_patch(baseline=app, target=target, output=patch)
            manifest = read_manifest_from_zip(patch)
            staging = stage_patch(app, patch, manifest)
            script = write_patch_job(
                app_dir=app,
                zip_path=patch,
                manifest=manifest,
                staging=staging,
                pid=0,
                relaunch=False,
            )
            job_path = staging / "patch_job.json"
            job = json.loads(job_path.read_text(encoding="utf-8"))
            job["fail_on"] = "_internal/data/added.txt"
            job_path.write_text(json.dumps(job), encoding="utf-8")
            failed = self._run_powershell(script, job_path)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual((app / "_internal" / "data" / "hero_details.csv").read_text(encoding="utf-8"), "old")
            self.assertTrue((app / "_internal" / "data" / "remove_me.txt").is_file())
            self.assertFalse((app / "_internal" / "data" / "added.txt").exists())

            staging = stage_patch(app, patch, manifest)
            script = write_patch_job(
                app_dir=app,
                zip_path=patch,
                manifest=manifest,
                staging=staging,
                pid=0,
                relaunch=False,
            )
            completed = self._run_powershell(script, staging / "patch_job.json")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(hashes(app), hashes(target))
            self.assertFalse(patch.is_file())
            self.assertEqual(load_applied_history(app)[-1]["from_version"], "1.0.0")

    def test_pending_patch_stages_without_starting_the_app(self) -> None:
        with self._temp_dir() as root:
            app = root / "app"
            target = root / "target"
            write_tree(app, "1.0.0", marker="old")
            write_tree(target, "1.0.1", marker="new")
            patch = app / "E7_BP_Helper_patch_1.0.0_to_1.0.1.zip"
            build_patch(baseline=app, target=target, output=patch)
            calls: list[Path] = []

            def fake_spawn(script_path: Path, job_path: Path) -> None:
                calls.append(job_path)
                self.assertTrue(script_path.is_file())

            import backend.windows_patch as windows_patch

            original = windows_patch.spawn_patch_applier
            windows_patch.spawn_patch_applier = fake_spawn
            try:
                exit_code = process_pending_patches(app, pid=123, relaunch=True)
            finally:
                windows_patch.spawn_patch_applier = original
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(calls), 1)
            job = json.loads(calls[0].read_text(encoding="utf-8"))
            self.assertEqual(job["wait_pid"], 123)
            self.assertTrue(job["relaunch"])

    def _job(self, app: Path, patch: Path, staging: Path, manifest) -> dict:
        return {
            "app_dir": str(app),
            "staging_dir": str(staging),
            "backup_dir": str(app / ".patch_backup" / manifest.patch_id),
            "zip_path": str(patch),
            "applied_dir": "patches_applied",
            "from_version": manifest.from_version,
            "files": [relative for relative, _digest in manifest.files],
            "deleted": list(manifest.deleted),
            "version": {"version": manifest.to_version, "patch_protocol": manifest.patch_protocol},
        }

    def _versioned_copy(self, root: Path, version: str) -> Path:
        path = root / f"tree-{version}"
        write_tree(path, version, marker=version)
        return path

    def _copy_tree(self, source: Path, destination: Path) -> None:
        import shutil

        shutil.copytree(source, destination)

    def _run_powershell(self, script: Path, job_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-JobPath",
                str(job_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def _temp_dir(self):
        import tempfile

        class _TempPath:
            def __enter__(self_inner):
                self_inner._directory = tempfile.TemporaryDirectory()
                return Path(self_inner._directory.__enter__())

            def __exit__(self_inner, exc_type, exc, traceback):
                return self_inner._directory.__exit__(exc_type, exc, traceback)

        return _TempPath()


class LauncherPatchOrderTests(unittest.TestCase):
    def test_development_launch_does_not_apply_patches(self) -> None:
        from backend.windows_patch import apply_pending_patch_and_maybe_exit

        self.assertIsNone(apply_pending_patch_and_maybe_exit())


if __name__ == "__main__":
    unittest.main()
