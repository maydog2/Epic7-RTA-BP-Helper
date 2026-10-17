# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
project_root = Path(SPECPATH)

runtime_data_files = [
    project_root / "data" / "hero_details.csv",
    project_root / "data" / "transformer" / "rec_model_position_warfare.weights.h5",
    project_root / "data" / "transformer" / "rec_variables.pkl",
    project_root / "data" / "transformer" / "training_report_position_warfare.json",
]

for required_file in runtime_data_files:
    if not required_file.is_file():
        raise SystemExit(f"Missing required runtime file: {required_file}")

if not (project_root / "data" / "runtime").is_dir():
    raise SystemExit("Missing data/runtime. Run workflow_scripts/build_runtime_artifacts.py first.")

if not (project_root / "frontend" / "dist" / "index.html").is_file():
    raise SystemExit("Missing frontend/dist. Run `npm run build` in frontend/ first.")

version_file = project_root / "build" / "windows_version.json"
if not version_file.is_file():
    raise SystemExit("Missing build/windows_version.json. Run .\\build_exe.ps1 so the packaged app can detect patches.")

datas = [
    (str(project_root / "data" / "hero_details.csv"), "data"),
    (str(project_root / "data" / "transformer" / "rec_model_position_warfare.weights.h5"), "data/transformer"),
    (str(project_root / "data" / "transformer" / "rec_variables.pkl"), "data/transformer"),
    (str(project_root / "data" / "transformer" / "training_report_position_warfare.json"), "data/transformer"),
    (str(project_root / "data" / "reranker"), "data/reranker"),
    (str(project_root / "data" / "runtime"), "data/runtime"),
    (str(project_root / "dataset"), "dataset"),
    (str(project_root / "CharacterUI" / "elements"), "CharacterUI/elements"),
    (str(project_root / "CharacterUI" / "roles"), "CharacterUI/roles"),
    (str(project_root / "frontend" / "dist"), "frontend/dist"),
    (str(version_file), "."),
] + collect_data_files("tf_keras")

hiddenimports = (
    collect_submodules("tf_keras")
    + collect_submodules("tensorflow")
    + collect_submodules("backend")
    + [
        "h5py",
        "transformer_draft_data",
        "transformer_draft_model",
        "match_history_utils",
    ]
)

a = Analysis(
    ["launcher.py"],
    pathex=[str(project_root), str(project_root / "workflow_scripts"), str(project_root / "backend")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="E7_BP_Helper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="E7_BP_Helper",
)
