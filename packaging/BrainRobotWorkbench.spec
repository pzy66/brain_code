# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path.cwd()

library_bin = ROOT / ".venv" / "Library" / "bin"
binaries = []
if library_bin.exists():
    binary_names = {
        "ffi-8.dll",
        "jpeg8.dll",
        "libbz2.dll",
        "libcrypto-3-x64.dll",
        "libexpat.dll",
        "liblzma.dll",
        "libpng16.dll",
        "libssl-3-x64.dll",
    }
    for path in library_bin.glob("Qt5*conda.dll"):
        binaries.append((str(path), "."))
    for name in sorted(binary_names):
        path = library_bin / name
        if path.exists():
            binaries.append((str(path), "."))

datas = []
robot_bundle = ROOT / "hybrid_controller" / "robot"
if robot_bundle.exists():
    datas.append((str(robot_bundle), "hybrid_controller/robot"))
brainflow_lib = ROOT / ".venv" / "lib" / "site-packages" / "brainflow" / "lib"
if brainflow_lib.exists():
    datas.append((str(brainflow_lib), "brainflow/lib"))

datas += collect_data_files("ultralytics", include_py_files=False)

for source, target in (
    (ROOT / "datasets" / "vision" / "models" / "best.pt", "datasets/vision/models"),
    (
        ROOT / "datasets" / "vision" / "calibration" / "current_profile.json",
        "datasets/vision/calibration",
    ),
    (
        ROOT
        / "datasets"
        / "profiles"
        / "hybrid_controller"
        / "vision_grasp"
        / "current_grasp_profile.json",
        "datasets/profiles/hybrid_controller/vision_grasp",
    ),
    (
        ROOT
        / "datasets"
        / "profiles"
        / "hybrid_controller"
        / "robot_pick_tuning"
        / "current_pick_tuning.json",
        "datasets/profiles/hybrid_controller/robot_pick_tuning",
    ),
    (
        ROOT
        / "datasets"
        / "profiles"
        / "hybrid_controller"
        / "ssvep_profiles"
        / "current_fbcca_profile.json",
        "datasets/profiles/hybrid_controller/ssvep_profiles",
    ),
    (
        ROOT
        / "datasets"
        / "profiles"
        / "hybrid_controller"
        / "ssvep_profiles"
        / "default_fbcca_profile.json",
        "datasets/profiles/hybrid_controller/ssvep_profiles",
    ),
):
    if source.exists():
        datas.append((str(source), target))

hiddenimports = [
    "PyQt5.sip",
    "robot_workbench.app",
    "robot_workbench.flow_ui",
    "hybrid_controller.adapters.rosbridge_client",
    "hybrid_controller.adapters.teleop_ros_channel",
    "hybrid_controller.adapters.robot_client",
    "hybrid_controller.robot.tools.jetmax_start_ros_runtime",
    "brainflow",
    "brainflow.board_shim",
    "brainflow_compat",
    "serial.tools.list_ports",
    "paramiko",
    "cv2",
    "lap",
    "numpy",
    "PIL",
    "PIL.Image",
    "torch",
    "torchvision",
    "ultralytics",
    "yaml",
    "twisted.internet.selectreactor",
]
hiddenimports += collect_submodules("ultralytics")

a = Analysis(
    [str(ROOT / "run_integrated_workbench.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "mne",
        "moabb",
        "pyriemann",
        "cupy",
        "jupyterlab",
        "notebook",
        "IPython",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BrainRobotWorkbench",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="BrainRobotWorkbench",
)
