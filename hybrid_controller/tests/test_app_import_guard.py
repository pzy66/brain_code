from __future__ import annotations

import importlib
import sys
from unittest import mock


def test_app_import_does_not_require_vision_deps_at_module_import_time() -> None:
    sys.modules.pop("hybrid_controller.app", None)
    for module_name in list(sys.modules):
        if module_name.startswith("hybrid_controller.vision"):
            sys.modules.pop(module_name, None)

    with mock.patch.dict(sys.modules, {"cv2": None, "torch": None}):
        module = importlib.import_module("hybrid_controller.app")

    assert hasattr(module, "main")
