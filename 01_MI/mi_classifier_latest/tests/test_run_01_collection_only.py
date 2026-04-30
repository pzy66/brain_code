import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
COLLECTION_ROOT = PROJECT_ROOT / "code" / "collection"
if str(COLLECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(COLLECTION_ROOT))

import run_01_collection_only
import run_collection_pycharm


class RootCollectionLauncherTests(unittest.TestCase):
    def test_main_runs_real_collector_entrypoint(self) -> None:
        with mock.patch("run_01_collection_only.runpy.run_path") as mocked_run_path:
            exit_code = run_01_collection_only.main()

        self.assertEqual(exit_code, 0)
        mocked_run_path.assert_called_once()
        target = str(mocked_run_path.call_args.args[0]).replace("\\", "/")
        self.assertTrue(target.endswith("code/collection/mi_data_collector.py"))
        self.assertEqual(mocked_run_path.call_args.kwargs, {"run_name": "__main__"})

    def test_pycharm_launcher_delegates_to_collector_main(self) -> None:
        with mock.patch("mi_data_collector.main", return_value=23) as mocked_main:
            exit_code = run_collection_pycharm.main()

        self.assertEqual(exit_code, 23)
        mocked_main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
