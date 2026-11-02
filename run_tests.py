"""テストを一時フォルダにコピーしてから実行する。

本家のテストは画像や csv を作業フォルダに書き出すので、フォルダを汚さないために分けている。
  python run_tests.py
一時フォルダは英数字のパスにする。cv2.imread は Windows で日本語のパスを読めない。
"""
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

FORK = Path(__file__).resolve().parent
MODULES = ["test_math", "test_tortuosity_measures", "test_tortuosity", "test_configuration", "test_retina",
           "test_landmarks", "test_retina_grayscale", "test_vessel_classification", "test_fork"]


def main():
    warnings.filterwarnings("ignore")
    work = tempfile.mkdtemp(prefix="retipy_fork_")
    try:
        shutil.copytree(FORK / "retipy", Path(work) / "retipy")
        os.chdir(work)
        sys.path.insert(0, str(Path(work) / "retipy"))
        import matplotlib
        matplotlib.use("Agg")
        suite = unittest.TestSuite()
        for name in sys.argv[1:] or MODULES:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName("test." + name))
        with open(os.devnull, "w") as quiet:
            result = unittest.TextTestRunner(verbosity=0, stream=quiet).run(suite)
        print(f"{result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
        for test, message in result.failures + result.errors:
            print(" -", test.id(), "|", [x for x in message.strip().splitlines() if x.strip()][-1][:120])
        return 1 if result.failures or result.errors else 0
    finally:
        os.chdir(FORK)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
