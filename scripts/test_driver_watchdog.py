"""Regression test for the hang that idled out a whole Kaggle session.

Version 3 of the tweet notebook stopped 34 minutes in, mid-print, and sat until the 12-hour limit.
The driver now sends every command's output to a log file instead of a pipe, copies that file to the
console from a separate thread, and kills a command that writes nothing for SILENCE_LIMIT_S. This
checks all three without a GPU or any corpus: output lands in the log with its exit code intact, a
silent command is killed long before it would have finished, and a slow command that keeps printing
is left alone.

    python scripts/test_driver_watchdog.py
"""
import importlib.util
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_driver():
    spec = importlib.util.spec_from_file_location("rb", os.path.join(ROOT, "kaggle", "run_benchmark.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rb"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    rb = load_driver()
    path = os.path.join(tempfile.mkdtemp(), "bench.log")
    rb.log_to_file(path)
    py = f'"{sys.executable}"'

    def log():
        return open(path, encoding="utf-8", errors="replace").read()

    checks = []
    rc = rb.sh(f'{py} -c "import sys; print(123456789); print(987654321, file=sys.stderr)"')
    checks.append(("a command's stdout and stderr both land in the log", "123456789" in log() and "987654321" in log(), True))
    checks.append(("a clean exit returns 0", rc, 0))

    rc = rb.sh(f'{py} -c "import sys; sys.exit(3)"', check=False)
    checks.append(("a failing command's exit code comes back", rc, 3))

    rb.SILENCE_LIMIT_S = 4
    t0 = time.time()
    rc = rb.sh(f'{py} -c "import time; time.sleep(120)"', check=False)
    checks.append(("a silent command is killed long before it would finish", time.time() - t0 < 60, True))
    checks.append(("and reported as a failure", rc != 0, True))
    checks.append(("with a line in the log saying why", "WATCHDOG" in log(), True))

    t0 = time.time()
    rc = rb.sh(f'{py} -c "import time; [(print(i, flush=True), time.sleep(1)) for i in range(8)]"', check=False)
    checks.append(("a slow command that keeps printing is left alone", rc == 0 and time.time() - t0 >= 7, True))

    ok = True
    for name, got, want in checks:
        good = got == want
        ok &= good
        print(("PASS " if good else "FAIL ") + name + ("" if good else f" (got {got!r}, wanted {want!r})"))
    print("DRIVER WATCHDOG TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
