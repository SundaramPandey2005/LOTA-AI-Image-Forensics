"""
LOTA Mathematical & Engineering Sanity Checks Runner
Executes full automated test suite verifying bit-planes, MGPS convolutions, and model architectures.
"""
import sys
import subprocess


def run_sanity_checks():
    print("=" * 70)
    print("  RUNNING LOTA AUTOMATED SANITY CHECKS & UNIT TESTS")
    print("=" * 70)

    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("  [SUCCESS] ALL 13 MATHEMATICAL & ARCHITECTURAL CHECKS PASSED!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("  [FAILED] SOME TESTS FAILED. CHECK TRACEBACK ABOVE.")
        print("=" * 70)
        sys.exit(result.returncode)


if __name__ == "__main__":
    run_sanity_checks()
