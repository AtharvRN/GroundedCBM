#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight MedicalCBM smoke tests.")
    parser.add_argument(
        "--pattern",
        default="test_medical*.py",
        help="unittest discovery pattern inside tests/.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use unittest verbosity=2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    tests_dir = repo_root / "tests"
    suite = unittest.defaultTestLoader.discover(str(tests_dir), pattern=str(args.pattern))
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
