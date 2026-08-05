#!/usr/bin/env python3
"""Fail when a Linux wheel contains an absolute ELF RPATH or RUNPATH."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def absolute_runtime_paths(binary):
    output = subprocess.check_output(["readelf", "-d", binary], text=True)
    paths = []
    for line in output.splitlines():
        if "(RPATH)" not in line and "(RUNPATH)" not in line:
            continue
        value = line.partition("[")[2].rpartition("]")[0]
        paths.extend(path for path in value.split(":") if os.path.isabs(path))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    failures = []

    with tempfile.TemporaryDirectory(prefix="biosaur2-wheel-rpath-") as directory:
        directory = Path(directory)
        for wheel in args.wheels:
            with zipfile.ZipFile(wheel) as archive:
                for member in archive.namelist():
                    if not member.endswith(".so"):
                        continue
                    target = directory / wheel.stem / member
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(member))
                    for path in absolute_runtime_paths(target):
                        failures.append("%s: %s" % (wheel, path))

    if failures:
        print("Absolute ELF runtime paths found:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
