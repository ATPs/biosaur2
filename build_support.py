"""Build helpers for producing portable native extensions."""

import os
import sys

from setuptools.command.build_ext import build_ext


def strip_absolute_runtime_paths(arguments):
    """Drop absolute linker rpaths inherited from the active Python build."""
    cleaned = []
    index = 0
    arguments = list(arguments)
    while index < len(arguments):
        argument = arguments[index]
        if argument.startswith("-Wl,-rpath,"):
            path = argument.removeprefix("-Wl,-rpath,")
            if os.path.isabs(path):
                index += 1
                continue
        elif argument.startswith("-Wl,-rpath="):
            path = argument.removeprefix("-Wl,-rpath=")
            if os.path.isabs(path):
                index += 1
                continue
        elif argument == "-Wl,-rpath" and index + 1 < len(arguments):
            path = arguments[index + 1]
            if os.path.isabs(path):
                index += 2
                continue
        elif argument.startswith("-R") and len(argument) > 2:
            path = argument[2:]
            if os.path.isabs(path):
                index += 1
                continue
        elif argument == "-R" and index + 1 < len(arguments):
            path = arguments[index + 1]
            if os.path.isabs(path):
                index += 2
                continue
        cleaned.append(argument)
        index += 1
    return cleaned


class PortableBuildExt(build_ext):
    """Avoid embedding a build machine's Python library path in extensions."""

    def build_extensions(self):
        if sys.platform.startswith("linux"):
            self.compiler.linker_so = strip_absolute_runtime_paths(
                self.compiler.linker_so
            )
        super().build_extensions()
