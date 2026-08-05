from build_support import strip_absolute_runtime_paths


def test_strip_absolute_runtime_paths_removes_conda_linker_flags():
    arguments = [
        "gcc",
        "-shared",
        "-Wl,-rpath,/data/p/anaconda3/lib",
        "-Wl,-rpath-link,/data/p/anaconda3/lib",
        "-L/data/p/anaconda3/lib",
    ]

    assert strip_absolute_runtime_paths(arguments) == [
        "gcc",
        "-shared",
        "-Wl,-rpath-link,/data/p/anaconda3/lib",
        "-L/data/p/anaconda3/lib",
    ]


def test_strip_absolute_runtime_paths_keeps_relative_paths_and_other_flags():
    arguments = [
        "-Wl,-rpath,$ORIGIN",
        "-Wl,-rpath",
        "relative/lib",
        "-Rrelative/lib",
        "-Wl,--as-needed",
    ]

    assert strip_absolute_runtime_paths(arguments) == arguments


def test_strip_absolute_runtime_paths_handles_split_and_r_option_forms():
    arguments = [
        "-Wl,-rpath",
        "/opt/runtime",
        "-R/usr/local/lib",
        "-R",
        "/tmp/runtime",
        "-pthread",
    ]

    assert strip_absolute_runtime_paths(arguments) == ["-pthread"]
