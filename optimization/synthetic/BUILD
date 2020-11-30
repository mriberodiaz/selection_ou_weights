load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])


py_library(
    name = "federated_synthetic",
    srcs = ["federated_synthetic.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:training_loop",
        "//utils:training_utils",
        "//utils/datasets:synthetic_dataset",
    ],
)

py_library(
    name = "federated_synthetic_random",
    srcs = ["federated_synthetic_random.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:training_loop_random",
        "//utils:training_utils",
        "//utils/datasets:synthetic_dataset",
    ],
)
