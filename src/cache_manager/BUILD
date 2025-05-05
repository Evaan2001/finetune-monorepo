load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(
    name="cache_manager",
    srcs=["cache_manager.py"],
    deps=[
        "@pypi//torch",
        "@pypi//transformers",
        "@pypi//accelerate",
        "@pypi//psutil",
        "//src/training:train_model",  # to access train_classifier
    ],
    imports=[".."],  # Add this line to allow imports relative to the workspace root
    visibility=["//visibility:public"],
)

sh_test(
    name="cache_manager_test",
    srcs=["run_cache_manager.sh"],
    data=[":cache_manager"],
)
