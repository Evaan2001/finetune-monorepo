load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(
    name="train_model",
    srcs=["train_model.py"],
    deps=[
        "@pypi//torch",
        "@pypi//transformers",
        "@pypi//datasets",
        "@pypi//accelerate",
        "@pypi//psutil",
    ],
    visibility=["//visibility:public"],
)

sh_test(
    name="training_test",
    srcs=["run_training.sh"],
    data=[
        ":train_model",
    ],
    size="large",
)
