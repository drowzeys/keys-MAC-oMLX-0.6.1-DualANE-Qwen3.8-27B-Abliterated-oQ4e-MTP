import os, sys
from setuptools import setup
from mlx import extension

os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "15.0")
cmake_args = os.environ.get("CMAKE_ARGS", "").strip()
extra = f"-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 -DPython_EXECUTABLE={sys.executable} -DPython3_EXECUTABLE={sys.executable}"
os.environ["CMAKE_ARGS"] = f"{cmake_args} {extra}".strip()

setup(
    name="omlx-qwen35-prefill-kernel",
    version="0.6.1",
    ext_modules=[
        extension.CMakeExtension(
            "omlx.custom_kernels.qwen35_prefill._ext",
            sourcedir="omlx/custom_kernels/qwen35_prefill/csrc",
        )
    ],
    cmdclass={"build_ext": extension.CMakeBuild},
)
