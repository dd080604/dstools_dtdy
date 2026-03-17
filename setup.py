from setuptools import setup, Extension
from setuptools.command.build_py import build_py
from Cython.Build import cythonize
import numpy as np
import os
import subprocess
import sys


class CustomBuildPy(build_py):
    def run(self):
        self.build_qr_shared_library()
        super().run()

    def build_qr_shared_library(self):
        root = os.path.abspath(os.path.dirname(__file__))
        src_dir = os.path.join(root, "src", "dstools_dtdy")
        c_file = os.path.join(src_dir, "qr.c")
        so_file = os.path.join(src_dir, "qr.so")

        if not os.path.exists(c_file):
            raise FileNotFoundError(f"Could not find {c_file}")

        if sys.platform.startswith("linux") or sys.platform == "darwin":
            cmd = [
                "gcc",
                "-shared",
                "-fPIC",
                "-O2",
                "-o",
                so_file,
                c_file,
            ]
        else:
            raise RuntimeError("Automatic qr.so build is only configured here for Linux/macOS.")

        print("Building qr.so with command:")
        print(" ".join(cmd))
        subprocess.check_call(cmd)

        if not os.path.exists(so_file):
            raise RuntimeError(f"Failed to create {so_file}")


extensions = [
    Extension(
        name="dstools_dtdy.linear_model_cython",
        sources=["src/dstools_dtdy/linear_model_cython.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
    cmdclass={"build_py": CustomBuildPy},
)
