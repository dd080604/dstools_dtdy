from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np
import os
import subprocess
import sys


class CustomBuildExt(build_ext):
    def run(self):
        super().run()
        self.build_qr_shared_library()

    def build_qr_shared_library(self):
        src_dir = os.path.join(os.path.dirname(__file__), "src", "dstools_dtdy")
        c_file = os.path.join(src_dir, "qr.c")
        so_file = os.path.join(src_dir, "qr.so")

        if not os.path.exists(c_file):
            raise FileNotFoundError(f"Could not find {c_file}")

        # Linux / Colab
        if sys.platform.startswith("linux"):
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
            raise RuntimeError(
                "Automatic qr.so build in setup.py is currently configured for Linux/macOS only."
            )

        print("Building qr.so with command:")
        print(" ".join(cmd))
        subprocess.check_call(cmd)


extensions = [
    Extension(
        name="dstools_dtdy.linear_model_cython",
        sources=["src/dstools_dtdy/linear_model_cython.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
    cmdclass={"build_ext": CustomBuildExt},
)
