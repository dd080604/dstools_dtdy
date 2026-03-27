from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="dstools_dtdy.linear_model_cython",
        sources=["src/dstools_dtdy/linear_model_cython.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
