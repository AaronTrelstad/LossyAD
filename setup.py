from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

cython_dir = "cython_modules"

pyx_files = [os.path.join(cython_dir, f) for f in os.listdir(cython_dir) if f.endswith(".pyx")]

setup(
    name="pip",
    ext_modules=cythonize(pyx_files, compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()],
)
