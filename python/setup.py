from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

bingham_extension = Extension(
    name="pybingham",
    sources=["bingham_c.pyx"],
    libraries=["bingham"],
)

setup(
    name="pybingham",
    ext_modules=cythonize([bingham_extension])
)

