from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="pybingham",
        sources=["bingham_c.pyx"],
        libraries=["bingham"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-w"]
    )
]

setup(
    name="pybingham",
    ext_modules=cythonize(extensions, language_level="3")
)

