from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [Extension('pyintersect',
                sources=['src/pyintersect.pyx',
                         'src/intersection.cc'], 
                         language='c++',
                         extra_compile_args=['-msse4', '-std=c++11',  '-pedantic', '-Wno-write-strings', '-O3'],
                         include_dirs=[numpy.get_include()])]

setup(
    ext_modules = cythonize(extensions)
)


