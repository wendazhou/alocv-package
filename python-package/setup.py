import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

extensions = [
    Extension("alocv._cholesky_c", ['alocv/_cholesky_c.pyx'],
              include_dirs=[np.get_include()])
]

setup(
    name='alocv',
    version='0.1.0',
    author='Wenda Zhou',
    author_email='wz2335@columbia.edu',
    packages=find_packages(),
    ext_modules=cythonize(extensions, include_path=[np.get_include()])
)
