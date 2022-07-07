import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

extensions = [
    Extension("alocv._helper_c",
              ['alocv/_helper_c.pyx',
               '../C/src/cholesky_utils.c',
               '../C/src/alo_enet.cpp',
               '../C/src/alo_lasso.cpp',
               '../C/src/gram_utils.cpp'],
              include_dirs=
              [np.get_include(),
               '../C/include',
               '../C/src'] +
              np.__config__.blas_opt_info['include_dirs'],
              library_dirs=np.__config__.blas_opt_info['library_dirs'],
              libraries=np.__config__.blas_opt_info['libraries'],
              define_macros=[('USE_MKL', 1)])
]

setup(
    name='alocv',
    version='0.1.0',
    author='Wenda Zhou',
    author_email='wz2335@columbia.edu',
    packages=find_packages(),
    ext_modules=cythonize(extensions, include_path=[np.get_include()])
)
