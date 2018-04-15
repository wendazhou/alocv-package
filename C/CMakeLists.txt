cmake_minimum_required(VERSION 3.5)
project(alocv)

add_library(alocv src/cholesky_utils.c)
target_include_directories(alocv PUBLIC include)
target_include_directories(alocv PRIVATE src "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018/windows/mkl/include")