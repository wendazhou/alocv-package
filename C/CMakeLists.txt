cmake_minimum_required(VERSION 3.5)
project(alocv)

find_library(mkl_core NAMES mkl_core
             HINTS ENV ICPP_COMPILER18
             PATH_SUFFIXES mkl/lib/intel64)

find_library(mkl_tbb_thread NAMES mkl_tbb_thread
             HINTS ENV ICPP_COMPILER18
             PATH_SUFFIXES mkl/lib/intel64)

find_library(mkl_intel_lp64 NAMES mkl_intel_lp64
             HINTS ENV ICPP_COMPILER18 
             PATH_SUFFIXES mkl/lib/intel64)

find_library(intel_tbb NAMES tbb
             HINTS ENV ICPP_COMPILER18
             PATH_SUFFIXES tbb/lib/intel64/vc14)

set(intel_tbb_dll "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.2.185/windows/redist/intel64/tbb/vc14/tbb.dll")
set(MKL_LIBRARIES ${mkl_core} ${mkl_tbb_thread} ${mkl_intel_lp64})

message(STATUS "Found MKL libs: ${MKL_LIBRARIES}")

add_library(alocv STATIC src/cholesky_utils.c
                         src/alo_lasso.cpp)
target_compile_definitions(alocv PUBLIC USE_MKL)
target_include_directories(alocv PUBLIC include)
target_include_directories(alocv PRIVATE src "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018/windows/mkl/include")
target_link_libraries(alocv PUBLIC ${MKL_LIBRARIES} ${intel_tbb})

add_executable(bench bench/main.c)
target_link_libraries(bench PRIVATE alocv)
target_compile_definitions(bench PRIVATE _CRT_SECURE_NO_WARNINGS)
add_custom_command(TARGET bench POST_BUILD COMMAND
                   ${CMAKE_COMMAND} -E copy_if_different 
                   "${intel_tbb_dll}"
                   $<TARGET_FILE_DIR:bench>)