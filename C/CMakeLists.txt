cmake_minimum_required(VERSION 3.9)
project(alocv)

include(CheckIPOSupported)

find_library(mkl_core NAMES mkl_core libmkl_core
             HINTS ENV ICPP_COMPILER18 ENV LIBRARY_PATH
             PATH_SUFFIXES mkl/lib/intel64)

find_library(mkl_tbb_thread NAMES mkl_tbb_thread libmkl_tbb_thread
             HINTS ENV ICPP_COMPILER18 ENV LIBRARY_PATH
             PATH_SUFFIXES mkl/lib/intel64)

find_library(mkl_intel_lp64 NAMES mkl_intel_lp64 libmkl_intel_lp64
             HINTS ENV ICPP_COMPILER18 ENV LIBRARY_PATH
             PATH_SUFFIXES mkl/lib/intel64)

find_library(intel_tbb NAMES tbb
             HINTS ENV ICPP_COMPILER18 ENV LIBRARY_PATH
             PATH_SUFFIXES tbb/lib/intel64/vc14)

find_path(intel_mkl_include NAMES mkl.h
          HINTS ENV ICPP_COMPILER18 ENV MKLROOT
		  PATH_SUFFIXES mkl/include include)

set(MKL_LIBRARIES ${mkl_core} ${mkl_tbb_thread} ${mkl_intel_lp64})

message(STATUS "Found MKL libs: ${MKL_LIBRARIES}")
message(STATUS "Found MKL headers: ${intel_mkl_include}")

add_library(alocv STATIC src/cholesky_utils.c
                         src/alo_lasso.cpp)
target_compile_definitions(alocv PUBLIC USE_MKL)
target_include_directories(alocv PUBLIC include)
target_include_directories(alocv PRIVATE src "${intel_mkl_include}")
target_link_libraries(alocv PUBLIC ${MKL_LIBRARIES} ${intel_tbb})

check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
	set_property(TARGET alocv PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

add_executable(bench bench/main.c)
target_link_libraries(bench PRIVATE alocv)
target_compile_definitions(bench PRIVATE _CRT_SECURE_NO_WARNINGS)
