cmake_minimum_required(VERSION 3.9)
project(alocv)

include(CheckIPOSupported)

find_library(mkl_rt NAMES mkl_rt libmkl_rt
             HINTS ENV ICPP_COMPILER19 ENV LIBRARY_PATH
	     PATH_SUFFIXES mkl/lib/intel64)

find_path(intel_mkl_include NAMES mkl.h
          HINTS ENV ICPP_COMPILER19 ENV MKLROOT
          PATH_SUFFIXES mkl/include include)

set(MKL_LIBRARIES ${mkl_rt})

message(STATUS "Found MKL libs: ${MKL_LIBRARIES}")
message(STATUS "Found MKL headers: ${intel_mkl_include}")

add_library(alocv STATIC src/cholesky_utils.c
                         src/alo_lasso.cpp
                         src/alo_enet.cpp
                         src/gram_utils.cpp
                         src/alo_svm.cpp)
target_compile_definitions(alocv PUBLIC USE_MKL)
target_include_directories(alocv PUBLIC include)
target_include_directories(alocv PRIVATE src "${intel_mkl_include}")
target_link_libraries(alocv PUBLIC ${MKL_LIBRARIES})


if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19)
        target_compile_options(alocv PRIVATE /permissive- /Zc:__cplusplus /std:c++17)
    endif()
endif()

add_executable(bench bench/main.c)
target_link_libraries(bench PRIVATE alocv)
target_compile_definitions(bench PRIVATE _CRT_SECURE_NO_WARNINGS)

check_ipo_supported(RESULT ipo_supported)
if(ipo_supported AND NOT(CMAKE_BUILD_TYPE EQUAL "DEBUG"))
	set(USE_INTERPROCEDURAL_OPT TRUE)
endif()

if(USE_INTERPROCEDURAL_OPT)
	set_property(TARGET alocv PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
	set_property(TARGET bench PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

add_subdirectory(tests)
