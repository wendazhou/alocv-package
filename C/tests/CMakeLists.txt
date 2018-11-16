cmake_minimum_required(VERSION 3.9)

add_executable(test main.cpp test_rfp.cpp test_alo_svm.cpp)
target_include_directories(test PRIVATE ../src "${intel_mkl_include}")
target_link_libraries(test PRIVATE alocv)