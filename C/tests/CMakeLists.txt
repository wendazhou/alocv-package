cmake_minimum_required(VERSION 3.9)

add_executable(test main.cpp test_rfp.cpp test_rfp_manipulation.cpp test_alo_svm.cpp test_alo_lasso.cpp)
target_include_directories(test PRIVATE ../src "${intel_mkl_include}")
target_link_libraries(test PRIVATE alocv)

if(USE_INTERPROCEDURAL_OPT)
	set_property(TARGET test PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()