include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonAMD ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_amd.cc LINK_LIBS TritonAMDGPUToLLVM TritonAMDGPUTransforms TritonAMDGPUDialectToLLVM)
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
