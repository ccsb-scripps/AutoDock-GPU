cmake_minimum_required(VERSION 3.18)
project(AutoDock-GPU CXX)

execute_process(COMMAND ${GIT_EXECUTABLE} describe --abbrev=40 --dirty --tags --always
    OUTPUT_VARIABLE GIT_REPO_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(${GIT_REPO_VERSION} STREQUAL "")
  file(READ BASE_VERSION GIT_REPO_VERSION)
  set(GIT_REPO_VERSION "${GIT_REPO_VERSION}-release")
endif()
add_definitions(-DVERSION=${GIT_REPO_VERSION})
message(STATUS "AutoDock-GPU Version ${GIT_REPO_VERSION}")

option(BUILD_OPENCL "Build AutoDock-GPU with OpenCL backend" ON)
option(BUILD_CUDA "Build AutoDock-GPU with CUDA backend" OFF)

include_directories(
  host/inc
  common)

file(GLOB ADGPU_SRC host/src/*.cpp)

if(BUILD_CUDA)
  message(STATUS "Building with CUDA")
  enable_language(CUDA)
  include_directories(cuda)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  file(COPY_FILE host/inc/performdocking.h.Cuda host/inc/performdocking.h)
  file(COPY_FILE host/src/performdocking.cpp.Cuda host/src/performdocking.cpp)
  list(APPEND ADGPU_SRC "cuda/kernels.cu" "host/src/performdocking.cpp")
elseif(BUILD_OPENCL)
  message(STATUS "Building with OpenCL")
  find_package(OpenCL REQUIRED)
  add_definitions(-DCL_TARGET_OPENCL_VERSION=220)
  file(GLOB ADGPU_SRC_OPENCL device/*.cl)
  file(GLOB ADGPU_SRC_OPENCL_CPP wrapcl/src/*.cpp)
  include_directories(${OpenCL_INCLUDE_DIRS} "device" "wrapcl/inc")
  link_directories(${OpenCL_LIBRARY})
  file(COPY_FILE host/inc/performdocking.h.OpenCL host/inc/performdocking.h)
  file(COPY_FILE host/src/performdocking.cpp.OpenCL host/src/performdocking.cpp)
  list(APPEND ADGPU_SRC ${ADGPU_SRC_OPENCL} ${ADGPU_SRC_OPENCL_CPP} "host/src/performdocking.cpp")
else()
  message(FATAL_ERROR "Please specify -DBUILD_OPENCL or -DBUILD_CUDA option when invoking CMake")
endif()

add_executable(adgpu ${ADGPU_SRC})
if(BUILD_CUDA)
  set_property(TARGET adgpu PROPERTY CUDA_STANDARD 17)
elseif(BUILD_OPENCL)
  target_link_libraries(adgpu
    ${OpenCL_LIBRARY}
    OpenCL::OpenCL)
  add_custom_target(
    stringify
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/stringify_ocl_krnls.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_dependencies(adgpu stringify)
endif()
