cmake_minimum_required(VERSION 3.18)
set (CMAKE_CXX_STANDARD 11)
project(AutoDock-GPU CXX)

find_package(Git REQUIRED)
if(OVERLAP)
  if(APPLE)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
      set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
      set(OpenMP_C_LIB_NAMES "omp")
      set(OpenMP_omp_LIBRARY omp)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
      set(OpenMP_CXX_LIB_NAMES "omp")
      set(OpenMP_omp_LIBRARY omp)
    endif()
  endif(APPLE)
  find_package(OpenMP REQUIRED)
endif()

execute_process(COMMAND ${GIT_EXECUTABLE} describe --abbrev=40 --dirty --tags --always
    OUTPUT_VARIABLE GIT_REPO_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(${GIT_REPO_VERSION} STREQUAL "")
  file(READ BASE_VERSION GIT_REPO_VERSION)
  set(GIT_REPO_VERSION "${GIT_REPO_VERSION}-release")
endif()
add_definitions(-DVERSION="${GIT_REPO_VERSION}")
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
  file(CREATE_LINK host/inc/performdocking.h.Cuda host/inc/performdocking.h COPY_ON_ERROR)
  file(CREATE_LINK host/src/performdocking.cpp.Cuda host/src/performdocking.cpp COPY_ON_ERROR)
  list(APPEND ADGPU_SRC "cuda/kernels.cu" "host/src/performdocking.cpp")
elseif(BUILD_OPENCL)
  message(STATUS "Building with OpenCL")
  if(NUMWI EQUAL 1)
    add_definitions(-DN1WI)
  elseif(NUMWI EQUAL 2)
    add_definitions(-DN2WI)
  elseif(NUMWI EQUAL 4)
    add_definitions(-DN4WI)
  elseif(NUMWI EQUAL 8)
    add_definitions(-DN8WI)
  elseif(NUMWI EQUAL 16)
    add_definitions(-DN16WI)
  elseif(NUMWI EQUAL 32)
    add_definitions(-DN32WI)
  elseif(NUMWI EQUAL 64)
    add_definitions(-DN64WI)
  elseif(NUMWI EQUAL 128)
    add_definitions(-DN128WI)
  elseif(NUMWI EQUAL 256)
    add_definitions(-DN256WI)
  else()
    add_definitions(-DN64WI)
  endif()
  find_package(OpenCL REQUIRED)
  add_definitions(-DCL_TARGET_OPENCL_VERSION=220)
  add_definitions(-DGPU_DEVICE)
  add_definitions(-DK1=gpu_calc_initpop -DK2=gpu_sum_evals -DK3=perform_LS -DK4=gpu_gen_and_eval_newpops -DK5=gradient_minSD -DK6=gradient_minFire -DK7=gradient_minAD)
  add_definitions(-DKRNL_SOURCE=./device/calcenergy.cl -DKRNL_DIRECTORY=./device -DKCMN_DIRECTORY=./common)
  file(GLOB ADGPU_SRC_OPENCL device/*.cl)
  file(GLOB ADGPU_SRC_OPENCL_CPP wrapcl/src/*.cpp)
  include_directories(${OpenCL_INCLUDE_DIRS} "device" "wrapcl/inc")
  link_directories(${OpenCL_LIBRARY})
  file(CREATE_LINK host/inc/performdocking.h.OpenCL host/inc/performdocking.h COPY_ON_ERROR)
  file(CREATE_LINK host/src/performdocking.cpp.OpenCL host/src/performdocking.cpp COPY_ON_ERROR)
  list(APPEND ADGPU_SRC ${ADGPU_SRC_OPENCL} ${ADGPU_SRC_OPENCL_CPP} "host/src/performdocking.cpp")
else()
  message(FATAL_ERROR "Please specify -DBUILD_OPENCL or -DBUILD_CUDA option when invoking CMake")
endif()

add_executable(adgpu ${ADGPU_SRC})

if(OVERLAP)
  add_definitions(-DUSE_PIPELINE)
  target_link_libraries(adgpu OpenMP::OpenMP_CXX)
endif()

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

include(CTest)
add_test(NAME test_ligand COMMAND
  ${CMAKE_CURRENT_BINARY_DIR}/adgpu
    -ffile ./input/3ce3/derived/3ce3_protein.maps.fld
    -lfile ./input/3ce3/derived/3ce3_ligand.pdbqt
    -nrun 100
    -ngen 27000
    -psize 150
    -resnam test
    -gfpop 0
    -lsmet sw
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
