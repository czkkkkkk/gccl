cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(gccl LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

include_directories(/usr/local/cuda/include)

option(GCCL_DEBUG "Debug mode" OFF)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if ($ENV{GCCL_GPU_TEST})
  set(GPU_TEST ON)
else()
  set(GPU_TEST OFF)
endif()

if ($ENV{GCCL_CUDA})
  set(HAVE_CUDA ON)
  message(STATUS "GCCL has cuda")
else()
  set(HAVE_CUDA OFF)
  message(STATUS "GCCL does not have cuda")
endif()

if ($ENV{GCCL_MPI_TEST})
  set(MPI_TEST ON)
else()
  set(MPI_TEST OFF)
endif()

# Cross machine test
if ($ENV{GCCL_CM_TEST})
  set(CM_TEST ON)
else()
  set(CM_TEST OFF)
endif()

# pThread
find_package(Threads)
list(APPEND GCCL_EXTERNAL_LIB ${CMAKE_THREAD_LIBS_INIT})

# ZMQ
find_path(ZMQ_INCLUDE_DIR NAMES zmq.hpp)
find_library(ZMQ_LIBRARY NAMES zmq)

if(ZMQ_INCLUDE_DIR AND ZMQ_LIBRARY)
  message(STATUS "Found ZeroMQ:")
  message(STATUS "  (Headers) ${ZMQ_INCLUDE_DIR}")
  message(STATUS "  (Library) ${ZMQ_LIBRARY}")
  list(APPEND GCCL_EXTERNAL_INCLUDES ${ZMQ_INCLUDE_DIR})
  list(APPEND GCCL_EXTERNAL_LIB ${ZMQ_LIBRARY})
else()
  message(FATAL_ERROR "ZeorMQ not found")
endif()
include_directories(ZMQ_INCLUDE_DIR)

# GFLAGS
### gflags
find_path(GFLAGS_DIR NAMES gflags/gflags.h)
find_library(GFLAGS_LIBRARY NAMES gflags)
# find_package(gflags REQUIRED)
if(GFLAGS_DIR AND GFLAGS_LIBRARY)
  message(STATUS "Found gflags:")
  message(STATUS " (Headers)      ${GFLAGS_DIR}")
  message(STATUS " (Library)      ${GFLAGS_LIBRARY}")
  list(APPEND GCCL_EXTERNAL_INCLUDES ${GFLAGS_DIR})
  list(APPEND GCCL_EXTERNAL_LIB ${GFLAGS_LIBRARY})
else()
  message(FATAL_ERROR "GFLAGS not found")
endif()

# GLOG
find_path(glog_INCLUDE_DIR NAMES glog/logging.h)
find_library(glog_LIBRARY NAMES glog)
if(glog_INCLUDE_DIR AND glog_LIBRARY)
  message(STATUS "Found glog:")
  message(STATUS "  (Headers)       ${glog_INCLUDE_DIR}")
  message(STATUS "  (Library)       ${glog_LIBRARY}")
  list(APPEND GCCL_EXTERNAL_INCLUDES ${glog_INCLUDE_DIR})
  list(APPEND GCCL_EXTERNAL_LIB ${glog_LIBRARY})
else()
  message(FATAL_ERROR "glog not found")
endif()

# METIS
if(DEFINED ENV{METIS_HOME})
  find_library(metis_LIBRARY NAMES metis PATHS $ENV{METIS_HOME}/lib NO_DEFAULT_PATH)
  set(metis_INCLUDE_DIR $ENV{METIS_HOME}/include)
  message(STATUS "Found metis:")
  message(STATUS "  (Headers)       ${metis_INCLUDE_DIR}")
  message(STATUS "  (Library)       ${metis_LIBRARY}")
  list(APPEND GCCL_EXTERNAL_INCLUDES ${metis_INCLUDE_DIR})
  list(APPEND GCCL_EXTERNAL_LIB ${metis_LIBRARY})
else()
  message(FATAL_ERROR "METIS_HOME is not defined in environment")
endif()

# MPI
if(DEFINED ENV{MPI_HOME})
  find_library(mpi_LIBRARY NAMES mpi PATHS $ENV{MPI_HOME}/lib NO_DEFAULT_PATH)
  set(mpi_INCLUDE_DIR $ENV{MPI_HOME}/include)
  message(STATUS "Found mpi:")
  message(STATUS "  (Headers)       ${mpi_INCLUDE_DIR}")
  message(STATUS "  (Library)       ${mpi_LIBRARY}")
  list(APPEND GCCL_EXTERNAL_INCLUDES ${mpi_INCLUDE_DIR})
  list(APPEND GCCL_EXTERNAL_LIB ${mpi_LIBRARY})
else()
  message(FATAL_ERROR "MPI_HOME is not defined in environment")
endif()

# NLOHMANN JSON
if(DEFINED ENV{NLOHMANN_HOME})
  set(nlohmann_INCLUDE_DIR $ENV{NLOHMANN_HOME})
  message(STATUS "Found nlohmann json:")
  message(STATUS "  (Headers)       ${nlohmann_INCLUDE_DIR}")
  list(APPEND GCCL_EXTERNAL_INCLUDES ${nlohmann_INCLUDE_DIR})
else()
  message(FATAL_ERROR "NLOHMANN_HOME is not defined in environment")
endif()

# NVML
list(APPEND GCCL_EXTERNAL_LIB /usr/local/cuda/lib64/stubs/libnvidia-ml.so)

file(GLOB gccl_src
          "src/*.cc"
          "src/base/*.cc"
          "src/comm/*.cc"
          "src/comm/pattern/*.cc"
          "src/topo/*.cc"
          "src/conn/*.cc"
          "src/gpu/*.cc"
          "src/gpu/*.cu"
          "src/gpu/kernel/*.cu")

include_directories(src ${GCCL_EXTERNAL_INCLUDES})
# STATIC
add_library(gccl_static ${gccl_src})
if(${HAVE_CUDA})
  target_compile_definitions(gccl_static PUBLIC HAVE_CUDA)
endif()
set_target_properties(gccl_static PROPERTIES PUBLIC_HEADER "src/gccl.h")
set_target_properties(gccl_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gccl_static PROPERTIES CUDA_STANDARD 11)
target_link_libraries(gccl_static PRIVATE ${GCCL_EXTERNAL_LIB})

if(GCCL_DEBUG)
  message(STATUS "GCCL Debug is enabled")
  target_compile_definitions(gccl_static PRIVATE GCCL_DEBUG)
endif()

# SO
add_library(gccl SHARED ${gccl_src})
if(${HAVE_CUDA})
  target_compile_definitions(gccl PUBLIC HAVE_CUDA)
endif()
set_target_properties(gccl PROPERTIES PUBLIC_HEADER "src/gccl.h")
set_target_properties(gccl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gccl PROPERTIES CUDA_STANDARD 11)
set_target_properties(gccl PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Disable cxx11 abi since pytorch does not support it 
# target_compile_options(gccl PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(gccl PRIVATE ${GCCL_EXTERNAL_LIB})

if(GCCL_DEBUG)
  message(STATUS "GCCL Debug is enabled")
  target_compile_definitions(gccl PRIVATE GCCL_DEBUG)
endif()

add_executable(gag examples/gag.cc)
set_target_properties(gag PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gag PROPERTIES CXX_STANDARD ${CMAKE_CXX_STANDARD})
target_link_libraries(gag gccl_static)
target_link_libraries(gag ${GCCL_EXTERNAL_LIB})

# Test
include(CTest)
add_subdirectory(test)

# Install
if (DEFINED GCCL_HOME)
  message(STATUS "Set install prefix to from command line " ${GCCL_HOME})
  set (CMAKE_INSTALL_PREFIX ${GCCL_HOME} CACHE STRING "Install prefix" FORCE)
elseif (DEFINED ENV{GCCL_HOME})
  message(STATUS "Set install prefix to " $ENV{GCCL_HOME})
  set (CMAKE_INSTALL_PREFIX $ENV{GCCL_HOME} CACHE STRING "Install prefix" FORCE)
endif()

INSTALL(TARGETS gccl
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include)
