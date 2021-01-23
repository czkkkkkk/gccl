find_package(GTest)

if(GTEST_FOUND)
  message(STATUS "Enable testing")
  enable_testing()
  file(GLOB_RECURSE UNIT_TEST_SOURCE ${PROJECT_SOURCE_DIR}/test/*_unittest.cc) 
  include_directories(${GTEST_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/test)
  add_executable(${PROJECT_NAME}_unit_tests ${UNIT_TEST_SOURCE} unittest_main.cc)
  set_property(TARGET ${PROJECT_NAME}_unit_tests PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
  target_link_libraries(${PROJECT_NAME}_unit_tests gccl_static ${GTEST_LIBRARY} ${GCCL_EXTERNAL_LIB})
  add_test(NAME AllTestsIn${PROJECT_NAME}UnitTests
    COMMAND ${PROJECT_NAME}_unit_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(tests COMMAND "${CMAKE_CTEST_COMMAND}" COMMENT "Running tests..." DEPENDS ${PROJECT_NAME}_unit_tests)

  if (GPU_TEST)
    message(STATUS "GPU test is enabled")
    file(GLOB_RECURSE UNIT_TEST_GPU_SOURCE ${PROJECT_SOURCE_DIR}/test/*unittest_gpu.cu ${PROJECT_SOURCE_DIR}/test/*.cu) 
    add_executable(${PROJECT_NAME}_gpu_unit_tests ${UNIT_TEST_GPU_SOURCE} unittest_main.cc)
    set_property(TARGET ${PROJECT_NAME}_gpu_unit_tests PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
    set_target_properties(${PROJECT_NAME}_gpu_unit_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${PROJECT_NAME}_gpu_unit_tests PROPERTIES CUDA_STANDARD 11)
    target_link_libraries(${PROJECT_NAME}_gpu_unit_tests gccl_static ${GTEST_LIBRARY} ${GCCL_EXTERNAL_LIB})
    add_test(NAME AllTestsIn${PROJECT_NAME}GPUUnitTests
      COMMAND ${PROJECT_NAME}_gpu_unit_tests
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_dependencies(tests ${PROJECT_NAME}_gpu_unit_tests)
  else()
    message(STATUS "GPU test is not enabled")
  endif()

  if (MPI_TEST)
    message(STATUS "MPI test is enabled")
    file(GLOB_RECURSE UNIT_TEST_MPI_SOURCE ${PROJECT_SOURCE_DIR}/test/*unittest_mpi.cc)
    list(APPEND UNIT_TEST_MPI_SOURCE ${PROJECT_SOURCE_DIR}/test/test_mpi_utils.cc)
    if (GPU_TEST)
      message(STATUS "MPI GPU test is enabled")
      list(APPEND UNIT_TEST_MPI_SOURCE api_unittest_mpi_gpu.cc)
    else()
      message(STATUS "MPI GPU test is not enabled")
    endif()
    add_executable(${PROJECT_NAME}_mpi_unit_tests ${UNIT_TEST_MPI_SOURCE} unittest_main.cc)
    set_property(TARGET ${PROJECT_NAME}_mpi_unit_tests PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
    set_target_properties(${PROJECT_NAME}_mpi_unit_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 11)
    target_link_libraries(${PROJECT_NAME}_mpi_unit_tests gccl_static ${GTEST_LIBRARY} ${GCCL_EXTERNAL_LIB})
    target_compile_definitions(${PROJECT_NAME}_mpi_unit_tests PUBLIC MPI_TEST)
    add_test(NAME AllTestsIn${PROJECT_NAME}MPIUnitTests
      COMMAND mpirun -np 3 --oversubscribe --allow-run-as-root ${PROJECT_NAME}_mpi_unit_tests
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_dependencies(tests ${PROJECT_NAME}_mpi_unit_tests)
  else()
    message(STATUS "MPI test is not enabled")
  endif()

  # Cross machine test
  if (CM_TEST)
    message(STATUS "Cross machine test is enabled")
    file(GLOB_RECURSE UNIT_TEST_CM_SOURCE ${PROJECT_SOURCE_DIR}/test/api_unittest_mpi_gpu.cc)
    list(APPEND UNIT_TEST_CM_SOURCE ${PROJECT_SOURCE_DIR}/test/test_mpi_utils.cc)
    add_executable(${PROJECT_NAME}_cm_unit_tests ${UNIT_TEST_CM_SOURCE} unittest_main.cc)
    set_property(TARGET ${PROJECT_NAME}_cm_unit_tests PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
    set_target_properties(${PROJECT_NAME}_cm_unit_tests PROPERTIES CUDA_SEPARABLE_COCMLATION ON CUDA_STANDARD 11)
    target_link_libraries(${PROJECT_NAME}_cm_unit_tests gccl_static ${GTEST_LIBRARY} ${GCCL_EXTERNAL_LIB})
    target_compile_definitions(${PROJECT_NAME}_cm_unit_tests PUBLIC MPI_TEST) # To setup mpi env
  else()
    message(STATUS "Cross machine test is not enabled")
  endif()

else()

  message(WARNING "Google Test not found")
endif()
