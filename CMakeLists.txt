cmake_minimum_required(VERSION 3.16)

project(nku_pcfg_serial_baseline VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)
option(PCFG_ENABLE_MPI "Build the MPI sampling frontend" OFF)

add_library(pcfg_core
    src/batch_plan_io.cpp
    src/consumer.cpp
    src/md5.cpp
    src/md5_avx2.cpp
    src/metrics.cpp
    src/model.cpp
    src/model_io.cpp
    src/sampler.cpp
    src/sha256.cpp
)
target_include_directories(pcfg_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(pcfg_core PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(pcfg_core PUBLIC OpenMP::OpenMP_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
    set_source_files_properties(src/md5_avx2.cpp PROPERTIES COMPILE_OPTIONS "-march=x86-64")
endif()

add_executable(pcfg src/main.cpp)
target_link_libraries(pcfg PRIVATE pcfg_core)
target_compile_options(pcfg PRIVATE -Wall -Wextra -Wpedantic)

if(PCFG_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_executable(pcfg_mpi src/mpi_main.cpp)
    target_link_libraries(pcfg_mpi PRIVATE pcfg_core MPI::MPI_CXX)
    target_compile_options(pcfg_mpi PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(pcfg_tests tests/test_main.cpp)
    target_link_libraries(pcfg_tests PRIVATE pcfg_core)
    target_compile_options(pcfg_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME pcfg_tests COMMAND pcfg_tests)
    if(PCFG_ENABLE_MPI)
        add_test(NAME pcfg_mpi_smoke
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                         $<TARGET_FILE:pcfg_mpi> self-test)
        set_tests_properties(pcfg_mpi_smoke PROPERTIES ENVIRONMENT "TMPDIR=/tmp")
    endif()
endif()
