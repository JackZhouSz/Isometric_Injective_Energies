cmake_minimum_required(VERSION 3.15.0)
project(Iso_TLC_SEA LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Do not use shared lib.")


include(FetchContent)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(Eigen3)
include(cli11)
include(nlohmann-json)
include(nlopt)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

add_library(injectiveEnergy STATIC ${SRC_LIST})
target_include_directories(injectiveEnergy PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(injectiveEnergy PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(injectiveEnergy PUBLIC OpenMP::OpenMP_CXX)
endif()

# solver_test
add_executable(solver_test apps/solver_test.cpp)
target_link_libraries(solver_test PUBLIC injectiveEnergy nlopt::nlopt)

# TLC (2D)
add_executable(TLC_2D apps/TLC_2D.cpp)
target_link_libraries(TLC_2D PUBLIC injectiveEnergy CLI11::CLI11 nlopt::nlopt)
# set_target_properties(TLC_2D PROPERTIES LINKER_LANGUAGE CXX)

# IsoTLC (2D)


# TLC (3D)

# IsoTLC (3D)

# SEA (2D)

# SEA (3D)



# suitesparse

set(SUITESPARSE_INCLUDE_DIR /usr/local/include CACHE PATH "suitesparse include directory")
target_include_directories(TLC_2D PUBLIC "${SUITESPARSE_INCLUDE_DIR}")
target_include_directories(solver_test PUBLIC "${SUITESPARSE_INCLUDE_DIR}")

if (WIN32)
    # ------------------------------------------------------------------
    # Detect SuiteSparse libraries:
    # If not found automatically, set SuiteSparse_DIR in CMake to the
    # directory where SuiteSparse-config.cmake was installed.
    # ------------------------------------------------------------------
    find_package(SuiteSparse CONFIG REQUIRED)
    target_link_libraries(TLC_2D PRIVATE SuiteSparse::CHOLMOD)
    target_link_libraries(solver_test PRIVATE SuiteSparse::CHOLMOD)
else()
    set(SUITESPARSE_LIB_DIR /usr/local/lib CACHE PATH "suitesparse library directory")
    # target_link_libraries(TLC_2D PUBLIC -L"${SUITESPARSE_LIB_DIR}" cholmod)
    target_link_libraries(TLC_2D PRIVATE -L"${SUITESPARSE_LIB_DIR}" cholmod)
    target_link_libraries(solver_test PRIVATE -L"${SUITESPARSE_LIB_DIR}" cholmod)
endif()


# Math library
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(injectiveEnergy PUBLIC ${MATH_LIBRARY})
endif()

