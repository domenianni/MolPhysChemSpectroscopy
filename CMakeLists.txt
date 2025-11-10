cmake_minimum_required(VERSION 3.22)
project(pyStitchCorr)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

include_directories(pySpec/pyStitchCorr/build)

find_package(NLopt CONFIG REQUIRED)

add_definitions(-DPYTHON_LIB)
find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
set(PYBIND11_PYTHON_VERSION 3.11)
pybind11_add_module(stitchCorr
        pySpec/pyStitchCorr/build/pyStitchCorr.h     pySpec/pyStitchCorr/build/pyStitchCorr.cpp
        pySpec/pyStitchCorr/build/BaseStitchCorr.cpp pySpec/pyStitchCorr/build/BaseStitchCorr.h)

target_link_libraries(stitchCorr PRIVATE NLopt::nlopt)
