cmake_minimum_required(VERSION 3.22)
project(pyStitchCorr)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

option(PYTHON_LIB "Compile as python lib" ON)
set(PYTHON_LIB ON)
option(STANDALONE_EXE "Compile as a standalone executable" OFF)

add_definitions(-fPIC)

find_package(NLopt CONFIG REQUIRED)

if(PYTHON_LIB)
    add_definitions(-DPYTHON_LIB)
    find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    set(PYBIND11_PYTHON_VERSION 3.11)
    pybind11_add_module(stitchCorr pyStitchCorr.cpp)
    target_link_libraries(stitchCorr PRIVATE NLopt::nlopt)
elseif(STANDALONE_EXE)
    add_executable(stitchCorr stitchCorrParser.cpp pyStitchCorr.cpp inputParser.cpp inputParser.h main.cpp)
    target_link_libraries(stitchCorr PRIVATE NLopt::nlopt)
endif()

