cmake_minimum_required(VERSION 3.22)
project(pyStitchCorr)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

option(PYTHON_LIB "Compile as python lib" OFF)
option(LIBRARY "Compile as dll" OFF)

# Uncomment the following line to compile as python package:
set(PYTHON_LIB ON)
set(LIBRARY OFF)
set(EXE OFF)

include_directories(pySpec/pyStitchCorr/build)

find_package(NLopt CONFIG REQUIRED)

if(PYTHON_LIB)
    add_definitions(-DPYTHON_LIB)
    find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    set(PYBIND11_PYTHON_VERSION 3.11)
    pybind11_add_module(stitchCorr
            pySpec/pyStitchCorr/build/pyStitchCorr.h     pySpec/pyStitchCorr/build/pyStitchCorr.cpp
            pySpec/pyStitchCorr/build/BaseStitchCorr.cpp pySpec/pyStitchCorr/build/BaseStitchCorr.h)

    target_link_libraries(stitchCorr PRIVATE NLopt::nlopt)
endif()

if(EXE)
    add_executable(stitchCorr
                   pySpec/pyStitchCorr/build/stitchCorrParser.cpp pySpec/pyStitchCorr/build/stitchCorrParser.h
                   pySpec/pyStitchCorr/build/inputParser.cpp      pySpec/pyStitchCorr/build/inputParser.h
                   pySpec/pyStitchCorr/build/BaseStitchCorr.cpp   pySpec/pyStitchCorr/build/BaseStitchCorr.h
                   pySpec/pyStitchCorr/build/main.cpp)
    target_link_libraries(stitchCorr PRIVATE NLopt::nlopt)
endif()

if(LIBRARY)
    #include_directories("C:/Program Files/MATLAB/R2024a/extern/include")

    add_library(stitchCorr STATIC
            BaseStitchCorr.cpp   BaseStitchCorr.h
            #mexStitchCorr.h mexStitchCorr.cpp
    )
    target_link_libraries(stitchCorr PRIVATE NLopt::nlopt)
endif()

