cmake_minimum_required(VERSION 3.22)
project(pyStitchCorr)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

option(PYTHON_LIB "Compile as python lib" ON)

# Uncomment the following line to compile as python package:
#set(PYTHON_LIB ON)

find_package(NLopt CONFIG REQUIRED)

if(PYTHON_LIB)
    add_definitions(-DPYTHON_LIB)
    find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    set(PYBIND11_PYTHON_VERSION 3.11)
    pybind11_add_module(stitchCorr
                        pyStitchCorr.h     pyStitchCorr.cpp
                        BaseStitchCorr.cpp BaseStitchCorr.h)
    target_link_libraries(stitchCorr PRIVATE NLopt::nlopt)

else()
    add_executable(stitchCorr
                   stitchCorrParser.cpp stitchCorrParser.h
                   inputParser.cpp      inputParser.h
                   BaseStitchCorr.cpp   BaseStitchCorr.h
                   main.cpp)
    target_link_libraries(stitchCorr PRIVATE NLopt::nlopt)
endif()

