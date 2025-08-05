% Compilation requires:
% 	- mexStitchCorr.h & mexStitchCorr.cpp
% 	- BaseStitchCorr.h
% 	- nlopt.h, nlopt.hpp, nlopt.lib, nlopt.dll (all accessible via vcpkg)
% 	- stitchCorr.lib (precompiled via turning on the compile parameter LIBRARY in the CMAKE file)
%
% All files needed to be placed in the same folder and the following commands to be run.
% One needs to make sure to use the same toolchain between the stitchCorr.lib precompile and the mex settings (eg. MSVC).
%
% To use the function, nlopt.dll has to be in the same folder.

mex -setup CPP
mex -output mexStitchCorr COMPFLAGS="$COMPFLAGS /std:c++20" mexStitchCorr.cpp  stitchCorr.lib nlopt.lib