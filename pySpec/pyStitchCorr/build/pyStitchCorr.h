#ifndef PYSTITCHCORR_PYSTITCHCORR_H
#define PYSTITCHCORR_PYSTITCHCORR_H

#define USING(x) ((1 x 1) == 2)
#define ON +
#define OFF -

#if defined(PYTHON_LIB)
#define PY ON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#else
#   define PY OFF
#endif

#include "inputParser.h"
#include "BaseStitchCorr.h"
#include "stitchCorrParser.h"
#include <fstream>

void callStitchCorr(py::array_t<double>& x,
                    py::array_t<double>& t,
                    py::array_t<double>& y,
                    unsigned int blockCount=4,
                    int reference=-1,
                    bool sortedInput=false,
                    bool isAsymmetric=false,
                    bool linear=false);

#endif //PYSTITCHCORR_PYSTITCHCORR_H
