//
// Created by bauer on 04.08.2025.
//

#ifndef PYSTITCHCORR_MEXSTITCHCORR_H
#define PYSTITCHCORR_MEXSTITCHCORR_H

#include "mex.hpp"
#include "mexAdapter.hpp"
#include "BaseStitchCorr.h"

class MexFunction final : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs);
    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs);
};


#endif //PYSTITCHCORR_MEXSTITCHCORR_H