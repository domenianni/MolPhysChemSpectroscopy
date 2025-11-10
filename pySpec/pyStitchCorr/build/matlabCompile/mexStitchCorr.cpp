//
// Created by bauer on 04.08.2025.
//

#include "mexStitchCorr.h"

void MexFunction::operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
    checkArguments(outputs, inputs);

    matlab::data::TypedArray<double> mx = std::move(inputs[0]);
    matlab::data::TypedArray<double> mt = std::move(inputs[1]);
    matlab::data::TypedArray<double> my = std::move(inputs[2]);

    const unsigned int blockAmount{inputs[3][0]};
    const double xStride{1};
    const double tStride{blockAmount*32.0};

    const bool sortedInput{inputs[4][0]};

    stitchCorrData data{
        std::vector<double>(mx.begin(), mx.end()),
        std::vector<double>(mt.begin(), mt.end()),
        std::vector<double>(my.begin(), my.end())
    };

    auto* sc = new BaseStitchCorr{data,
                                  blockAmount,
                                  static_cast<unsigned int>(xStride),
                                  static_cast<unsigned int>(tStride),
                                  sortedInput};

    sc->correctStitch(std::nullopt, false);

    int i {0};
    for (auto val: data.y) {
        my[i] = val;
        ++i;
    }

    outputs[0] = my;
}

void MexFunction::checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
    // Get pointer to engine
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    // Get array factory
    matlab::data::ArrayFactory factory;

    if (inputs.size() != 5){matlabPtr->feval(u"error",
        0,
        std::vector<matlab::data::Array>({ factory.createScalar("Input must be x, t, y,the block amount and whether the array is sorted.") }));
    }

    for (int i{0}; i < 3; i++){
    // Check array argument: the first three must be double arrays
    if (inputs[i].getType() != matlab::data::ArrayType::DOUBLE ||
        inputs[i].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE){
            matlabPtr->feval(u"error",
            0,
            std::vector<matlab::data::Array>({ factory.createScalar("Input must be double array") }));
        }
    }
    // Check number of outputs
    if (outputs.size() > 1) {
        matlabPtr->feval(u"error",
            0,
            std::vector<matlab::data::Array>({ factory.createScalar("Only one output is returned") }));
    }
}