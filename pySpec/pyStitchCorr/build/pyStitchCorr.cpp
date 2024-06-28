#include "pyStitchCorr.h"

#if (USING(PY))



void callStitchCorr(py::array_t<double>& x,
                    py::array_t<double>& t,
                    py::array_t<double>& y,
                    const unsigned int   blockCount,
                    const int            reference,
                    const bool           sortedInput,
                    const bool           separateEvenOdd,
                    const bool           linear){
    /**
     * @param x, t, y: numpy-arrays from python call. No changes necessary.
     * @param blockCount: Amount of stitching blocks utilised.
     * @param reference: Which stitching block should be used as reference. Select -1 for no reference.
     * @param sortedInput: Whether stitching blocks are already interlaced along the x-axis.
     * @param isAsymmetric: Whether even and odd pixels are treated in two separate stitching blocks.
     * @param linear: Whether a linear correction should be executed. Only works reliably with a reference, but for
     *                testing purposes this is not enforced.
     *
     * @returns numpy array of calculated offsets. y-values are changed also automatically, if copy=false.
     */

    if (separateEvenOdd && !sortedInput){
        throw pybind11::value_error("If even and odd pixels are to be treated "
                                            "separately, the stitching blocks have to be sorted "
                                            "and interlaced!");
    }

    stitchCorrData data{};

    // Read Data of the buffers and assign them to a vector to be used inside the correction program.
    py::buffer_info Xinfo = x.request();
    auto Xptr = static_cast<double *>(Xinfo.ptr);
    auto Xshape = static_cast<int>(Xinfo.shape[0]);
    std::vector<double> xVector;
    data.x.assign(Xptr, Xptr + Xshape);

    py::buffer_info Yinfo = y.request();
    auto Yptr = static_cast<double *>(Yinfo.ptr);
    auto Yshape = static_cast<int>(Yinfo.shape[0] * Yinfo.shape[1]);
    std::vector<double> yVector;
    data.y.assign(Yptr, Yptr + Yshape);

    py::buffer_info Tinfo = t.request();
    auto Tptr = static_cast<double *>(Tinfo.ptr);
    auto Tshape = static_cast<int>(Tinfo.shape[0]);
    std::vector<double> tVector;
    data.t.assign(Tptr, Tptr + Tshape);

    // std::ofstream debug {"debug.txt"};
    // debug << "X-Stride: " << Xinfo.strides[0] << ";\t X-shape: " << Xinfo.shape[0] << " " << std::endl;
    // debug << "T-stride: " << Tinfo.strides[0] << ";\t T-shape: " << Tinfo.shape[0] << " " << std::endl << std::endl;
    // debug << "Y-Strides: "<< Yinfo.strides[0] << "\t" << Yinfo.strides[1]
    //       << ";\t Y-shape: " << Yinfo.shape[0] << "\t" << Yinfo.shape[1] << std::endl << std::endl;
    // debug << (Xinfo.shape[0] == Yinfo.shape[0]) << std::endl;

    // The lengths of the arrays are compared and the strides assigned for the y-Array, since the orientation is not
    // known a priori.
    py::ssize_t xStride;
    py::ssize_t tStride;
    if (Xinfo.shape[0] == Yinfo.shape[0]) {
        xStride = Yinfo.strides[0];
        tStride = Yinfo.strides[1];
    }
    else {
        xStride = Yinfo.strides[1];
        tStride = Yinfo.strides[0];
    }
    // debug << "Y-Stride x: "<< xStride << "\t"
    //       << "Y-Stride t: "<< tStride << std::endl << std::endl;

    // Strides are given in byte length, therefore they need to be converted to index length.
    xStride = xStride / sizeof(long long);
    tStride = tStride / sizeof(long long);
    // debug << "Y-Stride x: "<< static_cast<unsigned int>(xStride) << "\t"
    //       << "Y-Stride t: "<< static_cast<unsigned int>(tStride) << std::endl << std::endl;

    auto* sc = new BaseStitchCorr{data,
                                  blockCount * (1 + separateEvenOdd),
                                  static_cast<unsigned int>(xStride),
                                  static_cast<unsigned int>(tStride),
                                  sortedInput};

    if (reference == -1) {
        sc->correctStitch(std::nullopt, linear);
    }
    else {
        sc->correctStitch(std::optional<unsigned int>(reference), linear);
    }

    // Since the values are only changed in the copied vectors, they have to be assigned to the numpy containers.
    int i {0};
    for (auto val: data.y) {
        Yptr[i] = val;
        ++i;
    }
}

// Generates bindings for the python module. Add function/class definitions here!
PYBIND11_MODULE(stitchCorr, m) {
m.def("stitchCorr", &callStitchCorr);
}

#endif