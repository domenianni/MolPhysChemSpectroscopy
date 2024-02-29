#include "pyStitchCorr.h"

double wrapper(const std::vector<double> &offset, std::vector<double>& grad, void * data){
    return reinterpret_cast<StitchCorr*>(data)->calcTV(offset, grad);
}

StitchCorr& StitchCorr::setup(){
/**
 * Allocates the x, y and t arrays and assigns them to the member variables. Is called if copy=true
 */
    m_x = new double[m_blockCount * s_pixelCount]{};
    m_y = new double[m_blockCount * s_pixelCount * m_timesteps]{};
    m_t = new double[m_timesteps];

    return *this;
}

StitchCorr& StitchCorr::compareBlocks(){
    /*
     * Stitch Correction algorithm with simple built-in logger.
     */

    std::ofstream log {"log.txt"};

    log << "Blocks: " << m_blockCount << "; Pixel: " << s_pixelCount << "; Timesteps: " << m_timesteps << std::endl;

    // Member variable (current time step) used to increment through time steps. Retrieved inside the target function
    while(m_currTimeStep < m_timesteps){
        std::vector<double> offset(2*m_blockCount, m_offsetInitValue);

        log << "Optimising timestep " << m_currTimeStep << "( " << m_t[m_currTimeStep] << " )" << std::endl;

        /* Settings for the minimization algorithm.
         * -Since the target function does not supply gradients LN prefix is necessary.
         *      Possible Algorithms include:
         *      -> LN_NELDERMEAD, LN_BOBYQA, LN_COBYLA
         * -Maxeval sets amount of maximal target function evaluation (SUCCESS CODE 5 if reached!)
         * -ftol and xtol are convergence criteria (target fuction difference / x-value difference). They are set to
         *  absolute values, since the minimization optimizes towards ZERO!
         *
         *  Target function wrapped outside the class to allow access to the class variables.
         */

        nlopt::opt opt{nlopt::opt(m_algorithm, 2*m_blockCount)};
        opt.set_min_objective(wrapper, this);
        opt.set_maxeval(m_maxeval);
        opt.set_ftol_abs(m_ftol);
        opt.set_xtol_abs(m_xtol);

        double minf{};
        try{
            nlopt::result result = opt.optimize(offset, minf);
            log << result << std::endl << "Offset: [ ";
            for (int i{0}; i < 2*m_blockCount; ++i){
                log << offset[i] << " ";
            }
            log << "]" << std::endl;

            addOffset(offset, log);
        }
        catch(std::exception &e) {
            log << "nlopt failed: " << e.what() << std::endl;
        }
        ++m_currTimeStep;
    }
    return *this;
}

double StitchCorr::calcTV(const std::vector<double>& offset, std::vector<double> &grad){
    double sum{0};
    int elements{0};

    for (int i{0}; i < m_blockCount; ++i){
        for (int j{0}; j < m_blockCount; ++j){

            if (i==j){continue;}
            /* skip calculations so that no block is compared against itself (even though it
             * would work for the odd-even comparison)
             *
             * Since wavelength difference for each pixel pair is equidistant between two blocks it can be calculated
             * early.
             */

            double dx_inv = 1 / std::abs(m_x[ravelIndex(0, i, 0)] - m_x[ravelIndex(0, j, 0)]);
            double dx_asm = 1 / std::abs(m_x[ravelIndex(1, i, 0)] - m_x[ravelIndex(0, j, 0)]);

            for (int k{0}; k < s_pixelCount - 1; ++k){
                /* Checks if any of the calculated pixel is a NAN and skips the evaluation in this case, since it would
                 * lead to undefined behaviour! */
                if (    std::isnan(m_y[ravelIndex(k, i, m_currTimeStep)]) |
                        std::isnan(m_y[ravelIndex(k, j, m_currTimeStep)]) |
                        std::isnan(m_y[ravelIndex(k+1, i, m_currTimeStep)]))
                {continue;}

                // calculates the total variance between the same pixels of different stitch blocks
                sum += std::abs(
                        (m_y[ravelIndex(k, i, m_currTimeStep)] +
                         offset[ravelIndex(static_cast<int>(k % 2 != 0), i, 0, 2)]) -
                        (m_y[ravelIndex(k, j, m_currTimeStep)] +
                         offset[ravelIndex(static_cast<int>(k % 2 != 0), j, 0, 2)])
                ) * dx_inv;

                // calculates the total variance between even-odd / odd-even adjacent pixels of different stitch blocks
                sum += std::abs(
                        (m_y[ravelIndex(k+1, i, m_currTimeStep)] +
                         offset[ravelIndex(static_cast<int>((k+1) % 2 != 0), i, 0, 2)]) -
                        (m_y[ravelIndex(k, j, m_currTimeStep)] +
                         offset[ravelIndex(static_cast<int>(k % 2 != 0), j, 0, 2)])
                ) * dx_asm;

                elements += 2;
            }
        }
    }
    /* for now only the sum is returned. Though I don't know if it is necessary to calculate the mean, since this would
     * account for the existence of NAN values. Though inclusion of taking the mean will slow down the program and
     * necessitate a check for complete NAN values, so that no undefined behaviour occurs.
     * if (elements == 0){return 0;}
     */
    return sum; // /elements;
}

StitchCorr& StitchCorr::addOffset(std::vector<double>& offset, std::ofstream& log){
    /*
     * @param offset: the calculated offset values from the minimization algorithm. Shape: blockCount * 2
     *
     * @param log: std::ofstream of the log-file.
     *
     * Corrects the member DOD values (m_y) by the calculated offset values. Also uses the built-in current timestep
     * counter.
     */

    int asym{2}; // Just used since the blockCount needs to be multiplied by two for odd-even stitch correction

    for (int i{0}; i < m_blockCount; ++i){
        for (int j{0}; j < s_pixelCount; ++j){
            m_y[ravelIndex(j, i, m_currTimeStep)] += offset[ravelIndex(static_cast<int>(j % 2 != 0), i, 0, asym)];
        }
    }

    for(int i{0}; i < m_blockCount * asym; ++i) {
        m_offset[i + m_currTimeStep * (m_blockCount * 2)] = offset[i];
    }
    return *this;
}

int StitchCorr::ravelIndex(const int pixelIdx, const int blockIdx, const int timeIdx, const int xLen) const{
    /*
     * Calculates the index from a 3D flattened array.
     *
     * @param pixelIdx, blockIdx, timeIdx: Indices of the positions.
     *
     * @param xLen: the length of the first axis.
     *
     * The length of the second axis is calculated from the block count stored in the member variable.
     */
    if (m_sortedInput){
        return (blockIdx + pixelIdx * m_blockCount + xLen * m_blockCount * timeIdx);
    }
    return (pixelIdx + (xLen) * blockIdx + (xLen * m_blockCount) * timeIdx);
}

// int StitchCorr::ravelIndexStride(const int pixelIdx, const int blockIdx, const int timeIdx, const int xLen) {}

#if (USING(PY))

int ravelIndexComplete(const int pixelIdx, const int blockIdx, int pixelCount){
    return (pixelIdx + (pixelCount) * blockIdx);
}

double calcTVFunc(py::array& offArray, py::array& xArray, py::array& yArray){
    /*
     * Pure target function to be called from python frontend and minimized within this context. Only one timestep at
     * a time is processed.
     *
     * @param offArray: Numpy array of offset values, which are added ontop of DOD values.
     *
     * @param xArray, yArray: Numpy array of the respective wavelength and DOD values.
     *
     * @returns Mean total variance (one double value).
     *
     * See documentation inside the class for more details on the evaluated function.
     */

    py::buffer_info offInfo = offArray.request();
    auto offset = static_cast<double *>(offInfo.ptr);
    auto offShape = static_cast<int>(offInfo.shape[0]);

    py::buffer_info Xinfo = xArray.request();
    auto x = static_cast<double *>(Xinfo.ptr);
    auto Xshape = static_cast<int>(Xinfo.shape[0]);

    py::buffer_info Yinfo = yArray.request();
    auto y = static_cast<double *>(Yinfo.ptr);
    auto Yshape = static_cast<int>(Yinfo.shape[0]);

    int blockCount{offShape/2};
    int pixelCount{Xshape / blockCount};

    if (Yshape != blockCount * pixelCount){throw std::exception();}

    double sum{0};
    int elements{0};

    for (int i{0}; i < blockCount; ++i){
        for (int j{0}; j < blockCount; ++j){
            if (i==j){
                continue;
            }
            double dx_inv = abs(1 / (x[ravelIndexComplete(0, i, pixelCount)] - x[ravelIndexComplete(0, j, pixelCount)]));

            for (int k{0}; k < pixelCount - 2; ++k){
                sum = sum + (
                        std::abs((y[ravelIndexComplete(k, i, pixelCount)] +
                                offset[ravelIndexComplete(static_cast<int>((k % 2 != 0)), i, 2)]) -
                               (y[ravelIndexComplete(k, j, pixelCount)] +
                                offset[ravelIndexComplete(static_cast<int>(k % 2 != 0), j, 2)])
                       )) * dx_inv;

                sum = sum + (
                        std::abs((y[ravelIndexComplete(k + 1, i, pixelCount)] +
                                offset[ravelIndexComplete(static_cast<int>((k + 1) % 2 != 0), i, 2)]) -
                               (y[ravelIndexComplete(k, j, pixelCount)] +
                                offset[ravelIndexComplete(static_cast<int>(k % 2 != 0), j, 2)])
                       )) * dx_inv;

                ++elements;
            }
        }
    }
    return sum / elements;
}

py::array_t<double> callStitchCorr(py::array& x, py::array& t, py::array& y, const int blockCount=4, bool copy=true){
    /**
     * @param x, t, y: numpy-arrays from python call. No changes necessary.
     *
     * @param blockCount: Amount of stitching blocks utilised.
     *
     * @param copy: flag, whether arrays should be copied inside the stitch corr class.
     *
     * @returns numpy array of calculated offsets. y-values are changed also automatically, if copy=false.
     */

    py::buffer_info Xinfo = x.request();
    auto Xptr = static_cast<double *>(Xinfo.ptr);
    auto Xshape = static_cast<int>(Xinfo.shape[0]);

    py::buffer_info Yinfo = y.request();
    auto Yptr = static_cast<double *>(Yinfo.ptr);
    auto Yshape = static_cast<int>(Yinfo.shape[0] * Yinfo.shape[1]);

    std::ofstream debug {"debug.txt"};
    debug << Yinfo.strides[0] << " " << Yinfo.strides[1] << Yinfo.shape[0] << " " << Yinfo.shape[1];

    // TODO: REMOVE AND CHANGE TO ON THE FLY SELECTION OF STRIDES
    int smallStride;
    if (Yinfo.strides[0] < Yinfo.strides[1]){smallStride = 0;}
    else {smallStride = 1;}
    if (Yinfo.shape[smallStride] != Xinfo.shape[0]){throw std::exception("Memory layout incorrect!");}

    // if (Xinfo.shape[0] == Yinfo.shape[0]) {
    //     int xStride{Yinfo.stride[0]};
    //     int tStride{Yinfo.stride[1]};
    // }
    // else {
    //     int xStride{Yinfo.stride[1]};
    //     int tStride{Yinfo.stride[0]};
    // }
    // if (xStride > tStride){
    //     xStride = xStride / tStride
    //     tStride = 1
    // }
    // else {
    //     xStride = 1
    //     tStride = tStride / xStride
    // }


    py::buffer_info Tinfo = t.request();
    auto Tptr = static_cast<double *>(Tinfo.ptr);
    auto Tshape = static_cast<int>(Tinfo.shape[0]);

    StitchCorr sc = StitchCorr(Xptr, Xshape, Tptr, Tshape, Yptr, Yshape, blockCount, copy);

    return sc.get_offset();
}

// Generates bindings for the python module. Add function/class definitions here!
PYBIND11_MODULE(stitchCorr, m) {
m.def("stitchCorr", &callStitchCorr);
m.def("stitchCorrFunc", &calcTVFunc);
}

#endif