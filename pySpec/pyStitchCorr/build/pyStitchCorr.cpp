#include "pyStitchCorr.h"

double wrapper_ref(const std::vector<double> &offset, std::vector<double>& grad, void* data){
    return reinterpret_cast<StitchCorr*>(data)->calcTVref(offset, grad);
}

double wrapper(const std::vector<double> &offset, std::vector<double>& grad, void* data){
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

    // Select the asymmetric kernel function for calcTV if the asymmetric flag is set.
    if (m_isAsymmetric) { m_kernel_ptr = &StitchCorr::kernel_asym; }

    log << "Blocks: " << m_blockCount << "; Pixel: " << s_pixelCount << "; Timesteps: " << m_timesteps << std::endl;

    // Member variable (current time step) used to increment through time steps. Retrieved inside the target function
    while(m_currTimeStep < m_timesteps){

        // offset is shorter by 1 if m_reference is not -1, meaning a reference block has been selected
        std::vector<double> offset(m_blockFactor*m_blockCount-(m_reference > -1), m_offsetInitValue);

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

        nlopt::opt opt{nlopt::opt(m_algorithm, m_blockFactor*m_blockCount-(m_reference > -1))};

        opt.set_min_objective(wrapper, this);
        if (m_reference > -1) { opt.set_min_objective(wrapper_ref, this); }

        opt.set_maxeval(m_maxeval);
        opt.set_ftol_abs(m_ftol);
        opt.set_xtol_abs(m_xtol);

        double minf{};
        try{
            nlopt::result result = opt.optimize(offset, minf);
            log << result << std::endl << "Offset: [ ";
            for (int i{0}; i < m_blockFactor*m_blockCount-(m_reference > -1); ++i){
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

double StitchCorr::calcTVref(const std::vector<double>& offset, std::vector<double> &grad) {
    // Calculates the total variance, with regard to one (signified by m_reference) block.
    // Only needs to loop then over one idx to cover all blocks.
    double sum{0};

    for (int i{0}; i < m_blockCount; ++i) {
        if (i == m_reference){continue;}

        sum = kernel_ref(sum, offset, i, m_reference);
        }

    return sum;
}

double StitchCorr::calcTV(const std::vector<double>& offset, std::vector<double> &grad){
    double sum{0};

    for (int i{0}; i < m_blockCount; ++i){
        for (int j{0}; j < m_blockCount; ++j){

            if (i==j){continue;}
            /* skip calculations so that no block is compared against itself (even though it
             * would work for the odd-even comparison)
             */

            sum = (this->*m_kernel_ptr)(sum, offset, i, j);
            }
        }
    /* for now only the sum is returned. Though I don't know if it is necessary to calculate the mean, since this would
     * account for the existence of NAN values. Though inclusion of taking the mean will slow down the program and
     * necessitate a check for complete NAN values, so that no undefined behaviour occurs.
     * if (elements == 0){return 0;}
     */
    return sum;
}

double StitchCorr::kernel_ref(double sum, const std::vector<double>& offset, int i, int ref) {
    /* Since wavelength difference for each pixel pair is equidistant between two blocks it can be calculated
     * early.
     */
    double dx_inv = 1 / std::abs(m_x[ravelIndex(0, i, 0)] - m_x[ravelIndex(0, ref, 0)]);

    for (int k{0}; k < s_pixelCount - 1; ++k){
        /* Checks if any of the calculated pixel is a NAN and skips the evaluation in this case, since it would
         * lead to undefined behaviour! */
        if (    std::isnan(m_y[ravelIndex(k, i, m_currTimeStep)]) |
                std::isnan(m_y[ravelIndex(k, ref, m_currTimeStep)]))
        {continue;}

        // calculates the total variance between the same pixels of different stitch blocks. If i > ref, the reference
        // has been passed and the offset index needs to be reduced by 1
        sum += std::abs( m_y[ravelIndex(k, i, m_currTimeStep)] +
                            offset[i - (i > ref)] -
                            m_y[ravelIndex(k, ref, m_currTimeStep)]
        ) * dx_inv;
    }

    return sum;
}

double StitchCorr::kernel(double sum, const std::vector<double>& offset, int i, int j) {
    /* Since wavelength difference for each pixel pair is equidistant between two blocks it can be calculated
     * early.
     */
    double dx_inv = 1 / std::abs(m_x[ravelIndex(0, i, 0)] - m_x[ravelIndex(0, j, 0)]);

    for (int k{0}; k < s_pixelCount - 1; ++k){
        /* Checks if any of the calculated pixel is a NAN and skips the evaluation in this case, since it would
         * lead to undefined behaviour! */
        if (    std::isnan(m_y[ravelIndex(k, i, m_currTimeStep)]) |
                std::isnan(m_y[ravelIndex(k, j, m_currTimeStep)]))
        {continue;}

        // calculates the total variance between the same pixels of different stitch blocks
        sum += std::abs(
                (m_y[ravelIndex(k, i, m_currTimeStep)] + offset[i]) -
                (m_y[ravelIndex(k, j, m_currTimeStep)] + offset[j])
        ) * dx_inv;
    }

    return sum;
}

double StitchCorr::kernel_asym(double sum, const std::vector<double> &offset, int i, int j) {
    /* Since wavelength difference for each pixel pair is equidistant between two blocks it can be calculated
     * early.
     */

    double dx_inv = 1 / std::abs(m_x[ravelIndex(0, i, 0)] - m_x[ravelIndex(0, j, 0)]);
    double dx_asm = 1 / std::abs(m_x[ravelIndex(1, i, 0)] - m_x[ravelIndex(0, j, 0)]);

    for (int k{0}; k < s_pixelCount - 1; ++k) {
        /* Checks if any of the calculated pixel is a NAN and skips the evaluation in this case, since it would
         * lead to undefined behaviour! */
        if (    std::isnan(m_y[ravelIndex(k, i, m_currTimeStep)]) |
                std::isnan(m_y[ravelIndex(k, j, m_currTimeStep)]) |
                std::isnan(m_y[ravelIndex(k+1, i, m_currTimeStep)]))
        {continue;}

        // calculates the total variance between the same pixels of different stitch blocks
        sum += std::abs(
                (m_y[ravelIndex(k, i, m_currTimeStep)] +
                 offset[ravelIndex(k % 2 != 0, i, 0, 2)]) -
                (m_y[ravelIndex(k, j, m_currTimeStep)] +
                 offset[ravelIndex(k % 2 != 0, j, 0, 2)])
        ) * dx_inv;

        // calculates the total variance between even-odd / odd-even adjacent pixels of different stitch blocks
        sum += std::abs(
                (m_y[ravelIndex(k+1, i, m_currTimeStep)] +
                 offset[ravelIndex((k+1) % 2 != 0, i, 0, 2)]) -
                (m_y[ravelIndex(k, j, m_currTimeStep)] +
                 offset[ravelIndex(k % 2 != 0, j, 0, 2)])
        ) * dx_asm;
    }

    return sum;
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
    bool ref_passed = false;
    for (int i{0}; i < m_blockCount; ++i){
        // If a reference was selected m_reference will be > -1
        if (m_reference == i) {
            ref_passed = true;
            continue;
        }

        for (int j{0}; j < s_pixelCount; ++j){
            m_y[ravelIndex(j, i, m_currTimeStep)] +=
                offset[ravelIndex(j % m_blockFactor != 0, i - ref_passed, 0, m_blockFactor)];
        }
    }

    for(int i{0}; i < m_blockCount * m_blockFactor - (m_reference > -1); ++i) {
        if (m_reference == i){continue;}

        m_offset[i + m_currTimeStep * (m_blockCount * m_blockFactor)] = offset[i];
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

py::array_t<double> callStitchCorr(py::array_t<double>& x,
                                   py::array_t<double>& t,
                                   py::array_t<double>& y,
                                   const int blockCount=4,
                                   const int reference=-1,
                                   const bool copy=true,
                                   const bool sortedInput=false,
                                   const bool isAsymmetric=true){
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

    auto sc = StitchCorr(Xptr, Xshape,
                         Tptr, Tshape,
                         Yptr, Yshape,
                         static_cast<int>(xStride), static_cast<int>(tStride),
                         blockCount, reference,
                         copy, sortedInput,
                         isAsymmetric);

    return sc.get_offset();
}

// Generates bindings for the python module. Add function/class definitions here!
PYBIND11_MODULE(stitchCorr, m) {
m.def("stitchCorr", &callStitchCorr);
}

#endif