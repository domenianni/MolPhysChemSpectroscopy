//
// Created by bauer on 25.06.2024.
//

#include "BaseStitchCorr.h"

unsigned int MAXBLOCKAMOUNT = 256;

struct fData {
    BaseStitchCorr*             obj{nullptr};
    std::optional<unsigned int> reference{std::nullopt};
    unsigned int                currentTimeIdx{0};
    bool                        isLinear{false};
};

double wrapper(const std::vector<double> &parameter, std::vector<double>& grad, void* data){
    // Wrapper necessary due to 'nlopt::opt.set_min_objective'. Here, f_data is simply unpacked and linear or static
    // selected.
    auto f_data = static_cast<fData *>(data);

    if (f_data->isLinear) {
        return f_data->obj->targetLinear(parameter, f_data->currentTimeIdx, f_data->reference);
    }
    return f_data->obj->targetStatic(parameter, f_data->currentTimeIdx, f_data->reference);
}

void BaseStitchCorr::calculateLinearOffset(std::vector<double>& offset, const double slope, const double intercept) {
    for (int i{0}; i < offset.size(); i++) {
        offset[i] = slope * i + intercept;
    }
}

double BaseStitchCorr::targetStatic(const std::vector<double>         &parameter,
                                    const unsigned int                currentTimeIdx,
                                    const std::optional<unsigned int> reference) {

    for (unsigned int i{0}; i < m_blockAmount; ++i) {
        // m_offsetVectors[reference] should be zero from the initialization, therefore just skipping it here should be
        // enough!
        if (i == reference.value_or(MAXBLOCKAMOUNT)){continue;}

        // since offsetVectors are not changing within one iteration, they can be pre-calculated.
        // To not have a free-floating parameter within the optimizer, the parameter vector is shortened if a reference
        // was selected.
        calculateLinearOffset(m_offsetVectors[i],
                                 0,
                                 parameter[i - (i > reference.value_or(MAXBLOCKAMOUNT))]);
    }

    if (reference.has_value()) {
        return calcTV(currentTimeIdx, reference.value());
    }

    return calcTV(currentTimeIdx);
}

double BaseStitchCorr::targetLinear(const std::vector<double> &parameter,
                                    const unsigned int currentTimeIdx,
                                    const std::optional<unsigned int> reference) {

    for (unsigned int i{0}; i < m_blockAmount; ++i) {
        // m_offsetVectors[reference] should be zero from the initialization, therefore just skipping it here should be
        // enough!
        if (i == reference.value_or(MAXBLOCKAMOUNT)){continue;}

        // since offsetVectors are not changing within one iteration, they can be pre-calculated.
        // To not have a free-floating parameter within the optimizer, the parameter vector is shortened if a reference
        // was selected
        calculateLinearOffset(m_offsetVectors[i],
                                 parameter[2 * (i - (i > reference.value_or(MAXBLOCKAMOUNT)))],
                                 parameter[2 * (i - (i > reference.value_or(MAXBLOCKAMOUNT))) + 1]);
    }

    if (reference.has_value()) {
        return calcTV(currentTimeIdx, reference.value());
    }

    return calcTV(currentTimeIdx);
}

//----------------------------------------------------------------------------------------------------------------------
double BaseStitchCorr::calcTV(unsigned int currentTimeIdx) {
    double sum{0};

    for (int i{0}; i < m_blockAmount; ++i){
        for (int j{0}; j < m_blockAmount; ++j){

            if (i==j){continue;}
            /* skip calculations so that no block is compared against itself (even though it
             * would work for the odd-even comparison)
             */
            sum = kernel(sum, i, m_offsetVectors[i], j, m_offsetVectors[j], currentTimeIdx);
        }
    }
    /* for now only the sum is returned. Though I don't know if it is necessary to calculate the mean, since this would
     * account for the existence of NAN values. Though inclusion of taking the mean will slow down the program and
     * necessitate a check for complete NAN values, so that no undefined behaviour occurs.
     * if (elements == 0){return 0;}
     */
    return sum;
}

double BaseStitchCorr::calcTV(unsigned int currentTimeIdx, unsigned int reference) {
    // Calculates the total variance, with regard to one (signified by m_reference) block.
    // Only needs to loop then over one idx to cover all blocks.
    double sum{0};

    for (int i{0}; i < m_blockAmount; ++i) {
        if (i == reference){continue;}

        sum = kernel(sum, i, m_offsetVectors[i], reference, m_offsetVectors[reference], currentTimeIdx);
    }

    return sum;
}
//----------------------------------------------------------------------------------------------------------------------

double BaseStitchCorr::kernel(double sum,
                  unsigned int i, std::vector<double>& offsetI,
                  unsigned int j, std::vector<double>& offsetJ,
                  unsigned int currentTimeIdx) {
    /* Since wavelength difference for each pixel pair is equidistant between two blocks it can be calculated
     * early.
     * However, calculating at each step might make the algorithm work even with wavenumbers...
     */
    double dx_inv = 1 / std::abs(m_data.x[ravelIdx(0, i, 0)] -
                                 m_data.x[ravelIdx(0, j, 0)]);

    for (unsigned int k{0}; k < s_pixelAmount; ++k){
        /* Checks if any of the calculated pixel is a NAN and skips the evaluation in this case, since it would
         * lead to undefined behaviour! */
        if (    std::isnan(m_data.y[ravelIdx(k, i, currentTimeIdx)]) |
                std::isnan(m_data.y[ravelIdx(k, j, currentTimeIdx)]))
        {continue;}

        // double val = ((m_data.y[ravelIdx(k, i, currentTimeIdx)] + offsetI[k]) -
        //            (m_data.y[ravelIdx(k, j, currentTimeIdx)] + offsetJ[k])) * dx_inv;
        // sum += val * val;

        // calculates the total variance between the same pixels of different stitch blocks
        sum += std::abs(
                   (m_data.y[ravelIdx(k, i, currentTimeIdx)] + offsetI[k]) -
                   (m_data.y[ravelIdx(k, j, currentTimeIdx)] + offsetJ[k])
        ) * dx_inv;
    }

    return sum;
}

unsigned int BaseStitchCorr::ravelIdx(const unsigned int pixelIdx,
                                      const unsigned int blockIdx,
                                      const unsigned int timeIdx) {
    return m_pixelStride * pixelIdx + m_blockStride * blockIdx + m_tStride * timeIdx;
}

//----------------------------------------------------------------------------------------------------------------------
BaseStitchCorr& BaseStitchCorr::correctStitch(std::optional<unsigned int> reference, bool isLinear){

    // parameterAmount one less if a reference was chosen, but doubled if the stitch correction should be linear.
    const unsigned int parameterAmount{
        (m_blockAmount - reference.has_value()) * (1 + isLinear)
    };

    // Struct to be passed to the wrapper (necessary due to 'nlopt::opt.set_min_objective')
    fData f_data{this, reference, 0, isLinear};

    // The logger setup. Will only log the last corrected measurement!!
    std::ofstream log {"log.txt"};
    log << "Blocks: " << m_blockAmount << "; Pixel: " << s_pixelAmount << "; Timesteps: " << m_data.t.size();
    log << "; Parameter Amount: " << parameterAmount << std::endl;
    log << "blockStride: " << m_blockStride << "; pixelStride: " << m_pixelStride << std::endl;

    // Loop over all times!
    while(f_data.currentTimeIdx < m_data.t.size()){

        // parameter all initialized to m_offsetInitValue (0.0)
        std::vector<double> parameter(parameterAmount, m_offsetInitValue);

        log << "Optimising timestep " << f_data.currentTimeIdx << "( " << m_data.t[f_data.currentTimeIdx] << " )" << std::endl;

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

        nlopt::opt opt{nlopt::opt(m_algorithm, parameterAmount)};

        opt.set_min_objective(wrapper, &f_data);

        opt.set_maxeval(m_maxeval);
        opt.set_ftol_abs(m_ftol);
        opt.set_xtol_abs(m_xtol);

        // minf is the last function value during the optimizer
        double minf{};
        try{
            nlopt::result result = opt.optimize(parameter, minf);

            // more logging of the parameter
            log << result << std::endl << "Offset: [ ";
            for (int i{0}; i < parameterAmount; ++i){
                log << parameter[i] << " ";
            }
            log << "]" << std::endl;

            // since the last iteration saves the offset vectors in m_offsetVectors, this can be utilized to now correct
            // the data. The vectors are accessed inside the following function.
            correctData(f_data.currentTimeIdx);
        }
        catch(std::exception &e) {
            log << "nlopt failed: " << e.what() << std::endl;
        }

        ++f_data.currentTimeIdx;

        // the last set of parameters is saved within the instance for now.
        m_parameter.push_back(parameter);
    }

    log.close();

    return *this;
}

//----------------------------------------------------------------------------------------------------------------------
void BaseStitchCorr::correctData(unsigned int currentTimeIdx) {
    for (int i{0}; i < m_blockAmount; ++i){
        for (int j{0}; j < s_pixelAmount; ++j){
            m_data.y[ravelIdx(j, i, currentTimeIdx)] += m_offsetVectors[i][j];
        }
    }
}
