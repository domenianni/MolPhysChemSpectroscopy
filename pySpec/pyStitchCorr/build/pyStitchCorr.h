#ifndef PYSTITCHCORR_PYSTITCHCORR_H
#define PYSTITCHCORR_PYSTITCHCORR_H

#define USING(x) ((1 x 1) == 2)
#define ON +
#define OFF -

#if defined(PYTHON_LIB)
#	define PY ON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#else
#   define PY OFF
#endif

#include <cmath>
#include <nlopt.hpp>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <iostream>

class StitchCorr{
    /** Stitch Correction Class. Performs Stitch Correction on one run of the UV-MIR Experiment
  *
  * Input arrays MUST BE FLATTENED; PixelCount is set to 32 in Source Code!
  *
  * @param x double[blockCount * pixelCount]: Unsorted wavelength axis of type double. Changes block after pixelCount.
  *
  * @param t double[time steps]: Unsorted timeaxis of time double. Only used to determine the amount of time steps for now.
  *
  * @param y double[blockCount * pixelCount * time steps]: Flattened array of the DOD values of type double.
  *
  * @param blockCount int: Amount of stitching blocks of type int.
  *
  * @param copy bool: Flag to set whether to copy the supplied arrays internally. If copied the supplied DOD array will
  *             not be changed by the class and the values have to be retrieved via the public get_y_val method.
  *
  * The stitch correction uses the open-source NLOPT non-linear-solver interface and >currently< the COBYLA-algorithm,
  * though this can be changed in the source code.
  *
  * It minimizes the total variance (defined by dy/dx) between the respective pixel intensity values of two stitching
  * blocks as well as between the pixel p and p+1 within all possible combinations of stitching blocks for EACH time
  * step by varying an additive offset value per even and odd pixel per stitching block.
  *
  * Finally it adds the calculated offset values to the internally saved m_y array. If copy is false this will modify
  * the supplied DOD array.
  */
public:
    StitchCorr(double* x, int xShape,
               double* t, int tShape,
               double* y, int yShape,
               const int blockCount=4,
               bool copy=true,
               bool sortedInput=false):

            m_blockCount{blockCount},
            m_copied{copy},
            m_timesteps{tShape},
            m_sortedInput{sortedInput}
    {
        if ((xShape != blockCount * s_pixelCount) && (yShape != blockCount * s_pixelCount * m_timesteps)){throw -99;}

        m_offset = new double[2 * blockCount * m_timesteps]{};

        if (m_copied){
            setup();

            for (int i{0}; i < m_timesteps; ++i){
                m_t[i] = t[i];
            }
            for (int i{0}; i < m_blockCount * s_pixelCount; ++i){
                m_x[i] = x[i];
            }
            for (int i{0}; i < m_blockCount * s_pixelCount * m_timesteps; ++i){
                m_y[i] = y[i];
            }
        }
        else{
            m_x = x;
            m_y = y;
            m_t = t;
        }

        compareBlocks();
    }

    ~StitchCorr(){
        delete[] m_offset;

        if (m_copied){
            delete[] m_x;
            delete[] m_y;
            delete[] m_t;
        }
    }

    double calcTV(const std::vector<double>& offset, std::vector<double> &grad);

#if (USING (PY))
py::array_t<double> get_offset(){
    return py::array(2 * m_blockCount * m_timesteps, m_offset);
}
#endif

private:
    const bool m_copied{};
    const static int s_pixelCount{32};
    int m_blockCount{4};
    int m_timesteps{0};

    /* Settings for the minimization algorithm.
     * -Since the target function does not supply gradients LN prefix is necessary.
     *      Possible Algorithms include:
     *      -> LN_NELDERMEAD, LN_BOBYQA, LN_COBYLA
     * - m_maxeval sets amount of maximal target function evaluation (SUCCESS CODE 5 if reached!)
     * - m_ftol and m_xtol are convergence criteria (target fuction difference / x-value difference). They are set to
     *  absolute values, since the minimization optimizes towards ZERO!
     */
    nlopt::algorithm m_algorithm{nlopt::LN_COBYLA};
    int m_maxeval{2000};
    double m_ftol{1e-8};
    double m_xtol{1e-8};
    double m_offsetInitValue{0.0};

    bool m_sortedInput{false};

    double* m_x{nullptr};
    double* m_y{nullptr};
    double* m_t{nullptr};

    double* m_offset{nullptr};

    int m_currTimeStep{0};

    std::ofstream m_log {"log.txt"};

    StitchCorr& setup();
    StitchCorr& compareBlocks();
    StitchCorr& addOffset(std::vector<double>& offset, std::ofstream& log);
    int ravelIndex(int pixelIdx, int blockIdx, int timeIdx, int x_len=s_pixelCount) const;
};

#endif //PYSTITCHCORR_PYSTITCHCORR_H
