//
// Created by bauer on 25.06.2024.
//

#ifndef STITCHCORR_H
#define STITCHCORR_H

#include <assert.h>
#include "nlopt.hpp"
#include <optional>
#include <iostream>
#include <fstream>

struct stitchCorrData{
    std::vector<double> x;
    std::vector<double> t;
    std::vector<double> y;
};

class BaseStitchCorr {
public:
    std::vector<std::vector<double>> m_parameter;

    double targetStatic(const std::vector<double> &parameter,
                        unsigned int currentTimeIdx,
                        std::optional<unsigned int> reference);
    double targetLinear(const std::vector<double> &parameter,
                        unsigned int currentTimeIdx,
                        std::optional<unsigned int> reference);

    double calcTV(unsigned int currentTimeIdx);
    double calcTV(unsigned int currentTimeIdx, unsigned int reference);

    BaseStitchCorr& correctStitch(std::optional<unsigned int> reference=std::nullopt, bool isLinear=false);

    BaseStitchCorr(stitchCorrData& data,
                   const unsigned int blockAmount,
                   const unsigned int xStride,
                   const unsigned int tStride,
                   const bool isSorted):
        m_data{data},
        m_blockAmount{blockAmount},
        m_xStride{xStride},
        m_tStride{tStride},
        s_pixelAmount{static_cast<unsigned int>(m_data.x.size() / m_blockAmount)}
    {
        assert(m_data.y.size() == m_data.t.size() * m_data.x.size());

        if (isSorted) {
            m_blockStride = m_xStride;
            m_pixelStride = m_xStride * m_blockAmount;
        } else {
            m_blockStride = m_xStride * s_pixelAmount;
            m_pixelStride = m_xStride;
        }

        for (int i{0}; i < m_blockAmount; ++i) {
            m_offsetVectors.emplace_back(s_pixelAmount, 0.0);
        }
    }

private:
    void correctData(unsigned int currentTimeIdx);

    void calculateLinearOffset(std::vector<double>& offset,
                               double slope,
                               double intercept);

    double kernel(double sum,
                  unsigned int i, std::vector<double>& offsetA,
                  unsigned int j, std::vector<double>& offsetB,
                  unsigned int currentTimeIdx);

    unsigned int ravelIdx(unsigned int pixelIdx,
                          unsigned int blockIdx,
                          unsigned int timeIdx);

    stitchCorrData& m_data;

    nlopt::algorithm m_algorithm{nlopt::LN_COBYLA};
    int m_maxeval{2000};
    double m_ftol{1e-8};
    double m_xtol{1e-8};
    double m_offsetInitValue{0.0};

    const unsigned int m_blockAmount;
    const unsigned int s_pixelAmount;
    const unsigned int m_xStride;
    const unsigned int m_tStride;

    unsigned int m_pixelStride;
    unsigned int m_blockStride;

    std::vector<std::vector<double>> m_offsetVectors;
};



#endif //STITCHCORR_H
