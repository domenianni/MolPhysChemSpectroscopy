//
// Created by Marku_000 on 05.02.2023.
//
#ifndef PYSTITCHCORR_STITCHCORRPARSER_H
#define PYSTITCHCORR_STITCHCORRPARSER_H

#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>
#include <filesystem>
#include "BaseStitchCorr.h"

class stitchCorrParser{
    std::string m_delimiter{" "};

    public:
        std::vector<stitchCorrData> m_data;

        stitchCorrParser()= default;

        stitchCorrParser(const std::string& delimiter){m_delimiter = delimiter;}

        stitchCorrParser& readData(const std::filesystem::path& path);
        stitchCorrParser& writeData(const std::filesystem::path& path);

    private:
        void _add_data(std::vector<double>& x, std::vector<double>& t, std::vector<double>& y);

};


#endif //PYSTITCHCORR_STITCHCORRPARSER_H
