//
// Created by Marku_000 on 05.02.2023.
//

#include "stitchCorrParser.h"

stitchCorrParser& stitchCorrParser::writeData(const std::filesystem::path& path) {
    std::string zero_placeholder{"0.00000"};
    std::string delimiter{" "};

    std::ofstream file(path, std::ios::out);

    if (!file.is_open()){
        std::cout << "Opening file: " << path << "failed!";
        return *this;
    }

    for (const stitchCorrData& data : m_data){

        file << zero_placeholder;
        for (const double& x : data.x){
            file << delimiter << x;
        }
        file << "\n";

        int row{0};
        for (const double& t : data.t){
            file << t;

            for (int i{0}; i < data.x.size(); i++){
                file << delimiter << data.y[row * data.x.size() + i];
            }
            file << "\n";
            ++row;
        }
        file << "\n";
    }

    return *this;
}

stitchCorrParser& stitchCorrParser::readData(const std::filesystem::path& path) {
    int row{0};
    int column{0};

    std::ifstream file(path, std::ios::in);

    if (!file.is_open()){
        std::cout << "Reading file: " << path << "failed!";
        return *this;
    }

    std::string str;

    std::vector<double> y;
    std::vector<double> x;
    std::vector<double> t;

    while(getline(file, str)){
        column = 0;
        size_t pos;
        std::string token;

        if (str.length() == 0 && !y.empty()){
            // Finds the empty rows between measurements!
            this->_add_data(x, t, y);
            row = 0;
            continue;
        }

        while ((pos = str.find(m_delimiter)) != std::string::npos) {

            token = str.substr(0, pos);
            str.erase(0, pos + m_delimiter.length());

            if (token.length() == 0){continue;}

            if (column != 0 && row == 0){
                x.push_back(std::stod(token));
            }
            else if (column == 0 && row != 0){
                t.push_back(std::stod(token));
            }
            else if (column != 0 && row > 0){
                y.push_back(std::stod(token));
            }

            ++column;
        }

        if (token.length() == 0){continue;}

        if (row == 0){
            x.push_back(std::stod(str));
        }
        else{
            y.push_back(std::stod(str));
        }

        ++row;
    }

    if (!y.empty()){
        // Finds the empty rows between measurements!
        this->_add_data(x, t, y);
    }

    file.close();
    return *this;
}

void stitchCorrParser::_add_data(std::vector<double> &x, std::vector<double> &t, std::vector<double> &y) {
    m_data.push_back(stitchCorrData{x, t, y});
    x.clear();
    t.clear();
    y.clear();
}

