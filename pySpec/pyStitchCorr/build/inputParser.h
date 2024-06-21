//
// Created by Marku_000 on 12.02.2023.
//

#ifndef PYSTITCHCORR_INPUTPARSER_H
#define PYSTITCHCORR_INPUTPARSER_H

#include <string>
#include <vector>
#include <filesystem>
#include <iostream>

class inputParser{

private:
    std::vector<std::filesystem::path> m_files;
    bool m_sorted_input = false;
    int m_block_amount = 4;

public:
    inputParser(int argc, char** argv){
        if (argc <= 1){throw std::runtime_error("No file selected!");}
        if (argc > 64) {throw std::runtime_error("too many input parameters!");}

        std::vector<std::string> args{argv + 1, argv + argc};

        for (const auto& arg : args) {
            if (m_files.empty()) {
                if (parse_opt_params(arg)){continue;}
            }

            if (!std::filesystem::exists(arg)) {
                throw std::runtime_error(std::string(arg) + ": No such file");
            }
            m_files.emplace_back(arg);
        }
    }

    bool parse_opt_params(const std::string& arg);
    bool get_next_path(std::filesystem::path& path);

    [[nodiscard]] int get_block_amount() const;
    [[nodiscard]] bool isSorted() const;
    [[nodiscard]] std::string get_delimiter() const;
};


#endif //PYSTITCHCORR_INPUTPARSER_H
