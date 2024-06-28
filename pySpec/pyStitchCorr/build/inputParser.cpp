//
// Created by Marku_000 on 12.02.2023.
//

#include "inputParser.h"

bool inputParser::parse_opt_params(const std::string& arg){
    static bool block_arg{false};
    static bool ref_arg{false};
    static bool delim_arg{false};

    if (m_files.empty()) {
        if (arg == "-h"){
            std::cout << "The call signature is:\n";
            std::cout << "<arguments> <file paths>\n";
            std::cout << "With the possible arguments:\n";
            std::cout << "-s / --sorted : If the input data has a sorted wavenumber axis.\n";
            std::cout << "-b <int> / --blocks <int> to specify the amount of blocks in the files.\n";
            std::cout << "-r <int> / --reference <int> to specify the block to serve as reference.\n";
            std::cout << "-l / --linear to calculate a linear function as an offset.\n";
            exit(0);
        }

        if (arg == "-s" || arg == "--sorted") {
            if (m_sorted_input) {
                throw std::runtime_error("cannot use -s/--sorted param twice!");
            }

            m_sorted_input = true;
            return true;
        }

        if (arg == "-l" || arg == "--linear") {
            if (m_linear) {
                throw std::runtime_error("cannot use -l/--linear param twice!");
            }

            m_linear = true;
            return true;
        }

        if (arg == "-b" || arg == "--blocks") {
            block_arg = true;
            return true;
        }

        if (block_arg) {
            m_block_amount = std::stoi(arg);
            block_arg = false;
            return true;
        }

        if (arg == "-r" || arg == "--reference") {
            ref_arg = true;
            return true;
        }

        if (ref_arg) {
            m_reference_idx = std::stoi(arg);
            ref_arg = false;
            return true;
        }

        if (arg == "-d" || arg == "--delimiter") {
            delim_arg = true;
            return true;
        }

        if (delim_arg) {
            m_delimiter = arg;
            delim_arg = false;
            return true;
        }
    }

    return false;
}

bool inputParser::get_next_path(std::filesystem::path& path){
    static int i{0};

    if (i >= m_files.size()){
        return false;
    }

    path = m_files[i];
    i++;

    return true;
}

[[nodiscard]] int inputParser::get_block_amount() const  {
    return m_block_amount;
}

[[nodiscard]] int inputParser::get_reference_idx() const  {
    return m_reference_idx;
}

[[nodiscard]] bool inputParser::isSorted() const{
    return m_sorted_input;
}

[[nodiscard]] std::string inputParser::get_delimiter() const {
    return m_delimiter;
}

[[nodiscard]] bool inputParser::isLinear() const {
    return m_linear;
}
