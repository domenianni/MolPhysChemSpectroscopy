//
// Created by Marku_000 on 12.02.2023.
//

#include "inputParser.h"
#include "stitchCorrParser.h"
#include "pyStitchCorr.h"

int main(int argc, char** argv){
/*
 *	The entry point for the standalone stitch correction executable.
 *	It takes in a list of measurements and creates in the same directory a new folder with the stitch corrected data.
 *
 *	Takes the following input:
 *		<arguments> <file paths>
 *		With the possible arguments:
 *		-s / --sorted : If the input data has a sorted wavenumber axis.
 *		-b <int> / --blocks <int> to specify the amount of blocks in the files.
 *
 *	This information is also accessible with the tag -h.
 */

    auto* args = new inputParser(argc, argv);
    std::filesystem::path path{};

    while(args->get_next_path(path)) {
        std::cout << path << std::endl;

        if (path.empty()){continue;}

        auto* parser = new stitchCorrParser{};

        parser->readData(path);

        for (auto& data : parser->m_data){
            auto* sc = new StitchCorr{
                    data.x.data(), static_cast<int>(data.x.size()),
                    data.t.data(), static_cast<int>(data.t.size()),
                    data.y.data(), static_cast<int>(data.y.size()),
                    args->get_block_amount(),
                    false,
                    args->isSorted()};

            delete sc;
        }

        std::filesystem::path result_root{path.parent_path()};

        std::filesystem::create_directory(result_root = result_root / std::filesystem::path{"stitch_corr"});

        parser->writeData(result_root / path.stem().concat("_stitchCorr.dat"));

        delete parser;
    }

    return 0;
}