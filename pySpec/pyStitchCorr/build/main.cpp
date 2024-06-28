//
// Created by Marku_000 on 12.02.2023.
//

#include "inputParser.h"
#include "stitchCorrParser.h"
#include "pyStitchCorr.h"
#include "BaseStitchCorr.h"

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
 *
 *  Careful: There exists a bug, where a linear correction cannot be performed, when the x-Axis has been sorted. ONLY in
 *  this case the stitching blocks NEED to be seperate...
 */

    auto* args = new inputParser(argc, argv);
    std::filesystem::path path{};

    while(args->get_next_path(path)) {
        std::cout << path << std::endl;

        if (path.empty()){continue;}

        auto* parser = new stitchCorrParser{args->get_delimiter()};

        parser->readData(path);

        for (auto& data : parser->m_data){

            auto* sc = new BaseStitchCorr{
                data,
                static_cast<unsigned int>(args->get_block_amount()),
                1, static_cast<unsigned int>(data.x.size()),
                args->isSorted()
            };

            if (args->get_reference_idx() == -1) {
                sc->correctStitch(std::nullopt, args->isLinear());
            }
            else {
                sc->correctStitch(args->get_reference_idx(), args->isLinear());
            }

            delete sc;
        }

        std::filesystem::path result_root{path.parent_path()};

        std::filesystem::create_directory(result_root = result_root / std::filesystem::path{"stitch_corr"});

        parser->writeData(result_root / path.stem().concat("_stitchCorr.dat"));

        delete parser;
    }

    return 0;
}