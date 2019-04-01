/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.clt.old;

import jargs.gnu.CmdLineParser;
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.original.Preprocessor;
import juicebox.tools.utils.original.norm.NormalizationVectorUpdater;

import java.io.File;

public class PreProcessing extends JuiceboxCLT {


    private String inputFile;
    private String outputFile;
    private Preprocessor preprocessor;
    private boolean noNorm = false;
    private boolean noFragNorm = false;
    private int genomeWide;

    public PreProcessing() {
        super(getBasicUsage()+"\n"
                + "           : -d only calculate intra chromosome (diagonal) [false]\n"
                + "           : -f <restriction site file> calculate fragment map\n"
                + "           : -m <int> only write cells with count above threshold m [0]\n"
                + "           : -q <int> filter by MAPQ score greater than or equal to q [not set]\n"
                + "           : -c <chromosome ID> only calculate map on specific chromosome [not set]\n"
                + "           : -r <comma-separated list of resolutions> Only calculate specific resolutions [not set]\n"
                + "           : -t <tmpDir> Set a temporary directory for writing\n"
                + "           : -s <statistics file> Add the text statistics file to the Hi-C file header\n"
                + "           : -g <graphs file> Add the text graphs file to the Hi-C file header\n"
                + "           : -n Don't normalize the matrices\n"
                + "           : -z <double> scale factor for hic file\n"
                + "           : -a <1, 2, 3, 4> filter based on inner, outer, left-left, right-right pairs respectively\n"
                + "           : --randomize_position randomize positions between fragment sites\n"
                + "           : --random_seed seed for random generator\n"
                + "           : --randomize_pos_maps fragment maps for randomization\n"


        );
    }

    public static String getBasicUsage() {
        return "pre [options] <infile> <outfile> <genomeID>";
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {
        CommandLineParser parser1 = (CommandLineParser) parser;

        String genomeId = "";
        try {
            genomeId = args[3];
        } catch (ArrayIndexOutOfBoundsException e) {
            System.err.println("No genome ID given");
            printUsageAndExit();
        }


        ChromosomeHandler chromHandler = HiCFileTools.loadChromosomes(genomeId);

        inputFile = args[1];
        outputFile = args[2];
        String tmpDir = parser1.getTmpdirOption();
        double hicFileScalingFactor = parser1.getScalingOption();

        preprocessor = new Preprocessor(new File(outputFile), genomeId, chromHandler, hicFileScalingFactor);
        preprocessor.setIncludedChromosomes(parser1.getChromosomeOption());
        preprocessor.setCountThreshold(parser1.getCountThresholdOption());
        preprocessor.setMapqThreshold(parser1.getMapqThresholdOption());
        preprocessor.setDiagonalsOnly(parser1.getDiagonalsOption());
        preprocessor.setFragmentFile(parser1.getFragmentOption());
        preprocessor.setExpectedVectorFile(parser1.getExpectedVectorOption());
        preprocessor.setTmpdir(tmpDir);
        preprocessor.setStatisticsFile(parser1.getStatsOption());
        preprocessor.setGraphFile(parser1.getGraphOption());
        preprocessor.setResolutions(parser1.getResolutionOption());
        preprocessor.setAlignmentFilter(parser1.getAlignmentOption());
        preprocessor.setRandomizePosition(parser1.getRandomizePositionsOption());
        preprocessor.setPositionRandomizerSeed(parser1.getRandomPositionSeedOption());
        preprocessor.setRandomizeFragMaps(parser1.getRandomizePositionMaps());

        noNorm = parser1.getNoNormOption();
        genomeWide = parser1.getGenomeWideOption();
        noFragNorm = parser1.getNoFragNormOption();
    }

    @Override
    public void run() {
        try {
            long currentTime = System.currentTimeMillis();
            preprocessor.preprocess(inputFile);
            if (HiCGlobals.printVerboseComments) {
                System.out.println("\nCalculating contact matrices took: " + (System.currentTimeMillis() - currentTime) + " milliseconds");
            }
            if (!noNorm) {
                NormalizationVectorUpdater.updateHicFile(outputFile, genomeWide, noFragNorm);
            }
            else {
                System.out.println("Done creating .hic file. Normalization not calculated due to -n flag.");
                System.out.println("To run normalization, run: juicebox addNorm <hicfile>");
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(56);
        }
    }
}