/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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
import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.original.NormalizationVectorUpdater;
import juicebox.tools.utils.original.Preprocessor;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.util.List;

public class PreProcessing extends JuiceboxCLT {

    private long genomeLength = 0;
    private String inputFile;
    private String outputFile;
    private Preprocessor preprocessor;

    public PreProcessing() {
        super(getBasicUsage()+"\n"
                + "           : -d only calculate intra chromosome (diagonal) [false]\n"
                + "           : -f <restriction site file> calculate fragment map\n"
                + "           : -m <int> only write cells with count above threshold m [0]\n"
                + "           : -q <int> filter by MAPQ score greater than or equal to q\n"
                + "           : -c <chromosome ID> only calculate map on specific chromosome\n"
                + "           : -h print help"
        );
    }

    public static String getBasicUsage() {
        return "pre [options] <infile> <outfile> <genomeID>";
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {
        CommandLineParser parser1 = (CommandLineParser) parser;
        if (parser1.getHelpOption()) {
            printUsageAndExit();
        }

        String genomeId = "";
        try {
            genomeId = args[3];
        } catch (ArrayIndexOutOfBoundsException e) {
            System.err.println("No genome ID given");
            printUsageAndExit();
        }


        List<Chromosome> chromosomes = HiCFileTools.loadChromosomes(genomeId);

        for (Chromosome c : chromosomes) {
            if (c != null)
                genomeLength += c.getLength();
        }
        chromosomes.set(0, new Chromosome(0, HiCFileTools.ALL_CHROMOSOME, (int) (genomeLength / 1000)));

        inputFile = args[1];
        outputFile = args[2];
        String tmpDir = parser1.getTmpdirOption();

        preprocessor = new Preprocessor(new File(outputFile), genomeId, chromosomes);
        preprocessor.setIncludedChromosomes(parser1.getChromosomeOption());
        preprocessor.setCountThreshold(parser1.getCountThresholdOption());
        preprocessor.setMapqThreshold(parser1.getMapqThresholdOption());
        preprocessor.setDiagonalsOnly(parser1.getDiagonalsOption());
        preprocessor.setFragmentFile(parser1.getFragmentOption());
        preprocessor.setTmpdir(tmpDir);
        preprocessor.setStatisticsFile(parser1.getStatsOption());
        preprocessor.setGraphFile(parser1.getGraphOption());
        preprocessor.setResolutions(parser1.getResolutionOption());

    }

    @Override
    public void run() {
        try {
            preprocessor.preprocess(inputFile);
            NormalizationVectorUpdater.updateHicFile(outputFile);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(56);
        }
    }
}