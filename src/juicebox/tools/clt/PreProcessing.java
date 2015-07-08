/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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

package juicebox.tools.clt;

import juicebox.tools.HiCTools;
import juicebox.tools.utils.common.HiCFileTools;
import juicebox.tools.utils.original.NormalizationVectorUpdater;
import juicebox.tools.utils.original.Preprocessor;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class PreProcessing extends JuiceboxCLT {

    private long genomeLength = 0;
    private String inputFile;
    private String outputFile;
    private Preprocessor preprocessor;

    public PreProcessing(){
        super("pre <options> <infile> <outfile> <genomeID>");
    }

    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {
        String genomeId;
        try {
            genomeId = args[3];
        } catch (ArrayIndexOutOfBoundsException e) {
            System.err.println("No genome ID given");
            throw new IOException("1");
        }

        List<Chromosome> chromosomes = HiCFileTools.loadChromosomes(genomeId);

        for (Chromosome c : chromosomes) {
            if (c != null)
                genomeLength += c.getLength();
        }
        chromosomes.set(0, new Chromosome(0, "All", (int) (genomeLength / 1000)));

        inputFile = args[1];
        outputFile = args[2];
        String tmpDir = parser.getTmpdirOption();

        preprocessor = new Preprocessor(new File(outputFile), genomeId, chromosomes);
        preprocessor.setIncludedChromosomes(parser.getChromosomeOption());
        preprocessor.setCountThreshold(parser.getCountThresholdOption());
        preprocessor.setMapqThreshold(parser.getMapqThresholdOption());
        preprocessor.setDiagonalsOnly(parser.getDiagonalsOption());
        preprocessor.setFragmentFile(parser.getFragmentOption());
        preprocessor.setTmpdir(tmpDir);
        preprocessor.setStatisticsFile(parser.getStatsOption());
        preprocessor.setGraphFile(parser.getGraphOption());

    }

    @Override
    public void run() throws IOException {
        preprocessor.preprocess(inputFile);
        NormalizationVectorUpdater.updateHicFile(outputFile);
    }
}