/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2024 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.clt.old;

import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.tools.clt.CommandLineParser;
import juicebox.tools.clt.JuiceboxCLT;
import juicebox.tools.utils.common.ShellCommandRunner;
import juicebox.tools.utils.original.MultithreadedPreprocessor;
import juicebox.tools.utils.original.MultithreadedPreprocessorHic;
import juicebox.tools.utils.original.Preprocessor;
import juicebox.windowui.HiCZoom;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class Summation extends JuiceboxCLT {

    private String inputFile;
    private String outputFile;
    private MultithreadedPreprocessorHic preprocessor;
    private String shell = "sh";

    public Summation() {
        super(getBasicUsage() + "\n" + PreProcessing.flags);
    }

    public static String getBasicUsage() {
        return "sum [options] <out.hic> <in1.hic> <in2.hic> ...";
    }

    @Override
    public void readArguments(String[] args, CommandLineParser parser) {

        outputFile = args[1];
        inputFile = getRemainingFiles(args);

        String[] genomeId = new String[1];
        List<String> resolutionStrings = new ArrayList<>();
        populateParameters(args[2], genomeId, resolutionStrings, parser.getGenomeOption());

        ChromosomeHandler chromHandler = HiCFileTools.loadChromosomes(genomeId[0]);

        String tmpDir = parser.getTmpdirOption();
        double hicFileScalingFactor = parser.getScalingOption();

        updateNumberOfCPUThreads(parser, 1);

        preprocessor = new MultithreadedPreprocessorHic(new File(outputFile), genomeId[0], chromHandler,
                hicFileScalingFactor, numCPUThreads);
        usingMultiThreadedVersion = true;
        preprocessor.setFromHIC(true);

        preprocessor.setGenome(genomeId[0]);

        List<String> customResolutions = parser.getResolutionOption();
        if (customResolutions != null && customResolutions.size() > 0) {
            preprocessor.setResolutions(customResolutions);
        } else {
            preprocessor.setResolutions(resolutionStrings);
        }

        preprocessor.setIncludedChromosomes(parser.getChromosomeSetOption());
        preprocessor.setCountThreshold(parser.getCountThresholdOption());
        preprocessor.setV9DepthBase(parser.getV9DepthBase());
        preprocessor.setMapqThreshold(parser.getMapqThresholdOption());
        preprocessor.setDiagonalsOnly(parser.getDiagonalsOption());
        preprocessor.setFragmentFile(parser.getFragmentOption());
        preprocessor.setExpectedVectorFile(parser.getExpectedVectorOption());
        preprocessor.setTmpdir(tmpDir);
        preprocessor.setAlignmentFilter(parser.getAlignmentOption());
        preprocessor.setRandomizePosition(parser.getRandomizePositionsOption());
        preprocessor.setPositionRandomizerSeed(parser.getRandomPositionSeedOption());
        preprocessor.setRandomizeFragMaps(parser.getRandomizePositionMaps());
        preprocessor.setThrowOutIntraFragOption(parser.getThrowIntraFragOption());
        preprocessor.setSubsampler(parser.getSubsampleOption());

        int blockCapacity = parser.getBlockCapacityOption();
        if (blockCapacity > 10) {
            Preprocessor.BLOCK_CAPACITY = blockCapacity;
        }

        String customShell = parser.getShellOption();
        if (customShell != null && customShell.length() > 0) {
            shell = customShell;
        }
    }

    @Override
    public void run() {
        HiCGlobals.allowDynamicBlockIndex = false;
        try {
            long currentTime = System.currentTimeMillis();

            preprocessor.preprocess(inputFile, null, null, null);
            ShellCommandRunner.runShellFile(shell, outputFile + MultithreadedPreprocessor.CAT_SCRIPT);

            if (HiCGlobals.printVerboseComments) {
                System.out.println("\nBinning contact matrices took: " + (System.currentTimeMillis() - currentTime) + " milliseconds");
            }

            System.out.println("Done creating .hic file. Normalization not calculated due to -n flag.");
            System.out.println("To run normalization, run: java -jar juicer_tools.jar addNorm <hicfile>");

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(56);
        }
    }

    private void populateParameters(String hicFile, String[] genomeId, List<String> resolutionStrings,
                                    String genomeOptionIfAvailable) {
        List<String> files = new ArrayList<>();
        files.add(hicFile);
        Dataset ds = HiCFileTools.extractDatasetForCLT(files, false);
        genomeId[0] = cleanUpGenome(ds.getGenomeId());
        if (genomeOptionIfAvailable != null && genomeOptionIfAvailable.length() > 1) {
            genomeId[0] = genomeOptionIfAvailable;
        }

        for (HiCZoom zoom : ds.getBpZooms()) {
            resolutionStrings.add("" + zoom.getBinSize());
        }
    }

    private String cleanUpGenome(String genomeId) {
        String g2 = genomeId.toLowerCase();
        if (g2.endsWith("hg19")) return "hg19";
        if (g2.endsWith("hg38")) return "hg38";
        if (g2.endsWith("mm9")) return "mm9";
        if (g2.endsWith("mm10")) return "mm10";
        return genomeId;
    }

    private String getRemainingFiles(String[] args) {
        StringBuilder result = new StringBuilder(args[2]);
        for (int i = 3; i < args.length; i++) {
            result.append("+").append(args[i]);
        }
        return result.toString();
    }
}