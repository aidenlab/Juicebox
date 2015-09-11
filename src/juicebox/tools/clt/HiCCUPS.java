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

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Floats;
import jargs.gnu.CmdLineParser;
import juicebox.HiCGlobals;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.Matrix;
import juicebox.data.MatrixZoomData;
import juicebox.tools.utils.common.ArrayTools;
import juicebox.tools.utils.juicer.hiccups.GPUController;
import juicebox.tools.utils.juicer.hiccups.GPUOutputContainer;
import juicebox.tools.utils.juicer.hiccups.HiCCUPSUtils;
import juicebox.track.feature.Feature2DList;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import java.awt.*;
import java.io.PrintWriter;
import java.util.*;
import java.util.List;

/**
 * HiC Computational Unbiased Peak Search
 *
 * @Created by muhammadsaadshamim on 1/20/15.
 *
 */
public class HiCCUPS extends JuiceboxCLT {

    public static final int regionMargin = 20;
    public static final int krNeighborhood = 5;
    //public static final int originalPixelClusterRadius = 20000; //TODO --> 10000? original 20000
    public static final Color defaultPeakColor = Color.cyan;
    public static final boolean shouldColorBeScaledByFDR = false;

    private static final int totalMargin = 2 * regionMargin;
    // w1 (40) corresponds to the number of expected bins (so the max allowed expected is 2^(40/3))
    // w2 (10000) corresponds to the number of reads (so it can't handle pixels with more than 10,000 reads)
    // TODO dimensions should be variably set
    private static final int w1 = 40;
    private static final int w2 = 10000;
    //private static final int fdr = 10;// TODO must be greater than 1, fdr percentage (change to)
    //private static final int peakWidth = 1;
    //private static final int window = 3;
    // defaults are set based on GM12878/IMR90
    public static double fdrsum = 0.02;
    public static double oeThreshold1 = 1.5;
    public static double oeThreshold2 = 1.75;


    /*
     * Reasonable Commands
     *
     * fdr = 10 for all resolutions
     * peak width = 1 for 25kb, 2 for 10kb, 4 for 5kb
     * window = 3 for 25kb, 5 for 10kb, 7 for 5kb
     *
     * cluster radius is 20kb for 5kb and 10kb res and 50kb for 25kb res
     * fdrsumthreshold is 0.02 for all resolutions
     * oeThreshold1 = 1.5 for all res
     * oeThreshold2 = 1.75 for all res
     * oeThreshold3 = 2 for all res
     *
     * published GM12878 looplist was only generated with 5kb and 10kb resolutions
     * same with published IMR90 looplist
     * published CH12 looplist only generated with 10kb
     */
    public static double oeThreshold3 = 2;
    //public static double pixelClusterRadius = originalPixelClusterRadius;
    private static boolean dataShouldBePostProcessed = true;
    private static int matrixSize = 512;// 540 original
    private static int regionWidth = matrixSize - totalMargin;
    private boolean chrSpecified = false;
    private Set<String> chromosomesSpecified = new HashSet<String>();
    private String inputHiCFileName;
    private String outputFDRFileName;
    private String outputEnrichedFileName;
    private String outputFinalLoopListFileName;
    private Configuration[] configurations;

    public HiCCUPS() { //TODO fdr, window, peakwidth flags
        super("hiccups [-m matrixSize] [-c chromosome(s)] [-r resolution(s)] [-f fdr] [-p peak width] [-w window] " +
                "[-t thresholds] [-d centroid distances] <hicFile(s)> <finalLoopsList>");
        HiCGlobals.useCache = false;
        // also
        // hiccups [-r resolution] [-c chromosome] [-m matrixSize] <hicFile> <outputFDRThresholdsFileName>

        //TODO hiccups [-m matrixSize] [-c chromosome(s)] [-r resolution(s)] [-f fdr] [-p peak width] [-w window]
        //[-t thresholds] [-d centroid distances] <hicFile(s)> <finalLoopsList>
    }

    private static void postProcess(Map<Integer, Feature2DList> looplists, Dataset ds,
                                    List<Chromosome> commonChromosomes, String outputFinalLoopListFileName,
                                    List<Configuration> configurations) {
        for (Configuration conf : configurations) {

            int res = conf.resolution;
            //pixelClusterRadius = originalPixelClusterRadius; // reset for different resolutions

            /*for (String s : new String[]{HiCCUPSUtils.notNearCentroidAttr, HiCCUPSUtils.centroidAttr, HiCCUPSUtils.nearCentroidAttr,
                    HiCCUPSUtils.nearDiagAttr, HiCCUPSUtils.StrongAttr, HiCCUPSUtils.FilterStage}) {
                looplists.get(res).addAttributeFieldToAll(s, "0");
            }*/

            HiCCUPSUtils.removeLowMapQFeatures(looplists.get(res), res, ds, commonChromosomes);
            HiCCUPSUtils.coalesceFeaturesToCentroid(looplists.get(res), res, conf.clusterRadius);
            HiCCUPSUtils.filterOutFeaturesByFDR(looplists.get(res));
        }

        Feature2DList finalList = HiCCUPSUtils.mergeAllResolutions(looplists);
        finalList.exportFeatureList(outputFinalLoopListFileName + "_postprocessing", false);
    }

    @Override
    public void readArguments(String[] args, CmdLineParser parser) {

        CommandLineParserForJuicer juicerParser = (CommandLineParserForJuicer) parser;

        if (args.length == 4) {
            dataShouldBePostProcessed = false;
        } else if (!(args.length == 3)) {
            printUsage();
        }

        inputHiCFileName = args[1];
        if (dataShouldBePostProcessed) {
            outputFinalLoopListFileName = args[2];
        } else {
            outputFDRFileName = args[2];
            outputEnrichedFileName = args[3];
        }

        determineValidMatrixSize(juicerParser);
        determineValidChromosomes(juicerParser);
        determineValidConfigurations(juicerParser);
    }

    /**
     * @param juicerParser
     */
    private void determineValidConfigurations(CommandLineParserForJuicer juicerParser) {

        try {
            int[] resolutions = HiCCUPSUtils.extractIntegerValues(juicerParser.getMultipleResolutionOptions(), -1, -1);
            double[] fdr = HiCCUPSUtils.extractFDRValues(juicerParser.getFDROptions(), resolutions.length, 0.1f); // becomes default 10
            int[] peaks = HiCCUPSUtils.extractIntegerValues(juicerParser.getPeakOptions(), resolutions.length, 2);
            int[] windows = HiCCUPSUtils.extractIntegerValues(juicerParser.getWindowOptions(), resolutions.length, 5);
            int[] radii = HiCCUPSUtils.extractIntegerValues(juicerParser.getClusterRadiusOptions(), resolutions.length, 20000);

            configurations = new Configuration[resolutions.length];
            for (int i = 0; i < resolutions.length; i++) {
                configurations[i] = new Configuration(resolutions[i], fdr[i], peaks[i], windows[i], radii[i]);
            }

        } catch (Exception e) {
            System.out.println("Either no resolution specified or other error. Defaults being used.");
            configurations = new Configuration[]{new Configuration(10000, 10, 2, 5, 20000),
                    new Configuration(5000, 10, 4, 7, 20000)};
        }

        try {
            List<String> t = juicerParser.getThresholdOptions();
            if (t.size() > 1) {
                double[] thresholds = HiCCUPSUtils.extractDoubleValues(t, 4, -1f);
                fdrsum = thresholds[0];
                oeThreshold1 = thresholds[1];
                oeThreshold2 = thresholds[2];
                oeThreshold3 = thresholds[3];

            }
        } catch (Exception e) {
            // do nothing - use default postprocessing thresholds
        }
    }


    private void determineValidChromosomes(CommandLineParserForJuicer juicerParser) {
        List<String> specifiedChromosomes = juicerParser.getChromosomeOption();
        if (specifiedChromosomes != null) {
            chromosomesSpecified = new HashSet<String>(specifiedChromosomes);
            chrSpecified = true;
        }
    }

    private void determineValidMatrixSize(CommandLineParserForJuicer juicerParser) {
        int specifiedMatrixSize = juicerParser.getMatrixSizeOption();
        if (specifiedMatrixSize > 2 * regionMargin) {
            matrixSize = specifiedMatrixSize;
            regionWidth = specifiedMatrixSize - totalMargin;
        }
        System.out.println("Using Matrix Size " + matrixSize);
    }

    @Override
    public void run() {

        Dataset ds = HiCFileTools.extractDatasetForCLT(Arrays.asList(inputHiCFileName.split("\\+")), true);

        // select zoom level closest to the requested one

        List<Chromosome> commonChromosomes = ds.getChromosomes();
        if (chrSpecified)
            commonChromosomes = new ArrayList<Chromosome>(HiCFileTools.stringToChromosomes(chromosomesSpecified,
                    commonChromosomes));

        Map<Integer, Feature2DList> looplists = new HashMap<Integer, Feature2DList>();

        List<Configuration> filteredConfigurations = filterConfigurations(configurations, ds);
        for (Configuration conf : filteredConfigurations) {


            looplists.put(conf.resolution, runHiccupsProcessing(ds, conf, commonChromosomes));
        }

        if (dataShouldBePostProcessed) {
            postProcess(looplists, ds, commonChromosomes, outputFinalLoopListFileName, filteredConfigurations);
        }
    }

    private List<Configuration> filterConfigurations(Configuration[] configurations, Dataset ds) {

        int[] resolutions = new int[configurations.length];
        for (int i = 0; i < configurations.length; i++) {
            resolutions[i] = configurations[i].resolution;
        }
        List<Integer> filteredResolutions = HiCFileTools.filterResolutions(ds, resolutions);

        // using map because duplicate resolutions will be removed while preserving order of respective configurations
        Map<Integer, Configuration> configurationMap = new HashMap<Integer, Configuration>();
        for (int i = 0; i < configurations.length; i++) {
            configurations[i].resolution = filteredResolutions.get(i);
            configurationMap.put(filteredResolutions.get(i), configurations[i]);
        }

        return new ArrayList<Configuration>(configurationMap.values());
    }

    private Feature2DList runHiccupsProcessing(Dataset ds, Configuration conf, List<Chromosome> commonChromosomes) {

        long begin_time = System.currentTimeMillis();

        HiCZoom zoom = ds.getZoomForBPResolution(conf.resolution);

        PrintWriter outputFDR = HiCFileTools.openWriter(outputFDRFileName + "_" + conf.resolution);

        // Loop through chromosomes
        int[][] histBL = new int[w1][w2];
        int[][] histDonut = new int[w1][w2];
        int[][] histH = new int[w1][w2];
        int[][] histV = new int[w1][w2];
        float[][] fdrLogBL = new float[w1][w2];
        float[][] fdrLogDonut = new float[w1][w2];
        float[][] fdrLogH = new float[w1][w2];
        float[][] fdrLogV = new float[w1][w2];
        float[] thresholdBL = ArrayTools.newValueInitializedFloatArray(w1, (float) w2);
        float[] thresholdDonut = ArrayTools.newValueInitializedFloatArray(w1, (float) w2);
        float[] thresholdH = ArrayTools.newValueInitializedFloatArray(w1, (float) w2);
        float[] thresholdV = ArrayTools.newValueInitializedFloatArray(w1, (float) w2);
        float[] boundRowIndex = new float[1];
        float[] boundColumnIndex = new float[1];


        GPUController gpuController = new GPUController(conf.windowWidth, matrixSize, conf.peakWidth, conf.divisor());

        Feature2DList globalList = new Feature2DList();

        for (int runNum : new int[]{0, 1}) {
            for (Chromosome chromosome : commonChromosomes) {

                // skip these matrices
                if (chromosome.getName().equals(Globals.CHR_ALL)) continue;
                Matrix matrix = ds.getMatrix(chromosome, chromosome);
                if (matrix == null) continue;

                // get matrix data access
                long start_time = System.currentTimeMillis();
                MatrixZoomData zd = matrix.getZoomData(zoom);

                NormalizationType preferredNormalization = HiCFileTools.determinePreferredNormalization(ds);
                double[] normalizationVector = ds.getNormalizationVector(chromosome.getIndex(), zoom,
                        NormalizationType.KR).getData();
                double[] expectedVector = HiCFileTools.extractChromosomeExpectedVector(ds, chromosome.getIndex(),
                        zoom, preferredNormalization);

                // need overall bounds for the chromosome
                int chrLength = chromosome.getLength();
                int chrMatrixWdith = (int) Math.ceil((double) chrLength / conf.resolution);
                long load_time = System.currentTimeMillis();
                System.out.println("Time to load chr " + chromosome.getName() + " matrix: " + (load_time - start_time) + "ms");


                for (int i = 0; i < Math.ceil(chrMatrixWdith * 1.0 / regionWidth) + 1; i++) {
                    int[] rowBounds = calculateRegionBounds(i, regionWidth, chrMatrixWdith);

                    if (rowBounds[4] < chrMatrixWdith - regionMargin) {
                        for (int j = i; j < Math.ceil(chrMatrixWdith * 1.0 / regionWidth) + 1; j++) {
                            int[] columnBounds = calculateRegionBounds(j, regionWidth, chrMatrixWdith);

                            if (columnBounds[4] < chrMatrixWdith - regionMargin) {
                                GPUOutputContainer gpuOutputs = gpuController.process(zd, normalizationVector, expectedVector,
                                        rowBounds, columnBounds, matrixSize,
                                        thresholdBL, thresholdDonut, thresholdH, thresholdV,
                                        boundRowIndex, boundColumnIndex, preferredNormalization);

                                int diagonalCorrection = (rowBounds[4] - columnBounds[4]) + conf.peakWidth + 2;

                                if (runNum == 0) {

                                    gpuOutputs.cleanUpBinNans();
                                    gpuOutputs.cleanUpBinDiagonal(diagonalCorrection);
                                    gpuOutputs.updateHistograms(histBL, histDonut, histH, histV, w1, w2);


                                } else { // runNum = 1

                                    gpuOutputs.cleanUpPeakNaNs();
                                    gpuOutputs.cleanUpPeakDiagonal(diagonalCorrection);

                                    Feature2DList peaksList = gpuOutputs.extractPeaks(chromosome.getIndex(), chromosome.getName(),
                                            w1, w2, rowBounds[4], columnBounds[4], conf.resolution);
                                    peaksList.calculateFDR(fdrLogBL, fdrLogDonut, fdrLogH, fdrLogV);

                                    globalList.add(peaksList);
                                }
                            }
                        }
                    }
                }

                long segmentTime = System.currentTimeMillis();

                if (runNum == 0) {
                    System.out.println("Time to calculate chr " + chromosome.getName() + " expecteds and add to hist: " + (segmentTime - load_time) + "ms");
                } else { // runNum = 1
                    System.out.println("Time to print chr" + chromosome.getName() + " peaks: " + (segmentTime - load_time) + "ms");
                }

            }
            if (runNum == 0) {

                long thresh_time0 = System.currentTimeMillis();

                int[][] rcsHistBL = ArrayTools.makeReverse2DCumulativeArray(histBL);
                int[][] rcsHistDonut = ArrayTools.makeReverse2DCumulativeArray(histDonut);
                int[][] rcsHistH = ArrayTools.makeReverse2DCumulativeArray(histH);
                int[][] rcsHistV = ArrayTools.makeReverse2DCumulativeArray(histV);

                for (int i = 0; i < w1; i++) {
                    float[] unitPoissonPMF = Floats.toArray(Doubles.asList(ArrayTools.generatePoissonPMF(i, w2)));
                    calculateThresholdAndFDR(i, w2, conf.fdrThreshold, unitPoissonPMF, rcsHistBL, thresholdBL, fdrLogBL);
                    calculateThresholdAndFDR(i, w2, conf.fdrThreshold, unitPoissonPMF, rcsHistDonut, thresholdDonut, fdrLogDonut);
                    calculateThresholdAndFDR(i, w2, conf.fdrThreshold, unitPoissonPMF, rcsHistH, thresholdH, fdrLogH);
                    calculateThresholdAndFDR(i, w2, conf.fdrThreshold, unitPoissonPMF, rcsHistV, thresholdV, fdrLogV);
                }

                long thresh_time1 = System.currentTimeMillis();
                System.out.println("Time to calculate thresholds: " + (thresh_time1 - thresh_time0) + "ms");
            }
        }
        long final_time = System.currentTimeMillis();
        System.out.println("Total time: " + (final_time - begin_time));

        if (!dataShouldBePostProcessed) {
            globalList.exportFeatureList(outputEnrichedFileName + "_" + conf.resolution, true);
            if (outputFDR != null) {
                for (int i = 0; i < w1; i++) {
                    outputFDR.println(i + "\t" + thresholdBL[i] + "\t" + thresholdDonut[i] + "\t" + thresholdH[i] + "\t" + thresholdV[i]);
                }
            }
        }

        if (outputFDR != null) {
            outputFDR.close();
        }
        return globalList;
    }

    private int[] calculateRegionBounds(int index, int regionWidth, int chrMatrixWidth) {

        int bound1R = Math.min(regionMargin + (index * regionWidth), chrMatrixWidth - regionMargin);
        int bound1 = bound1R - regionMargin;
        int bound2R = Math.min(bound1R + regionWidth, chrMatrixWidth - regionMargin);
        int bound2 = bound2R + regionMargin;

        int diff1 = bound1R - bound1;
        int diff2 = bound2 - bound2R;

        return new int[]{bound1, bound2, diff1, diff2, bound1R, bound2R};
    }

    private void calculateThresholdAndFDR(int index, int width, double fdr, float[] poissonPMF,
                                          int[][] rcsHist, float[] threshold, float[][] fdrLog) {
        //System.out.println("");
        //System.out.println("index is "+index);
        //System.out.println("rcsHist is "+rcsHist[index][0]);
        if (rcsHist[index][0] > 0) {
            float[] expected = ArrayTools.scalarMultiplyArray(rcsHist[index][0], poissonPMF);
            float[] rcsExpected = ArrayTools.makeReverseCumulativeArray(expected);
            for (int j = 0; j < width; j++) {
                if (fdr * rcsExpected[j] <= rcsHist[index][j]) {
                    threshold[index] = (j - 1);
                    break;
                }
            }

            for (int j = (int) threshold[index]; j < width; j++) {
                float sum1 = rcsExpected[j];
                float sum2 = rcsHist[index][j];
                if (sum2 > 0) {
                    fdrLog[index][j] = sum1 / sum2;
                } else {
                    break;
                }
            }
        }
    }

    private class Configuration {
        final int windowWidth, peakWidth, clusterRadius;
        final double fdrThreshold;
        int resolution;

        Configuration(int resolution, double fdrThreshold, int peakWidth, int windowWidth, int clusterRadius) {
            this.resolution = resolution;
            this.fdrThreshold = fdrThreshold;
            this.windowWidth = windowWidth;
            this.peakWidth = peakWidth;
            this.clusterRadius = clusterRadius;
        }

        public int divisor() {
            return (windowWidth - peakWidth) * (windowWidth + peakWidth);
        }
    }
}

