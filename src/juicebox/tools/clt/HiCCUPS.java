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

import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.tools.HiCTools;
import juicebox.tools.utils.Common.ArrayTools;
import juicebox.tools.utils.Common.HiCFileTools;
import juicebox.tools.utils.Juicer.HiCCUPS.GPUController;
import juicebox.tools.utils.Juicer.HiCCUPS.GPUOutputContainer;
import juicebox.tools.utils.Juicer.HiCCUPS.HiCCUPSPeak;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;


/**
 * Created by muhammadsaadshamim on 1/20/15.
 */
public class HiCCUPS extends JuiceboxCLT {

    private int[] resolutions = new int[]{25000, 10000, 5000};

    private boolean chrSpecified = false;
    Set<String> chromosomesSpecified = new HashSet<String>();

    private String inputHiCFileName;
    private String outputFDRFileName;
    private String outputEnrichedFileName;

    // w1 (40) corresponds to the number of expected bins (so the max allowed expected is 2^(40/3))
    // w2 (10000) corresponds to the number of reads (so it can't handle pixels with more than 10,000 reads)
    // TODO dimensions should be variably set
    private static int w1 = 40, w2 = 10000;

    private static int regionWidth = 500;
    private static int regionMargin = 20;
    private static int matrixSize = regionWidth + regionMargin + regionMargin;
    private static int fdr = 10;
    private static int window = 20;
    private static int peakWidth = 20;

    private static int divisor() {
        return (window - peakWidth) * (window + peakWidth);
    }

    public HiCCUPS() {
        super("hiccups [-r resolution] [-c chromosome] <hic file> <outputFDRThresholdsFileName> <outputEnrichedPixelsFileName>");
        // -i input file custom
    }

    @Override
    public void readArguments(String[] args, HiCTools.CommandLineParser parser) throws IOException {
        // read
        System.out.println(args);

        if (!(args.length == 4)) {
            throw new IOException("1");
        }

        inputHiCFileName = args[1];
        outputFDRFileName = args[2];
        outputEnrichedFileName = args[3];

        Set<String> specifiedChromosomes = parser.getChromosomeOption();
        Set<String> specifiedResolutions = parser.getMultipleResolutionOptions();

        if (specifiedResolutions != null) {
            resolutions = new int[specifiedResolutions.size()];

            int index = 0;
            for(String res : specifiedResolutions){
                resolutions[index] = Integer.parseInt(res);
                index++;
            }
        }

        if (specifiedChromosomes != null) {
            chromosomesSpecified = new HashSet<String>(specifiedChromosomes);
            chrSpecified = true;
        }
    }

    @Override
    public void run() {

        //Calculate parameters that will need later

        try {
            System.out.println("Accessing " + inputHiCFileName);
            DatasetReaderV2 reader = new DatasetReaderV2(inputHiCFileName);
            Dataset ds = reader.read();
            HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());

            // select zoom level closest to the requested one

            List<Chromosome> commonChromosomes = ds.getChromosomes();
            if (chrSpecified)
                commonChromosomes = new ArrayList<Chromosome>(HiCFileTools.stringToChromosomes(chromosomesSpecified,
                        commonChromosomes));

            Set<HiCZoom> actualResolutionsFound = new HashSet<HiCZoom>();

            for (int resolution : resolutions) {
                actualResolutionsFound.add(HiCFileTools.getZoomLevel(ds, resolution));
            }

            for (HiCZoom zoom : actualResolutionsFound) {
                runHiccupsProcessing(ds, zoom, commonChromosomes);
            }

        } catch (IOException e) {
            System.out.println("Unable to extract APA data");
            e.printStackTrace();
            System.exit(-3);
        }
    }

    /**
     * 
     * @param ds
     * @param zoom
     * @param commonChromosomes
     */
    private void runHiccupsProcessing(Dataset ds, HiCZoom zoom, List<Chromosome> commonChromosomes) {

        long begin_time = System.currentTimeMillis();
        int resolution = zoom.getBinSize();

        // take in input sheet
        PrintWriter outputFDR = HiCFileTools.openWriter(outputFDRFileName + "_" + resolution);
        PrintWriter outputEnriched = HiCFileTools.openWriter(outputEnrichedFileName + "_" + resolution);

        // Loop through chromosomes
        int[][] histBL = new int[w1][w2];
        int[][] histDonut = new int[w1][w2];
        int[][] histH = new int[w1][w2];
        int[][] histV = new int[w1][w2];
        float[][] fdrLogBL = new float[w1][w2];
        float[][] fdrLogDonut = new float[w1][w2];
        float[][] fdrLogH = new float[w1][w2];
        float[][] fdrLogV = new float[w1][w2];
        float[] thresholdBL = new float[w1];
        float[] thresholdDonut = new float[w1];
        float[] thresholdH = new float[w1];
        float[] thresholdV = new float[w1];
        float[] boundRowIndex = new float[1];
        float[] boundColumnIndex = new float[1];

        GPUController gpuController = new GPUController(window, matrixSize, peakWidth, divisor());

        // get the kernel function from the compiled module
        // TODO?
        //matrixmul = mod.get_function("BasicPeakCallingKernel");

        for (int runNum : new int[]{0, 1}) {
            for (Chromosome chromosome : commonChromosomes) {

                // skip these matrices
                if (chromosome.getName().equals(Globals.CHR_ALL)) continue;
                Matrix matrix = ds.getMatrix(chromosome, chromosome);
                if (matrix == null) continue;

                // get matrix data access
                long start_time = System.currentTimeMillis();
                MatrixZoomData zd = matrix.getZoomData(zoom);

                // need to extract KR and expected vector for HiCCUPS calculations; entire vector extracted, but only segments passed at a time corresponding to sliced region
                NormalizationVector krNormalizationVector = ds.getNormalizationVector(chromosome.getIndex(), zoom, NormalizationType.KR);
                if (krNormalizationVector == null)
                    krNormalizationVector = ds.getNormalizationVector(chromosome.getIndex(), zoom, NormalizationType.KR); // TODO

                double[] expectedVector = ds.getExpectedValues(zoom, NormalizationType.KR).getExpectedValues();

                // need overall bounds for the chromosome
                int chrLength = chromosome.getLength();
                int chrMatrixWdith = (int) Math.ceil((double) chrLength / resolution);
                long load_time = System.currentTimeMillis();
                System.out.println("Time to load chr" + chromosome.getName() + " matrix: " + (load_time - start_time));


                for (int i = 0; i < (chrMatrixWdith / regionWidth) + 1; i++) {
                    int[] rowBounds = calculateRegionBounds(i, regionWidth, regionMargin, matrixSize, chrMatrixWdith);

                    for (int j = i; j < (chrMatrixWdith / regionWidth) + 1; j++){
                        int[] columnBounds = calculateRegionBounds(j, regionWidth, regionMargin, matrixSize, chrMatrixWdith);

                        GPUOutputContainer gpuOutputs = gpuController.process(zd, krNormalizationVector, expectedVector,
                                rowBounds, columnBounds, matrixSize,
                                thresholdBL, thresholdDonut, thresholdH, thresholdV,
                                boundRowIndex, boundColumnIndex);

                        int diagonalCorrection = (rowBounds[4] - columnBounds[4]) + peakWidth + 2;
                        
                        if (runNum == 0) {

                            gpuOutputs.cleanUpBinNans();
                            gpuOutputs.cleanUpBinDiagonal(diagonalCorrection);
                            gpuOutputs.updateHistograms(histBL, histDonut, histH, histV, w1, w2);

                        } else { // runNum = 1

                            gpuOutputs.cleanUpPeakNaNs();
                            gpuOutputs.cleanUpPeakDiagonal(diagonalCorrection);

                            List<HiCCUPSPeak> peaksList = gpuOutputs.extractPeaks(w1, w2,
                                    rowBounds[4], columnBounds[4], resolution);

                            for(HiCCUPSPeak peak : peaksList){
                                peak.calculateFDR(fdrLogBL, fdrLogDonut, fdrLogH, fdrLogV);
                                outputEnriched.println(peak);
                            }
                        }
                    }
                }

                long segmentTime = System.currentTimeMillis();

                if (runNum == 0) {
                    System.out.println("Time to calculate chr " + chromosome.getName() + " expecteds and add to hist: " + (segmentTime - load_time) + "s");
                } else { // runNum = 1
                    System.out.println("Time to print chr" + chromosome.getName() + " peaks: " + (segmentTime - load_time) + "s");
                }

            }
            if (runNum == 0) {

                long thresh_time0 = System.currentTimeMillis();

                runZeroProcessHistogram(histBL, w1, w2, fdr, thresholdBL, fdrLogBL);
                runZeroProcessHistogram(histDonut, w1, w2, fdr, thresholdDonut, fdrLogDonut);
                runZeroProcessHistogram(histH, w1, w2, fdr, thresholdH, fdrLogH);
                runZeroProcessHistogram(histV, w1, w2, fdr, thresholdV, fdrLogV);

                for (int i = 0; i < w1; i++) {
                    outputFDR.println(i + "\t" + thresholdBL[i] + "\t" + thresholdDonut[i] + "\t" + thresholdH[i] + "\t" + thresholdV[i]);
                }

                long thresh_time1 = System.currentTimeMillis();
                System.out.println("Time to calculate thresholds: " + (thresh_time1 - thresh_time0));
            }
        }
        long final_time = System.currentTimeMillis();
        System.out.println("Total time: " + (final_time - begin_time));

        outputFDR.close();
        outputEnriched.close();
    }



    private int[] calculateRegionBounds(int index, int regionWidth, int regionMargin, int matrixSize, int chrMatrixWdith) {

        int bound1R = Math.max((index * regionWidth), 0);
        int bound2R = Math.min((index + 1) * regionWidth, chrMatrixWdith);
        int bound1 = Math.max((index * regionWidth) - regionMargin, 0);
        int bound2 = Math.min(((index + 1) * regionWidth) + regionMargin, chrMatrixWdith);
        if (bound1 == 0) {
            bound2 = matrixSize;
        }
        if (bound2 == chrMatrixWdith) {
            bound1 = chrMatrixWdith - matrixSize;
        }
        int diff1 = bound1R - bound1;
        int diff2 = bound2 - bound2R;

        return new int[]{bound1, bound2, diff1, diff2, bound1R, bound2R};
    }


    private void runZeroProcessHistogram(int[][] hist, int w1, int w2, int fdr, float[] threshold, float[][] fdrLog) {

        int[][] rcsHist = new int[w1][w2];
        for (int i = 0; i < w1; i++) {
            rcsHist[i] = ArrayTools.makeReverseCumulativeArray(hist[i]);
        }

        for (int i = 0; i < w1; i++) {
            calculateThresholdAndFDR(i, w2, fdr, rcsHist, threshold, fdrLog);
        }

    }

    private void calculateThresholdAndFDR(int index, int width, int fdr, int[][] rcsHist,
                                          float[] threshold, float[][] fdrLog) {
        if (rcsHist[index][0] > 0) {
            float[] expected = ArrayTools.doubleArrayToFloatArray(
                    ArrayTools.generateScaledPoissonPMF(index, rcsHist[index][0], width));
            float[] rcsExpected = ArrayTools.makeReverseCumulativeArray(expected);
            for (int j = 0; j < width; j++) {
                if (fdr * rcsExpected[j] <= rcsHist[index][j]) {
                    threshold[index] = (j - 1);
                    break;
                }
            }
            for (int j = (int)threshold[index]; j < width; j++) {
                float sum1 = rcsExpected[j];
                float sum2 = rcsHist[index][j];
                if (sum2 > 0) {
                    fdrLog[index][j] = (sum1 / (sum2 * 1f));
                } else {
                    break;
                }
            }
        } else {
            threshold[index] = width;
        }
    }
}

