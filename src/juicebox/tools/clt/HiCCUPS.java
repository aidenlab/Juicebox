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
import juicebox.HiCGlobals;
import juicebox.data.Dataset;
import juicebox.data.DatasetReaderV2;
import juicebox.data.Matrix;
import juicebox.data.MatrixZoomData;
import juicebox.tools.HiCTools;
import juicebox.tools.utils.common.ArrayTools;
import juicebox.tools.utils.common.HiCFileTools;
import juicebox.tools.utils.juicer.hiccups.GPUController;
import juicebox.tools.utils.juicer.hiccups.GPUOutputContainer;
import juicebox.track.feature.Feature2DList;
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

    public static final int regionMargin = 20;
    private static final int totalMargin = 2 * regionMargin;
    // w1 (40) corresponds to the number of expected bins (so the max allowed expected is 2^(40/3))
    // w2 (10000) corresponds to the number of reads (so it can't handle pixels with more than 10,000 reads)
    // TODO dimensions should be variably set
    private static int w1 = 40, w2 = 10000;
    private static int matrixSize = 512;// 540 original
    private static int regionWidth = matrixSize - totalMargin;
    private static int fdr = 10;// TODO must be greater than 1, fdr percentage (change to)
    private static int window = 3;
    private static int peakWidth = 1;
    private int[] resolutions = new int[]{25000, 10000, 5000};
    private boolean chrSpecified = false;
    private Set<String> chromosomesSpecified = new HashSet<String>();
    private String inputHiCFileName;
    private String outputFDRFileName;
    private String outputEnrichedFileName;

    public HiCCUPS() {
        super("hiccups [-r resolution] [-c chromosome] [-m matrixSize] <hicFile> <outputFDRThresholdsFileName> <outputEnrichedPixelsFileName>");
        // -i input file custom
    }

    private static int divisor() {
        return (window - peakWidth) * (window + peakWidth);
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
        int specifiedMatrixSize = parser.getMatrixSizeOption();

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

        if(specifiedMatrixSize > 2*regionMargin){
            matrixSize = specifiedMatrixSize;
            regionWidth = specifiedMatrixSize - totalMargin;
        }
        System.out.println("Using Matrix Size "+ matrixSize);
    }

    @Override
    public void run() {

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
            System.out.println("Unable to run hiccups");
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

        PrintWriter outputFDR = HiCFileTools.openWriter(outputFDRFileName + "_" + resolution);

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


        GPUController gpuController = new GPUController(window, matrixSize, peakWidth, divisor());

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
                int chrMatrixWdith = (int) Math.ceil((double) chrLength / resolution);
                long load_time = System.currentTimeMillis();
                System.out.println("Time to load chr " + chromosome.getName() + " matrix: " + (load_time - start_time) + "ms");


                for (int i = 0; i < Math.ceil(chrMatrixWdith * 1.0 / regionWidth) + 1; i++) {
                    int[] rowBounds = calculateRegionBounds(i, regionWidth, regionMargin, chrMatrixWdith);

                    if(rowBounds[4] < chrMatrixWdith - regionMargin) {
                        for (int j = i; j < Math.ceil(chrMatrixWdith * 1.0 / regionWidth) + 1; j++) {
                            int[] columnBounds = calculateRegionBounds(j, regionWidth, regionMargin, chrMatrixWdith);

                            if (columnBounds[4] < chrMatrixWdith - regionMargin) {
                                GPUOutputContainer gpuOutputs = gpuController.process(zd, normalizationVector, expectedVector,
                                        rowBounds, columnBounds, matrixSize,
                                        thresholdBL, thresholdDonut, thresholdH, thresholdV,
                                        boundRowIndex, boundColumnIndex, preferredNormalization);

                                int diagonalCorrection = (rowBounds[4] - columnBounds[4]) + peakWidth + 2;

                                if (runNum == 0) {

                                    gpuOutputs.cleanUpBinNans();
                                    gpuOutputs.cleanUpBinDiagonal(diagonalCorrection);
                                    gpuOutputs.updateHistograms(histBL, histDonut, histH, histV, w1, w2);


                                } else { // runNum = 1

                                    gpuOutputs.cleanUpPeakNaNs();
                                    gpuOutputs.cleanUpPeakDiagonal(diagonalCorrection);

                                    Feature2DList peaksList = gpuOutputs.extractPeaks(chromosome.getIndex(), chromosome.getName(),
                                            w1, w2, rowBounds[4], columnBounds[4], resolution);
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
                    calculateThresholdAndFDR(i, w2, fdr, unitPoissonPMF, rcsHistBL, thresholdBL, fdrLogBL);
                    calculateThresholdAndFDR(i, w2, fdr, unitPoissonPMF, rcsHistDonut, thresholdDonut, fdrLogDonut);
                    calculateThresholdAndFDR(i, w2, fdr, unitPoissonPMF, rcsHistH, thresholdH, fdrLogH);
                    calculateThresholdAndFDR(i, w2, fdr, unitPoissonPMF, rcsHistV, thresholdV, fdrLogV);
                }

                for (int i = 0; i < w1; i++) {
                    outputFDR.println(i + "\t" + thresholdBL[i] + "\t" + thresholdDonut[i] + "\t" + thresholdH[i] + "\t" + thresholdV[i]);
                }

                long thresh_time1 = System.currentTimeMillis();
                System.out.println("Time to calculate thresholds: " + (thresh_time1 - thresh_time0) + "ms");
            }
        }
        long final_time = System.currentTimeMillis();
        System.out.println("Total time: " + (final_time - begin_time));

        globalList.exportFeatureList(outputEnrichedFileName + "_" + resolution);

        outputFDR.close();
    }



    private int[] calculateRegionBounds(int index, int regionWidth, int regionMargin, int chrMatrixWidth) {

        int bound1R = Math.min(regionMargin + (index * regionWidth), chrMatrixWidth - regionMargin);
        int bound1 = bound1R - regionMargin;
        int bound2R = Math.min(bound1R + regionWidth, chrMatrixWidth - regionMargin);
        int bound2 = bound2R + regionMargin;

        int diff1 = bound1R - bound1;
        int diff2 = bound2 - bound2R;

        return new int[]{bound1, bound2, diff1, diff2, bound1R, bound2R};
    }

    private void calculateThresholdAndFDR(int index, int width, int fdr, float[] poissonPMF,
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

            for (int j = (int)threshold[index]; j < width; j++) {
                float sum1 = rcsExpected[j];
                float sum2 = rcsHist[index][j];
                if (sum2 > 0) {
                    fdrLog[index][j] = sum1/sum2;
                } else {
                    break;
                }
            }
        }
    }
}

