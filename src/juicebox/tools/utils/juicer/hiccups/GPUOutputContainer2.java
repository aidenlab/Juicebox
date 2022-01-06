/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.juicer.hiccups;

import juicebox.HiCGlobals;
import juicebox.tools.utils.common.ArrayTools;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 5/12/15.
 */
public class GPUOutputContainer2 {

    private final float[][] observed;
    private final float[][] peak;
    private final float[][] binBL;
    private final float[][] binDonut;
    private final float[][] binH;
    private final float[][] binV;
    private final float[][] expectedBL;
    private final float[][] expectedDonut;
    private final float[][] expectedH;
    private final float[][] expectedV;
    private final float[][] pvalBL;
    private final float[][] pvalDonut;
    private final float[][] pvalH;
    private final float[][] pvalV;
    private final int numRows;
    private final int numColumns;


    public GPUOutputContainer2(float[][] observed, float[][] peak,
                               float[][] binBL, float[][] binDonut, float[][] binH, float[][] binV,
                               float[][] expectedBL, float[][] expectedDonut, float[][] expectedH, float[][] expectedV,
                               float[][] pvalBL, float[][] pvalDonut, float[][] pvalH, float[][] pvalV) {
        this.observed = ArrayTools.deepCopy(observed);
        this.peak = ArrayTools.deepCopy(peak);
        this.numRows = observed.length;
        this.numColumns = observed[0].length;

        this.binBL = ArrayTools.deepCopy(binBL);
        this.binDonut = ArrayTools.deepCopy(binDonut);
        this.binH = ArrayTools.deepCopy(binH);
        this.binV = ArrayTools.deepCopy(binV);

        this.expectedBL = ArrayTools.deepCopy(expectedBL);
        this.expectedDonut = ArrayTools.deepCopy(expectedDonut);
        this.expectedH = ArrayTools.deepCopy(expectedH);
        this.expectedV = ArrayTools.deepCopy(expectedV);

        this.pvalBL = ArrayTools.deepCopy(pvalBL);
        this.pvalDonut = ArrayTools.deepCopy(pvalDonut);
        this.pvalH = ArrayTools.deepCopy(pvalH);
        this.pvalV = ArrayTools.deepCopy(pvalV);
    }

    /**
     * Ensure NaN entries are uniform across the various arrays
     */
    public void cleanUpBinNans() {

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numColumns; j++) {

                if (Float.isNaN(expectedBL[i][j]) || Float.isNaN(expectedDonut[i][j]) ||
                        Float.isNaN(expectedH[i][j]) || Float.isNaN(expectedV[i][j])) {

                    binBL[i][j] = Float.NaN;
                    binDonut[i][j] = Float.NaN;
                    binH[i][j] = Float.NaN;
                    binV[i][j] = Float.NaN;
                }
            }
        }
    }

    public void cleanUpPvalNans() {

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numColumns; j++) {

                if (Float.isNaN(expectedBL[i][j]) || Float.isNaN(expectedDonut[i][j]) ||
                        Float.isNaN(expectedH[i][j]) || Float.isNaN(expectedV[i][j])) {

                    pvalBL[i][j] = Float.NaN;
                    pvalDonut[i][j] = Float.NaN;
                    pvalH[i][j] = Float.NaN;
                    pvalV[i][j] = Float.NaN;
                }

                if (expectedBL[i][j] == 0 || expectedDonut[i][j] == 0 ||
                expectedH[i][j] == 0 || expectedV[i][j] == 0) {
                    pvalBL[i][j] = Float.NaN;
                    pvalDonut[i][j] = Float.NaN;
                    pvalH[i][j] = Float.NaN;
                    pvalV[i][j] = Float.NaN;
                }
            }
        }
    }

    public void cleanUpPeakNaNs() {

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numColumns; j++) {

                if (Float.isNaN(expectedBL[i][j]) || Float.isNaN(expectedDonut[i][j]) ||
                        Float.isNaN(expectedH[i][j]) || Float.isNaN(expectedV[i][j])) {

                    peak[i][j] = Float.NaN;
                }
            }
        }
    }

    public synchronized void updateHistograms(long[][] histBL, long[][] histDonut, long[][] histH, long[][] histV, int maxRows) {
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numColumns; j++) {

                if (Float.isNaN(observed[i][j]) || Float.isInfinite(observed[i][j]))
                    continue;

                int val = (int) observed[i][j];
                processHistogramValue(binBL[i][j], val, histBL, maxRows);
                processHistogramValue(binDonut[i][j], val, histDonut, maxRows);
                processHistogramValue(binH[i][j], val, histH, maxRows);
                processHistogramValue(binV[i][j], val, histV, maxRows);
            }
        }
    }

    private void processHistogramValue(float potentialRowIndex, int columnIndex, long[][] histogram, int maxRows) {
        if (Float.isNaN(potentialRowIndex))
            return;

        int rowIndex = (int) potentialRowIndex;
        if (rowIndex >= 0 && rowIndex < maxRows) {
                histogram[rowIndex][0] += 1;
                if (HiCGlobals.printVerboseComments && histogram[rowIndex][0] < 0) {
                    System.out.println("earlier source row " + rowIndex + " col " + columnIndex + " -- " + histogram[rowIndex][0]);
                }
        }
    }

    public synchronized void updatePvalLists(Map<Integer, Map<Long, Integer>> pvalBLMap, Map<Integer,Map<Long, Integer>> pvalDonutMap, Map<Integer,Map<Long, Integer>> pvalHMap, Map<Integer,Map<Long, Integer>> pvalVMap,
                                             long[][] histBL, long[][] histDonut, long[][] histH, long[][] histV, int maxRows, double FDRthreshold) {
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numColumns; j++) {

                if (Float.isNaN(observed[i][j]) || Float.isInfinite(observed[i][j]))
                    continue;

                processPValue(binBL[i][j], pvalBL[i][j], histBL, pvalBLMap, maxRows, FDRthreshold);
                processPValue(binDonut[i][j], pvalDonut[i][j], histDonut, pvalDonutMap, maxRows, FDRthreshold);
                processPValue(binH[i][j], pvalH[i][j], histH, pvalHMap, maxRows, FDRthreshold);
                processPValue(binV[i][j], pvalV[i][j], histV, pvalVMap, maxRows, FDRthreshold);
            }
        }
    }

    private synchronized void processPValue(float potentialRowIndex, float potentialPValue, long[][] hist, Map<Integer,Map<Long, Integer>> kHist, int maxRows, double FDRthreshold) {
        if (Float.isNaN(potentialRowIndex))
            return;

        if (Float.isNaN(potentialPValue))
            return;

        int rowIndex = (int) potentialRowIndex;
        if (rowIndex >= 0 && rowIndex < maxRows) {
            if (potentialPValue < FDRthreshold) {
                hist[rowIndex][1]++;
                double ktemp = (potentialPValue * hist[rowIndex][0]) / FDRthreshold;
                long k = ktemp == Math.ceil(ktemp) ? (long) ktemp + 1 : (long) ktemp;
                if (kHist.get(rowIndex).get(k) == null) {
                    kHist.get(rowIndex).put(k, 1);
                } else {
                    kHist.get(rowIndex).put(k, kHist.get(rowIndex).get(k) + 1);
                }
            }
        }
    }


    public void cleanUpBinDiagonal(int relativeDiagonal) {

        // correction for diagonal (don't need to double number of calculations)
        if (relativeDiagonal >= (-1 * numRows)) {

            // TODO optimize so only necessary region eliminated
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numColumns; j++) {
                    if (j - i <= relativeDiagonal) {
                        binBL[i][j] = Float.NaN;
                        binDonut[i][j] = Float.NaN;
                        binH[i][j] = Float.NaN;
                        binV[i][j] = Float.NaN;
                    }
                }
            }
        }


    }

    public void cleanUpPeakDiagonal(int relativeDiagonal) {
        if (relativeDiagonal >= (-1 * numRows)) {

            // TODO optimize so only necessary region eliminated
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numColumns; j++) {
                    if (j - i <= relativeDiagonal) {
                        peak[i][j] = Float.NaN;
                    }
                }
            }
        }
    }

    public Feature2DList extractPeaks(int chrIndex, String chrName, int w1, int w2,
                                      int rowOffset, int columnOffset, int resolution) {

        Feature2DList peaks = new Feature2DList();

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numColumns; j++) {

                float peakVal = peak[i][j];

                if (Float.isNaN(peakVal) || peakVal <= 0 ||
                        Float.isNaN(pvalBL[i][j]) || Float.isNaN(pvalDonut[i][j]) || Float.isNaN(pvalH[i][j]) || Float.isNaN(pvalV[i][j]))
                    continue;

                float observedVal = observed[i][j];
                float expectedBLVal = expectedBL[i][j];
                float expectedDonutVal = expectedDonut[i][j];
                float expectedHVal = expectedH[i][j];
                float expectedVVal = expectedV[i][j];
                float binBLVal = binBL[i][j];
                float binDonutVal = binDonut[i][j];
                float binHVal = binH[i][j];
                float binVVal = binV[i][j];
                float pvalBLVal = pvalBL[i][j];
                float pvalDonutVal = pvalDonut[i][j];
                float pvalHVal = pvalH[i][j];
                float pvalVVal = pvalV[i][j];

                int rowPos = (i + rowOffset) * resolution;
                int colPos = (j + columnOffset) * resolution;

                if (!(Float.isNaN(observedVal) ||
                        Float.isNaN(expectedBLVal) || Float.isNaN(expectedDonutVal) || Float.isNaN(expectedHVal) || Float.isNaN(expectedVVal) ||
                        Float.isNaN(binBLVal) || Float.isNaN(binDonutVal) || Float.isNaN(binHVal) || Float.isNaN(binVVal) ||
                        Float.isNaN(pvalBLVal) || Float.isNaN(pvalDonutVal) || Float.isNaN(pvalHVal) || Float.isNaN(pvalVVal))) {
                    if (observedVal < w2 && binBLVal < w1 && binDonutVal < w1 && binHVal < w1 && binVVal < w1) {

                        peaks.add(chrIndex, chrIndex, HiCCUPSUtils.generatePeak(chrName, observedVal, peakVal,
                                rowPos, colPos, expectedBLVal, expectedDonutVal, expectedHVal, expectedVVal,
                                binBLVal, binDonutVal, binHVal, binVVal, resolution, pvalBLVal, pvalDonutVal, pvalHVal, pvalVVal));
                    }
                }
            }
        }

        return peaks;
    }

    public Feature2DList extractPeaksListGiven(int chrIndex, String chrName, int w1, int w2,
                                               int rowOffset, int columnOffset, int resolution, List<Feature2D> inputListFoundFeatures) {

        Feature2DList peaks = new Feature2DList();

        for (Feature2D f : inputListFoundFeatures) {

            int i = (int) ((f.getStart1() + f.getEnd1()) / (2 * resolution)) - rowOffset;
            int j = (int) ((f.getStart2() + f.getEnd2()) / (2 * resolution)) - columnOffset;
            float peakVal = peak[i][j];


            float observedVal = observed[i][j];
            float expectedBLVal = expectedBL[i][j];
            float expectedDonutVal = expectedDonut[i][j];
            float expectedHVal = expectedH[i][j];
            float expectedVVal = expectedV[i][j];
            float binBLVal = binBL[i][j];
            float binDonutVal = binDonut[i][j];
            float binHVal = binH[i][j];
            float binVVal = binV[i][j];
            float pvalBLVal = pvalBL[i][j];
            float pvalDonutVal = pvalDonut[i][j];
            float pvalHVal = pvalH[i][j];
            float pvalVVal = pvalV[i][j];

            int rowPos = (i + rowOffset) * resolution;
            int colPos = (j + columnOffset) * resolution;

            if (!(Float.isNaN(observedVal) ||
                    Float.isNaN(expectedBLVal) || Float.isNaN(expectedDonutVal) || Float.isNaN(expectedHVal) || Float.isNaN(expectedVVal) ||
                    Float.isNaN(binBLVal) || Float.isNaN(binDonutVal) || Float.isNaN(binHVal) || Float.isNaN(binVVal))) {
                if (observedVal < w2 && binBLVal < w1 && binDonutVal < w1 && binHVal < w1 && binVVal < w1) {

                    peaks.add(chrIndex, chrIndex, HiCCUPSUtils.generatePeak(chrName, observedVal, peakVal,
                            rowPos, colPos, expectedBLVal, expectedDonutVal, expectedHVal, expectedVVal,
                            binBLVal, binDonutVal, binHVal, binVVal, resolution, pvalBLVal, pvalDonutVal, pvalHVal, pvalVVal));
                }
            }
        }


        return peaks;
    }
}
