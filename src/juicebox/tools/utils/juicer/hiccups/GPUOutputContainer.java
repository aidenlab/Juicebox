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

package juicebox.tools.utils.juicer.hiccups;

import juicebox.tools.utils.common.ArrayTools;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;

import java.util.List;

/**
 * Created by muhammadsaadshamim on 5/12/15.
 */
public class GPUOutputContainer {

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
    private final int numRows;
    private final int numColumns;


    public GPUOutputContainer(float[][] observed, float[][] peak,
                              float[][] binBL, float[][] binDonut, float[][] binH, float[][] binV,
                              float[][] expectedBL, float[][] expectedDonut, float[][] expectedH, float[][] expectedV) {
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

    public void updateHistograms(int[][] histBL, int[][] histDonut, int[][] histH, int[][] histV, int maxRows, int maxColumns) {
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numColumns; j++) {

                if (Float.isNaN(observed[i][j]))
                    continue;

                int val = (int) observed[i][j];
                processHistogramValue(binBL[i][j], val, histBL, maxRows, maxColumns);
                processHistogramValue(binDonut[i][j], val, histDonut, maxRows, maxColumns);
                processHistogramValue(binH[i][j], val, histH, maxRows, maxColumns);
                processHistogramValue(binV[i][j], val, histV, maxRows, maxColumns);
            }
        }
    }

    private void processHistogramValue(float potentialRowIndex, int columnIndex, int[][] histogram, int maxRows, int maxColumns) {
        if (Float.isNaN(potentialRowIndex))
            return;

        int rowIndex = (int) potentialRowIndex;
        if (rowIndex >= 0 && rowIndex < maxRows) {
            if (columnIndex >= 0 && columnIndex < maxColumns) {
                histogram[rowIndex][columnIndex] += 1;
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

                if (Float.isNaN(peakVal) || peakVal <= 0)
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

                int rowPos = (i + rowOffset) * resolution;
                int colPos = (j + columnOffset) * resolution;

                if (!(Float.isNaN(observedVal) ||
                        Float.isNaN(expectedBLVal) || Float.isNaN(expectedDonutVal) || Float.isNaN(expectedHVal) || Float.isNaN(expectedVVal) ||
                        Float.isNaN(binBLVal) || Float.isNaN(binDonutVal) || Float.isNaN(binHVal) || Float.isNaN(binVVal))) {
                    if (observedVal < w2 && binBLVal < w1 && binDonutVal < w1 && binHVal < w1 && binVVal < w1) {

                        peaks.add(chrIndex, chrIndex, HiCCUPSUtils.generatePeak(chrName, observedVal, peakVal,
                                rowPos, colPos, expectedBLVal, expectedDonutVal, expectedHVal, expectedVVal,
                                binBLVal, binDonutVal, binHVal, binVVal, resolution));
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

            int i = ((f.getStart1() + f.getEnd1()) / (2 * resolution)) - rowOffset;
            int j = ((f.getStart2() + f.getEnd2()) / (2 * resolution)) - columnOffset;
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

            int rowPos = (i + rowOffset) * resolution;
            int colPos = (j + columnOffset) * resolution;

            if (!(Float.isNaN(observedVal) ||
                    Float.isNaN(expectedBLVal) || Float.isNaN(expectedDonutVal) || Float.isNaN(expectedHVal) || Float.isNaN(expectedVVal) ||
                    Float.isNaN(binBLVal) || Float.isNaN(binDonutVal) || Float.isNaN(binHVal) || Float.isNaN(binVVal))) {
                if (observedVal < w2 && binBLVal < w1 && binDonutVal < w1 && binHVal < w1 && binVVal < w1) {

                    peaks.add(chrIndex, chrIndex, HiCCUPSUtils.generatePeak(chrName, observedVal, peakVal,
                            rowPos, colPos, expectedBLVal, expectedDonutVal, expectedHVal, expectedVVal,
                            binBLVal, binDonutVal, binHVal, binVVal, resolution));
                }
            }
        }


        return peaks;
    }
}
