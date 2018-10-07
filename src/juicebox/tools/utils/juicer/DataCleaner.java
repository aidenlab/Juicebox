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

package juicebox.tools.utils.juicer;

import java.util.HashMap;
import java.util.Map;

public class DataCleaner {

    final private double[][] originalData;
    final private double[][] cleanData;
    Map<Integer, Integer> cleanIndexRowToOriginalIndexRow = new HashMap<>();
    Map<Integer, Integer> cleanIndexColToOriginalIndexCol = new HashMap<>();
    private double coverageThreshold = 0.3;

    public DataCleaner(double[][] data, double threshold) {
        coverageThreshold = threshold;
        if (threshold > 0) {
            coverageThreshold = threshold;
        }

        originalData = data;

        cleanData = cleanUpData();
    }

    private double[][] cleanUpData() {

        boolean[][] isZeroNanOrInf = new boolean[originalData.length][originalData[0].length];
        int[] numZerosRowIndx = new int[originalData.length];
        int[] numZerosColIndx = new int[originalData[0].length];

        for (int i = 0; i < originalData.length; i++) {
            for (int j = 0; j < originalData[0].length; j++) {
                isZeroNanOrInf[i][j] = Double.isNaN(originalData[i][j])
                        || Double.isInfinite(originalData[i][j])
                        || isCloseToZero(originalData[i][j]);

                if (isZeroNanOrInf[i][j]) {
                    originalData[i][j] = 0;
                    numZerosRowIndx[i]++;
                    numZerosColIndx[j]++;
                }
            }
        }

        calculateWhichIndicesToKeep(numZerosRowIndx, cleanIndexRowToOriginalIndexRow);
        calculateWhichIndicesToKeep(numZerosColIndx, cleanIndexColToOriginalIndexCol);

        return makeCleanMatrix();
    }

    private double[][] makeCleanMatrix() {

        int numRows = cleanIndexRowToOriginalIndexRow.keySet().size();
        int numCols = cleanIndexColToOriginalIndexCol.keySet().size();

        double[][] cleanMatrx = new double[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            int i0 = getOriginalIndexRow(i);
            for (int j = 0; j < numCols; j++) {
                int j0 = getOriginalIndexCol(j);
                cleanMatrx[i][j] = originalData[i0][j0];
            }
        }
        return cleanMatrx;
    }


    private void calculateWhichIndicesToKeep(int[] numZerosIndxCount, Map<Integer, Integer> cleanIndexToOriginalIndex) {

        int cutOff = (int) (numZerosIndxCount.length * coverageThreshold);
        int counter = 0;

        for (int i0 = 0; i0 < numZerosIndxCount.length; i0++) {
            if (numZerosIndxCount[i0] < cutOff) {
                cleanIndexToOriginalIndex.put(counter, i0);
                counter++;
            }
        }
    }

    private boolean isCloseToZero(double v) {
        return Math.abs(v) < 1E-30;
    }


    public double[][] getCleanedData() {
        return cleanData;
    }

    public int getOriginalIndexRow(int i) {
        return cleanIndexRowToOriginalIndexRow.get(i);
    }

    public int getOriginalIndexCol(int i) {
        return cleanIndexColToOriginalIndexCol.get(i);
    }
}
