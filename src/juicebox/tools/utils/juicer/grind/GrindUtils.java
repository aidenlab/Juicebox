/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.juicer.grind;

import org.apache.commons.math.linear.RealMatrix;

public class GrindUtils {

    public static double[][] generateDefaultDistortionLabelsFile(int length, int numSuperDiagonals) {
        double[][] labels = new double[length][length];
        for (int i = 0; i < length; i++) {
            labels[i][i] = 1;
        }

        for (int k = 1; k < numSuperDiagonals + 1; k++) {
            double scale = (numSuperDiagonals - k + 2) / (numSuperDiagonals + 2.0);
            for (int i = 0; i < length - k; i++) {
                labels[i][i + k] = scale;
                labels[i + k][i] = scale;
            }
        }

        return labels;
    }

    /**
     * if nans or zeros on diagonals, make that similar region 0s or empty
     *
     * @param labelsMatrix
     * @param compositeMatrix
     */
    public static void cleanUpLabelsMatrixBasedOnData(double[][] labelsMatrix, double[][] compositeMatrix) {
        // todo - right now try without making changes?
    }

    public static boolean mapRegionIsProblematic(RealMatrix localizedRegionData, double maxAllowedPercentZeroedOutColumns) {
        double[][] data = localizedRegionData.getData();
        int[] rowSums = new int[data.length];
        int numNonZeroRows = 0;

        for (int i = 0; i < data.length; i++) {
            int sum = 0;
            for (int j = 0; j < data[i].length; j++) {
                sum += Math.round(data[i][j]);
            }
            if (sum < 1) {
                numNonZeroRows++;
            }
        }

        return numNonZeroRows > data.length * maxAllowedPercentZeroedOutColumns;
    }
}
