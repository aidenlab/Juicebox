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

import juicebox.tools.utils.common.MatrixTools;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.util.Pair;

import java.io.IOException;
import java.io.Writer;
import java.util.Random;

public class GrindUtils {

    private static final Random generator = new Random(0);

    public static double[][] generateDefaultDistortionLabelsFile(int length, int numSuperDiagonals, boolean isContinous) {
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

        if (!isContinous) {
            int midPt = length / 2;
            for (int i = midPt; i < midPt + 2 * numSuperDiagonals; i++) {
                for (int j = midPt - 2 * numSuperDiagonals; j < midPt; j++) {
                    labels[i][j] = 0;
                }
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
        double[] rowSums = MatrixTools.getRowSums(compositeMatrix);
        zeroOutLabelsBasedOnNeighboringRowSums(labelsMatrix, rowSums);
    }

    private static void zeroOutLabelsBasedOnNeighboringRowSums(double[][] labelsMatrix, double[] rowSums) {
        for (int i = 0; i < rowSums.length - 1; i++) {
            if (rowSums[i] <= 0 && rowSums[i + 1] <= 0) {
                for (int j = 0; j < rowSums.length; j++) {
                    labelsMatrix[i][j] = 0;
                    labelsMatrix[j][i] = 0;
                }
            }
        }

        for (int i = 1; i < rowSums.length; i++) {
            if (rowSums[i] <= 0 && rowSums[i - 1] <= 0) {
                for (int j = 0; j < rowSums.length; j++) {
                    labelsMatrix[i][j] = 0;
                    labelsMatrix[j][i] = 0;
                }
            }
        }
    }

    public static boolean mapRegionIsProblematic(RealMatrix localizedRegionData, double maxAllowedPercentZeroedOutColumns) {
        double[][] data = localizedRegionData.getData();
        int numNonZeroRows = 0;

        for (double[] datum : data) {
            int sum = 0;
            for (int j = 0; j < datum.length; j++) {
                sum += Math.round(datum[j]);
            }
            if (sum < 1) {
                numNonZeroRows++;
            }
        }

        return numNonZeroRows > data.length * maxAllowedPercentZeroedOutColumns;
    }

    public static void saveGrindMatrixDataToFile(String fileName, String path, double[][] data, Writer writer) throws IOException {
        MatrixTools.saveMatrixTextV2(path + "/" + fileName, data);
        writer.write(fileName + "\n");
    }

    public static Pair<double[][], double[][]> randomlyManipulateMatrix(double[][] data, double[][] labels) {
        double[][] newData, newLabels;
        Pair<Integer, Integer> boundaries = randomlyPickTwoIndices(data.length);
        int lengthTranslocation = boundaries.getSecond() - boundaries.getFirst() + 1;
        int newPosition = generator.nextInt(data.length - lengthTranslocation);

        if (generator.nextBoolean()) {
            // create inversion
            newData = invertMatrixRegion(data, boundaries);
            newLabels = invertMatrixRegion(labels, boundaries);

            // both with low probability
            if (false && generator.nextBoolean() && generator.nextBoolean()) {
                // create translocation
                newData = translocateMatrixRegion(newData, boundaries, newPosition);
                newLabels = translocateMatrixRegion(newLabels, boundaries, newPosition);
            }
        } else {
            // create translocation
            newData = translocateMatrixRegion(data, boundaries, newPosition);
            newLabels = translocateMatrixRegion(labels, boundaries, newPosition);
        }

        return new Pair<>(newData, newLabels);
    }

    private static double[][] invertMatrixRegion(double[][] data, Pair<Integer, Integer> boundaries) {
        double[][] transformedData = flipRowsInBoundaries(data, boundaries);
        transformedData = MatrixTools.transpose(transformedData);
        return flipRowsInBoundaries(transformedData, boundaries);
    }

    private static double[][] flipRowsInBoundaries(double[][] data, Pair<Integer, Integer> boundaries) {
        double[][] transformedRegion = MatrixTools.deepClone(data);
        for (int i = boundaries.getFirst(); i <= boundaries.getSecond(); i++) {
            int copyIndex = boundaries.getSecond() - i + boundaries.getFirst();
            System.arraycopy(data[i], 0, transformedRegion[copyIndex], 0, data[i].length);
        }
        return transformedRegion;
    }

    private static double[][] translocateMatrixRegion(double[][] data, Pair<Integer, Integer> boundaries, int newIndex) {
        double[][] transformedData = translateRowsInBoundaries(data, boundaries, newIndex);
        transformedData = MatrixTools.transpose(transformedData);
        return translateRowsInBoundaries(transformedData, boundaries, newIndex);
    }

    private static double[][] translateRowsInBoundaries(double[][] source, Pair<Integer, Integer> boundaries, int newIndex) {
        Pair<double[][], double[][]> splitMatrix = splitApartRowsOfMatrix(source, boundaries);
        return insertRowsAndReformMatrix(splitMatrix, newIndex);
    }

    private static double[][] insertRowsAndReformMatrix(Pair<double[][], double[][]> splitMatrix, int newIndex) {
        double[][] overall = splitMatrix.getFirst();
        double[][] region = splitMatrix.getSecond();
        int n = overall.length + region.length;

        double[][] finalMatrix = new double[n][n];
        int iter = 0;
        for (int i = 0; i < newIndex; i++) {
            System.arraycopy(overall[i], 0, finalMatrix[iter], 0, overall[i].length);
            iter++;
        }
        for (double[] row : region) {
            System.arraycopy(row, 0, finalMatrix[iter], 0, row.length);
            iter++;
        }
        for (int i = newIndex; i < overall.length; i++) {
            System.arraycopy(overall[i], 0, finalMatrix[iter], 0, overall[i].length);
            iter++;
        }
        return finalMatrix;
    }

    private static Pair<double[][], double[][]> splitApartRowsOfMatrix(double[][] source, Pair<Integer, Integer> boundaries) {
        int lengthOfTranslocation = boundaries.getSecond() - boundaries.getFirst() + 1;
        double[][] copyRegionBeingTranslated = new double[lengthOfTranslocation][source[0].length];
        int iterI1 = 0;
        for (int i = boundaries.getFirst(); i <= boundaries.getSecond(); i++) {
            System.arraycopy(source[i], 0, copyRegionBeingTranslated[iterI1], 0, source[0].length);
            iterI1++;
        }

        double[][] copyRegionNOTBeingTranslated = new double[source.length - lengthOfTranslocation][source[0].length];

        int iterI2 = 0;
        for (int i = 0; i < boundaries.getFirst(); i++) {
            System.arraycopy(source[i], 0, copyRegionNOTBeingTranslated[iterI2], 0, source[0].length);
            iterI2++;
        }
        for (int i = boundaries.getSecond() + 1; i < source.length; i++) {
            System.arraycopy(source[i], 0, copyRegionNOTBeingTranslated[iterI2], 0, source[0].length);
            iterI2++;
        }

        return new Pair<>(copyRegionNOTBeingTranslated, copyRegionBeingTranslated);
    }

    private static Pair<Integer, Integer> randomlyPickTwoIndices(int length) {
        Integer a = generator.nextInt(length);
        Integer b = generator.nextInt(length);
        while (a.equals(b)) {
            b = generator.nextInt(length);
        }
        if (a < b) {
            return new Pair<>(a, b);
        } else {
            return new Pair<>(b, a);
        }
    }
}
