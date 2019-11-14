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

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.Random;

public class GrindUtils {

    public static int[][] appropriatelyTransformVerticalStripes(int[][] data) {
        int[][] transformedData = new int[data[0].length][data.length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                transformedData[data[0].length - j - 1][data.length - i - 1] = data[i][j];
            }
        }
        return transformedData;
    }

    public static double[][] appropriatelyTransformVerticalStripes(double[][] data) {
        double[][] transformedData = new double[data[0].length][data.length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                transformedData[data[0].length - j - 1][data.length - i - 1] = data[i][j];
            }
        }
        return transformedData;
    }

    public static float[][] appropriatelyTransformVerticalStripes(float[][] data) {
        float[][] transformedData = new float[data[0].length][data.length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                transformedData[data[0].length - j - 1][data.length - i - 1] = data[i][j];
            }
        }
        return transformedData;
    }

    public static float[][] generateDefaultDistortionLabelsFile(int length, int numSuperDiagonals, boolean isContinous) {
        float[][] labels = new float[length][length];
        for (int i = 0; i < length; i++) {
            labels[i][i] = 1;
        }

        for (int k = 1; k < numSuperDiagonals + 1; k++) {
            float scale = (numSuperDiagonals - k + 2f) / (numSuperDiagonals + 2f);
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
                    labels[j][i] = 0;
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
    public static void cleanUpLabelsMatrixBasedOnData(float[][] labelsMatrix, float[][] compositeMatrix) {
        float[] rowSums = MatrixTools.getRowSums(compositeMatrix);
        zeroOutLabelsBasedOnNeighboringRowSums(labelsMatrix, rowSums);
    }

    private static void zeroOutLabelsBasedOnNeighboringRowSums(float[][] labelsMatrix, float[] rowSums) {
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

    public static void saveGrindMatrixDataToFile(String fileName, String path, int[][] labels, Writer writer, boolean useTxtInsteadOfNPY) throws IOException {
        if (useTxtInsteadOfNPY) {
            String txtFileName = fileName + ".txt";
            MatrixTools.saveMatrixTextV2(path + "/" + txtFileName, labels);
        } else {
            String npyFileName = fileName + ".npy";
            MatrixTools.saveMatrixTextNumpy(path + "/" + npyFileName, labels);
        }
        synchronized (writer) {
            writer.write(fileName + "\n");
        }
    }

    public static void saveGrindMatrixDataToFile(String fileName, String path, RealMatrix matrix, Writer writer, boolean useTxtInsteadOfNPY) throws IOException {
        saveGrindMatrixDataToFile(fileName, path, matrix.getData(), writer, useTxtInsteadOfNPY);
    }

    public static void saveGrindMatrixDataToFile(String fileName, String path, double[][] data, Writer writer, boolean useTxtInsteadOfNPY) throws IOException {
        if (useTxtInsteadOfNPY) {
            String txtFileName = fileName + ".txt";
            MatrixTools.saveMatrixTextV2(path + "/" + txtFileName, data);
        } else {
            String npyFileName = fileName + ".npy";
            MatrixTools.saveMatrixTextNumpy(path + "/" + npyFileName, data);
        }
        synchronized (writer) {
            writer.write(fileName + "\n");
        }
    }

    public static void saveGrindMatrixDataToFile(String fileName, String path, float[][] data, Writer writer, boolean useTxtInsteadOfNPY) throws IOException {
        if (useTxtInsteadOfNPY) {
            String txtFileName = fileName + ".txt";
            MatrixTools.saveMatrixTextV2(path + "/" + txtFileName, data);
        } else {
            String npyFileName = fileName + ".npy";
            MatrixTools.saveMatrixTextNumpy(path + "/" + npyFileName, data);
        }
        synchronized (writer) {
            writer.write(fileName + "\n");
        }
    }

    public static void saveGrindMatrixDataToImage(String fileName, String path, float[][] data, Writer writer,
                                                  boolean isLabelMatrix) throws IOException {
        float meanToScaleWithR = 1, meanToScaleWithG = 1, meanToScaleWithB = 1;
        if (!isLabelMatrix) {
            float meanToScaleWith = 0;
            for (int i = 0; i < data.length; i++) {
                meanToScaleWith += data[i][i];
            }
            meanToScaleWith = meanToScaleWith / data.length;
            meanToScaleWithB = meanToScaleWith / 2;
            meanToScaleWithG = meanToScaleWith / 4;
            meanToScaleWithR = meanToScaleWith / 8;
        }

        File myNewPNGFile = new File(path + "/" + fileName);
        BufferedImage image = new BufferedImage(data.length, data[0].length, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                float val = data[i][j];
                int r = Math.min(255, (int) Math.round(255. * val / meanToScaleWithR));
                int g = Math.min(255, (int) Math.round(255. * val / meanToScaleWithG));
                int b = Math.min(255, (int) Math.round(255. * val / meanToScaleWithB));
                Color myColor = new Color(r, g, b);
                image.setRGB(i, j, myColor.getRGB());
            }
        }

        ImageIO.write(image, "PNG", myNewPNGFile);
    }

    public static void saveGrindMatrixDataToImage(String fileName, String path, double[][] data, Writer writer,
                                                  boolean isLabelMatrix) throws IOException {
        double meanToScaleWithR = 1, meanToScaleWithG = 1, meanToScaleWithB = 1;
        if (!isLabelMatrix) {
            double meanToScaleWith = 0;
            for (int i = 0; i < data.length; i++) {
                meanToScaleWith += data[i][i];
            }
            meanToScaleWith = meanToScaleWith / data.length;
            meanToScaleWithB = meanToScaleWith / 2;
            meanToScaleWithG = meanToScaleWith / 4;
            meanToScaleWithR = meanToScaleWith / 8;
        }

        File myNewPNGFile = new File(path + "/" + fileName);
        BufferedImage image = new BufferedImage(data.length, data[0].length, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                double val = data[i][j];
                int r = Math.min(255, (int) Math.round(255. * val / meanToScaleWithR));
                int g = Math.min(255, (int) Math.round(255. * val / meanToScaleWithG));
                int b = Math.min(255, (int) Math.round(255. * val / meanToScaleWithB));
                Color myColor = new Color(r, g, b);
                image.setRGB(i, j, myColor.getRGB());
            }
        }

        ImageIO.write(image, "PNG", myNewPNGFile);
    }

    public static Pair<float[][], float[][]> randomlyManipulateMatrix(float[][] data, float[][] labels, Random generator) {
        float[][] newData, newLabels;
        Pair<Integer, Integer> boundaries = randomlyPickTwoIndices(data.length, generator);
        int lengthTranslocation = boundaries.getSecond() - boundaries.getFirst() + 1;
        int newPosition = generator.nextInt(data.length - lengthTranslocation);

        if (generator.nextBoolean()) {
            // create inversion
            newData = invertMatrixRegion(data, boundaries);
            newLabels = invertMatrixRegion(labels, boundaries);

            // both with lower probability
            if (generator.nextBoolean()) {
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

    private static float[][] invertMatrixRegion(float[][] data, Pair<Integer, Integer> boundaries) {
        float[][] transformedData = flipRowsInBoundaries(data, boundaries);
        transformedData = MatrixTools.transpose(transformedData);
        return flipRowsInBoundaries(transformedData, boundaries);
    }

    private static float[][] flipRowsInBoundaries(float[][] data, Pair<Integer, Integer> boundaries) {
        float[][] transformedRegion = MatrixTools.deepClone(data);
        for (int i = boundaries.getFirst(); i <= boundaries.getSecond(); i++) {
            int copyIndex = boundaries.getSecond() - i + boundaries.getFirst();
            System.arraycopy(data[i], 0, transformedRegion[copyIndex], 0, data[i].length);
        }
        return transformedRegion;
    }

    private static float[][] translocateMatrixRegion(float[][] data, Pair<Integer, Integer> boundaries, int newIndex) {
        float[][] transformedData = translateRowsInBoundaries(data, boundaries, newIndex);
        transformedData = MatrixTools.transpose(transformedData);
        return translateRowsInBoundaries(transformedData, boundaries, newIndex);
    }

    private static float[][] translateRowsInBoundaries(float[][] source, Pair<Integer, Integer> boundaries, int newIndex) {
        Pair<float[][], float[][]> splitMatrix = splitApartRowsOfMatrix(source, boundaries);
        return insertRowsAndReformMatrix(splitMatrix, newIndex);
    }

    private static float[][] insertRowsAndReformMatrix(Pair<float[][], float[][]> splitMatrix, int newIndex) {
        float[][] overall = splitMatrix.getFirst();
        float[][] region = splitMatrix.getSecond();
        int n = overall.length + region.length;

        float[][] finalMatrix = new float[n][n];
        int iter = 0;
        for (int i = 0; i < newIndex; i++) {
            System.arraycopy(overall[i], 0, finalMatrix[iter], 0, overall[i].length);
            iter++;
        }
        for (float[] row : region) {
            System.arraycopy(row, 0, finalMatrix[iter], 0, row.length);
            iter++;
        }
        for (int i = newIndex; i < overall.length; i++) {
            System.arraycopy(overall[i], 0, finalMatrix[iter], 0, overall[i].length);
            iter++;
        }
        return finalMatrix;
    }

    private static Pair<float[][], float[][]> splitApartRowsOfMatrix(float[][] source, Pair<Integer, Integer> boundaries) {
        int lengthOfTranslocation = boundaries.getSecond() - boundaries.getFirst() + 1;
        float[][] copyRegionBeingTranslated = new float[lengthOfTranslocation][source[0].length];
        int iterI1 = 0;
        for (int i = boundaries.getFirst(); i <= boundaries.getSecond(); i++) {
            System.arraycopy(source[i], 0, copyRegionBeingTranslated[iterI1], 0, source[0].length);
            iterI1++;
        }

        float[][] copyRegionNOTBeingTranslated = new float[source.length - lengthOfTranslocation][source[0].length];

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

    private static Pair<Integer, Integer> randomlyPickTwoIndices(int length, Random generator) {
        Integer a = generator.nextInt(length);
        Integer b = generator.nextInt(length);
        while (Math.abs(a - b) < 2) {
            b = generator.nextInt(length);
        }
        if (a < b) {
            return new Pair<>(a, b);
        } else {
            return new Pair<>(b, a);
        }
    }
}
