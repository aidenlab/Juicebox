/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.tools.utils.common;

import juicebox.data.ContactRecord;
import juicebox.tools.utils.juicer.apa.APARegionStatistics;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.jetbrains.bio.npy.NpyFile;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


/**
 * Helper methods to handle matrix operations
 */
public class MatrixTools {

    /**
     * @return matrix initialized with 0s
     */
    public static RealMatrix cleanArray2DMatrix(int n) {
        return cleanArray2DMatrix(n, n);
    }

    /**
     * @return matrix initialized with 0s
     */
    public static RealMatrix cleanArray2DMatrix(int rows, int cols) {
        return presetValueMatrix(rows, cols, 0);
    }

    /**
     * @return matrix initialized with 1s
     */
    public static RealMatrix ones(int n) {
        return ones(n, n);
    }

    /**
     * @return matrix initialized with 1s
     */
    private static RealMatrix ones(int rows, int cols) {
        return presetValueMatrix(rows, cols, 1);
    }

    /**
     * @return matrix of size m x n initialized with a specified value
     */
    private static RealMatrix presetValueMatrix(int numRows, int numCols, int val) {
        RealMatrix matrix = new Array2DRowRealMatrix(numRows, numCols);
        for (int r = 0; r < numRows; r++)
            for (int c = 0; c < numCols; c++)
                matrix.setEntry(r, c, val);
        return matrix;
    }

    /**
     * @return matrix randomly initialized with 1s and 0s
     */
    public static RealMatrix randomUnitMatrix(int n) {
        return randomUnitMatrix(n, n);
    }

    /**
     * Generate a matrix with randomly initialized 1s and 0s
     *
     * @param rows number of rows
     * @param cols number of columns
     * @return randomized binary matrix
     */
    private static RealMatrix randomUnitMatrix(int rows, int cols) {
        Random generator = new Random();
        RealMatrix matrix = cleanArray2DMatrix(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                if (generator.nextBoolean())
                    matrix.setEntry(r, c, 1);
        return matrix;
    }

    /**
     * @return minimal positive entry in the matrix greater than 0
     */
    public static double minimumPositive(RealMatrix data) {
        return minimumPositive(data.getData());
    }

    /**
     * @return minimal positive entry in the matrix greater than 0
     */
    private static double minimumPositive(double[][] data) {
        double minVal = Double.MAX_VALUE;
        for (double[] row : data) {
            for (double val : row) {
                if (val > 0 && val < minVal)
                    minVal = val;
            }
        }
        if (minVal == Double.MAX_VALUE)
            minVal = 0;
        return minVal;
    }

    /**
     * @return mean of matrix
     */
    public static double mean(RealMatrix matrix) {
        return APARegionStatistics.statistics(matrix.getData()).getMean();
    }

    /**
     * Flatten a 2D double matrix into a double array
     *
     * @param matrix
     * @return 1D double array in row major order
     */
    public static double[] flattenedRowMajorOrderMatrix(RealMatrix matrix) {
        int m = matrix.getRowDimension();
        int n = matrix.getColumnDimension();
        int numElements = m * n;
        double[] flattenedMatrix = new double[numElements];

        int index = 0;
        for (int i = 0; i < m; i++) {
            System.arraycopy(matrix.getRow(i), 0, flattenedMatrix, index, n);
            index += n;
        }
        return flattenedMatrix;
    }

    public static double[] flattenedRowMajorOrderMatrix(double[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
    
        int numElements = m * n;
        double[] flattenedMatrix = new double[numElements];
    
        int index = 0;
        for (double[] doubles : matrix) {
            System.arraycopy(doubles, 0, flattenedMatrix, index, n);
            index += n;
        }
        return flattenedMatrix;
    }

    public static float[] flattenedRowMajorOrderMatrix(float[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
    
        int numElements = m * n;
        float[] flattenedMatrix = new float[numElements];
    
        int index = 0;
        for (float[] floats : matrix) {
            System.arraycopy(floats, 0, flattenedMatrix, index, n);
            index += n;
        }
        return flattenedMatrix;
    }

    public static int[] flattenedRowMajorOrderMatrix(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
    
        int numElements = m * n;
        int[] flattenedMatrix = new int[numElements];
    
        int index = 0;
        for (int[] ints : matrix) {
            System.arraycopy(ints, 0, flattenedMatrix, index, n);
            index += n;
        }
        return flattenedMatrix;
    }


    /**
     * Write data from matrix out to specified file with each row on a separate line
     *
     * @param filename
     * @param realMatrix
     */
    public static void saveMatrixText(String filename, RealMatrix realMatrix) {
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), StandardCharsets.UTF_8));
            double[][] matrix = realMatrix.getData();
            for (double[] row : matrix) {
                writer.write(Arrays.toString(row) + "\n");
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if (writer != null)
                    writer.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    /**
     * Reshape array into square matrix
     *
     * @param flatMatrix
     * @param n
     * @return properly dimensioned matrix
     */
    public static float[][] reshapeFlatMatrix(float[] flatMatrix, int n) {
        return reshapeFlatMatrix(flatMatrix, n, n);
    }

    /**
     * Reshape array into a matrix
     *
     * @param flatMatrix
     * @param numRows
     * @param numCols
     * @return properly dimensioned matrix
     */
    private static float[][] reshapeFlatMatrix(float[] flatMatrix, int numRows, int numCols) {
        float[][] matrix = new float[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            System.arraycopy(flatMatrix, i * numCols, matrix[i], 0, numCols);
        }
        return matrix;
    }

    /**
     * From Matrix M, extract out M[r1:r2,c1:c2]
     * r2, c2 not inclusive (~python numpy)
     *
     * @return extracted matrix region M[r1:r2,c1:c2]
     */
    public static float[][] extractLocalMatrixRegion(float[][] matrix, int r1, int r2, int c1, int c2) {

        int numRows = r2 - r1;
        int numColumns = c2 - c1;
        float[][] extractedRegion = new float[numRows][numColumns];

        for (int i = 0; i < numRows; i++) {
            System.arraycopy(matrix[r1 + i], c1, extractedRegion[i], 0, numColumns);
        }

        return extractedRegion;
    }

    /**
     * From Matrix M, extract out M[r1:r2,c1:c2]
     * r2, c2 not inclusive (~python numpy, not like Matlab)
     *
     * @return extracted matrix region M[r1:r2,c1:c2]
     */
    public static RealMatrix extractLocalMatrixRegion(RealMatrix matrix, int r1, int r2, int c1, int c2) {

        int numRows = r2 - r1;
        int numColumns = c2 - c1;
        RealMatrix extractedRegion = cleanArray2DMatrix(numRows, numColumns);

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numColumns; j++) {
                extractedRegion.setEntry(i, j, matrix.getEntry(r1 + i, c1 + j));
            }
        }
        return extractedRegion;
    }

    /**
     * Returns the values along the diagonal of the matrix
     *
     * @param matrix
     * @return diagonal
     */
    public static RealMatrix extractDiagonal(RealMatrix matrix) {
        int n = Math.min(matrix.getColumnDimension(), matrix.getRowDimension());
        RealMatrix diagonal = MatrixTools.cleanArray2DMatrix(n);
        for (int i = 0; i < n; i++) {
            diagonal.setEntry(i, i, matrix.getEntry(i, i));
        }
        return diagonal;
    }

    /**
     * Returns the values along the diagonal of the matrix
     *
     * @param matrix
     * @return diagonal
     */
    public static RealMatrix makeSymmetricMatrix(RealMatrix matrix) {
        RealMatrix symmetricMatrix = extractDiagonal(matrix);
        int n = symmetricMatrix.getRowDimension();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double val = matrix.getEntry(i, j);
                symmetricMatrix.setEntry(i, j, val);
                symmetricMatrix.setEntry(j, i, val);
            }
        }

        return symmetricMatrix;
    }

    /**
     * @return matrix flipped across the antidiagonal
     */
    public static RealMatrix flipAcrossAntiDiagonal(RealMatrix matrix) {
        int n = Math.min(matrix.getColumnDimension(), matrix.getRowDimension());
        RealMatrix antiDiagFlippedMatrix = cleanArray2DMatrix(n, n);
        int maxIndex = n - 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                antiDiagFlippedMatrix.setEntry(maxIndex - j, maxIndex - i, matrix.getEntry(i, j));
            }
        }
        return antiDiagFlippedMatrix;
    }

    /**
     * @return matrix flipped Left-Right
     */
    public static RealMatrix flipLeftRight(RealMatrix matrix) {
        int r = matrix.getRowDimension(), c = matrix.getColumnDimension();
        RealMatrix leftRightFlippedMatrix = cleanArray2DMatrix(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                leftRightFlippedMatrix.setEntry(i, c - 1 - j, matrix.getEntry(i, j));
            }
        }
        return leftRightFlippedMatrix;
    }

    /**
     * @return matrix flipped Top-Bottom
     */
    public static RealMatrix flipTopBottom(RealMatrix matrix) {
        int r = matrix.getRowDimension(), c = matrix.getColumnDimension();
        RealMatrix topBottomFlippedMatrix = cleanArray2DMatrix(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                topBottomFlippedMatrix.setEntry(r - 1 - i, j, matrix.getEntry(i, j));
            }
        }
        return topBottomFlippedMatrix;
    }

    /**
     * @return Element-wise multiplication of matrices i.e. M.*N in Matlab
     */
    public static RealMatrix elementBasedMultiplication(RealMatrix matrix1, RealMatrix matrix2) {
        // chooses minimal intersection of dimensions
        int r = Math.min(matrix1.getRowDimension(), matrix2.getRowDimension());
        int c = Math.min(matrix1.getColumnDimension(), matrix2.getColumnDimension());

        RealMatrix elementwiseMultipliedMatrix = cleanArray2DMatrix(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                elementwiseMultipliedMatrix.setEntry(i, j, matrix1.getEntry(i, j) * matrix2.getEntry(i, j));
            }
        }
        return elementwiseMultipliedMatrix;
    }


    /**
     * @return Element-wise division of matrices i.e. M./N in Matlab
     */
    public static RealMatrix elementBasedDivision(RealMatrix matrix1, RealMatrix matrix2) {
        // chooses minimal intersection of dimensions
        int r = Math.min(matrix1.getRowDimension(), matrix2.getRowDimension());
        int c = Math.min(matrix1.getColumnDimension(), matrix2.getColumnDimension());

        RealMatrix elementwiseDividedMatrix = cleanArray2DMatrix(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                elementwiseDividedMatrix.setEntry(i, j, matrix1.getEntry(i, j) / matrix2.getEntry(i, j));
            }
        }
        return elementwiseDividedMatrix;
    }


    /**
     * Replace NaNs in given matrix with given value
     */
    public static void setNaNs(RealMatrix matrix, int val) {
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                if (Double.isNaN(matrix.getEntry(i, j))) {
                    matrix.setEntry(i, j, val);
                }
            }
        }
    }

    /**
     * Return sign of values in matrix:
     * val > 0 : 1
     * val = 0 : 0
     * val < 0 : -1
     */
    public static RealMatrix sign(RealMatrix matrix) {
        int r = matrix.getRowDimension();
        int c = matrix.getColumnDimension();
        RealMatrix signMatrix = cleanArray2DMatrix(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                double val = matrix.getEntry(i, j);
                if (val > 0) {
                    signMatrix.setEntry(i, j, 1);
                } else if (val < 0) {
                    signMatrix.setEntry(i, j, -1);
                }
            }
        }
        return signMatrix;
    }

    /**
     * Replace all of a given value in a matrix with a new value
     */
    public static void replaceValue(RealMatrix matrix, int initialVal, int newVal) {
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                if (matrix.getEntry(i, j) == initialVal) {
                    matrix.setEntry(i, j, newVal);
                }
            }
        }
    }

    /**
     * Normalize matrix by dividing by max element
     *
     * @return matrix * (1/max_element)
     */
    public static RealMatrix normalizeByMax(RealMatrix matrix) {
        double max = calculateMax(matrix);
        return matrix.scalarMultiply(1 / max);
    }

    /**
     * @return max element in matrix
     */
    public static double calculateMax(RealMatrix matrix) {
        double max = matrix.getEntry(0, 0);
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                double val = matrix.getEntry(i, j);
                if (max < val) {
                    max = val;
                }
            }
        }
        return max;
    }

    /**
     * @return min element in matrix
     */
    public static double calculateMin(RealMatrix matrix) {
        double min = matrix.getEntry(0, 0);
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                double val = matrix.getEntry(i, j);
                if (min > val) {
                    min = val;
                }
            }
        }
        return min;
    }

    /**
     * print for matrix
     */
    public static void print(RealMatrix matrix) {
        print(matrix.getData());
    }

    /**
     * print for 2D array
     */
    private static void print(double[][] data) {
        for (double[] row : data) {
            System.out.println(Arrays.toString(row));
        }
    }

    /**
     * print for 2D array
     */
    private static void print(float[][] data) {
        for (float[] row : data) {
            System.out.println(Arrays.toString(row));
        }
    }

    /**
     * @return region within matrix specified by indices
     */
    public static RealMatrix getSubMatrix(RealMatrix matrix, int[] indices) {
        return matrix.getSubMatrix(indices[0], indices[1], indices[2], indices[3]);
    }

    /**
     * Fill lower left triangle with values from upper right triangle
     *
     * @param matrix
     * @return
     */
    public static RealMatrix fillLowerLeftTriangle(RealMatrix matrix) {
        for (int r = 0; r < matrix.getRowDimension(); r++)
            for (int c = 0; c < matrix.getColumnDimension(); c++)
                matrix.setEntry(c, r, matrix.getEntry(r, c));
        return matrix;
    }

    public static void thresholdValues(RealMatrix matrix, int val) {
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                if (matrix.getEntry(i, j) > val) {
                    matrix.setEntry(i, j, val);
                }
            }
        }
    }

    public static void thresholdValuesDouble(RealMatrix matrix, double lowVal, double highVal) {
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                if (matrix.getEntry(i, j) > highVal) {
                    matrix.setEntry(i, j, highVal);
                }
                if (matrix.getEntry(i, j) < lowVal) {
                    matrix.setEntry(i, j, lowVal);
                }
            }
        }
    }

    public static int[][] normalizeMatrixUsingColumnSum(int[][] matrix) {
        int[][] newMatrix = new int[matrix.length][matrix[0].length];
        int[] columnSum = new int[matrix[0].length];
        for (int[] row : matrix) {
            for (int i = 0; i < row.length; i++) {
                columnSum[i] += row[i];
            }
        }

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                newMatrix[i][j] = matrix[i][j] / columnSum[j];
            }
        }

        return newMatrix;
    }

    public static int[][] normalizeMatrixUsingRowSum(int[][] matrix) {
        int[][] newMatrix = new int[matrix.length][matrix[0].length];
        int[] rowSum = getRowSums(matrix);

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                newMatrix[i][j] = matrix[i][j] / rowSum[i];
            }
        }

        return newMatrix;
    }

    public static int[] getRowSums(int[][] matrix) {
        int[] rowSum = new int[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int val : matrix[i]) {
                rowSum[i] += val;
            }
        }
        return rowSum;
    }

    public static double[] getRowSums(double[][] matrix) {
        double[] rowSum = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (double val : matrix[i]) {
                rowSum[i] += val;
            }
        }
        return rowSum;
    }

    public static float[] getAbsValColSums(float[][] matrix) {
        float[] colSum = new float[matrix[0].length];
        for (float[] floats : matrix) {
            for (int j = 0; j < floats.length; j++) {
                colSum[j] += Math.abs(floats[j]);
            }
        }
        return colSum;
    }

    public static int[] getAbsValColSums(int[][] matrix) {
        int[] colSum = new int[matrix[0].length];
        for (int[] ints : matrix) {
            for (int j = 0; j < ints.length; j++) {
                colSum[j] += Math.abs(ints[j]);
            }
        }
        return colSum;
    }

    public static float[] getRowSums(float[][] matrix) {
        float[] rowSum = new float[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (float val : matrix[i]) {
                rowSum[i] += val;
            }
        }
        return rowSum;
    }

    public static double[] getRowSums(List<ContactRecord> unNormedRecordList, double scalar, double[] normVector) {
        double[] rowSum = new double[normVector.length];
        for (ContactRecord record : unNormedRecordList) {
            int x = record.getBinX();
            int y = record.getBinY();
            float counts = record.getCounts();

            double normVal = counts * scalar / (normVector[x] * normVector[y]);
            rowSum[x] += normVal;
            if (x != y) {
                rowSum[y] += normVal;
            }

        }
        return rowSum;
    }

    public static void cleanUpNaNs(RealMatrix matrix) {
        for (int r = 0; r < matrix.getRowDimension(); r++) {
            for (int c = 0; c < matrix.getColumnDimension(); c++) {
                if (Double.isNaN(matrix.getEntry(r, c))) {
                    matrix.setEntry(r, c, 0);
                }
            }
        }
    }

    public static void cleanUpNaNs(double[][] matrix) {
        for (int r = 0; r < matrix.length; r++) {
            for (int c = 0; c < matrix[r].length; c++) {
                if (Double.isNaN(matrix[r][c])) {
                    matrix[r][c] = 0;
                }
            }
        }
    }

    public static void cleanUpNaNs(float[][] matrix) {
        for (int r = 0; r < matrix.length; r++) {
            for (int c = 0; c < matrix[r].length; c++) {
                if (Float.isNaN(matrix[r][c])) {
                    matrix[r][c] = 0;
                }
            }
        }
    }

    public static double sum(double[][] data) {
        double sum = 0;
        for (double[] row : data) {
            for (double val : row) {
                sum += val;
            }
        }
        return sum;
    }

    public static double getAverage(RealMatrix data) {
        return getAverage(data.getData());
    }

    private static double getAverage(double[][] data) {
        double average = 0;
        if (data.length > 0) {
            double total = 0;
            for (double[] vals : data) {
                for (double val : vals) {
                    total += val;
                }
            }
            average = (total / data.length) / data[0].length;
        }
        return average;
    }

    public static void exportData(double[][] data, File file) {
        try {
            DecimalFormat df = new DecimalFormat("##.###");

            final FileWriter fw = new FileWriter(file);
            for (double[] row : data) {
                for (double val : row) {
                    if (Double.isNaN(val)) {
                        fw.write("NaN, ");
                    } else {
                        fw.write(Double.valueOf(df.format(val)) + ", ");
                    }
                }
                fw.write("0\n");
            }
            fw.close();
        } catch (Exception e) {
            System.err.println("Error exporting matrix");
            e.printStackTrace();
            System.exit(86);
        }
    }

    public static double[][] transpose(double[][] matrix) {
        int h0 = matrix.length;
        int w0 = matrix[0].length;
        double[][] transposedMatrix = new double[w0][h0];

        for (int i = 0; i < h0; i++) {
            for (int j = 0; j < w0; j++) {
                transposedMatrix[j][i] = matrix[i][j];
            }
        }
        return transposedMatrix;
    }

    public static float[][] transpose(float[][] matrix) {
        int h0 = matrix.length;
        int w0 = matrix[0].length;
        float[][] transposedMatrix = new float[w0][h0];

        for (int i = 0; i < h0; i++) {
            for (int j = 0; j < w0; j++) {
                transposedMatrix[j][i] = matrix[i][j];
            }
        }
        return transposedMatrix;
    }

    public static double[][] convertToDoubleMatrix(boolean[][] adjacencyMatrix) {
        double[][] matrix = new double[adjacencyMatrix.length][adjacencyMatrix[0].length];
        for (int i = 0; i < adjacencyMatrix.length; i++) {
            for (int j = 0; j < adjacencyMatrix[0].length; j++) {
                if (adjacencyMatrix[i][j]) {
                    matrix[i][j] = 1;
                }
            }
        }
        return matrix;
    }

    public static double[][] convertToDoubleMatrix(int[][] adjacencyMatrix) {
        double[][] matrix = new double[adjacencyMatrix.length][adjacencyMatrix[0].length];
        for (int i = 0; i < adjacencyMatrix.length; i++) {
            for (int j = 0; j < adjacencyMatrix[0].length; j++) {
                matrix[i][j] = adjacencyMatrix[i][j];
            }
        }
        return matrix;
    }

    public static float[][] convertToFloatMatrix(double[][] dataMatrix) {
        float[][] matrix = new float[dataMatrix.length][dataMatrix[0].length];
        for (int i = 0; i < dataMatrix.length; i++) {
            for (int j = 0; j < dataMatrix[0].length; j++) {
                matrix[i][j] = (float) dataMatrix[i][j];
            }
        }
        return matrix;
    }

    public static void copyFromAToBRegion(double[][] source, double[][] destination, int rowOffSet, int colOffSet) {
        for (int i = 0; i < source.length; i++) {
            System.arraycopy(source[i], 0, destination[i + rowOffSet], colOffSet, source[0].length);
        }
    }

    public static void copyFromAToBRegion(float[][] source, float[][] destination, int rowOffSet, int colOffSet) {
        for (int i = 0; i < source.length; i++) {
            System.arraycopy(source[i], 0, destination[i + rowOffSet], colOffSet, source[0].length);
        }
    }

    public static void saveMatrixTextV2(String filename, RealMatrix realMatrix) {
        saveMatrixTextV2(filename, realMatrix.getData());
    }

    public static void saveMatrixTextV2(String filename, double[][] matrix) {
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), StandardCharsets.UTF_8));
            for (double[] row : matrix) {
                String s = Arrays.toString(row);//.replaceAll().replaceAll("]","").trim();
                s = s.replaceAll("\\[", "").replaceAll("\\]", "").trim();
                writer.write(s + "\n");
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if (writer != null)
                    writer.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static void saveMatrixTextV2(String filename, float[][] matrix) {
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), StandardCharsets.UTF_8));
            for (float[] row : matrix) {
                String s = Arrays.toString(row);//.replaceAll().replaceAll("]","").trim();
                s = s.replaceAll("\\[", "").replaceAll("\\]", "").trim();
                writer.write(s + "\n");
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if (writer != null)
                    writer.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static void saveMatrixTextV2(String filename, int[][] matrix) {
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), StandardCharsets.UTF_8));
            for (int[] row : matrix) {
                String s = Arrays.toString(row);//.replaceAll().replaceAll("]","").trim();
                s = s.replaceAll("\\[", "").replaceAll("\\]", "").trim();
                writer.write(s + "\n");
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if (writer != null)
                    writer.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static void saveMatrixTextNumpy(String filename, double[][] matrix) {
        int numRows = matrix.length;
        int numCols = matrix[0].length;
        double[] flattenedArray = MatrixTools.flattenedRowMajorOrderMatrix(matrix);

        NpyFile.write(Paths.get(filename), flattenedArray, new int[]{numRows, numCols});
    }

    public static void saveMatrixTextNumpy(String filename, float[][] matrix) {
        int numRows = matrix.length;
        int numCols = matrix[0].length;
        float[] flattenedArray = MatrixTools.flattenedRowMajorOrderMatrix(matrix);

        NpyFile.write(Paths.get(filename), flattenedArray, new int[]{numRows, numCols});
    }

    public static void saveMatrixTextNumpy(String filename, int[][] matrix) {
        int numRows = matrix.length;
        int numCols = matrix[0].length;
        int[] flattenedArray = MatrixTools.flattenedRowMajorOrderMatrix(matrix);

        NpyFile.write(Paths.get(filename), flattenedArray, new int[]{numRows, numCols});
    }

    public static void saveMatrixTextNumpy(String filename, int[] matrix) {
        NpyFile.write(Paths.get(filename), matrix, new int[]{1, matrix.length});
    }

    public static void saveMatrixTextNumpy(String filename, double[] matrix) {
        NpyFile.write(Paths.get(filename), matrix, new int[]{1, matrix.length});
    }

    public static float[][] generateCompositeMatrixWithNansCleaned(RealMatrix matrixDiag1, RealMatrix matrixDiag2, RealMatrix matrix1vs2) {
        return generateCompositeMatrixWithNansCleaned(
                convertToFloatMatrix(matrixDiag1.getData()),
                convertToFloatMatrix(matrixDiag2.getData()),
                convertToFloatMatrix(matrix1vs2.getData()));
    }

    private static float[][] generateCompositeMatrixWithNansCleaned(float[][] matrixDiag1, float[][] matrixDiag2, float[][] matrix1vs2) {
        int newLength = matrixDiag1.length + matrixDiag2.length;
        float[][] compositeMatrix = new float[newLength][newLength];

        copyFromAToBRegion(matrixDiag1, compositeMatrix, 0, 0);
        copyFromAToBRegion(matrixDiag2, compositeMatrix, matrixDiag1.length, matrixDiag1.length);

        for (int i = 0; i < matrix1vs2.length; i++) {
            for (int j = 0; j < matrix1vs2[0].length; j++) {
                compositeMatrix[i][matrixDiag1.length + j] = matrix1vs2[i][j];
                compositeMatrix[matrixDiag1.length + j][i] = matrix1vs2[i][j];
            }
        }

        MatrixTools.cleanUpNaNs(compositeMatrix);
        return compositeMatrix;
    }

    public static double[][] deepClone(double[][] data) {
        double[][] copy = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, copy[i], 0, data[i].length);
        }
        return copy;
    }

    public static float[][] deepClone(float[][] data) {
        float[][] copy = new float[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, copy[i], 0, data[i].length);
        }
        return copy;
    }

    public static void labelRegionWithOnes(int[][] labelsMatrix, int rowLength, int numRows, int colLength, int numCols, int startRowOf1, int startColOf1) {
        for (int i = 0; i < Math.min(rowLength, numRows); i++) {
            for (int j = 0; j < Math.min(colLength, numCols); j++) {
                labelsMatrix[startRowOf1 + i][startColOf1 + j] = 1;
            }
        }
    }

    public static void labelEnrichedRegionWithOnes(int[][] labelsMatrix, double[][] data, int rowLength, int numRows, int colLength, int numCols, int startRowOf1, int startColOf1) {
        double total = 0;
        int numVals = 0;

        for (int i = 0; i < Math.min(rowLength, numRows); i++) {
            for (int j = 0; j < Math.min(colLength, numCols); j++) {
                total += data[startRowOf1 + i][startColOf1 + j];
                numVals++;
            }
        }
        double average = total / numVals;

        for (int i = 0; i < Math.min(rowLength, numRows); i++) {
            for (int j = 0; j < Math.min(colLength, numCols); j++) {
                if (data[startRowOf1 + i][startColOf1 + j] > average) {
                    labelsMatrix[startRowOf1 + i][startColOf1 + j] = 1;
                }
            }
        }
    }

    // column length assumed identical and kept the same
    public static double[][] stitchMultipleMatricesTogetherByRowDim(List<double[][]> data) {
        // todo currently assuming each one identical...

        int colNums = data.get(0)[0].length;
        int rowNums = 0;
        for (double[][] mtrx : data) {
            rowNums += mtrx.length;
        }

        double[][] aggregate = new double[rowNums][colNums];

        int rowOffSet = 0;
        for (double[][] region : data) {
            MatrixTools.copyFromAToBRegion(region, aggregate, rowOffSet, 0);
            rowOffSet += region.length;
        }

        return aggregate;
    }

    public static double[][] takeDerivativeDownColumn(double[][] data) {
        double[][] derivative = new double[data.length][data[0].length - 1];

        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, derivative[i], 0, derivative[i].length);
        }
        for (int i = 0; i < derivative.length; i++) {
            for (int j = 0; j < derivative[i].length; j++) {
                derivative[i][j] -= data[i][j + 1];
            }
        }

        return derivative;
    }

    public static double[][] smoothAndAppendDerivativeDownColumn(double[][] data, double[] convolution) {

        int numColumns = data[0].length;
        if (convolution != null && convolution.length > 1) {
            numColumns -= (convolution.length - 1);
        }

        double[][] appendedDerivative = new double[data.length][2 * numColumns - 1];

        if (convolution != null && convolution.length > 1) {
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < numColumns; j++) {
                    for (int k = 0; k < convolution.length; k++) {
                        appendedDerivative[i][j] += convolution[k] * data[i][j + k];
                    }
                }
            }
        } else {
            for (int i = 0; i < data.length; i++) {
                System.arraycopy(data[i], 0, appendedDerivative[i], 0, numColumns);
            }
        }

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < numColumns - 1; j++) {
                appendedDerivative[i][numColumns + j] = appendedDerivative[i][j] - appendedDerivative[i][j + 1];
            }
        }

        return appendedDerivative;
    }

    public static float[][] getNormalizedThresholdedAndAppendedDerivativeDownColumn(float[][] data, float maxVal, float scaleDerivFactor, float derivativeThreshold) {

        double[] averageVal = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            for (float val : data[i]) {
                averageVal[i] += val;
            }
        }

        for (int i = 0; i < data.length; i++) {
            averageVal[i] = averageVal[i] / data[i].length;
        }

        float[][] thresholdedData = new float[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                thresholdedData[i][j] = (float) Math.min(maxVal, data[i][j] / averageVal[i]);
            }
        }

        return getMainAppendedDerivativeScaledPosDownColumn(thresholdedData, scaleDerivFactor, derivativeThreshold);
    }

    public static float[][] getNormalizedThresholdedByMedian(float[][] data, float maxVal) {

        double[] medianVal = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            medianVal[i] = getMedian(data[i]);
        }

        float[][] thresholdedData = new float[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                thresholdedData[i][j] = (float) Math.min(maxVal, data[i][j] / medianVal[i]);
            }
        }

        return thresholdedData;
    }

    public static double getMedian(float[] values) {
        double[] array = new double[values.length];
        for (int k = 0; k < values.length; k++) {
            array[k] = values[k];
        }
        Median median = new Median();
        return median.evaluate(array);
    }


    public static float[][] getMainAppendedDerivativeScaledPosDownColumn(float[][] data, float scaleDerivFactor, float threshold) {

        int numColumns = data[0].length;
        float[][] derivative = getRelevantDerivativeScaledPositive(data, scaleDerivFactor, threshold);
        float[][] appendedDerivative = new float[data.length][numColumns + derivative[0].length];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, appendedDerivative[i], 0, numColumns);
        }

        for (int i = 0; i < data.length; i++) {
            System.arraycopy(derivative[i], 0, appendedDerivative[i], numColumns, derivative[i].length);
        }

        return appendedDerivative;
    }

    public static float[][] getMainAppendedDerivativeDownColumnV2(float[][] data, float scaleDerivFactor, float threshold) {

        int numColumns = data[0].length;
        float[][] derivative = getRelevantDerivative(data, scaleDerivFactor, threshold);
        float[][] appendedDerivative = new float[data.length][numColumns + derivative[0].length];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, appendedDerivative[i], 0, numColumns);
        }

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                appendedDerivative[i][j] = Math.min(.5f, Math.max(-.5f, appendedDerivative[i][j]));
            }
        }

        for (int i = 0; i < data.length; i++) {
            System.arraycopy(derivative[i], 0, appendedDerivative[i], numColumns, derivative[i].length);
        }

        return appendedDerivative;
    }

    public static float[][] getMainAppendedDerivativeDownColumn(float[][] data, float scaleDerivFactor, float threshold) {

        int numColumns = data[0].length;
        float[][] derivative = getRelevantDerivative(data, scaleDerivFactor, threshold);
        float[][] appendedDerivative = new float[data.length][numColumns + derivative[0].length];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, appendedDerivative[i], 0, numColumns);
        }

        for (int i = 0; i < data.length; i++) {
            System.arraycopy(derivative[i], 0, appendedDerivative[i], numColumns, derivative[i].length);
        }

        return appendedDerivative;
    }

    public static float[][] getRelevantDerivativeScaledPositive(float[][] data, float scaleDerivFactor, float threshold) {

        float[][] derivative = new float[data.length][data[0].length - 1];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length - 1; j++) {
                derivative[i][j] = data[i][j] - data[i][j + 1];
            }
        }

        float[] columnSums = getAbsValColSums(derivative);
        List<Integer> indicesToUse = new ArrayList<>();
        for (int k = 0; k < columnSums.length; k++) {
            if (columnSums[k] > 0) {
                indicesToUse.add(k);
            }
        }

        float[][] importantDerivative = new float[data.length][indicesToUse.size()];

        for (int i = 0; i < data.length; i++) {
            for (int k = 0; k < indicesToUse.size(); k++) {
                int indexToUse = indicesToUse.get(k);
                importantDerivative[i][k] = Math.min(threshold, Math.max(-threshold, derivative[i][indexToUse] * scaleDerivFactor)) + threshold;
            }
        }

        return importantDerivative;
    }

    public static float[][] getRelevantDerivative(float[][] data, float scaleDerivFactor, float threshold) {

        float[][] derivative = new float[data.length][data[0].length - 1];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length - 1; j++) {
                derivative[i][j] = data[i][j] - data[i][j + 1];
            }
        }

        float[] columnSums = getAbsValColSums(derivative);
        List<Integer> indicesToUse = new ArrayList<>();
        for (int k = 0; k < columnSums.length; k++) {
            if (columnSums[k] > 0) {
                indicesToUse.add(k);
            }
        }

        float[][] importantDerivative = new float[data.length][indicesToUse.size()];

        for (int i = 0; i < data.length; i++) {
            for (int k = 0; k < indicesToUse.size(); k++) {
                int indexToUse = indicesToUse.get(k);
                importantDerivative[i][k] = Math.min(threshold, Math.max(-threshold, derivative[i][indexToUse] * scaleDerivFactor));
            }
        }

        return importantDerivative;
    }

    public static float[][] getRelevantDiscreteIntDerivativeScaledPositive(float[][] data, float scaleDerivFactor, float threshold) {

        int[][] derivative = new int[data.length][data[0].length - 1];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length - 1; j++) {
                float tempVal = (data[i][j] - data[i][j + 1]);
                tempVal = Math.min(threshold, Math.max(-threshold, tempVal * scaleDerivFactor));
                derivative[i][j] = Math.round(tempVal);
            }
        }

        int[] columnSums = getAbsValColSums(derivative);
        List<Integer> indicesToUse = new ArrayList<>();
        for (int k = 0; k < columnSums.length; k++) {
            if (columnSums[k] > 0) {
                indicesToUse.add(k);
            }
        }

        float[][] importantDerivative = new float[data.length][indicesToUse.size()];

        for (int i = 0; i < data.length; i++) {
            for (int k = 0; k < indicesToUse.size(); k++) {
                int indexToUse = indicesToUse.get(k);
                importantDerivative[i][k] = derivative[i][indexToUse] + threshold;
            }
        }

        return importantDerivative;
    }
}
