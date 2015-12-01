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

package juicebox.tools.utils.common;

import juicebox.tools.utils.juicer.apa.APARegionStatistics;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

import java.io.*;
import java.util.Arrays;
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
     * @param rows
     * @param cols
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
        int n = matrix.getColumnDimension();
        int m = matrix.getRowDimension();
        int numElements = n * m;
        double[] flattenedMatrix = new double[numElements];

        int index = 0;
        for (int i = 0; i < m; i++) {
            System.arraycopy(matrix.getRow(i), 0, flattenedMatrix, index, n);
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
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "utf-8"));
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
        int[] rowSum = new int[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                rowSum[i] += matrix[i][j];
            }
        }

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                newMatrix[i][j] = matrix[i][j] / rowSum[i];
            }
        }

        return newMatrix;
    }
}
