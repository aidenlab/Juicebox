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

package juicebox.tools.utils.Common;

import juicebox.tools.utils.Juicer.APA.APARegionStatistics;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

import java.io.*;
import java.util.Random;

/**
 * Created by muhammadsaadshamim on 5/11/15.
 */
public class MatrixTools {

    private static RealMatrix presetValueMatrix(int rows, int cols, int val) {
        RealMatrix matrix = new Array2DRowRealMatrix(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                matrix.setEntry(r, c, val);
        return matrix;
    }

    public static RealMatrix cleanArray2DMatrix(int n) {
        return cleanArray2DMatrix(n,n);
    }

    public static RealMatrix cleanArray2DMatrix(int rows, int cols) {
        return presetValueMatrix(rows, cols, 0);
    }

    public static RealMatrix ones(int n) {
        return ones(n, n);
    }

    private static RealMatrix ones(int rows, int cols) {
        return presetValueMatrix(rows, cols, 1);
    }

    public static double minimumPositive(RealMatrix data) {
        return minimumPositive(data.getData());
    }

    public static double minimumPositive(double[][] data) {
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
     * Flatten a 2D double matrix into a double array
     * @param matrix
     * @return 1D double array in row major order
     */
    public static double[] flattenedRowMajorOrderMatrix(RealMatrix matrix) {
        int n = matrix.getColumnDimension();
        int m = matrix.getRowDimension();
        int numElements = n * m;
        double[] flattenedMatrix = new double[numElements];

        int index = 0;
        for(int i = 0; i < m; i++){
        //for (double[] row : matrix.getData()) {
            System.arraycopy(matrix.getRow(i), 0, flattenedMatrix, index, n);
            index += n;
        }
        return flattenedMatrix;
    }

    public static double mean(RealMatrix x) {
        return APARegionStatistics.statistics(x.getData()).getMean();
    }

    public static void saveMatrixText(String filename, RealMatrix realMatrix) {
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "utf-8"));
            double[][] matrix = realMatrix.getData();
            for (double[] row : matrix) {
                for (double val : row) {
                    writer.write(val + " ");
                }
                writer.write("\n");
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

    public static double[] sliceFromVector(double[]  vector, int bound1, int bound2) {

        int n = bound2 - bound1;
        double[] slicedVector = new double[n];

        for(int i = 0; i < n; i++){
            slicedVector[i] = vector[bound1+i];
        }

        return slicedVector;
    }

    public static float[][] reshapeFlatMatrix(float[] flatMatrix, int n) {
        float[][] squareMatrix = new float[n][n];

        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                squareMatrix[i][j] = flatMatrix[i*n+j];
            }
        }
        return squareMatrix;
    }

    /**
     * From Matrix M, extract out M[r1:r2,c1:c2]
     * r2, c2 not inclusive (~python numpy)
     *
     * @param matrix
     * @param r1
     * @param r2
     * @param c1
     * @param c2
     * @return extracted matrix region M[r1:r2,c1:c2]
     */
    public static float[][] extractLocalMatrixRegion(float[][] matrix, int r1, int r2, int c1, int c2) {

        int numRows = r2 - r1;
        int numColumns = c2 - c1;
        float[][] extractedRegion = new float[numRows][numColumns];

        for(int i = 0; i < numRows; i++){
            for(int j = 0; j < numColumns; j++){
                extractedRegion[i][j] = matrix[r1+i][c1+j];
            }
        }

        return extractedRegion;
    }

    /**
     * From Matrix M, extract out M[r1:r2,c1:c2]
     * r2, c2 not inclusive (~python numpy, not like Matlab)
     *
     * @param matrix
     * @param r1
     * @param r2
     * @param c1
     * @param c2
     * @return extracted matrix region M[r1:r2,c1:c2]
     */
    public static RealMatrix extractLocalMatrixRegion(RealMatrix matrix, int r1, int r2, int c1, int c2) {

        int numRows = r2 - r1;
        int numColumns = c2 - c1;
        RealMatrix extractedRegion = cleanArray2DMatrix(numRows, numColumns);

        for(int i = 0; i < numRows; i++){
            for(int j = 0; j < numColumns; j++){
                extractedRegion.setEntry(i,j, matrix.getEntry(r1+i,c1+j));
            }
        }
        return extractedRegion;
    }

    /**
     * Returns the values along the diagonal of the matrix
     * @param matrix
     * @return diagonal
     */
    public static RealMatrix extractDiagonal(RealMatrix matrix) {
        int n = Math.min(matrix.getColumnDimension(), matrix.getRowDimension());
        RealMatrix diagonal = MatrixTools.cleanArray2DMatrix(n, n);
        for(int i = 0; i < n; i ++){
            diagonal.setEntry(i,i,matrix.getEntry(i,i));
        }
        return diagonal;
    }

    /**
     * Returns the values along the diagonal of the matrix
     * @param matrix
     * @return diagonal
     */
    public static RealMatrix makeSymmetricMatrix(RealMatrix matrix) {
        RealMatrix symmetricMatrix = extractDiagonal(matrix);
        int n = symmetricMatrix.getRowDimension();
        for(int i = 0; i < n; i++){
            for(int j = i+1; j < n; j++){
                double val = matrix.getEntry(i, j);
                symmetricMatrix.setEntry(i,j,val);
                symmetricMatrix.setEntry(j,i,val);
            }
        }

        return symmetricMatrix;
    }

    /**
     * Returns the matrix flipped across the antidiagonal
     * @param matrix
     * @return antiDiagFlippedMatrix
     */
    public static RealMatrix flipAcrossAntiDiagonal(RealMatrix matrix) {
        int n = Math.min(matrix.getColumnDimension(), matrix.getRowDimension());
        RealMatrix antiDiagFlippedMatrix = cleanArray2DMatrix(n, n);
        int maxIndex = n-1;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                antiDiagFlippedMatrix.setEntry(maxIndex-j,maxIndex-i, matrix.getEntry(i,j));
            }
        }
        return antiDiagFlippedMatrix;
    }

    /**
     * Returns the matrix flipped Left-Right
     * @param matrix
     * @return leftRightFlippedMatrix
     */
    public static RealMatrix flipLeftRight(RealMatrix matrix) {
        int r = matrix.getRowDimension(), c = matrix.getColumnDimension();
        RealMatrix leftRightFlippedMatrix = cleanArray2DMatrix(r,c);
        for(int i = 0; i < r; i++){
            for(int j = 0; j < c; j++){
                leftRightFlippedMatrix.setEntry(i, c-1-j, matrix.getEntry(i,j));
            }
        }
        return leftRightFlippedMatrix;
    }

    /**
     * Returns the matrix flipped Top-Bottom
     * @param matrix
     * @return topBottomFlippedMatrix
     */
    public static RealMatrix flipTopBottom(RealMatrix matrix) {
        int r = matrix.getRowDimension(), c = matrix.getColumnDimension();
        RealMatrix topBottomFlippedMatrix = cleanArray2DMatrix(r,c);
        for(int i = 0; i < r; i++){
            for(int j = 0; j < c; j++){
                topBottomFlippedMatrix.setEntry(r-1-i, j, matrix.getEntry(i,j));
            }
        }
        return topBottomFlippedMatrix;
    }

    /**
     * Element-wise multiplication of matrices i.e. M.*N in Matlab
     * @param matrix1
     * @param matrix2
     * @return elementwiseMultipliedMatrix
     */
    public static RealMatrix elementBasedMultiplication(RealMatrix matrix1, RealMatrix matrix2) {
        int r = Math.min(matrix1.getRowDimension(), matrix2.getRowDimension());
        int c = Math.min(matrix1.getColumnDimension(), matrix2.getColumnDimension());
        RealMatrix elementwiseMultipliedMatrix = cleanArray2DMatrix(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                elementwiseMultipliedMatrix.setEntry(i, j,
                        matrix1.getEntry(i, j) * matrix2.getEntry(i, j));
            }
        }
        return elementwiseMultipliedMatrix;
    }


    /**
     * Element-wise division of matrices i.e. M./N in Matlab
     * @param matrix1
     * @param matrix2
     * @return elementwiseMultipliedMatrix
     */
    public static RealMatrix elementBasedDivision(RealMatrix matrix1, RealMatrix matrix2) {
        int r = Math.min(matrix1.getRowDimension(), matrix2.getRowDimension());
        int c = Math.min(matrix1.getColumnDimension(), matrix2.getColumnDimension());
        RealMatrix elementwiseDividedMatrix = cleanArray2DMatrix(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                elementwiseDividedMatrix.setEntry(i, j,
                        matrix1.getEntry(i, j) / matrix2.getEntry(i, j));
            }
        }
        return elementwiseDividedMatrix;
    }


    /**
     * Set NaNs in matrix to given value
     * @param matrix
     * @param val
     */
    public static void setNaNs(RealMatrix matrix, int val) {
        for(int i = 0; i < matrix.getRowDimension(); i++){
            for(int j = 0; j < matrix.getColumnDimension(); j++){
                if(Double.isNaN(matrix.getEntry(i,j))){
                    matrix.setEntry(i,j,val);
                }
            }
        }
    }

    /**
     * Return sign of values in matrix:
     * val > 0 : 1
     * val = 0 : 0
     * val < 0 : -1
     * @param matrix
     * @return signMatrix
     */
    public static RealMatrix sign(RealMatrix matrix) {
        int r = matrix.getRowDimension();
        int c = matrix.getColumnDimension();
        RealMatrix signMatrix = cleanArray2DMatrix(r,c);
        for(int i = 0; i < r; i++){
            for(int j = 0; j < c; j++){
                double val = matrix.getEntry(i,j);
                if(val > 0){
                    matrix.setEntry(i,j,1);
                }
                else if(val < 0){
                    matrix.setEntry(i,j,-1);
                }
            }
        }
        return signMatrix;
    }

    /**
     * Replace all of a given value in a matrix with a new value
     * @param matrix
     * @param initialVal
     * @param newVal
     */
    public static void replaceValue(RealMatrix matrix, int initialVal, int newVal) {
        for(int i = 0; i < matrix.getRowDimension(); i++){
            for(int j = 0; j < matrix.getColumnDimension(); j++){
                if(matrix.getEntry(i,j) == initialVal){
                    matrix.setEntry(i,j,newVal);
                }
            }
        }
    }

    /**
     * Normalize matrix by dividing by max element
     * @param matrix
     * @return matrix * (1/max_element)
     */
    public static RealMatrix normalizeByMax(RealMatrix matrix) {
        double max = calculateMax(matrix);
        return matrix.scalarMultiply(1/max);
    }

    /**
     * Calculate max element in matrix
     * @param matrix
     * @return max
     */
    public static double calculateMax(RealMatrix matrix) {
        double max = matrix.getEntry(0, 0);
        for(int i = 0; i < matrix.getRowDimension(); i++){
            for(int j = 0; j < matrix.getColumnDimension(); j++){
                double val = matrix.getEntry(i,j);
                if(max < val){
                    max = val;
                }
            }
        }
        return max;
    }

    public static RealMatrix randomUnitMatrix(int n) {
        return randomUnitMatrix(n,n);
    }

    private static RealMatrix randomUnitMatrix(int rows, int cols) {
        Random generator = new Random();
        RealMatrix matrix = cleanArray2DMatrix(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                if(generator.nextBoolean())
                    matrix.setEntry(r, c, 1);
        return matrix;
    }

    public static void print(RealMatrix matrix) {
        double[][] data = matrix.getData();
        for(double[] row : data){
            for(double entry : row){
                System.out.print(entry + " ");
            }
            System.out.println("");
        }
    }

    public static RealMatrix getSubMatrix(RealMatrix matrix, int[] indices) {
        return matrix.getSubMatrix(indices[0], indices[1], indices[2], indices[3]);
    }
}
