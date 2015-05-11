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

import juicebox.tools.utils.Juicer.APARegionStatistics;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

import java.io.*;

/**
 * Created by muhammadsaadshamim on 5/11/15.
 */
public class MatrixTools {

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


    public static RealMatrix cleanArray2DMatrix(int rows, int cols) {
        RealMatrix matrix = new Array2DRowRealMatrix(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                matrix.setEntry(r, c, 0);
        return matrix;
    }

    /**
     * Flatten a 2D double matrix into a double array
     * @param matrix
     * @return 1D double array in row major order
     */
    public static double[] flattenSquareMatrix(RealMatrix matrix) {
        int n = matrix.getColumnDimension();
        int numElements = n * n;
        double[] flattenedMatrix = new double[numElements];

        int index = 0;
        for (double[] row : matrix.getData()) {
            System.arraycopy(row, 0, flattenedMatrix, index, n);
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
}
