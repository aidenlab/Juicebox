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

package juicebox.tools.utils.Juicer;

import juicebox.data.Block;
import juicebox.data.ContactRecord;
import juicebox.data.MatrixZoomData;
import juicebox.tools.utils.Common.StatPercentile;
import juicebox.track.Feature2D;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.Array2DRowRealMatrix;

import java.io.*;
import java.util.*;

/**
 * Created by Muhammad Shamim on 1/21/15.
 */
public class APAUtils {

    private final static double epsilon = 1e-6;

    /**
     * creates a range of integers
     *
     * @param start
     * @param stop
     * @return
     */
    private static int[] range(int start, int stop) {
        int[] result = new int[stop - start];
        for (int i = 0; i < stop - start; i++)
            result[i] = start + i;
        return result;
    }

    /**
     * @param filename
     * @param matrix
     */
    public static void saveMeasures(String filename, Array2DRowRealMatrix matrix) {

        Writer writer = null;

        APARegionStatistics apaStats = new APARegionStatistics(matrix);

        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "utf-8"));
            writer.write("P2M" + '\t' + apaStats.getPeak2mean() + '\n');
            writer.write("P2UL" + '\t' + apaStats.getPeak2UL() + '\n');
            writer.write("P2UR" + '\t' + apaStats.getPeak2UR() + '\n');
            writer.write("P2LL" + '\t' + apaStats.getPeak2LL() + '\n');
            writer.write("P2LR" + '\t' + apaStats.getPeak2LR() + '\n');
            writer.write("ZscoreLL" + '\t' + apaStats.getZscoreLL());
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


    public static void saveMatrixText(String filename, Array2DRowRealMatrix realMatrix) {
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

    public static void saveListText(String filename, List<Double> array) {
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "utf-8"));
            for (double val : array) {
                writer.write(val + " ");
            }
            writer.write("\n");
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

    public static Array2DRowRealMatrix standardNormalization(Array2DRowRealMatrix matrix) {
        Array2DRowRealMatrix normeddata = cleanArray2DMatrix(matrix.getRowDimension(),
                matrix.getColumnDimension()).add(matrix);
        normeddata.scalarMultiply(1. / Math.max(1., APAUtils.mean(matrix)));
        return normeddata;
    }

    private static double mean(Array2DRowRealMatrix x) {
        return APARegionStatistics.statistics(x.getData()).getMean();
    }

    public static Array2DRowRealMatrix centerNormalization(Array2DRowRealMatrix matrix) {

        int center = matrix.getRowDimension() / 2;
        double centerVal = matrix.getEntry(center, center);

        if (centerVal == 0) {
            centerVal = minimumPositive(matrix.getData());
            if (centerVal == 0)
                centerVal = 1;
        }

        Array2DRowRealMatrix normeddata = cleanArray2DMatrix(matrix.getRowDimension(),
                matrix.getColumnDimension()).add(matrix);
        normeddata.scalarMultiply(1. / centerVal);
        return normeddata;
    }

    public static double peakEnhancement(Array2DRowRealMatrix matrix) {
        int rows = matrix.getRowDimension();
        int center = rows / 2;
        double centerVal = matrix.getEntry(center, center);
        double remainingSum = APARegionStatistics.sum(matrix.getData()) - centerVal;
        double remainingAverage = remainingSum / (rows * rows - 1);
        return centerVal / remainingAverage;
    }

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


    public static Array2DRowRealMatrix cleanArray2DMatrix(int rows, int cols) {
        Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                matrix.setEntry(r, c, 0);
        return matrix;
    }




    /**
     *
     * @param data
     * @return
     */
    public static Array2DRowRealMatrix rankPercentile(Array2DRowRealMatrix data) {
        int n = data.getColumnDimension();
        StatPercentile percentile = new StatPercentile(flattenSquareMatrix(data));

        Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(n, n);
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                double currValue = data.getEntry(r, c);
                if (currValue == 0) {
                    matrix.setEntry(r, c, 0);
                } else {
                    matrix.setEntry(r, c, percentile.evaluate(currValue));
                }
                //matrix.setEntry(r, c, percentile.evaluate());
            }
        }
        return matrix;
    }

    /**
     * Flatten a 2D double matrix into a double array
     * @param matrix
     * @return 1D double array in row major order
     */
    private static double[] flattenSquareMatrix(Array2DRowRealMatrix matrix) {
        int n = matrix.getColumnDimension();
        int numElements = n * n;
        double[] flatMatrix = new double[numElements];

        int index = 0;
        for (double[] row : matrix.getData()) {
            System.arraycopy(row, 0, flatMatrix, index, n);
            index += n;
        }
        return flatMatrix;
    }

    public static Array2DRowRealMatrix extractLocalizedData(MatrixZoomData zd, Feature2D loop,
                                                            int L, int resolution, int window) {
        //long time = System.nanoTime();
        int loopX = loop.getStart1() / resolution;
        int loopY = loop.getStart2() / resolution;
        int binXStart = loopX - (window + 1);
        int binXEnd = loopX + (window + 1);
        int binYStart = loopY - (window + 1);
        int binYEnd = loopY + (window + 1);

        //System.out.println((System.nanoTime()-time)/1000000000.);

        List<Block> blocks = zd.getNormalizedBlocksOverlapping(binXStart, binYStart, binXEnd, binYEnd,
                NormalizationType.NONE);

        Array2DRowRealMatrix data = APAUtils.cleanArray2DMatrix(L, L);

        for (Block b : blocks) {
            for (ContactRecord rec : b.getContactRecords()) {
                //, rec.getBinY(), rec.getCounts()

                int relativeX = window + (rec.getBinX() - loopX);
                int relativeY = window + (rec.getBinY() - loopY);

                if (relativeX >= 0 && relativeX < L) {
                    if (relativeY >= 0 && relativeY < L) {
                        data.addToEntry(relativeX, relativeY, rec.getCounts());
                    }
                }
            }
        }
        //System.out.println((System.nanoTime()-time)/1000000000.);
        return data;
    }
}