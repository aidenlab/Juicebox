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


package juicebox.data;

import juicebox.matrix.BasicMatrix;
import juicebox.matrix.InMemoryMatrix;
import juicebox.matrix.SparseVector;
import juicebox.matrix.SymmetricMatrix;

import java.util.BitSet;
import java.util.Set;

/**
 * @author jrobinso
 *         Date: 2/28/14
 *         Time: 10:24 AM
 */
class Pearsons {


    public static SymmetricMatrix computePearsons(SymmetricMatrix matrix) {
        int nCols = matrix.getColumnDimension();
        int nRows = matrix.getRowDimension();
        SymmetricMatrix pearsons = new SymmetricMatrix(nCols);
        pearsons.fill(Float.NaN);

        Set<Integer> nullColumns = matrix.getNullColumns();
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {

                if (nullColumns.contains(i) || nullColumns.contains(j)) {
                    pearsons.setEntry(i, j, Float.NaN);
                } else {
                    double corr = Pearsons.computePearsons1(matrix, i, j);

                    pearsons.setEntry(i, j, (float) corr);
                }
            }
        }
        for (int i = 0; i < nCols; i++) {
            pearsons.setEntry(i, i, 1.0f);
        }
        return pearsons;
    }

    private static double computePearsons1(BasicMatrix matrix, int col1, int col2) {
        // sum of col1 * col2 (dot product)  - sum(X)sum(Y)/n      divided by
        // square root of   ((sum x^2)-(sumx)^2/n)  ((sum y^2)-(sumy)^2/n)
        double length = matrix.getRowDimension();
        double sum_xsq = 0;
        double sum_ysq = 0;
        double sumx = 0;
        double sumy = 0;
        for (int i = 0; i < length; i++) sum_xsq += matrix.getEntry(i, col1) * matrix.getEntry(i, col1);
        for (int i = 0; i < length; i++) sum_ysq += matrix.getEntry(i, col2) * matrix.getEntry(i, col2);
        for (int i = 0; i < length; i++) sumx += matrix.getEntry(i, col1);
        for (int i = 0; i < length; i++) sumy += matrix.getEntry(i, col2);
        double denominator = (sum_xsq - (sumx * sumx / length)) * (sum_ysq - (sumy * sumy / length));
        double numerator = 0;
        for (int i = 0; i < length; i++)
            numerator += matrix.getEntry(i, col1) * matrix.getEntry(i, col2); // dot product
        return (numerator - (sumx * sumy / length)) / Math.sqrt(denominator);


    }

    public static double computePearsons(BasicMatrix matrix, int col1, int col2) {

        double length = matrix.getRowDimension();
        double result;
        double sum_sq_x = 0;
        double sum_sq_y = 0;
        double sum_coproduct = 0;
        double mean_x = matrix.getEntry(0, col1);
        double mean_y = matrix.getEntry(0, col2);

        int count = 1;
        for (int i = 1; i < length; i++) {

            double sweep = ((double) count) / (count + 1);
            final float v1 = matrix.getEntry(i, col1);
            final float v2 = matrix.getEntry(i, col2);
            if (!Float.isNaN(v1) && !Float.isNaN(v2)) {
                double delta_x = v1 - mean_x;
                double delta_y = v2 - mean_y;

                sum_sq_x += delta_x * delta_x * sweep;
                sum_sq_y += delta_y * delta_y * sweep;
                sum_coproduct += delta_x * delta_y * sweep;

                mean_x += delta_x / (count + 1);
                mean_y += delta_y / (count + 1);

                count++;
            }
        }

        if (count == 1 || mean_x == 0 || mean_y == 0) {
            return Float.NaN;
        }

        double pop_sd_x = Math.sqrt(sum_sq_x / count);
        double pop_sd_y = Math.sqrt(sum_sq_y / count);
        double cov_x_y = sum_coproduct / length;
        result = cov_x_y / (pop_sd_x * pop_sd_y);
        return result;
    }


    private static double computePearsons(double[] scores1, double[] scores2) {

//        double length = scores1.length;
//        double sum_xsq = 0;
//        double sum_ysq = 0;
//        double sumx = 0;
//        double sumy = 0;
//        for (int i=0; i<length; i++) sum_xsq += scores1[i]*scores1[i];
//        for (int i=0; i<length; i++) sum_ysq += scores2[i]*scores2[i];
//        for (int i=0; i<length; i++) sumx += scores1[i];
//        for (int i=0; i<length; i++) sumy += scores2[i];
//        double denominator = (sum_xsq - (sumx*sumx/length))*(sum_ysq - (sumy*sumy/length));
//        double numerator = 0;
//        for (int i=0; i<length; i++) numerator += scores1[i]*scores2[i]; // dot product
//        return (numerator - (sumx*sumy/length))/Math.sqrt(denominator);

        double result;
        double sum_sq_x = 0;
        double sum_sq_y = 0;
        double sum_coproduct = 0;
        double mean_x = scores1[0];
        double mean_y = scores2[0];
        for (int i = 1; i < scores1.length; i++) {

            double sweep = ((double) i) / (i + 1);
            double delta_x = scores1[i] - mean_x;
            double delta_y = scores2[i] - mean_y;

            sum_sq_x += delta_x * delta_x * sweep;
            sum_sq_y += delta_y * delta_y * sweep;
            sum_coproduct += delta_x * delta_y * sweep;

            mean_x += delta_x / (i + 1);
            mean_y += delta_y / (i + 1);
        }
        double pop_sd_x = Math.sqrt(sum_sq_x / scores1.length);
        double pop_sd_y = Math.sqrt(sum_sq_y / scores1.length);
        double cov_x_y = sum_coproduct / scores1.length;
        result = cov_x_y / (pop_sd_x * pop_sd_y);
        return result;
    }


    public static BasicMatrix computePearsons(SparseVector[] columns, int dim) {

        BasicMatrix pearsons = new InMemoryMatrix(dim);
        //pearsons.fill(Float.NaN);

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if (i == j) continue;

                SparseVector v1 = columns[i];
                SparseVector v2 = columns[j];
                if (v1 == null || v2 == null) {
                    pearsons.setEntry(i, j, Float.NaN);
                } else {
                    double corr = Pearsons.computePearsons(columns[i], columns[j]);
                    pearsons.setEntry(i, j, (float) corr);
                }
            }
        }
        for (int i = 0; i < dim; i++) {
            pearsons.setEntry(i, i, 1.0f);
        }
        return pearsons;
    }


    public static BasicMatrix computePearsons(double[][] columns, int dim) {

        BasicMatrix pearsons = new InMemoryMatrix(dim);
        //pearsons.fill(Float.NaN);

        BitSet bitSet = new BitSet(dim);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if (i == j) continue;

                double[] v1 = columns[i];
                double[] v2 = columns[j];
                if (v1 == null || v2 == null) {
                    pearsons.setEntry(i, j, Float.NaN);
                } else {
                    double corr = Pearsons.computePearsons(columns[i], columns[j]);
                    pearsons.setEntry(i, j, (float) corr);
                    bitSet.set(i);
                }
            }
        }
        // Set diagonal to 1, set centromere to NaN
        for (int i = 0; i < dim; i++) {
            if (bitSet.get(i)) pearsons.setEntry(i, i, 1.0f);
            else pearsons.setEntry(i, i, Float.NaN);
        }
        return pearsons;
    }

    private static double computePearsons(SparseVector scores1, SparseVector scores2) {

        int size = scores1.getLength();
        if (size != scores2.getLength()) {
            throw new IllegalArgumentException("Vectors must be same size");
        }

        double result;
        double sum_sq_x = 0;
        double sum_sq_y = 0;
        double sum_coproduct = 0;
        double mean_x = scores1.get(0);
        double mean_y = scores2.get(0);
        for (int i = 1; i < size; i++) {

            double sweep = ((double) i) / (i + 1);
            double delta_x = scores1.get(i) - mean_x;
            double delta_y = scores2.get(i) - mean_y;

            sum_sq_x += delta_x * delta_x * sweep;
            sum_sq_y += delta_y * delta_y * sweep;
            sum_coproduct += delta_x * delta_y * sweep;

            mean_x += delta_x / (i + 1);
            mean_y += delta_y / (i + 1);
        }
        double pop_sd_x = Math.sqrt(sum_sq_x / size);
        double pop_sd_y = Math.sqrt(sum_sq_y / size);
        double cov_x_y = sum_coproduct / size;
        result = cov_x_y / (pop_sd_x * pop_sd_y);
        return result;
    }


    public void testStdDev() {

        int n = 5;
        double[] data = new double[n];
        for (int i = 0; i < n; i++) data[i] = Math.random();


        double sum = 0;
        double sumSq = 0;
        for (int i = 0; i < n; i++) {
            sum += data[i];
            sumSq += data[i] * data[i];
        }
        double mean = sum / n;
        double varEst = sumSq / n - mean * mean;

        System.out.println("V2 = " + varEst);

        double sum2 = 0;
        for (int i = 0; i < n; i++) {
            sum2 += (data[i] - mean) * (data[i] - mean);
        }
        double varExact = sum2 / (n - 1);

        System.out.println("V3 = " + varExact);
    }

}
