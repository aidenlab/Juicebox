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

package juicebox.matrix;


import org.broad.igv.util.collections.FloatArrayList;
import org.broad.igv.util.collections.IntArrayList;

import java.io.PrintWriter;
import java.util.Arrays;

public class SparseSymmetricMatrix implements BasicMatrix {

    private final int totSize;
    private IntArrayList rows1 = null;
    private IntArrayList cols1 = null;
    private FloatArrayList values1 = null;
    private IntArrayList rows2 = null;
    private IntArrayList cols2 = null;
    private FloatArrayList values2 = null;


    public SparseSymmetricMatrix(int totSize) {
        rows1 = new IntArrayList();
        cols1 = new IntArrayList();
        values1 = new FloatArrayList();
        this.totSize = totSize;
    }

    public void set(int row, int col, float v) {

        if (!Float.isNaN(v)) {
            if (rows2 == null) {
                try {
                    rows1.add(row);
                    cols1.add(col);
                    values1.add(v);
                } catch (NegativeArraySizeException error) {
                    rows2 = new IntArrayList();
                    cols2 = new IntArrayList();
                    values2 = new FloatArrayList();
                    rows2.add(row);
                    cols2.add(col);
                    values2.add(v);
                }
            } else {
                rows2.add(row);
                cols2.add(col);
                values2.add(v);
            }
        }
    }

    public float[] getRow(int rowNum) {

        float[] result = new float[totSize];

        int size = rows1.size();
        for (int i = 0; i < size; i++) {
            if (rows1.get(i) == rowNum) result[cols1.get(i)] = values1.get(i);
        }
        if (rows2 != null) {
            size = rows2.size();
            for (int i = 0; i < size; i++) {
                if (rows2.get(i) == rowNum) result[cols2.get(i)] = values2.get(i);
            }
        }
        return result;

    }


    public double[] multiply(double[] vector) {

        double[] result = new double[vector.length];
        Arrays.fill(result, 0);

        int[] rowArray1 = rows1.toArray();
        int[] colArray1 = cols1.toArray();
        float[] valueArray1 = values1.toArray();

        int n = rowArray1.length;
        for (int i = 0; i < n; i++) {
            int row = rowArray1[i];
            int col = colArray1[i];
            float value = valueArray1[i];
            result[row] += vector[col] * value;

            if (row != col) {
                result[col] += vector[row] * value;
            }
        }
        if (rows2 != null) {
            int[] rowArray2 = rows2.toArray();
            int[] colArray2 = cols2.toArray();
            float[] valueArray2 = values2.toArray();
            int n2 = rowArray2.length;
            for (int j = 0; j < n2; j++) {
                int row = rowArray2[j];
                int col = colArray2[j];
                float value = valueArray2[j];
                result[row] += vector[col] * value;

                if (row != col) {
                    result[col] += vector[row] * value;
                }
            }
        }

        return result;
    }

    public void print() {
        print(new PrintWriter(System.out));
    }

    public void print(PrintWriter pw) {
        for (int i = 0; i < totSize; i++) {
            float[] row = getRow(i);
            for (int j = 0; j < totSize; j++) {
                pw.print(row[j] + " ");
            }
            pw.println();
        }
        pw.close();
    }

    @Override
    public float getEntry(int row, int col) {
        return 0;
    }

    @Override
    public int getRowDimension() {
        return 0;
    }

    @Override
    public int getColumnDimension() {
        return 0;
    }

    @Override
    public float getLowerValue() {
        return 0;
    }

    @Override
    public float getUpperValue() {
        return 0;
    }

    @Override
    public void setEntry(int i, int j, float corr) {

    }
}