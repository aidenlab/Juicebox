package juicebox.matrix;


import org.broad.igv.util.collections.FloatArrayList;
import org.broad.igv.util.collections.IntArrayList;

import java.io.PrintWriter;
import java.util.Arrays;

public class SparseSymmetricMatrix implements BasicMatrix {

    IntArrayList rows1 = null;
    IntArrayList cols1 = null;
    FloatArrayList values1 = null;
    IntArrayList rows2 = null;
    IntArrayList cols2 = null;
    FloatArrayList values2 = null;
    final int totSize;


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
                }
                catch (NegativeArraySizeException error) {
                    rows2 = new IntArrayList();
                    cols2 = new IntArrayList();
                    values2 = new FloatArrayList();
                    rows2.add(row);
                    cols2.add(col);
                    values2.add(v);
                }
            }
            else {
                rows2.add(row);
                cols2.add(col);
                values2.add(v);
            }
        }
    }

    public float[] getRow(int rowNum) {

        float[] result = new float[totSize];

        int size = rows1.size();
        for (int i=0; i<size; i++) {
            if (rows1.get(i) == rowNum) result[cols1.get(i)] = values1.get(i);
        }
        if (rows2 != null) {
            size = rows2.size();
            for (int i=0; i<size; i++) {
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
        for (int i=0; i<totSize; i++) {
            float[] row = getRow(i);
            for (int j=0; j<totSize; j++) {
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