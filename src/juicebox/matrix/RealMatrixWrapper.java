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

import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.stat.StatUtils;
import org.broad.igv.util.collections.DoubleArrayList;

/**
 * Wrpas a apache commons RealMatrix.  We don't expose the apache class so we can use other implementations.
 *
 * @author jrobinso
 *         Date: 7/13/12
 *         Time: 1:02 PM
 */
public class RealMatrixWrapper implements BasicMatrix {

    private final RealMatrix matrix;
    private float lowerValue = -1;
    private float upperValue = 1;

    public RealMatrixWrapper(RealMatrix matrix) {
        this.matrix = matrix;
        computePercentiles();
    }

    @Override
    public float getEntry(int row, int col) {
        return (float) matrix.getEntry(row, col);
    }

    @Override
    public int getRowDimension() {
        return matrix.getRowDimension();
    }

    @Override
    public int getColumnDimension() {
        return matrix.getColumnDimension();
    }

    @Override
    public float getLowerValue() {
        return lowerValue;
    }

    @Override
    public float getUpperValue() {
        return upperValue;
    }

    @Override
    public void setEntry(int i, int j, float corr) {

    }

    private void computePercentiles() {

        // Statistics, other attributes
        DoubleArrayList flattenedDataList = new DoubleArrayList(matrix.getColumnDimension() * matrix.getRowDimension());
        double min = 1;
        double max = -1;
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                double value = matrix.getEntry(i, j);
                if (!Double.isNaN(value) && value != 1) {
                    min = value < min ? value : min;
                    max = value > max ? value : max;
                    flattenedDataList.add(value);
                }
            }
        }

        // Stats
        double[] flattenedData = flattenedDataList.toArray();
        lowerValue = (float) StatUtils.percentile(flattenedData, 5);
        upperValue = (float) StatUtils.percentile(flattenedData, 95);
        System.out.println(lowerValue + "  " + upperValue);

    }


    public RealMatrix getMatrix() {
        return matrix;
    }
}
