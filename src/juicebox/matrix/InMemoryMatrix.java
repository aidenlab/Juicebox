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

import org.apache.commons.math.stat.StatUtils;
import org.broad.igv.util.collections.DoubleArrayList;

import java.util.Arrays;

/**
 * Assumptions -- matrix is square
 *
 * @author jrobinso
 *         Date: 7/13/12
 *         Time: 1:48 PM
 */
public class InMemoryMatrix implements BasicMatrix {


    private final int dim;
    private final float[] data;
    private float lowerValue = Float.NaN;
    private float upperValue = Float.NaN;


    public InMemoryMatrix(int dim) {
        this.dim = dim;
        this.data = new float[dim * dim];
    }

    public InMemoryMatrix(int dim, float[] data) {
        this.data = data;
        this.dim = dim;
    }

    public void fill(float value) {
        Arrays.fill(data, value);

        // Invalidate bounds
        lowerValue = Float.NaN;
        upperValue = Float.NaN;
    }


    @Override
    public float getEntry(int row, int col) {

        int idx = row * dim + col;
        return idx < data.length ? data[idx] : Float.NaN;
    }

    public void setEntry(int row, int col, float value) {

        int idx = row * dim + col;
        data[idx] = value;

        // Invalidate bounds
        lowerValue = Float.NaN;
        upperValue = Float.NaN;

    }

    @Override
    public int getRowDimension() {
        return dim;
    }

    @Override
    public int getColumnDimension() {
        return dim;
    }

    @Override
    public float getLowerValue() {
        if (Float.isNaN(lowerValue)) {
            computeBounds();

        }
        return lowerValue;
    }

    @Override
    public float getUpperValue() {
        if (Float.isNaN(upperValue)) {
            computeBounds();
        }
        return upperValue;
    }

    private void computeBounds() {
        DoubleArrayList tmpList = new DoubleArrayList(data.length);
        for (float datum : data) {
            if (!Float.isNaN(datum)) tmpList.add(datum);
        }
        double[] tmp = tmpList.toArray();
        lowerValue = (float) StatUtils.percentile(tmp, 5);
        upperValue = (float) StatUtils.percentile(tmp, 95);
    }


}
