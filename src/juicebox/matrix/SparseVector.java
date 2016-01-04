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

import java.util.Collection;
import java.util.HashMap;

/**
 * Simple representation of a sparse vector.   C
 */
public class SparseVector {

    private final int length;
    private final HashMap<Integer, Double> values;

    public SparseVector(int length) {
        this.length = length;
        values = new HashMap<Integer, Double>();
    }

    public void set(Integer i, Double v) {
        if (i >= length) {
            throw new IndexOutOfBoundsException("Index " + i + " is >= length " + length);
        }
        values.put(i, v);
    }

    public int getLength() {
        return length;
    }

    public Double get(Integer idx) {
        return values.containsKey(idx) ? values.get(idx) : 0.0;
    }

    public Collection<Integer> getIndeces() {
        return values.keySet();
    }

    /**
     * Computes the mean of occupied elements
     *
     * @return
     */
    public Double getMean() {

        if (values.size() == 0) return Double.NaN;

        double sum = 0;
        for (Double v : values.values()) {
            sum += v;
        }
        return sum / values.size();

    }
}
