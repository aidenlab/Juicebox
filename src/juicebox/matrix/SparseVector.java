package juicebox.matrix;
/*
 * Copyright (c) 2007-2014 by The Broad Institute of MIT and Harvard.  All Rights Reserved.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
 *
 * THE SOFTWARE IS PROVIDED "AS IS." THE BROAD AND MIT MAKE NO REPRESENTATIONS OR
 * WARRANTES OF ANY KIND CONCERNING THE SOFTWARE, EXPRESS OR IMPLIED, INCLUDING,
 * WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, NONINFRINGEMENT, OR THE ABSENCE OF LATENT OR OTHER DEFECTS, WHETHER
 * OR NOT DISCOVERABLE.  IN NO EVENT SHALL THE BROAD OR MIT, OR THEIR RESPECTIVE
 * TRUSTEES, DIRECTORS, OFFICERS, EMPLOYEES, AND AFFILIATES BE LIABLE FOR ANY DAMAGES
 * OF ANY KIND, INCLUDING, WITHOUT LIMITATION, INCIDENTAL OR CONSEQUENTIAL DAMAGES,
 * ECONOMIC DAMAGES OR INJURY TO PROPERTY AND LOST PROFITS, REGARDLESS OF WHETHER
 * THE BROAD OR MIT SHALL BE ADVISED, SHALL HAVE OTHER REASON TO KNOW, OR IN FACT
 * SHALL KNOW OF THE POSSIBILITY OF THE FOREGOING.
 */


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
        return values.containsKey(idx) ? values.get(idx) : 0;
    }

    public Collection<Integer> getIndeces() {
        return values.keySet();
    }

    /**
     * Computes the mean of occupied elements
     * @return
     */
    public Double getMean() {

        if(values.size() == 0) return Double.NaN;

        double sum=0;
        for(Double v : values.values()) {
            sum += v;
        }
        return sum / values.size();

    }
}
