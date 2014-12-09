/*
 * Copyright (C) 2011-2014 Aiden Lab - All Rights Reserved
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Aiden Lab All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. Aiden Lab is not responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
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
        return values.containsKey(idx) ? values.get(idx) : 0;
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
