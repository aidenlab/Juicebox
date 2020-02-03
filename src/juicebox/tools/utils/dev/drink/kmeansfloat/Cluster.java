/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.dev.drink.kmeansfloat;


import java.util.Arrays;

/**
 * Class to represent a cluster of coordinates.
 */
public class Cluster {

    // Indices of the member coordinates.
    private final int[] memberIndexes;
    // The cluster center.
    private final float[] center;

    /**
     * Constructor.
     *
     * @param memberIndexes indices of the member coordinates.
     * @param center        the cluster center.
     */
    public Cluster(int[] memberIndexes, float[] center) {
        this.memberIndexes = memberIndexes;
        this.center = center;
    }

    /**
     * Get the member indices.
     *
     * @return an array containing the indices of the member coordinates.
     */
    public int[] getMemberIndexes() {
        return memberIndexes;
    }

    /**
     * Get the cluster center.
     *
     * @return a reference to the cluster center array.
     */
    public float[] getCenter() {
        return center;
    }

    public Cluster getClone() {
        if (memberIndexes != null && center != null) {
            int[] membersCopy = Arrays.copyOf(memberIndexes, memberIndexes.length);
            float[] centerCopy = Arrays.copyOf(center, center.length);
            return new Cluster(membersCopy, centerCopy);
        }

        return null;
    }
}

