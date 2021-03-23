/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.data;

import juicebox.windowui.NormalizationType;

import java.util.Objects;

/**
 * @author jrobinso
 * @since Aug 3, 2010
 */
public class ContactRecord implements Comparable<ContactRecord> {
    
    /**
     * Bin number in x coordinate
     */
    private final int binX;

    /**
     * Bin number in y coordinate
     */
    private final int binY;

    /**
     * Total number of counts, or cumulative score
     */
    private float counts;

    public ContactRecord(int binX, int binY, float counts) {
        this.binX = binX;
        this.binY = binY;
        this.counts = counts;
    }

    public void incrementCount(float score) {
        counts += score;
    }


    public int getBinX() {
        return binX;
    }

    public int getBinY() {
        return binY;
    }

    public float getCounts() {
        return counts;
    }

    @Override
    public int compareTo(ContactRecord contactRecord) {
        if (this.binX != contactRecord.binX) {
            return binX - contactRecord.binX;
        } else if (this.binY != contactRecord.binY) {
            return binY - contactRecord.binY;
        } else return 0;
    }

    public String toString() {
        return "" + binX + " " + binY + " " + counts;
    }

    @Override
    public int hashCode() {
        return Objects.hash(binX, binY, counts);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        return compareTo((ContactRecord) obj) == 0;
    }


    public String getKey(NormalizationType normalizationType) {
        return binX + "_" + binY + "_" + normalizationType;
    }
}
