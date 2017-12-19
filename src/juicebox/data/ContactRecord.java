/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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
    private float counts = 0;
    private float rCounts = 0;
    private float gCounts = 0;
    private float bCounts = 0;
    private String key;

    public ContactRecord(int binX, int binY, float counts, RGBButton.Channel channel) {
        this.binX = binX;
        this.binY = binY;
        incrementCount(channel, counts);
    }

    public void incrementCount(RGBButton.Channel channel, float score) {
        counts += score;
        switch (channel) {
            case RED:
                rCounts += score;
                break;
            case GREEN:
                gCounts += score;
                break;
            case BLUE:
                bCounts += score;
                break;
        }
    }


    public int getBinX() {
        return binX;
    }

    public int getBinY() {
        return binY;
    }

    public float getBaseCounts() {
        return counts;
    }

    public double[] getRGBCounts() {
        return new double[]{rCounts, gCounts, bCounts};
    }

    public float getCounts(RGBButton.Channel channel) {
        switch (channel) {
            case RED:
                return rCounts;
            case GREEN:
                return gCounts;
            case BLUE:
                return bCounts;
        }
        return Float.NaN;
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
        return "" + binX + " " + binY + " R-" + rCounts + " G-" + gCounts + " B-" + bCounts;
    }

    public String getKey() {
        if (key == null) {
            key = binX + "_" + binY;
        }
        return key;
    }
}
