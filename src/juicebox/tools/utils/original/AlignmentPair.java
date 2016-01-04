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


package juicebox.tools.utils.original;

//import htsjdk.tribble.util.LittleEndianOutputStream;
//import java.io.IOException;

/**
 * @author Jim Robinson
 * @date 9/24/11
 */
public class AlignmentPair {

    private final boolean strand1;  // true if forward strand
    private final boolean strand2;
    private final int frag1;
    private final int frag2;
    private final int chr1;
    private final int pos1;
    private final int chr2;
    private final int pos2;
    private final int mapq1;
    private final int mapq2;
    private float score = 1.0f;  // The score (or count)
    private boolean isContigPair = false;


    public AlignmentPair(boolean strand1, int chr1, int pos1, int frag1, int mapq1, boolean strand2, int chr2, int pos2, int frag2, int mapq2) {
        this.strand1 = strand1;
        this.chr1 = chr1;
        this.pos1 = pos1;
        this.frag1 = frag1;
        this.mapq1 = mapq1;
        this.strand2 = strand2;
        this.chr2 = chr2;
        this.pos2 = pos2;
        this.frag2 = frag2;
        this.mapq2 = mapq2;
        this.isContigPair = false;
    }

    public AlignmentPair() {
        this(false, -1, -1, -1, -1, false, -1, -1, -1, -1);
        this.isContigPair = true;
    }


    public int getChr1() {
        return chr1;
    }

    public int getPos1() {
        return pos1;
    }

    public int getChr2() {
        return chr2;
    }

    public int getPos2() {
        return pos2;
    }

    public int getMapq1() {
        return mapq1;
    }

    public int getMapq2() {
        return mapq2;
    }

    public boolean getStrand1() {
        return strand1;
    }

    private int getStrand1AsInt() {
        return strand1 ? 0 : 16;          // 0 is the forward strand, so true; 16 is the reverse strand
    }

    private int getStrand2AsInt() {
        return strand2 ? 0 : 16;       // 0 is the forward strand, so true; 16 is the reverse strand
    }

    public boolean getStrand2() {
        return strand2;
    }

    public int getFrag1() {
        return frag1;
    }

    public int getFrag2() {
        return frag2;
    }

    public float getScore() {
        return score;
    }

    public void setScore(float score1) {
        this.score = score1;
    }

    public String toString() {
        int str1 = getStrand1AsInt();
        int str2 = getStrand2AsInt();
        return str1 + "\t" + chr1 + "\t" + pos1 + "\t" + frag1 + "\t" + mapq1 + "\t" +
                str2 + "\t" + chr2 + "\t" + pos2 + "\t" + frag2 + "\t" + mapq2 + "\t" + score;
    }

    public boolean isContigPair() {
        return this.isContigPair;
    }

}
