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


package juicebox.tools.utils.original.mnditerator;

//import htsjdk.tribble.util.LittleEndianOutputStream;
//import java.io.IOException;

public class AlignmentTriple {

    private boolean strand1 = true;  // true if forward strand
    private boolean strand2 = false;
    private boolean strand3 = true;
    private final int chr1;
    private final int pos1;
    private final int chr2;
    private final int pos2;
    private final int chr3;
    private final int pos3;
    private int mapq1 = 1000;
    private int mapq2 = 1000;
    private int mapq3 = 1000;
    private float score = 1.0f;  // The score (or count)
//    private boolean isContigPair;
    private boolean isValidForStats = true;

    public AlignmentTriple(boolean strand1, int chr1, int pos1, int mapq1, boolean strand2, int chr2, int pos2,
                           int mapq2, boolean strand3, int chr3, int pos3, int mapq3) {
        this.strand1 = strand1;
        this.chr1 = chr1;
        this.pos1 = pos1;
        this.mapq1 = mapq1;
        this.strand2 = strand2;
        this.chr2 = chr2;
        this.pos2 = pos2;
        this.mapq2 = mapq2;
        this.strand3 = strand3;
        this.chr3 = chr3;
        this.pos3 = pos3;
        this.mapq3 = mapq3;
//        isContigPair = false;
    }

    public AlignmentTriple() {
        this(false, -1, -1, -1, false, -1, -1, -1, false, -1, -1, -1);
//        isContigPair = true;
        isValidForStats = false;
    }

    public AlignmentTriple(boolean ignore) {
        this();
//        isContigPair = false;
        isValidForStats = false;
    }

    public AlignmentTriple(int chr1, int pos1, int chr2, int pos2, int chr3, int pos3) {
        this.chr1 = chr1;
        this.pos1 = pos1;
        this.chr2 = chr2;
        this.pos2 = pos2;
        this.chr3 = chr3;
        this.pos3 = pos3;
//        isContigPair = false;
    }

    public AlignmentTriple(boolean strand1, int chr1, int pos1, boolean strand2, int chr2, int pos2,
                           boolean strand3, int chr3, int pos3) {
        this.strand1 = strand1;
        this.chr1 = chr1;
        this.pos1 = pos1;
        this.strand2 = strand2;
        this.chr2 = chr2;
        this.pos2 = pos2;
        this.strand3 = strand3;
        this.chr3 = chr3;
        this.pos3 = pos3;
//        isContigPair = false;
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

    public int getChr3() {
        return chr3;
    }

    public int getPos3() {
        return pos3;
    }

    public int getMapq1() {
        return mapq1;
    }

    public int getMapq2() {
        return mapq2;
    }

    public int getMapq3() {
        return mapq3;
    }

    public boolean getStrand1() {
        return strand1;
    }

    public boolean getStrand2() {
        return strand2;
    }

    public boolean getStrand3() {
        return strand3;
    }

    private int getStrand1AsInt() {
        return strand1 ? 0 : 16;          // 0 is the forward strand, so true; 16 is the reverse strand
    }

    private int getStrand2AsInt() {
        return strand2 ? 0 : 16;       // 0 is the forward strand, so true; 16 is the reverse strand
    }

    private int getStrand3AsInt() {
        return strand3 ? 0 : 16;       // 0 is the forward strand, so true; 16 is the reverse strand
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
        int str3 = getStrand3AsInt();
        return str1 + "\t" + chr1 + "\t" + pos1 + "\t" + "\t" + mapq1 + "\t" +
                str2 + "\t" + chr2 + "\t" + pos2 + "\t" + "\t" + mapq2 + "\t" +
                str3 + "\t" + chr3 + "\t" + pos3 + "\t" + "\t" + mapq3 + "\t" + score;
    }

//    public boolean isContigPair() {
//        return this.isContigPair;
//    }

    public boolean isValid() {
        return isValidForStats;
    }

    public boolean isShort() {
        return mapq1 == 1000 && mapq2 == 1000 && mapq3 == 1000;
    }

    public void updateStrands(boolean strand1, boolean strand2, boolean strand3) {
        this.strand1 = strand1;
        this.strand2 = strand2;
        this.strand3 = strand3;
    }

    public void updateMAPQs(int mapq1, int mapq2, int mapq3) {
        this.mapq1 = mapq1;
        this.mapq2 = mapq2;
        this.mapq3 = mapq3;
    }
}
