/*
 * Copyright (c) 2007-2011 by The Broad Institute of MIT and Harvard.  All Rights Reserved.
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

package juicebox.tools;

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
        this(false,-1,-1,-1,-1,false,-1,-1,-1,-1);
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

    public int getMapq1() { return mapq1; }

    public int getMapq2() { return mapq2; }

    public boolean getStrand1() {
        return strand1;
    }

    public int getStrand1AsInt() {
        return strand1 ? 0 : 16;          // 0 is the forward strand, so true; 16 is the reverse strand
    }

    public int getStrand2AsInt() {
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

    public void setScore(float score1) {
        this.score = score1;
    }

    public float getScore() {
        return score;
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
