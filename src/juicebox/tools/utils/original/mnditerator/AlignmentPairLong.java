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

public class AlignmentPairLong extends AlignmentPair {

    private final String seq1;
    private final String seq2;

    public AlignmentPairLong(boolean strand1, int chr1, int pos1, int frag1, int mapq1, String seq1,
                             boolean strand2, int chr2, int pos2, int frag2, int mapq2, String seq2) {
        super(strand1, chr1, pos1, frag1, mapq1, strand2, chr2, pos2, frag2, mapq2);
        this.seq1 = seq1;
        this.seq2 = seq2;
    }

    public AlignmentPairLong(AlignmentPair np, String seq1, String seq2) {
        this(np.getStrand1(), np.getChr1(), np.getPos1(), np.getFrag1(), np.getMapq1(), seq1,
                np.getStrand2(), np.getChr2(), np.getPos2(), np.getFrag2(), np.getMapq2(), seq2);
    }

    public String getSeq1() {
        return seq1;
    }

    public String getSeq2() {
        return seq2;
    }

}

