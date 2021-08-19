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

import java.util.HashMap;
import java.util.Map;

public abstract class MNDLineParser {

    /**
     * A map of chromosome name -> chromosome string.  A private "intern" pool.  The java "intern" pool stores string
     * in perm space, which is rather limited and can cause us to run out of memory.
     */
    private final Map<String, String> stringInternPool = new HashMap<>();

    abstract protected int getChromosomeOrdinal(String chrom);

    abstract public String getChromosomeNameFromIndex(int chrIndex);

    public abstract AlignmentPair generateBasicPair(String[] tokens, int chrom1Index, int chrom2Index, int pos1Index, int pos2Index);

    protected AlignmentPair createPair(String[] tokens, String chrom1, String chrom2, int pos1Index, int pos2Index) {
        int chr1 = getChromosomeOrdinal(chrom1);
        int chr2 = getChromosomeOrdinal(chrom2);
        int pos1 = Integer.parseInt(tokens[pos1Index]);
        int pos2 = Integer.parseInt(tokens[pos2Index]);
        return new AlignmentPair(chr1, pos1, chr2, pos2);
    }

    public AlignmentPair generateMediumPair(String[] tokens, int chrom1Index, int chrom2Index,
                                            int pos1Index, int pos2Index, int frag1Index, int frag2Index,
                                            int mapq1Index, int mapq2Index, int strand1Index, int strand2Index) {
        AlignmentPair nextPair = generateBasicPair(tokens, chrom1Index, chrom2Index, pos1Index, pos2Index);
        updateFragmentsForPair(nextPair, tokens, frag1Index, frag2Index);
        updateMAPQsForPair(nextPair, tokens, mapq1Index, mapq2Index);
        updateStrandsForPair(nextPair, tokens, strand1Index, strand2Index);
        return nextPair;
    }

    public void updateStrandsForPair(AlignmentPair nextPair, String[] tokens, int strand1Index, int strand2Index) {
        boolean strand1 = Integer.parseInt(tokens[strand1Index]) == 0;
        boolean strand2 = Integer.parseInt(tokens[strand2Index]) == 0;
        nextPair.updateStrands(strand1, strand2);
    }

    public void updateDCICStrandsForPair(AlignmentPair nextPair, String[] tokens, int strand1Index, int strand2Index) {
        boolean strand1 = tokens[strand1Index].equals("+");
        boolean strand2 = tokens[strand2Index].equals("+");
        nextPair.updateStrands(strand1, strand2);
    }

    public void updateMAPQsForPair(AlignmentPair nextPair, String[] tokens, int mapq1Index, int mapq2Index) {
        int mapq1 = Integer.parseInt(tokens[mapq1Index]);
        int mapq2 = Integer.parseInt(tokens[mapq2Index]);
        nextPair.updateMAPQs(mapq1, mapq2);
    }

    public void updateFragmentsForPair(AlignmentPair nextPair, String[] tokens, int frag1Index, int frag2Index) {
        int frag1 = Integer.parseInt(tokens[frag1Index]);
        int frag2 = Integer.parseInt(tokens[frag2Index]);
        nextPair.updateFragments(frag1, frag2);
    }

    public void updatePairScoreIfNeeded(boolean includeScore, AlignmentPair nextPair, String[] tokens, int scoreIndex) {
        if (includeScore) {
            nextPair.setScore(Float.parseFloat(tokens[scoreIndex]));
        }
    }

    /**
     * Replace "aString" with a stored equivalent object, if it exists.  If it does not store it.  The purpose
     * of this class is to avoid running out of memory storing zillions of equivalent string.
     */
    protected String getInternedString(String aString) {
        String s = stringInternPool.get(aString);
        if (s == null) {
            //noinspection RedundantStringConstructorCall
            s = new String(aString); // The "new" will break any dependency on larger strings if this is a "substring"
            stringInternPool.put(aString, s);
        }
        return s;
    }
}
