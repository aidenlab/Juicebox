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

public class SimpleLineParser extends MNDLineParser {

    private final Map<String, Integer> chrNameToIndex = new HashMap<>();
    private final Map<Integer, String> chrIndexToName = new HashMap<>();
    private int nextChromIndex = 1;

    @Override
    protected int getChromosomeOrdinal(String chrom) {
        if (!chrNameToIndex.containsKey(chrom)) {
            chrNameToIndex.put(chrom, nextChromIndex);
            chrNameToIndex.put(chrom.toLowerCase(), nextChromIndex);
            chrNameToIndex.put(chrom.toUpperCase(), nextChromIndex);
            chrIndexToName.put(nextChromIndex, chrom);
            nextChromIndex++;
        }

        return chrNameToIndex.get(chrom);
    }

    @Override
    public AlignmentPair generateBasicPair(String[] tokens, int chrom1Index, int chrom2Index, int pos1Index, int pos2Index) {
        String chrom1 = getInternedString(tokens[chrom1Index]);
        String chrom2 = getInternedString(tokens[chrom2Index]);
        return createPair(tokens, chrom1, chrom2, pos1Index, pos2Index);
    }

    @Override
    public AlignmentTriple generateBasicTriple(String[] tokens, int chrom1Index, int chrom2Index, int chrom3Index, int pos1Index, int pos2Index, int pos3Index) {
        String chrom1 = getInternedString(tokens[chrom1Index]);
        String chrom2 = getInternedString(tokens[chrom2Index]);
        String chrom3 = getInternedString(tokens[chrom3Index]);
        return createTriple(tokens, chrom1, chrom2, chrom3, pos1Index, pos2Index, pos3Index);
    }

    @Override
    public String getChromosomeNameFromIndex(int chrIndex) {
        return chrIndexToName.get(chrIndex);
    }
}
