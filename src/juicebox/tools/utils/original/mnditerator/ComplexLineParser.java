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

import juicebox.data.ChromosomeHandler;

import java.util.Map;

public class ComplexLineParser extends MNDLineParser {

    private final Map<String, Integer> chromosomeOrdinals;
    private final boolean allowNewChroms;
    private final ChromosomeHandler handler;
    private int chromCounter = -1;

    ComplexLineParser(Map<String, Integer> chromosomeOrdinals, ChromosomeHandler handler,
                      boolean allowNewChroms, boolean shouldUpdateCounter) {
        this.chromosomeOrdinals = chromosomeOrdinals;
        this.allowNewChroms = allowNewChroms;
        this.handler = handler;
        if (shouldUpdateCounter) {
            updateChromCounter();
        }
    }

    @Override
    protected int getChromosomeOrdinal(String chrom) {
        return chromosomeOrdinals.get(chrom);
    }

    @Override
    public AlignmentPair generateBasicPair(String[] tokens, int chrom1Index, int chrom2Index, int pos1Index, int pos2Index) {
        String chrom1 = handler.cleanUpName(getInternedString(tokens[chrom1Index]));
        String chrom2 = handler.cleanUpName(getInternedString(tokens[chrom2Index]));
        if (isValid(chrom1, chrom2)) {
            return createPair(tokens, chrom1, chrom2, pos1Index, pos2Index);
        }
        return new AlignmentPair(); // sets dummy values, sets isContigPair
    }

    @Override
    public String getChromosomeNameFromIndex(int chrIndex) {
        return handler.getChromosomeFromIndex(chrIndex).getName();
    }

    private void updateOrdinalsMap(String chrom) {
        if (!chromosomeOrdinals.containsKey(chrom)) {
            chromosomeOrdinals.put(chrom, chromCounter++);
        }
    }

    private void updateChromCounter() {
        for (Integer val : chromosomeOrdinals.values()) {
            chromCounter = Math.max(chromCounter, val);
        }
        chromCounter++;
    }

    private boolean isValid(String chrom1, String chrom2) {
        if (chromosomeOrdinals.containsKey(chrom1) &&
                chromosomeOrdinals.containsKey(chrom2)) {
            return true;
        }

        if (allowNewChroms) {
            updateOrdinalsMap(chrom1);
            updateOrdinalsMap(chrom2);
            return true;
        }

        return false;
    }
}
