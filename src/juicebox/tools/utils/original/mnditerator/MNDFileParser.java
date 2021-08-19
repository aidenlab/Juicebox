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

import juicebox.tools.clt.JuiceboxCLT;

import java.io.IOException;

public class MNDFileParser {

    protected static final int dcicF1 = 0, dcicF2 = 1, dcicM1 = 2, dcicM2 = 3;
    public static String[] dcicCategories = new String[]{"frag1", "frag2", "mapq1", "mapq2"};
    protected final int[] dcicIndices = new int[]{-1, -1, -1, -1};
    private final MNDLineParser pg;
    private Format format = null;

    MNDFileParser(MNDLineParser pg) {
        this.pg = pg;
    }

    public static Format getFileFormat(int nTokens, String nextLine) throws IOException {
        if (nTokens == 4) {
            return Format.SUPER_SHORT;
        } else if (nTokens == 5) {
            return Format.SUPER_SHORT_WITH_SCORE;
        } else if (nTokens == 8) {
            return Format.SHORT;
        } else if (nTokens == 9) {
            return Format.SHORT_WITH_SCORE;
        } else if (nTokens == 16) {
            return Format.LONG;
        } else if (nTokens == 11) {
            return Format.MEDIUM;
        } else {
            throw new IOException("Unexpected number of columns: " + nTokens + "\n" +
                    "Check line containing:\n" + nextLine);
        }
    }

    /**
     * formats detailed: https://github.com/aidenlab/juicer/wiki/Pre#file-format
     */
    public AlignmentPair parse(String nextLine) throws IOException {
        String[] tokens = JuiceboxCLT.splitToList(nextLine);
        if (format == null) {
            int nTokens = tokens.length;
            if (nextLine.startsWith("#")) { // header line, skip; DCIC files MUST have header
                format = Format.DCIC;
                updateDCICIndicesIfApplicable(nextLine, tokens);
                return new AlignmentPair(true);
            } else {
                format = getFileFormat(nTokens, nextLine);
            }
        }

        if (format == Format.MEDIUM) {
            return parseMediumFormat(tokens);
        } else if (format == Format.LONG) {
            return parseLongFormat(tokens);
        } else if (format == Format.DCIC) {
            return parseDCICFormat(tokens);
        } else if (format == Format.SUPER_SHORT || format == Format.SUPER_SHORT_WITH_SCORE) {
            return parseSuperShortFormat(tokens, format == Format.SUPER_SHORT_WITH_SCORE);
        } else {
            return parseShortFormat(tokens, format == Format.SHORT_WITH_SCORE);
        }
    }

    public void updateDCICIndicesIfApplicable(String nextLine, String[] tokens) {
        if (nextLine.contains("column")) {
            for (int i = 0; i < tokens.length; i++) {
                for (int k = 0; k < MNDFileParser.dcicCategories.length; k++) {
                    if (tokens[i].contains(MNDFileParser.dcicCategories[k])) {
                        dcicIndices[k] = i - 1;
                    }
                }
            }
        }
    }

    public AlignmentPair parseShortFormat(String[] tokens, boolean includeScore) {
        AlignmentPair nextPair = pg.generateBasicPair(tokens, 1, 5, 2, 6);
        pg.updatePairScoreIfNeeded(includeScore, nextPair, tokens, 8);
        pg.updateFragmentsForPair(nextPair, tokens, 3, 7);
        pg.updateStrandsForPair(nextPair, tokens, 0, 4);
        return nextPair;
    }

    public AlignmentPair parseSuperShortFormat(String[] tokens, boolean includeScore) {
        AlignmentPair nextPair = pg.generateBasicPair(tokens, 0, 2, 1, 3);
        pg.updatePairScoreIfNeeded(includeScore, nextPair, tokens, 4);
        return nextPair;
    }

    public AlignmentPair parseDCICFormat(String[] tokens) {
        AlignmentPair nextPair = pg.generateBasicPair(tokens, 1, 3, 2, 4);
        pg.updateDCICStrandsForPair(nextPair, tokens, 5, 6);
        if (dcicIndices[dcicF1] != -1 && dcicIndices[dcicF2] != -1) {
            pg.updateFragmentsForPair(nextPair, tokens, dcicIndices[dcicF1], dcicIndices[dcicF2]);
        }
        if (dcicIndices[dcicM1] != -1 && dcicIndices[dcicM2] != -1) {
            pg.updateMAPQsForPair(nextPair, tokens, dcicIndices[dcicM1], dcicIndices[dcicM2]);
        }
        return nextPair;
    }

    public AlignmentPair parseLongFormat(String[] tokens) {
        AlignmentPair nextPair = pg.generateMediumPair(tokens, 1, 5, 2, 6,
                3, 7, 8, 11, 0, 4);
        return new AlignmentPairLong(nextPair, tokens[10], tokens[13]);
    }

    public AlignmentPair parseMediumFormat(String[] tokens) {
        return pg.generateMediumPair(tokens, 2, 6, 3, 7,
                4, 8, 9, 10, 1, 5);
    }

    public String getChromosomeNameFromIndex(int chrIndex) {
        return pg.getChromosomeNameFromIndex(chrIndex);
    }

    enum Format {SUPER_SHORT, SUPER_SHORT_WITH_SCORE, SHORT, LONG, MEDIUM, SHORT_WITH_SCORE, DCIC}

}
