/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2022 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.tools.utils.original.merge.merger;

import java.io.BufferedWriter;
import java.io.IOException;
import java.text.NumberFormat;
import java.util.Locale;
import java.util.StringTokenizer;

public abstract class StatsMerger extends Merger {
    public static int NUM_FIELDS = 28;
    private final NumberFormat nf = NumberFormat.getNumberInstance(Locale.US);
    private final long[] statistics = new long[NUM_FIELDS];

    public void parse(String s) {
        StringTokenizer lines = new StringTokenizer(s, "\n");
        while (lines.hasMoreTokens()) {
            String current = lines.nextToken();
            processLine(current);
        }
    }

    protected long getStatistic(STATS_LABEL label) {
        return statistics[label.ordinal()];
    }

    protected void processLine(String current) {
        int key = parseLabel(current).ordinal();
        if (key == STATS_LABEL.IGNORE.ordinal()) return;
        String[] colonSplit = current.split(":");
        if (colonSplit.length == 2) {
            if (hasNA(colonSplit[1])) return;
            String[] parenthesesSplit = colonSplit[1].split("\\(");
            if (key == STATS_LABEL.PAIR_PERCENTS.ordinal()) {
                if (statistics[key] != -1) {
                    if (checkAll25(colonSplit[1])) {
                        statistics[key] = 25;
                    } else {
                        statistics[key] = -1;
                    }
                }
            } else {
                try {
                    long value = Long.parseLong(parenthesesSplit[0].replaceAll(",", "").trim());
                    if (key == STATS_LABEL.CONVERGENCE.ordinal()) {
                        statistics[key] = Math.max(statistics[key], value);
                    } else {
                        statistics[key] += value;
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

        } else { // Appears to be correct format, convert files as appropriate
            System.err.println("Incorrect form in original statistics attribute. Offending line:");
            System.err.println(current);
        }
    }

    private boolean checkAll25(String s) {
        int start = s.length();
        int end = s.replaceAll("25%", "").length();
        return start - end == 12;
    }

    private boolean hasNA(String s) {
        String na = "N/A";
        return s.toLowerCase().contains(na.toLowerCase());
    }

    protected STATS_LABEL parseLabel(String s) {
        if (containsIgnoreCase(s, "No chimera found:")) return STATS_LABEL.NO_CHIMERA;
        if (containsIgnoreCase(s, "3 or more alignments")) return STATS_LABEL.THREE_PLUS;
        if (containsIgnoreCase(s, "Ligation Motif Present:")) return STATS_LABEL.LIGATION_MOTIF;
        if (containsIgnoreCase(s, "Total Unique:")) return STATS_LABEL.TOTAL_UNIQUE;
        if (containsIgnoreCase(s, "Total Duplicates:")) return STATS_LABEL.TOTAL_DUPS;
        if (containsIgnoreCase(s, "Library Complexity")) return STATS_LABEL.IGNORE;
        if (containsIgnoreCase(s, "Intra-fragment Reads:")) return STATS_LABEL.INTRA_FRAG_READS;
        if (containsIgnoreCase(s, "Below MAPQ Threshold:")) return STATS_LABEL.BELOW_MAPQ;
        if (containsIgnoreCase(s, "Hi-C Contacts:")) return STATS_LABEL.HIC_CONTACTS;
        if (containsIgnoreCase(s, "3' Bias (Long Range):")) return STATS_LABEL.THREE_BIAS;
        if (containsIgnoreCase(s, "Pair Type %(L-I-O-R):")) return STATS_LABEL.PAIR_PERCENTS;
        if (containsIgnoreCase(s, "L-I-O-R Convergence:")) return STATS_LABEL.CONVERGENCE;
        if (containsIgnoreCase(s, "Inter-chromosomal:")) return STATS_LABEL.INTER;
        if (containsIgnoreCase(s, "Intra-chromosomal:")) return STATS_LABEL.INTRA;
        if (containsIgnoreCase(s, "Short Range (<20Kb):")) return STATS_LABEL.IGNORE;
        if (containsIgnoreCase(s, "<500BP:")) return STATS_LABEL.FIVE_HUNDRED_BP;
        if (containsIgnoreCase(s, "500BP-5kB:")) return STATS_LABEL.FIVE_KB;
        if (containsIgnoreCase(s, "5kB-20kB:")) return STATS_LABEL.TWENTY_KB;
        if (containsIgnoreCase(s, "Long Range (>20Kb):")) return STATS_LABEL.LONG_RANGE;
        return STATS_LABEL.IGNORE;
    }

    protected void write(BufferedWriter statsOut, String description, STATS_LABEL valType, long denom, long denom2) throws IOException {
        long value = statistics[valType.ordinal()];
        if (value > 0) {
            String valString = commify(value);
            if (denom > 0) {
                if (denom2 > 0) {
                    statsOut.write(description + valString + " (" + percent(value, denom) + " / " + percent(value, denom2) + ")\n");
                } else {
                    statsOut.write(description + valString + " (" + percent(value, denom) + ")\n");
                }
            } else {
                statsOut.write(description + valString + "\n");
            }
        }
    }

    private String percent(long num, long total) {
        return String.format("%.2f", num * 100f / total) + "%";
    }

    private String commify(long value) {
        return nf.format(value);
    }

    public enum STATS_LABEL {
        TOTAL_SEQ, UNMAPPED, TWO_ALIGN, TWO_ALIGN_A, TWO_ALIGN_B,
        ONE_ALIGN, ONE_UNIQUE, ONE_DUPS, TWO_UNIQUE, TWO_DUPS,
        IGNORE, NO_CHIMERA, THREE_PLUS, LIGATION_MOTIF, TOTAL_UNIQUE, TOTAL_DUPS,
        INTRA_FRAG_READS, BELOW_MAPQ, HIC_CONTACTS, THREE_BIAS, PAIR_PERCENTS, CONVERGENCE,
        INTER, INTRA, FIVE_HUNDRED_BP, FIVE_KB, TWENTY_KB, LONG_RANGE
    }
}
