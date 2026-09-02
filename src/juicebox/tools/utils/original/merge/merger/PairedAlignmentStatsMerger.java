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
import java.io.FileWriter;
import java.io.IOException;

public class PairedAlignmentStatsMerger extends StatsMerger {
    @Override
    public void printToMergedFile(String filename) {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            out.write("Read type: Paired End\n");
            long denom1 = getStatistic(STATS_LABEL.TOTAL_SEQ);
            write(out, "Sequenced Read Pairs: ", STATS_LABEL.TOTAL_SEQ, -1, -1);
            write(out, "No chimera found: ", STATS_LABEL.NO_CHIMERA, denom1, -1);
            write(out, "One or both reads unmapped: ", STATS_LABEL.UNMAPPED, denom1, -1);
            write(out, "2 alignments: ", STATS_LABEL.TWO_ALIGN, denom1, -1);
            write(out, "\t2 alignments (A...B): ", STATS_LABEL.TWO_ALIGN_A, denom1, -1);
            write(out, "\t2 alignments (A1...A2B; A1B2...B1A2): ", STATS_LABEL.TWO_ALIGN_B, denom1, -1);
            write(out, "3 or more alignments: ", STATS_LABEL.THREE_PLUS, denom1, -1);
            write(out, "Ligation Motif Present: ", STATS_LABEL.LIGATION_MOTIF, denom1, -1);
            long denomAlignable = getStatistic(STATS_LABEL.TWO_ALIGN);
            long denom2AUniq = getStatistic(STATS_LABEL.TOTAL_UNIQUE);
            write(out, "Total Unique: ", STATS_LABEL.TOTAL_UNIQUE, denom1, denomAlignable);
            write(out, "Total Duplicates: ", STATS_LABEL.TOTAL_DUPS, denom1, denomAlignable);
            write(out, "Below MAPQ Threshold: ", STATS_LABEL.BELOW_MAPQ, denom1, denom2AUniq);
            write(out, "Hi-C Contacts: ", STATS_LABEL.HIC_CONTACTS, denom1, denom2AUniq);
            write(out, "\t3' Bias (Long Range): ", STATS_LABEL.THREE_BIAS, denom1, denom2AUniq);
            if (getStatistic(STATS_LABEL.PAIR_PERCENTS) == 25L) {
                out.write("\tPair Type %(L-I-O-R): 25% - 25% - 25% - 25%\n");
                write(out, "\tL-I-O-R Convergence: ", STATS_LABEL.CONVERGENCE, -1, -1);
            }
            write(out, "Inter-chromosomal: ", STATS_LABEL.INTER, denom1, denom2AUniq);
            write(out, "Intra-chromosomal: ", STATS_LABEL.INTRA, denom1, denom2AUniq);
            out.write("Short Range (<20Kb):\n");
            write(out, "\t<500BP: ", STATS_LABEL.FIVE_HUNDRED_BP, denom1, denom2AUniq);
            write(out, "\t500BP-5kB: ", STATS_LABEL.FIVE_KB, denom1, denom2AUniq);
            write(out, "\t5kB-20kB: ", STATS_LABEL.TWENTY_KB, denom1, denom2AUniq);
            write(out, "Long Range (>20Kb): ", STATS_LABEL.LONG_RANGE, denom1, denom2AUniq);
            out.close();
        } catch (IOException error) {
            error.printStackTrace();
        }
    }

    @Override
    protected STATS_LABEL parseLabel(String s) {
        if (containsIgnoreCase(s, "Sequenced Read Pairs:")) return STATS_LABEL.TOTAL_SEQ;
        if (containsIgnoreCase(s, "One or both reads unmapped:")) return STATS_LABEL.UNMAPPED;
        if (containsIgnoreCase(s, "2 alignments:")) return STATS_LABEL.TWO_ALIGN;
        if (containsIgnoreCase(s, "2 alignments (A...B):")) return STATS_LABEL.TWO_ALIGN_A;
        if (containsIgnoreCase(s, "2 alignments (A1...A2B; A1B2...B1A2):")) return STATS_LABEL.TWO_ALIGN_B;
        if (containsIgnoreCase(s, "Average insert size:")) return STATS_LABEL.IGNORE;
        return super.parseLabel(s);
    }
}
