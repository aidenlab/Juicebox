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

package juicebox.tools.utils.original.merge;

import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.tools.utils.original.Preprocessor;
import juicebox.tools.utils.original.merge.merger.GraphsMerger;
import juicebox.tools.utils.original.merge.merger.Merger;
import juicebox.tools.utils.original.merge.merger.PairedAlignmentStatsMerger;
import juicebox.tools.utils.original.merge.merger.SingleAlignmentStatsMerger;

import java.util.ArrayList;
import java.util.List;

public class HiCMergeTools {

    public static void mergeStatsAndGraphs(List<String> dsPaths, Preprocessor builder, String stem) {
        List<String> statsList = new ArrayList<>();
        List<String> graphsList = new ArrayList<>();
        for (String dsPath : dsPaths) {
            List<String> newList = new ArrayList<>();
            newList.add(dsPath);
            Dataset ds = HiCFileTools.extractDatasetForCLT(newList, false);
            addToListIfValidString(ds.getStatistics(), statsList);
            addToListIfValidString(ds.getGraphs(), graphsList);
        }

        if (statsList.size() > 0) {
            boolean isSingleAlignment = confirmAllSingleAlignment(statsList);
            String statsPath = stem + "merged_stats.txt";
            if (isSingleAlignment) {
                StatsUtils.merge(statsList, new SingleAlignmentStatsMerger(), statsPath);
            } else {
                StatsUtils.merge(statsList, new PairedAlignmentStatsMerger(), statsPath);
            }
            builder.setStatisticsFile(statsPath);
        }

        if (graphsList.size() > 0) {
            String graphsPath = stem + "merged_graphs_hists.m";
            StatsUtils.merge(graphsList, new GraphsMerger(), graphsPath);
            builder.setGraphFile(graphsPath);
        }
    }

    private static void addToListIfValidString(String info, List<String> infoList) {
        if (info != null && info.length() > 1) {
            infoList.add(info);
        }
    }

    private static boolean confirmAllSingleAlignment(List<String> statsList) {
        boolean hasSingleAlignment = false;
        boolean hasPairedAlignment = false;
        for (String s : statsList) {
            hasSingleAlignment = hasSingleAlignment | Merger.containsIgnoreCase(s, "Read type: Single End");
            hasPairedAlignment = hasPairedAlignment | Merger.containsIgnoreCase(s, "Read type: Paired End");
        }
        if (hasSingleAlignment && hasPairedAlignment) {
            System.err.println("Cannot mix single-end and paired-end files together");
            System.exit(9);
        }
        return hasSingleAlignment;
    }
}
