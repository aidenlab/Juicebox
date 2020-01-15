/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.utils.dev.drink;

import juicebox.data.feature.FeatureFilter;
import juicebox.data.feature.GenomeWideList;

import java.util.*;

public class DrinkUtils {

    public static void reSort(GenomeWideList<SubcompartmentInterval> subcompartments) {
        subcompartments.filterLists(new FeatureFilter<SubcompartmentInterval>() {
            @Override
            public List<SubcompartmentInterval> filter(String chr, List<SubcompartmentInterval> featureList) {
                Collections.sort(featureList);
                return featureList;
            }
        });
    }

    public static void collapseGWList(GenomeWideList<SubcompartmentInterval> intraSubcompartments) {
        intraSubcompartments.filterLists(new FeatureFilter<SubcompartmentInterval>() {
            @Override
            public List<SubcompartmentInterval> filter(String chr, List<SubcompartmentInterval> featureList) {
                return collapseSubcompartmentIntervals(featureList);
            }
        });
    }

    private static List<SubcompartmentInterval> collapseSubcompartmentIntervals(List<SubcompartmentInterval> intervals) {
        if (intervals.size() > 0) {

            Collections.sort(intervals);
            SubcompartmentInterval collapsedInterval = (SubcompartmentInterval) intervals.get(0).deepClone();

            Set<SubcompartmentInterval> newIntervals = new HashSet<>();
            for (SubcompartmentInterval nextInterval : intervals) {
                if (collapsedInterval.overlapsWith(nextInterval)) {
                    collapsedInterval = collapsedInterval.absorbAndReturnNewInterval(nextInterval);
                } else {
                    newIntervals.add(collapsedInterval);
                    collapsedInterval = (SubcompartmentInterval) nextInterval.deepClone();
                }
            }
            newIntervals.add(collapsedInterval);

            List<SubcompartmentInterval> newIntervalsSorted = new ArrayList<>(newIntervals);
            Collections.sort(newIntervalsSorted);

            return newIntervalsSorted;
        }
        return intervals;
    }

    public static String cleanUpPath(String filePath) {
        String[] breakUpFileName = filePath.split("/");
        return breakUpFileName[breakUpFileName.length - 1].replaceAll(".hic", "");
    }

    /*
    public static void saveFileBeforeAndAfterCollapsing(GenomeWideList<SubcompartmentInterval> subcompartments,
                                                        File outputDirectory, String preCollapsingFileName,
                                                        String postCollapsingFileName) {
        File outputFile = new File(outputDirectory, preCollapsingFileName);
        subcompartments.simpleExport(outputFile);

        collapseGWList(subcompartments);

        File outputFile2 = new File(outputDirectory, postCollapsingFileName);
        subcompartments.simpleExport(outputFile2);
    }

    /*
        // find the most common
        List<SubcompartmentInterval> frequentFliers = new ArrayList();
        for(Integer x : allSubcompartmentIntervalsMap.keySet()) {

            Map<Integer, Integer> counts = new HashMap<>();
            for (Pair<Integer, Integer> pair :allSubcompartmentIntervalsMap.get(x)){
                int value = pair.getValue();
                if(counts.containsKey(value)){
                    counts.put(value,counts.get(value)+1);
                }
                else {
                    counts.put(value,1);
                }
            }
            int maxFrequency = Ints.max(Ints.toArray(counts.values()));
            if(maxFrequency > 1){
                int commonClusterID = -1;
                for(Integer clusterID : counts.keySet()){
                    if(counts.get(clusterID) >= maxFrequency){
                        commonClusterID = clusterID;
                        break;
                    }
                }

                int x2 = x + getResolution();
                frequentFliers.add(new SubcompartmentInterval(chromosome.getIndex(), chromosome.getName(), x, x2, commonClusterID));
            }
        }
        mostFrequentSubcompartment.addAll(frequentFliers);
        */


}
