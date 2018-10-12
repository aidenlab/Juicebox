/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.juicer.curse;

import juicebox.data.feature.Feature;
import juicebox.data.feature.FeatureFilter;
import juicebox.data.feature.GenomeWideList;

import java.awt.*;
import java.util.*;
import java.util.List;

public class SubcompartmentInterval extends Feature implements Comparable<SubcompartmentInterval> {

    private static Color[] colors = new Color[]{
            new Color(230, 25, 75),
            new Color(60, 180, 75),
            new Color(255, 225, 25),
            new Color(0, 130, 200),
            new Color(245, 130, 48),
            new Color(145, 30, 180),
            new Color(70, 240, 240),
            new Color(240, 50, 230),
            new Color(210, 245, 60),
            new Color(250, 190, 190),
            new Color(0, 128, 128),
            new Color(230, 190, 255),
            new Color(170, 110, 40),
            new Color(255, 250, 200),
            new Color(128, 0, 0),
            new Color(170, 255, 195),
            new Color(128, 128, 0),
            new Color(255, 215, 180),
            new Color(0, 0, 128),
            new Color(128, 128, 128),
            new Color(0, 0, 0)
    };

    private static Color[] colors2 = new Color[]{
            new Color(255, 191, 191),
            new Color(178, 48, 0),
            new Color(255, 162, 128),
            new Color(204, 133, 51),
            new Color(76, 50, 19),
            new Color(51, 47, 38),
            new Color(229, 195, 57),
            new Color(127, 108, 32),
            new Color(51, 48, 13),
            new Color(166, 163, 124),
            new Color(229, 255, 128),
            new Color(48, 179, 0),
            new Color(22, 89, 31),
            new Color(57, 230, 172),
            new Color(0, 64, 60),
            new Color(35, 140, 133),
            new Color(115, 207, 230),
            new Color(57, 96, 115),
            new Color(0, 27, 51),
            new Color(128, 196, 255),
            new Color(0, 46, 115),
            new Color(102, 129, 204),
            new Color(0, 0, 102),
            new Color(92, 51, 204),
            new Color(191, 115, 230),
            new Color(204, 0, 255),
            new Color(54, 16, 64),
            new Color(107, 0, 115),
            new Color(213, 163, 217),
            new Color(89, 67, 88),
            new Color(230, 0, 153),
            new Color(229, 0, 92),
            new Color(242, 0, 32),
            new Color(89, 22, 31),
            new Color(140, 70, 79)
    };



    private final Integer x1;
    private Integer x2;
    private final Integer chrIndex;
    private final Integer clusterID;
    private String chrName;

    public SubcompartmentInterval(int chrIndex, String chrName, int x1, int x2, Integer clusterID) {
        this.chrIndex = chrIndex;
        this.chrName = chrName;
        this.x1 = x1;
        this.x2 = x2;
        this.clusterID = clusterID;
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
                    collapsedInterval.absorb(nextInterval);
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

    public static void reSort(GenomeWideList<SubcompartmentInterval> subcompartments) {
        subcompartments.filterLists(new FeatureFilter<SubcompartmentInterval>() {
            @Override
            public List<SubcompartmentInterval> filter(String chr, List<SubcompartmentInterval> featureList) {
                Collections.sort(featureList);
                return featureList;
            }
        });
    }

    private boolean overlapsWith(SubcompartmentInterval o) {
        return chrIndex.equals(o.chrIndex) && clusterID.equals(o.clusterID) && x2.equals(o.x1);
    }

    private void absorb(SubcompartmentInterval interval) {
        x2 = new Integer(interval.x2);
    }

    @Override
    public int compareTo(SubcompartmentInterval o) {
        int comparison = chrIndex.compareTo(o.chrIndex);
        if (comparison == 0) comparison = x1.compareTo(o.x1);
        if (comparison == 0) comparison = x2.compareTo(o.x2);
        if (comparison == 0) comparison = clusterID.compareTo(o.clusterID);
        return comparison;
    }

    @Override
    public String getKey() {
        return "" + chrIndex;
    }

    @Override
    public Feature deepClone() {
        return new SubcompartmentInterval(chrIndex, chrName, x1, x2, clusterID);
    }

    public Integer getChrIndex() {
        return chrIndex;
    }

    public Integer getX1() {
        return x1;
    }

    public Integer getX2() {
        return x2;
    }

    public Integer getClusterID() {
        return clusterID;
    }

    public String getChrName() {
        return chrName;
    }

    //    chr19	0	200000	NA	0	.	0	200000	255,255,255
    //    chr19	200000	500000	B1	-1	.	200000	500000	220,20,60
    @Override
    public String toString() {
        Color color = colors[clusterID % colors.length];
        String colorString = color.getRed() + "," + color.getGreen() + "," + color.getBlue();
        return "chr" + chrName + "\t" + x1 + "\t" + x2 + "\t" + clusterID + "\t" + clusterID
                + "\t.\t" + x1 + "\t" + x2 + "\t" + colorString;
    }
}