/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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

package juicebox.track.anchor;

import java.util.*;

/**
 * Created by muhammadsaadshamim on 9/28/15.
 */
public class AnchorTools {

    private static int gapThreshold = 15000;

    /**
     * BEDTools port of merge based on
     * http://bedtools.readthedocs.org/en/latest/content/tools/merge.html
     * <p/>
     * NOTE - only default functionality supported at present (no additional flags)
     *
     * @param anchors
     * @return merged list of anchors
     */
    public static List<FeatureAnchor> merge(List<FeatureAnchor> anchors) {
        Collections.sort(anchors);

        Set<FeatureAnchor> merged = new HashSet<FeatureAnchor>();
        FeatureAnchor current = anchors.get(0).clone();

        for (FeatureAnchor anchor : anchors) {
            if (anchor.hasOverlapWith(current)) {
                current.mergeWith(anchor);
            } else {
                merged.add(current);
                current = anchor.clone();
            }
        }
        merged.add(current); // in case last merger missed (i.e. boolean evaluated to true)

        return new ArrayList<FeatureAnchor>(merged);
    }

    /**
     * BEDTools port of intersect based on
     * http://bedtools.readthedocs.org/en/latest/content/tools/intersect.html
     * <p/>
     * NOTE - only default functionality supported at present (no additional flags)
     *
     * @param topAnchors
     * @param bottomAnchors
     * @return intersection of two anchor lists
     */
    public static List<FeatureAnchor> intersect(List<FeatureAnchor> topAnchors, List<FeatureAnchor> bottomAnchors) {
        Collections.sort(topAnchors);
        Collections.sort(bottomAnchors);

        Set<FeatureAnchor> intersected = new HashSet<FeatureAnchor>();

        int topIndex = 0;
        int bottomIndex = 0;
        int maxTopIndex = topAnchors.size();
        int maxBottomIndex = bottomAnchors.size();

        while (topIndex < maxTopIndex && bottomIndex < maxBottomIndex) {
            FeatureAnchor topAnchor = topAnchors.get(topIndex);
            FeatureAnchor bottomAnchor = bottomAnchors.get(bottomIndex);
            if (topAnchor.hasOverlapWith(bottomAnchor)) {
                // iterate over all possible intersections with top element
                for (int i = bottomIndex; i < maxBottomIndex; i++) {
                    FeatureAnchor newAnchor = bottomAnchors.get(i);
                    if (topAnchor.hasOverlapWith(newAnchor)) {
                        intersected.add(intersection(topAnchor, newAnchor));
                    } else {
                        break;
                    }
                }

                // iterate over all possible intersections with bottom element
                // start from +1 because +0 checked in the for loop above
                for (int i = topIndex + 1; i < maxTopIndex; i++) {
                    FeatureAnchor newAnchor = topAnchors.get(i);
                    if (bottomAnchor.hasOverlapWith(newAnchor)) {
                        intersected.add(intersection(bottomAnchor, newAnchor));
                    } else {
                        break;
                    }
                }

                // increment both
                topIndex++;
                bottomIndex++;
            } else if (topAnchor.isStrictlyToTheLeftOf(bottomAnchor)) {
                topIndex++;
            } else if (topAnchor.isStrictlyToTheRightOf(bottomAnchor)) {
                bottomIndex++;
            } else {
                System.err.println("Error while intersecting anchors.");
                System.err.println(topAnchor + " & " + bottomAnchor);
            }
        }

        return new ArrayList<FeatureAnchor>(intersected);
    }

    /**
     * @param anchor1
     * @param anchor2
     * @return intersection of anchor1 and anchor2
     */
    private static FeatureAnchor intersection(FeatureAnchor anchor1, FeatureAnchor anchor2) {
        if (anchor1.getChr().equals(anchor2.getChr())) {
            return new FeatureAnchor(anchor1.getChr(), Math.max(anchor1.getX1(), anchor2.getX1()),
                    Math.min(anchor1.getX2(), anchor2.getX2()));
        } else {
            System.err.println("Error calculating intersection of anchors");
            System.err.println(anchor1 + " & " + anchor2);
        }
        return null;
    }

    /**
     * Guarantees that all anchors have minimum width of 15000
     * PreProcessing step for anchors in MotifFinder code
     * equivalent to:
     * (awk on BED file) ... if($3-$2<15000){d=15000-($3-$2); print $1 \"\\t\" $2-int(d/2) \"\\t\" $3+int(d/2)
     *
     * @param anchors
     */
    public static void expandSmallAnchors(List<FeatureAnchor> anchors) {
        for (FeatureAnchor anchor : anchors) {
            int width = anchor.getWidth();
            if (width < gapThreshold) {
                anchor.widenMargins(gapThreshold - width);
            }
        }
    }
}
