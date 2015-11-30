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

package juicebox.data.anchor;

import java.util.*;

/**
 * Created by muhammadsaadshamim on 11/17/15.
 */
public class BEDTools {
    /**
     * BEDTools port of merge based on
     * http://bedtools.readthedocs.org/en/latest/content/tools/merge.html
     * <p/>
     * NOTE - only default functionality supported at present (no additional flags)
     *
     * @param anchors
     * @return merged list of anchors
     */
    public static List<MotifAnchor> merge(List<MotifAnchor> anchors) {
        Collections.sort(anchors);

        Set<MotifAnchor> merged = new HashSet<MotifAnchor>();
        if (anchors.size() > 0) {
            MotifAnchor current = (MotifAnchor) anchors.get(0).deepClone();

            for (MotifAnchor anchor : anchors) {
                if (anchor.hasOverlapWith(current)) {
                    current.mergeWith(anchor);
                } else {
                    merged.add(current);
                    current = (MotifAnchor) anchor.deepClone();
                }
            }
            merged.add(current); // in case last merger missed (i.e. boolean evaluated to true)
        }
        return new ArrayList<MotifAnchor>(merged);
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
    public static List<MotifAnchor> intersect(List<MotifAnchor> topAnchors, List<MotifAnchor> bottomAnchors,
                                              boolean conductFullIntersection) {
        Collections.sort(topAnchors);
        Collections.sort(bottomAnchors);

        Set<MotifAnchor> intersected = new HashSet<MotifAnchor>();

        int topIndex = 0;
        int bottomIndex = 0;
        int maxTopIndex = topAnchors.size();
        int maxBottomIndex = bottomAnchors.size();

        while (topIndex < maxTopIndex && bottomIndex < maxBottomIndex) {
            MotifAnchor topAnchor = topAnchors.get(topIndex);
            MotifAnchor bottomAnchor = bottomAnchors.get(bottomIndex);
            if (topAnchor.hasOverlapWith(bottomAnchor) || bottomAnchor.hasOverlapWith(topAnchor)) {

                //List<MotifAnchor> tempIntersected = new ArrayList<MotifAnchor>();

                // iterate over all possible intersections with top element
                for (int i = bottomIndex; i < maxBottomIndex; i++) {
                    MotifAnchor newAnchor = bottomAnchors.get(i);
                    if (topAnchor.hasOverlapWith(newAnchor) || newAnchor.hasOverlapWith(topAnchor)) {
                        intersected.add(intersection(topAnchor, newAnchor, conductFullIntersection));
                    } else {
                        break;
                    }
                }

                // iterate over all possible intersections with bottom element
                // start from +1 because +0 checked in the for loop above
                for (int i = topIndex + 1; i < maxTopIndex; i++) {
                    MotifAnchor newAnchor = topAnchors.get(i);
                    if (bottomAnchor.hasOverlapWith(newAnchor) || newAnchor.hasOverlapWith(bottomAnchor)) {
                        intersected.add(intersection(bottomAnchor, newAnchor, conductFullIntersection));
                    } else {
                        break;
                    }
                }

                //intersected.addAll(merge(tempIntersected));

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

        return new ArrayList<MotifAnchor>(intersected);
    }

    /**
     * @param anchor1
     * @param anchor2
     * @return intersection of anchor1 and anchor2
     */
    private static MotifAnchor intersection(MotifAnchor anchor1, MotifAnchor anchor2, boolean conductFullIntersection) {
        if (anchor1.getChr() == anchor2.getChr()) {

            int start = Math.max(anchor1.getX1(), anchor2.getX1());
            int end = Math.min(anchor1.getX2(), anchor2.getX2());

            if (start > end) {
                System.err.println("err _ " + start + " " + end);
            }

            MotifAnchor intersectedMotif = new MotifAnchor(anchor1.getChr(), start, end);

            // if all secondary attributes are also to be copied
            if (conductFullIntersection) {
                if (anchor1.hasFIMOAttributes()) {
                    intersectedMotif.addFIMOAttributesFrom(anchor1);
                } else if (anchor2.hasFIMOAttributes()) {
                    intersectedMotif.addFIMOAttributesFrom(anchor2);
                }

                intersectedMotif.addFeatureReferencesFrom(anchor1);
                intersectedMotif.addFeatureReferencesFrom(anchor2);
            }

            return intersectedMotif;
        } else {
            System.err.println("Error calculating intersection of anchors");
            System.err.println(anchor1 + " & " + anchor2);
        }
        return null;
    }

    public static List<MotifAnchor> preservativeIntersect(List<MotifAnchor> topAnchors, List<MotifAnchor> bottomAnchors,
                                                          boolean conductFullIntersection) {
        Collections.sort(topAnchors);
        Collections.sort(bottomAnchors);

        Set<MotifAnchor> intersected = new HashSet<MotifAnchor>();

        int topIndex = 0;
        int bottomIndex = 0;
        int maxTopIndex = topAnchors.size();
        int maxBottomIndex = bottomAnchors.size();

        while (topIndex < maxTopIndex && bottomIndex < maxBottomIndex) {
            MotifAnchor topAnchor = topAnchors.get(topIndex);
            MotifAnchor bottomAnchor = bottomAnchors.get(bottomIndex);
            if (topAnchor.hasOverlapWith(bottomAnchor) || bottomAnchor.hasOverlapWith(topAnchor)) {
                // iterate over all possible intersections with top element

                //List<MotifAnchor> tempIntersection = new ArrayList<MotifAnchor>();

                for (int i = bottomIndex; i < maxBottomIndex; i++) {
                    MotifAnchor newAnchor = bottomAnchors.get(i);
                    if (topAnchor.hasOverlapWith(newAnchor) || newAnchor.hasOverlapWith(topAnchor)) {
                        intersected.add(preservativeIntersection(topAnchor, newAnchor, conductFullIntersection));
                    } else {
                        break;
                    }
                }

                // iterate over all possible intersections with bottom element
                // start from +1 because +0 checked in the for loop above
                for (int i = topIndex + 1; i < maxTopIndex; i++) {
                    MotifAnchor newAnchor = topAnchors.get(i);
                    if (bottomAnchor.hasOverlapWith(newAnchor) || newAnchor.hasOverlapWith(bottomAnchor)) {
                        intersected.add(preservativeIntersection(newAnchor, bottomAnchor, conductFullIntersection));
                    } else {
                        break;
                    }
                }

                //intersected.addAll(merge(tempIntersection));

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

        return new ArrayList<MotifAnchor>(intersected);
    }

    /**
     * @param anchor1
     * @param anchor2
     * @return preservative intersection of anchor1 and anchor2
     */
    private static MotifAnchor preservativeIntersection(MotifAnchor anchor1, MotifAnchor anchor2, boolean conductFullIntersection) {
        if (anchor1.getChr() == anchor2.getChr()) {

            MotifAnchor intersectedMotif = (MotifAnchor) anchor1.deepClone();

            // if all secondary attributes are also to be copied
            if (conductFullIntersection) {
                if (anchor2.hasFIMOAttributes()) {
                    intersectedMotif.addFIMOAttributesFrom(anchor2);
                }

                intersectedMotif.addFeatureReferencesFrom(anchor2);
            }

            return intersectedMotif;
        } else {
            System.err.println("Error calculating preservative intersection of anchors");
            System.err.println(anchor1 + " & " + anchor2);
        }
        return null;
    }
}
