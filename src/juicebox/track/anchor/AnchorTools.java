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
        MotifAnchor current = anchors.get(0).deepClone();

        for (MotifAnchor anchor : anchors) {
            if (anchor.hasOverlapWith(current)) {
                current.mergeWith(anchor);
            } else {
                merged.add(current);
                current = anchor.deepClone();
            }
        }
        merged.add(current); // in case last merger missed (i.e. boolean evaluated to true)

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
    public static List<MotifAnchor> intersect(List<MotifAnchor> topAnchors, List<MotifAnchor> bottomAnchors) {
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
            if (topAnchor.hasOverlapWith(bottomAnchor)) {
                // iterate over all possible intersections with top element
                for (int i = bottomIndex; i < maxBottomIndex; i++) {
                    MotifAnchor newAnchor = bottomAnchors.get(i);
                    if (topAnchor.hasOverlapWith(newAnchor)) {
                        intersected.add(intersection(topAnchor, newAnchor));
                    } else {
                        break;
                    }
                }

                // iterate over all possible intersections with bottom element
                // start from +1 because +0 checked in the for loop above
                for (int i = topIndex + 1; i < maxTopIndex; i++) {
                    MotifAnchor newAnchor = topAnchors.get(i);
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

        return new ArrayList<MotifAnchor>(intersected);
    }

    /**
     * @param anchor1
     * @param anchor2
     * @return intersection of anchor1 and anchor2
     */
    private static MotifAnchor intersection(MotifAnchor anchor1, MotifAnchor anchor2) {
        if (anchor1.getChr().equals(anchor2.getChr())) {
            return new MotifAnchor(anchor1.getChr(), Math.max(anchor1.getX1(), anchor2.getX1()),
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
    public static void expandSmallAnchors(List<MotifAnchor> anchors, int gapThreshold) {
        for (MotifAnchor anchor : anchors) {
            int width = anchor.getWidth();
            if (width < gapThreshold) {
                anchor.widenMargins(gapThreshold - width);
            }
        }
    }

    /**
     * @param anchors
     * @param threshold
     * @return unique motifs within a given threshold from a given AnchorList
     */
    public static AnchorList extractUniqueMotifs(AnchorList anchors, final int threshold) {

        AnchorList uniqueAnchors = anchors.deepClone();
        uniqueAnchors.filterLists(new AnchorFilter() {
            @Override
            public List<MotifAnchor> filter(String chr, List<MotifAnchor> anchorList) {

                // bin the motifs within resolution/threshold
                Map<String, List<MotifAnchor>> uniqueMapping = new HashMap<String, List<MotifAnchor>>();
                for (MotifAnchor motif : anchorList) {
                    String key = (motif.getX1() / threshold) + "_" + (motif.getX2() / threshold);
                    if (uniqueMapping.containsKey(key)) {
                        uniqueMapping.get(key).add(motif);
                    } else {
                        List<MotifAnchor> motifList = new ArrayList<MotifAnchor>();
                        motifList.add(motif);
                        uniqueMapping.put(key, motifList);
                    }
                }

                // select for bins with only one value
                List<MotifAnchor> uniqueMotifs = new ArrayList<MotifAnchor>();
                for (List<MotifAnchor> motifList : uniqueMapping.values()) {
                    if (motifList.size() == 1) {
                        uniqueMotifs.add(motifList.get(0));
                    }
                }

                return uniqueMotifs;
            }
        });

        return uniqueAnchors;
    }

    /**
     * @param anchors
     * @param threshold
     * @return best (highest scoring) motifs within a given threshold from a given anchors list
     */
    public static AnchorList extractBestMotifs(AnchorList anchors, final int threshold) {
        AnchorList bestAnchors = anchors.deepClone();
        bestAnchors.filterLists(new AnchorFilter() {
            @Override
            public List<MotifAnchor> filter(String chr, List<MotifAnchor> anchorList) {

                // bin the motifs within resolution/threshold, saving only the highest scoring motif
                Map<String, MotifAnchor> bestMapping = new HashMap<String, MotifAnchor>();
                for (MotifAnchor motif : anchorList) {
                    String key = (motif.getX1() / threshold) + "_" + (motif.getX2() / threshold);
                    if (bestMapping.containsKey(key)) {
                        if (bestMapping.get(key).getScore() < motif.getScore()) {
                            bestMapping.put(key, motif);
                        }
                    } else {
                        bestMapping.put(key, motif);
                    }
                }

                return new ArrayList<MotifAnchor>(bestMapping.values());
            }
        });

        return bestAnchors;
    }
}
