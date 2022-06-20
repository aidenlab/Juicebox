/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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

import juicebox.HiCGlobals;
import juicebox.data.feature.FeatureFilter;
import juicebox.data.feature.GenomeWideList;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.FeatureFunction;

import java.util.*;

/**
 * Created by muhammadsaadshamim on 9/28/15.
 */
public class MotifAnchorTools extends GenericLocusTools {

    /**
     * update the original features that the motifs belong to
     */
    public static void updateOriginalFeatures(GenomeWideList<MotifAnchor> anchorList, final boolean uniqueStatus,
                                              final int specificStatus) {
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> anchorList) {
                for (MotifAnchor anchor : anchorList) {
                    anchor.updateOriginalFeatures(uniqueStatus, specificStatus);
                }
            }
        });
    }

    /**
     * Guarantees that all anchors have minimum width of gapThreshold
     */
    private static void expandSmallAnchors(GenomeWideList<MotifAnchor> anchorList, final int gapThreshold) {
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> anchorList) {
                expandSmallAnchors(anchorList, gapThreshold);
            }
        });
    }

    /**
     * Guarantees that all anchors have minimum width of gapThreshold
     * PreProcessing step for anchors in MotifFinder code
     * derived from:
     * (awk on BED file) ... if($3-$2<15000){d=15000-($3-$2); print $1 \"\\t\" $2-int(d/2) \"\\t\" $3+int(d/2)
     *
     * @param anchors
     */
    private static void expandSmallAnchors(List<MotifAnchor> anchors, int gapThreshold) {
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
    public static GenomeWideList<MotifAnchor> extractUniqueMotifs(GenomeWideList<MotifAnchor> anchors, final int threshold) {

        GenomeWideList<MotifAnchor> uniqueAnchors = anchors.deepClone();
        uniqueAnchors.filterLists(new FeatureFilter<MotifAnchor>() {
            @Override
            public List<MotifAnchor> filter(String chr, List<MotifAnchor> anchorList) {

                // bin the motifs within resolution/threshold
                Map<String, List<MotifAnchor>> uniqueMapping = new HashMap<>();
                for (MotifAnchor motif : anchorList) {
                    String key = (motif.getX1() / threshold) + "_" + (motif.getX2() / threshold);
                    if (uniqueMapping.containsKey(key)) {
                        uniqueMapping.get(key).add(motif);
                    } else {
                        List<MotifAnchor> motifList = new ArrayList<>();
                        motifList.add(motif);
                        uniqueMapping.put(key, motifList);
                    }
                }

                // select for bins with only one value
                List<MotifAnchor> uniqueMotifs = new ArrayList<>();
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
    public static GenomeWideList<MotifAnchor> extractBestMotifs(GenomeWideList<MotifAnchor> anchors, final int threshold) {
        GenomeWideList<MotifAnchor> bestAnchors = anchors.deepClone();
        bestAnchors.filterLists(new FeatureFilter<MotifAnchor>() {
            @Override
            public List<MotifAnchor> filter(String chr, List<MotifAnchor> anchorList) {

                // bin the motifs within resolution/threshold, saving only the highest scoring motif
                Map<String, MotifAnchor> bestMapping = new HashMap<>();
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

                return new ArrayList<>(bestMapping.values());
            }
        });

        return bestAnchors;
    }

    public static MotifAnchor searchForFeature(final String chrID, final String sequence, GenomeWideList<MotifAnchor> anchorList) {
        final MotifAnchor[] anchor = new MotifAnchor[1];
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> featureList) {
                for (MotifAnchor motif : featureList) {
                    if (motif.getChr().equalsIgnoreCase(chrID) && motif.getSequence().equals(sequence)) {
                        anchor[0] = (MotifAnchor) motif.deepClone();
                    }
                }
            }
        });
        return anchor[0];
    }

    public static MotifAnchor searchForFeature(final String chrID, final int start, final int end, GenomeWideList<MotifAnchor> anchorList) {
        final MotifAnchor[] anchor = new MotifAnchor[1];
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> featureList) {
                for (MotifAnchor motif : featureList) {
                    if (motif.getChr().equalsIgnoreCase(chrID) && motif.getX1() == start && motif.getX2() == end) {
                        anchor[0] = (MotifAnchor) motif.deepClone();
                    }
                }
            }
        });
        return anchor[0];
    }

    public static MotifAnchor searchForFeatureWithin(final String chrID, final int start, final int end, GenomeWideList<MotifAnchor> anchorList) {
        final MotifAnchor[] anchor = new MotifAnchor[1];
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> featureList) {
                for (MotifAnchor motif : featureList) {
                    if (motif.getChr().equalsIgnoreCase(chrID) && motif.getX1() >= start && motif.getX2() <= end) {
                        anchor[0] = (MotifAnchor) motif.deepClone();
                    }
                }
            }
        });
        return anchor[0];
    }

    public static List<MotifAnchor> searchForFeaturesWithin(final String chrID, final int start, final int end, GenomeWideList<MotifAnchor> anchorList) {
        final List<MotifAnchor> anchors = new ArrayList<>();
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> featureList) {
                for (MotifAnchor motif : featureList) {
                    if (motif.getChr().equalsIgnoreCase(chrID) && motif.getX1() >= start && motif.getX2() <= end) {
                        anchors.add((MotifAnchor) motif.deepClone());
                    }
                }
            }
        });
        return anchors;
    }


    public static void retainProteinsInLocus(final GenomeWideList<GenericLocus> firstList, final GenomeWideList<GenericLocus> secondList,
                                             final boolean retainUniqueSites, final boolean copyFeatureReferences) {
        firstList.filterLists(new FeatureFilter<GenericLocus>() {
            @Override
            public List<GenericLocus> filter(String key, List<GenericLocus> anchorList) {
                if (secondList.containsKey(key)) {
                    return retainProteinsInLocus(anchorList, secondList.getFeatures(key), retainUniqueSites, copyFeatureReferences);
                } else {
                    return new ArrayList<>();
                }
            }
        });
    }

    private static List<GenericLocus> retainProteinsInLocus(List<GenericLocus> topAnchors, List<GenericLocus> baseList,
                                                            boolean retainUniqueSites, boolean copyFeatureReferences) {
        Map<GenericLocus, Set<GenericLocus>> bottomListToTopList = new HashMap<>();

        for (GenericLocus anchor : baseList) {
            bottomListToTopList.put(anchor, new HashSet<>());
        }

        int topIndex = 0;
        int bottomIndex = 0;
        int maxTopIndex = topAnchors.size();
        int maxBottomIndex = baseList.size();
        Collections.sort(topAnchors);
        Collections.sort(baseList);


        while (topIndex < maxTopIndex && bottomIndex < maxBottomIndex) {
            GenericLocus topAnchor = topAnchors.get(topIndex);
            GenericLocus bottomAnchor = baseList.get(bottomIndex);
            if (topAnchor.hasOverlapWith(bottomAnchor) || bottomAnchor.hasOverlapWith(topAnchor)) {

                bottomListToTopList.get(bottomAnchor).add(topAnchor);

                // iterate over all possible intersections with top element
                for (int i = bottomIndex; i < maxBottomIndex; i++) {
                    GenericLocus newAnchor = baseList.get(i);
                    if (topAnchor.hasOverlapWith(newAnchor) || newAnchor.hasOverlapWith(topAnchor)) {
                        bottomListToTopList.get(newAnchor).add(topAnchor);
                    } else {
                        break;
                    }
                }

                // iterate over all possible intersections with bottom element
                // start from +1 because +0 checked in the for loop above
                for (int i = topIndex + 1; i < maxTopIndex; i++) {
                    GenericLocus newAnchor = topAnchors.get(i);
                    if (bottomAnchor.hasOverlapWith(newAnchor) || newAnchor.hasOverlapWith(bottomAnchor)) {
                        bottomListToTopList.get(bottomAnchor).add(newAnchor);
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

        List<GenericLocus> uniqueAnchors = new ArrayList<>();

        if (copyFeatureReferences) {
            for (GenericLocus anchor : bottomListToTopList.keySet()) {
                for (GenericLocus anchor2 : bottomListToTopList.get(anchor)) {
                    anchor2.addFeatureReferencesFrom(anchor);
                }
            }
        }

        if (retainUniqueSites) {
            for (Set<GenericLocus> motifs : bottomListToTopList.values()) {
                if (motifs.size() == 1) {
                    uniqueAnchors.addAll(motifs);
                }
            }
        } else {
            for (Set<GenericLocus> motifs : bottomListToTopList.values()) {
                if (motifs.size() > 1) {
                    uniqueAnchors.addAll(motifs);
                }
            }
        }
        return uniqueAnchors;
    }

    public static void retainBestMotifsInLocus(final GenomeWideList<MotifAnchor> firstList, final GenomeWideList<GenericLocus> secondList) {
        firstList.filterLists(new FeatureFilter<MotifAnchor>() {
            @Override
            public List<MotifAnchor> filter(String key, List<MotifAnchor> anchorList) {
                if (secondList.containsKey(key)) {
                    return retainBestMotifsInLocus(anchorList, secondList.getFeatures(key));
                } else {
                    return new ArrayList<>();
                }
            }
        });
    }

    private static List<MotifAnchor> retainBestMotifsInLocus(List<MotifAnchor> topAnchors, List<GenericLocus> baseList) {
        Map<GenericLocus, Set<MotifAnchor>> bottomListToTopList = new HashMap<>();

        for (GenericLocus anchor : baseList) {
            bottomListToTopList.put(anchor, new HashSet<>());
        }

        int topIndex = 0;
        int bottomIndex = 0;
        int maxTopIndex = topAnchors.size();
        int maxBottomIndex = baseList.size();
        Collections.sort(topAnchors);
        Collections.sort(baseList);


        while (topIndex < maxTopIndex && bottomIndex < maxBottomIndex) {
            MotifAnchor topAnchor = topAnchors.get(topIndex);
            GenericLocus bottomAnchor = baseList.get(bottomIndex);
            if (topAnchor.hasOverlapWith(bottomAnchor) || bottomAnchor.hasOverlapWith(topAnchor)) {

                bottomListToTopList.get(bottomAnchor).add(topAnchor);

                // iterate over all possible intersections with top element
                for (int i = bottomIndex; i < maxBottomIndex; i++) {
                    GenericLocus newAnchor = baseList.get(i);
                    if (topAnchor.hasOverlapWith(newAnchor) || newAnchor.hasOverlapWith(topAnchor)) {
                        bottomListToTopList.get(newAnchor).add(topAnchor);
                    } else {
                        break;
                    }
                }

                // iterate over all possible intersections with bottom element
                // start from +1 because +0 checked in the for loop above
                for (int i = topIndex + 1; i < maxTopIndex; i++) {
                    MotifAnchor newAnchor = topAnchors.get(i);
                    if (bottomAnchor.hasOverlapWith(newAnchor) || newAnchor.hasOverlapWith(bottomAnchor)) {
                        bottomListToTopList.get(bottomAnchor).add(newAnchor);
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

        for (GenericLocus anchor : bottomListToTopList.keySet()) {
            for (MotifAnchor anchor2 : bottomListToTopList.get(anchor)) {
                anchor2.addFeatureReferencesFrom(anchor);
                if (HiCGlobals.printVerboseComments) {
                    if (anchor2.getSequence().equals("TGAGTCACTAGAGGGAGGCA")) {
                        System.out.println(bottomListToTopList.get(anchor));
                    }
                }
            }
        }

        List<MotifAnchor> uniqueAnchors = new ArrayList<>();
        for (Set<MotifAnchor> motifs : bottomListToTopList.values()) {
            if (motifs.size() == 1) {
                uniqueAnchors.addAll(motifs);
            } else if (motifs.size() > 1) {
                MotifAnchor best = motifs.iterator().next();
                for (MotifAnchor an : motifs) {
                    if (an.getScore() > best.getScore()) {
                        best = an;
                    }
                }
                uniqueAnchors.add(best);
            }
        }
        return uniqueAnchors;
    }

    public static int[] calculateConvergenceHistogram(Feature2DList features) {

        // ++, +- (convergent), -+ (divergent), --, other (incomplete)
        final int[] results = new int[6];

        features.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D feature : feature2DList) {
                    results[feature.toFeature2DWithMotif().getConvergenceStatus()]++;
                }
            }
        });

        return results;
    }

}
