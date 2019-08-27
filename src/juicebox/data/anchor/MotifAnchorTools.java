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

package juicebox.data.anchor;

import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.feature.FeatureFilter;
import juicebox.data.feature.GenomeWideList;
import juicebox.gui.SuperAdapter;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DWithMotif;
import juicebox.track.feature.FeatureFunction;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.ui.util.MessageUtils;

import java.util.*;

/**
 * Created by muhammadsaadshamim on 9/28/15.
 */
public class MotifAnchorTools {

    /**
     * @param features
     * @return anchor list from features (i.e. split anchor1 and anchor2)
     */
    public static GenomeWideList<MotifAnchor> extractAnchorsFromIntrachromosomalFeatures(Feature2DList features,
                                                                                         final boolean onlyUninitializedFeatures,
                                                                                         final ChromosomeHandler handler) {

        final GenomeWideList<MotifAnchor> extractedAnchorList = new GenomeWideList<>(handler);
        features.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                List<MotifAnchor> anchors = new ArrayList<>();
                for (Feature2D f : feature2DList) {
                    anchors.addAll(f.getAnchors(onlyUninitializedFeatures, handler));
                }
                String newKey = chr.split("_")[0].replace("chr", "");
                extractedAnchorList.setFeatures(newKey, anchors);
            }
        });

        MotifAnchorTools.mergeAnchors(extractedAnchorList);
        MotifAnchorTools.expandSmallAnchors(extractedAnchorList, 15000);

        return extractedAnchorList;
    }

    public static GenomeWideList<MotifAnchor> extractAllAnchorsFromAllFeatures(Feature2DList features, final ChromosomeHandler handler) {

        final GenomeWideList<MotifAnchor> extractedAnchorList = new GenomeWideList<>(handler);
        features.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D f : feature2DList) {
                    Chromosome chrom = handler.getChromosomeFromName((f.getChr1()));
                    extractedAnchorList.addFeature(chrom.getName(), new MotifAnchor(chrom.getIndex(), f.getStart1(), f.getEnd1()));
                    chrom = handler.getChromosomeFromName((f.getChr2()));
                    extractedAnchorList.addFeature(chrom.getName(), new MotifAnchor(chrom.getIndex(), f.getStart2(), f.getEnd2()));
                }
            }
        });

        mergeAndExpandSmallAnchors(extractedAnchorList, getMinSizeForExpansionFromGUI());

        return extractedAnchorList;
    }

    public static int getMinSizeForExpansionFromGUI() {
        int minSize = 10000;
        String newSize = MessageUtils.showInputDialog("Specify a minimum size for 1D anchors", "" + minSize);
        try {
            minSize = Integer.parseInt(newSize);
        } catch (Exception e) {
            if (HiCGlobals.guiIsCurrentlyActive) {
                SuperAdapter.showMessageDialog("Invalid integer, using default size " + minSize);
            } else {
                MessageUtils.showMessage("Invalid integer, using default size " + minSize);
            }
        }
        return minSize;
    }

    /**
     * Merge anchors which have overlap
     */
    private static void mergeAnchors(GenomeWideList<MotifAnchor> anchorList) {
        anchorList.filterLists(new FeatureFilter<MotifAnchor>() {
            @Override
            public List<MotifAnchor> filter(String chr, List<MotifAnchor> anchorList) {
                return BEDTools.merge(anchorList);
            }
        });
    }

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
     * Merge anchors which have overlap
     */
    public static void intersectLists(final GenomeWideList<MotifAnchor> firstList, final GenomeWideList<MotifAnchor> secondList,
                                      final boolean conductFullIntersection) {
        firstList.filterLists(new FeatureFilter<MotifAnchor>() {
            @Override
            public List<MotifAnchor> filter(String key, List<MotifAnchor> anchorList) {
                if (secondList.containsKey(key)) {
                    return BEDTools.intersect(anchorList, secondList.getFeatures(key), conductFullIntersection);
                } else {
                    return new ArrayList<>();
                }
            }
        });
    }

    public static void preservativeIntersectLists(final GenomeWideList<MotifAnchor> firstList, final GenomeWideList<MotifAnchor> secondList,
                                                  final boolean conductFullIntersection) {
        firstList.filterLists(new FeatureFilter<MotifAnchor>() {
            @Override
            public List<MotifAnchor> filter(String key, List<MotifAnchor> anchorList) {
                if (secondList.containsKey(key)) {
                    return BEDTools.preservativeIntersect(anchorList, secondList.getFeatures(key), conductFullIntersection);
                } else {
                    return new ArrayList<>();
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

    public static MotifAnchor searchForFeature(final int chrID, final String sequence, GenomeWideList<MotifAnchor> anchorList) {
        final MotifAnchor[] anchor = new MotifAnchor[1];
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> featureList) {
                for (MotifAnchor motif : featureList) {
                    if (motif.getChr() == chrID && motif.getSequence().equals(sequence)) {
                        anchor[0] = (MotifAnchor) motif.deepClone();
                    }
                }
            }
        });
        return anchor[0];
    }

    public static MotifAnchor searchForFeature(final int chrID, final int start, final int end, GenomeWideList<MotifAnchor> anchorList) {
        final MotifAnchor[] anchor = new MotifAnchor[1];
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> featureList) {
                for (MotifAnchor motif : featureList) {
                    if (motif.getChr() == chrID && motif.getX1() == start && motif.getX2() == end) {
                        anchor[0] = (MotifAnchor) motif.deepClone();
                    }
                }
            }
        });
        return anchor[0];
    }

    public static MotifAnchor searchForFeatureWithin(final int chrID, final int start, final int end, GenomeWideList<MotifAnchor> anchorList) {
        final MotifAnchor[] anchor = new MotifAnchor[1];
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> featureList) {
                for (MotifAnchor motif : featureList) {
                    if (motif.getChr() == chrID && motif.getX1() >= start && motif.getX2() <= end) {
                        anchor[0] = (MotifAnchor) motif.deepClone();
                    }
                }
            }
        });
        return anchor[0];
    }

    public static List<MotifAnchor> searchForFeaturesWithin(final int chrID, final int start, final int end, GenomeWideList<MotifAnchor> anchorList) {
        final List<MotifAnchor> anchors = new ArrayList<>();
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> featureList) {
                for (MotifAnchor motif : featureList) {
                    if (motif.getChr() == chrID && motif.getX1() >= start && motif.getX2() <= end) {
                        anchors.add((MotifAnchor) motif.deepClone());
                    }
                }
            }
        });
        return anchors;
    }


    public static void retainProteinsInLocus(final GenomeWideList<MotifAnchor> firstList, final GenomeWideList<MotifAnchor> secondList,
                                             final boolean retainUniqueSites, final boolean copyFeatureReferences) {
        firstList.filterLists(new FeatureFilter<MotifAnchor>() {
            @Override
            public List<MotifAnchor> filter(String key, List<MotifAnchor> anchorList) {
                if (secondList.containsKey(key)) {
                    return retainProteinsInLocus(anchorList, secondList.getFeatures(key), retainUniqueSites, copyFeatureReferences);
                } else {
                    return new ArrayList<>();
                }
            }
        });
    }

    private static List<MotifAnchor> retainProteinsInLocus(List<MotifAnchor> topAnchors, List<MotifAnchor> baseList,
                                                           boolean retainUniqueSites, boolean copyFeatureReferences) {
        Map<MotifAnchor, Set<MotifAnchor>> bottomListToTopList = new HashMap<>();

        for (MotifAnchor anchor : baseList) {
            bottomListToTopList.put(anchor, new HashSet<MotifAnchor>());
        }

        int topIndex = 0;
        int bottomIndex = 0;
        int maxTopIndex = topAnchors.size();
        int maxBottomIndex = baseList.size();
        Collections.sort(topAnchors);
        Collections.sort(baseList);


        while (topIndex < maxTopIndex && bottomIndex < maxBottomIndex) {
            MotifAnchor topAnchor = topAnchors.get(topIndex);
            MotifAnchor bottomAnchor = baseList.get(bottomIndex);
            if (topAnchor.hasOverlapWith(bottomAnchor) || bottomAnchor.hasOverlapWith(topAnchor)) {

                bottomListToTopList.get(bottomAnchor).add(topAnchor);

                // iterate over all possible intersections with top element
                for (int i = bottomIndex; i < maxBottomIndex; i++) {
                    MotifAnchor newAnchor = baseList.get(i);
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

        List<MotifAnchor> uniqueAnchors = new ArrayList<>();

        if (copyFeatureReferences) {
            for (MotifAnchor anchor : bottomListToTopList.keySet()) {
                for (MotifAnchor anchor2 : bottomListToTopList.get(anchor)) {
                    anchor2.addFeatureReferencesFrom(anchor);
                }
            }
        }

        if (retainUniqueSites) {
            for (Set<MotifAnchor> motifs : bottomListToTopList.values()) {
                if (motifs.size() == 1) {
                    uniqueAnchors.addAll(motifs);
                }
            }
        } else {
            for (Set<MotifAnchor> motifs : bottomListToTopList.values()) {
                if (motifs.size() > 1) {
                    uniqueAnchors.addAll(motifs);
                }
            }
        }
        return uniqueAnchors;
    }

    // true --> upstream
    public static GenomeWideList<MotifAnchor> extractDirectionalAnchors(GenomeWideList<MotifAnchor> featureAnchors,
                                                                        final boolean direction) {
        final GenomeWideList<MotifAnchor> directionalAnchors = new GenomeWideList<>();
        featureAnchors.processLists(new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> featureList) {
                for (MotifAnchor anchor : featureList) {
                    if (anchor.isDirectionalAnchor(direction)) {
                        directionalAnchors.addFeature(chr, anchor);
                    }
                }
            }
        });

        return directionalAnchors;
    }

    public static void retainBestMotifsInLocus(final GenomeWideList<MotifAnchor> firstList, final GenomeWideList<MotifAnchor> secondList) {
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

    private static List<MotifAnchor> retainBestMotifsInLocus(List<MotifAnchor> topAnchors, List<MotifAnchor> baseList) {
        Map<MotifAnchor, Set<MotifAnchor>> bottomListToTopList = new HashMap<>();

        for (MotifAnchor anchor : baseList) {
            bottomListToTopList.put(anchor, new HashSet<MotifAnchor>());
        }

        int topIndex = 0;
        int bottomIndex = 0;
        int maxTopIndex = topAnchors.size();
        int maxBottomIndex = baseList.size();
        Collections.sort(topAnchors);
        Collections.sort(baseList);


        while (topIndex < maxTopIndex && bottomIndex < maxBottomIndex) {
            MotifAnchor topAnchor = topAnchors.get(topIndex);
            MotifAnchor bottomAnchor = baseList.get(bottomIndex);
            if (topAnchor.hasOverlapWith(bottomAnchor) || bottomAnchor.hasOverlapWith(topAnchor)) {

                bottomListToTopList.get(bottomAnchor).add(topAnchor);

                // iterate over all possible intersections with top element
                for (int i = bottomIndex; i < maxBottomIndex; i++) {
                    MotifAnchor newAnchor = baseList.get(i);
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

        for (MotifAnchor anchor : bottomListToTopList.keySet()) {
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
        final int[] results = new int[5];

        features.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D feature : feature2DList) {
                    Feature2DWithMotif feature2DWithMotif = (Feature2DWithMotif) feature;
                    results[feature2DWithMotif.getConvergenceStatus()]++;
                }
            }
        });

        return results;
    }

    public static void mergeAndExpandSmallAnchors(GenomeWideList<MotifAnchor> regionsInCustomChromosome, int minSize) {
        MotifAnchorTools.mergeAnchors(regionsInCustomChromosome);
        MotifAnchorTools.expandSmallAnchors(regionsInCustomChromosome, minSize);
        MotifAnchorTools.mergeAnchors(regionsInCustomChromosome);
    }
}
