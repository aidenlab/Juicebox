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

import juicebox.data.feature.FeatureFilter;
import juicebox.data.feature.GenomeWideList;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DWithMotif;
import juicebox.track.feature.FeatureFunction;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 9/28/15.
 */
public class MotifAnchorTools {

    /**
     * @param features
     * @return anchor list from features (i.e. split anchor1 and anchor2)
     */
    public static GenomeWideList<MotifAnchor> extractAnchorsFromFeatures(Feature2DList features,
                                                                         final boolean onlyUninitializedFeatures) {

        final GenomeWideList<MotifAnchor> extractedAnchorList = new GenomeWideList<MotifAnchor>();
        features.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                List<MotifAnchor> anchors = new ArrayList<MotifAnchor>();
                for (Feature2D f : feature2DList) {
                    anchors.addAll(((Feature2DWithMotif) f).getAnchors(onlyUninitializedFeatures));
                }
                String newKey = chr.split("_")[0];
                extractedAnchorList.setFeatures(newKey, anchors);
            }
        });

        return extractedAnchorList;
    }

    /**
     * Merge anchors which have overlap
     */
    public static void mergeAnchors(GenomeWideList<MotifAnchor> anchorList) {
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
    public static void updateOriginalFeatures(GenomeWideList<MotifAnchor> anchorList, final boolean uniqueStatus) {
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> anchorList) {
                for (MotifAnchor anchor : anchorList) {
                    anchor.updateOriginalFeatures(uniqueStatus);
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
                    return new ArrayList<MotifAnchor>();
                }
            }
        });
    }


    /**
     * Guarantees that all anchors have minimum width of gapThreshold
     */
    public static void expandSmallAnchors(GenomeWideList<MotifAnchor> anchorList, final int gapThreshold) {
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
    public static GenomeWideList<MotifAnchor> extractUniqueMotifs(GenomeWideList<MotifAnchor> anchors, final int threshold) {

        GenomeWideList<MotifAnchor> uniqueAnchors = anchors.deepClone();
        uniqueAnchors.filterLists(new FeatureFilter<MotifAnchor>() {
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
    public static GenomeWideList<MotifAnchor> extractBestMotifs(GenomeWideList<MotifAnchor> anchors, final int threshold) {
        GenomeWideList<MotifAnchor> bestAnchors = anchors.deepClone();
        bestAnchors.filterLists(new FeatureFilter<MotifAnchor>() {
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

    public static MotifAnchor searchForFeature(final int chrID, final String sequence, GenomeWideList<MotifAnchor> anchorList) {
        final MotifAnchor[] anchor = new MotifAnchor[1];
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
            @Override
            public void process(String chr, List<MotifAnchor> featureList) {
                for (MotifAnchor motif : featureList) {
                    if (motif.getChr().contains("" + chrID) && motif.getSequence().equals(sequence)) {
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
                    if (motif.getChr().contains("" + chrID) && motif.getX1() == start && motif.getX2() == end) {
                        anchor[0] = (MotifAnchor) motif.deepClone();
                    }
                }
            }
        });
        return anchor[0];
    }
}
