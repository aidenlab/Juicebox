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
import juicebox.data.ChromosomeHandler;
import juicebox.data.basics.Chromosome;
import juicebox.data.feature.FeatureFilter;
import juicebox.data.feature.GenomeWideList;
import juicebox.gui.SuperAdapter;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.FeatureFunction;
import org.broad.igv.ui.util.MessageUtils;

import java.util.ArrayList;
import java.util.List;

public class GenericLocusTools {

    /**
     * @param features
     * @return anchor list from features (i.e. split anchor1 and anchor2)
     */
    public static GenomeWideList<GenericLocus> extractAnchorsFromIntrachromosomalFeatures(Feature2DList features,
                                                                                          final boolean onlyUninitializedFeatures,
                                                                                          final ChromosomeHandler handler, int expandSize) {

        final GenomeWideList<GenericLocus> extractedAnchorList = new GenomeWideList<>(handler);
        features.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                List<GenericLocus> anchors = new ArrayList<>();
                for (Feature2D f : feature2DList) {
                    anchors.addAll(f.getAnchors(onlyUninitializedFeatures, handler));
                }
                String newKey = chr.split("_")[0].replace("chr", "");
                extractedAnchorList.setFeatures(newKey, anchors);
            }
        });

        mergeAnchors(extractedAnchorList);
        expandSmallAnchors(extractedAnchorList, expandSize);

        return extractedAnchorList;
    }

    public static GenomeWideList<GenericLocus> extractAllAnchorsFromAllFeatures(Feature2DList features, final ChromosomeHandler handler) {

        final GenomeWideList<GenericLocus> extractedAnchorList = new GenomeWideList<>(handler);
        features.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D f : feature2DList) {
                    Chromosome chrom = handler.getChromosomeFromName((f.getChr1()));
                    extractedAnchorList.addFeature(chrom.getName(), new GenericLocus(chrom.getName(), f.getStart1(), f.getEnd1()));
                    chrom = handler.getChromosomeFromName((f.getChr2()));
                    extractedAnchorList.addFeature(chrom.getName(), new GenericLocus(chrom.getName(), f.getStart2(), f.getEnd2()));
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
    private static void mergeAnchors(GenomeWideList<GenericLocus> anchorList) {
        anchorList.filterLists(new FeatureFilter<GenericLocus>() {
            @Override
            public List<GenericLocus> filter(String chr, List<GenericLocus> anchorList) {
                return BEDTools.merge(anchorList);
            }
        });
    }

    private static void mergeAnchorsTakeSmaller(GenomeWideList<GenericLocus> anchorList) {
        anchorList.filterLists(new FeatureFilter<GenericLocus>() {
            @Override
            public List<GenericLocus> filter(String chr, List<GenericLocus> anchorList) {
                return BEDTools.mergeTakeSmaller(anchorList);
            }
        });
    }

    /**
     * update the original features that the motifs belong to
     */
    public static void updateOriginalFeatures(GenomeWideList<GenericLocus> anchorList, String prefix) {
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<GenericLocus>() {
            @Override
            public void process(String chr, List<GenericLocus> anchorList) {
                for (GenericLocus anchor : anchorList) {
                    anchor.updateOriginalFeatures(prefix);
                }
            }
        });
    }

    /**
     * Merge anchors which have overlap
     */
    public static void intersectLists(final GenomeWideList<GenericLocus> firstList, final GenomeWideList<GenericLocus> secondList,
                                      final boolean conductFullIntersection) {
        firstList.filterLists(new FeatureFilter<GenericLocus>() {
            @Override
            public List<GenericLocus> filter(String key, List<GenericLocus> anchorList) {
                if (secondList.containsKey(key)) {
                    return BEDTools.intersect(anchorList, secondList.getFeatures(key), conductFullIntersection);
                } else {
                    return new ArrayList<>();
                }
            }
        });
    }

    public static void preservativeIntersectLists(final GenomeWideList<GenericLocus> firstList, final GenomeWideList<GenericLocus> secondList,
                                                  final boolean conductFullIntersection) {
        firstList.filterLists(new FeatureFilter<GenericLocus>() {
            @Override
            public List<GenericLocus> filter(String key, List<GenericLocus> anchorList) {
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
    private static void expandSmallAnchors(GenomeWideList<GenericLocus> anchorList, final int gapThreshold) {
        anchorList.processLists(new juicebox.data.feature.FeatureFunction<GenericLocus>() {
            @Override
            public void process(String chr, List<GenericLocus> anchorList) {
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
    private static void expandSmallAnchors(List<GenericLocus> anchors, int gapThreshold) {
        for (GenericLocus anchor : anchors) {
            int width = anchor.getWidth();
            if (width < gapThreshold) {
                anchor.widenMargins(gapThreshold - width);
            }
        }
    }

    // true --> upstream
    public static GenomeWideList<GenericLocus> extractDirectionalAnchors(GenomeWideList<GenericLocus> featureAnchors,
                                                                         final boolean direction) {
        final GenomeWideList<GenericLocus> directionalAnchors = new GenomeWideList<>();
        featureAnchors.processLists(new juicebox.data.feature.FeatureFunction<GenericLocus>() {
            @Override
            public void process(String chr, List<GenericLocus> featureList) {
                for (GenericLocus anchor : featureList) {
                    if (anchor.isDirectionalAnchor(direction)) {
                        directionalAnchors.addFeature(chr, anchor);
                    }
                }
            }
        });

        return directionalAnchors;
    }

    public static void mergeAndExpandSmallAnchors(GenomeWideList<GenericLocus> regionsInCustomChromosome, int minSize) {
        mergeAnchors(regionsInCustomChromosome);
        expandSmallAnchors(regionsInCustomChromosome, minSize);
        mergeAnchors(regionsInCustomChromosome);
    }

    public static void callMergeAnchors(GenomeWideList<GenericLocus> locusList) {
        mergeAnchorsTakeSmaller(locusList);
    }
}
