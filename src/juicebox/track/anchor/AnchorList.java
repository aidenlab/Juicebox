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


import juicebox.data.HiCFileTools;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DWithMotif;
import juicebox.track.feature.FeatureFunction;
import org.broad.igv.feature.Chromosome;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 9/29/15.
 */
public class AnchorList {

    private Map<String, List<MotifAnchor>> anchorLists = new HashMap<String, List<MotifAnchor>>();

    /**
     * Private constructor only used for cloning
     */
    private AnchorList() {
    }

    /**
     * @param chromosomes for genome
     */
    public AnchorList(List<Chromosome> chromosomes) {
        for (Chromosome chr : chromosomes) {
            anchorLists.put(chr.getName(), new ArrayList<MotifAnchor>());
        }
    }

    /**
     * @param genomeID
     */
    public AnchorList(String genomeID) {
        this(HiCFileTools.loadChromosomes(genomeID));
    }

    /**
     * @param chromosomes for genome
     * @param anchors     to be added to list
     */
    public AnchorList(List<Chromosome> chromosomes, List<MotifAnchor> anchors) {
        this(chromosomes);
        addAll(anchors);
    }

    /**
     * @param genomeID
     * @param anchors  to be added to list
     */
    public AnchorList(String genomeID, List<MotifAnchor> anchors) {
        this(HiCFileTools.loadChromosomes(genomeID), anchors);
    }

    /**
     * @param features
     * @return anchor list from features (i.e. split anchor1 and anchor2)
     */
    public static AnchorList extractAnchorsFromFeatures(Feature2DList features, final boolean onlyUninitializedFeatures) {

        final AnchorList extractedAnchorList = new AnchorList();

        features.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                List<MotifAnchor> anchors = new ArrayList<MotifAnchor>();
                for (Feature2D f : feature2DList) {
                    anchors.addAll(((Feature2DWithMotif) f).getAnchors(onlyUninitializedFeatures));
                }
                String newKey = chr.split("_")[0];
                extractedAnchorList.anchorLists.put(newKey, anchors);
            }
        });

        return extractedAnchorList;
    }

    /**
     * @param chr
     * @return motifs for correspoding chromosome
     */
    private List<MotifAnchor> getAnchors(String chr) {
        return anchorLists.get(chr);
    }

    /**
     * @param anchors to be added to this list
     */
    private void addAll(List<MotifAnchor> anchors) {
        for (MotifAnchor anchor : anchors) {
            anchorLists.get(anchor.getChr()).add(anchor);
        }
    }

    /**
     * pass interface implementing a filter for anchors
     *
     * @param filter
     */
    public void filterLists(AnchorFilter filter) {
        for (String chr : anchorLists.keySet()) {
            anchorLists.put(chr, filter.filter(chr, anchorLists.get(chr)));
        }
    }

    /**
     * pass interface implementing a process for all anchors
     * TODO - alter above functions to use this
     *
     * @param function
     */
    public void processLists(AnchorFunction function) {
        for (String chr : anchorLists.keySet()) {
            function.process(chr, anchorLists.get(chr));
        }
    }

    /**
     * Merge anchors which have overlap
     */
    public void merge() {
        filterLists(new AnchorFilter() {
            @Override
            public List<MotifAnchor> filter(String chr, List<MotifAnchor> anchorList) {
                return AnchorTools.merge(anchorList);
            }
        });
    }

    /**
     * Merge anchors which have overlap
     */
    public void intersectWith(final AnchorList secondList, final boolean conductFullIntersection) {
        filterLists(new AnchorFilter() {
            @Override
            public List<MotifAnchor> filter(String chr, List<MotifAnchor> anchorList) {
                return AnchorTools.intersect(anchorList, secondList.getAnchors(chr), conductFullIntersection);
            }
        });
    }

    /**
     * Expand anchors which are too small
     */
    public void expandSmallAnchors(final int threshold) {
        processLists(new AnchorFunction() {
            @Override
            public void process(String chr, List<MotifAnchor> anchorList) {
                AnchorTools.expandSmallAnchors(anchorList, threshold);
            }
        });
    }

    public void updateOriginalMotifs(final boolean uniqueStatus) {
        processLists(new AnchorFunction() {
            @Override
            public void process(String chr, List<MotifAnchor> anchorList) {
                for (MotifAnchor motifAnchor : anchorList) {
                    motifAnchor.updateOriginalMotifs(uniqueStatus);
                }
            }
        });
    }

    /**
     * @return deep copy of the anchor list
     */
    public AnchorList deepClone() {
        AnchorList clone = new AnchorList();
        for (String key : anchorLists.keySet()) {
            clone.anchorLists.put(key, cloneMotifList(anchorLists.get(key)));
        }
        return clone;
    }

    /**
     * @param motifs
     * @return deep copy of the list of motifs
     */
    private List<MotifAnchor> cloneMotifList(List<MotifAnchor> motifs) {
        List<MotifAnchor> clonedMotifs = new ArrayList<MotifAnchor>();
        for (MotifAnchor anchor : motifs) {
            clonedMotifs.add(anchor.deepClone());
        }
        return clonedMotifs;
    }

    /**
     * @return total number of anchors
     */
    public int size() {
        int size = 0;
        for (List<MotifAnchor> anchors : anchorLists.values()) {
            size += anchors.size();

        }
        return size;
    }
}
