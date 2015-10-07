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
import org.broad.igv.feature.Chromosome;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 9/29/15.
 */
public class AnchorList {

    private Map<String, List<FeatureAnchor>> anchorLists = new HashMap<String, List<FeatureAnchor>>();

    /**
     * @param chromosomes for genome
     */
    public AnchorList(List<Chromosome> chromosomes) {
        for (Chromosome chr : chromosomes) {
            anchorLists.put(chr.getName(), new ArrayList<FeatureAnchor>());
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
    public AnchorList(List<Chromosome> chromosomes, List<FeatureAnchor> anchors) {
        this(chromosomes);
        addAll(anchors);
    }

    /**
     * @param genomeID
     * @param anchors  to be added to list
     */
    public AnchorList(String genomeID, List<FeatureAnchor> anchors) {
        this(HiCFileTools.loadChromosomes(genomeID), anchors);
    }

    /**
     * @param anchors to be added to this list
     */
    private void addAll(List<FeatureAnchor> anchors) {
        for (FeatureAnchor anchor : anchors) {
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
            public List<FeatureAnchor> filter(String chr, List<FeatureAnchor> anchorList) {
                return AnchorTools.merge(anchorList);
            }
        });
    }

    /**
     * Expand anchors which are too small
     */
    public void expandSmallAnchors() {
        processLists(new AnchorFunction() {
            @Override
            public void process(String chr, List<FeatureAnchor> anchorList) {
                AnchorTools.expandSmallAnchors(anchorList);
            }
        });
    }
}
