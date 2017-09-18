/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

package juicebox.data.censoring;

import gnu.trove.procedure.TIntProcedure;
import juicebox.data.ChromosomeHandler;
import juicebox.data.anchor.MotifAnchor;
import juicebox.windowui.HiCZoom;
import net.sf.jsi.SpatialIndex;
import net.sf.jsi.rtree.RTree;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;

import java.util.*;

public class RegionsRTreeHandler {

    private final Map<Integer, SpatialIndex> regionsRtree = new HashMap<>();
    private final Map<Integer, Pair<List<MotifAnchor>, List<MotifAnchor>>> allRegionsForChr = new HashMap<>();
    private final List<Integer> boundariesOfCustomChromosomeX = new ArrayList<>();
    private final List<Integer> boundariesOfCustomChromosomeY = new ArrayList<>();

    private static Pair<List<MotifAnchor>, List<MotifAnchor>> getAllRegionsFromSubChromosomes(
            final ChromosomeHandler handler, Chromosome chr) {

        if (handler.isCustomChromosome(chr)) {
            final List<MotifAnchor> allRegions = new ArrayList<>();
            final List<MotifAnchor> translatedRegions = new ArrayList<>();

            handler.getListOfRegionsInCustomChromosome(chr.getIndex()).processLists(
                    new juicebox.data.feature.FeatureFunction<MotifAnchor>() {
                        @Override
                        public void process(String chr, List<MotifAnchor> featureList) {
                            for (MotifAnchor anchor : featureList) {
                                allRegions.add(anchor);
                            }
                        }
                    });
            Collections.sort(allRegions);

            int previousEnd = 0;
            for (MotifAnchor anchor : allRegions) {
                int currentEnd = previousEnd + anchor.getWidth();
                MotifAnchor anchor2 = new MotifAnchor(chr.getIndex(), previousEnd, currentEnd);
                translatedRegions.add(anchor2);
                previousEnd = currentEnd;
            }

            return new Pair<>(allRegions, translatedRegions);
        } else {
            // just a standard chromosome
            final List<MotifAnchor> allRegions = new ArrayList<>();
            final List<MotifAnchor> translatedRegions = new ArrayList<>();
            allRegions.add(new MotifAnchor(chr.getIndex(), 0, chr.getLength()));
            translatedRegions.add(new MotifAnchor(chr.getIndex(), 0, chr.getLength()));
            return new Pair<>(allRegions, translatedRegions);
        }
    }

    /**
     * @param handler
     */
    public void initializeRTree(Chromosome chr1, Chromosome chr2, HiCZoom zoom, ChromosomeHandler handler) {
        regionsRtree.clear();
        allRegionsForChr.clear();
        boundariesOfCustomChromosomeX.clear();
        boundariesOfCustomChromosomeY.clear();

        populateRTreeWithRegions(chr1, handler, boundariesOfCustomChromosomeX, zoom);
        if (chr1.getIndex() != chr2.getIndex()) {
            populateRTreeWithRegions(chr2, handler, boundariesOfCustomChromosomeY, zoom);
        } else {
            boundariesOfCustomChromosomeY.addAll(boundariesOfCustomChromosomeX);
        }
    }

    public List<Integer> getBoundariesOfCustomChromosomeX() {
        return boundariesOfCustomChromosomeX;
    }

    public List<Integer> getBoundariesOfCustomChromosomeY() {
        return boundariesOfCustomChromosomeY;
    }

    private void populateRTreeWithRegions(Chromosome chr, ChromosomeHandler handler, List<Integer> boundaries, HiCZoom zoom) {
        int chrIndex = chr.getIndex();
        Pair<List<MotifAnchor>, List<MotifAnchor>> allRegionsInfo = getAllRegionsFromSubChromosomes(handler, chr);

        if (allRegionsInfo != null) {
            allRegionsForChr.put(chrIndex, allRegionsInfo);
            SpatialIndex si = new RTree();
            si.init(null);
            List<MotifAnchor> translatedRegions = allRegionsInfo.getSecond();
            for (int i = 0; i < translatedRegions.size(); i++) {
                MotifAnchor anchor = translatedRegions.get(i);
                boundaries.add(anchor.getX2() / zoom.getBinSize());
                si.add(new net.sf.jsi.Rectangle((float) anchor.getX1(), (float) anchor.getX1(),
                        (float) anchor.getX2(), (float) anchor.getX2()), i);
            }
            regionsRtree.put(chrIndex, si);
        }
    }

    public List<Pair<MotifAnchor, MotifAnchor>> getIntersectingFeatures(final int chrIdx, net.sf.jsi.Rectangle selectionWindow) {
        final List<Pair<MotifAnchor, MotifAnchor>> foundFeatures = new ArrayList<>();

        if (allRegionsForChr.containsKey(chrIdx) && regionsRtree.containsKey(chrIdx)) {
            try {
                regionsRtree.get(chrIdx).intersects(
                        selectionWindow,
                        new TIntProcedure() {     // a procedure whose execute() method will be called with the results
                            public boolean execute(int i) {
                                MotifAnchor anchor = allRegionsForChr.get(chrIdx).getFirst().get(i);
                                MotifAnchor anchor2 = allRegionsForChr.get(chrIdx).getSecond().get(i);
                                foundFeatures.add(new Pair<>(anchor, anchor2));
                                return true;      // return true here to continue receiving results
                            }
                        }
                );
            } catch (Exception e) {
                System.err.println("Error encountered getting intersecting anchors for custom chr " + e.getLocalizedMessage());
            }
        }
        return foundFeatures;
    }

}
