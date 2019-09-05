/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

import juicebox.data.ChromosomeHandler;
import juicebox.data.anchor.MotifAnchor;
import juicebox.windowui.HiCZoom;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;

import java.util.*;

public class CustomMZDRegionHandler {

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
                            allRegions.addAll(featureList);
                        }
                    });
            Collections.sort(allRegions);

            int previousEnd = 0;
            for (MotifAnchor anchor : allRegions) {
                int currentEnd = previousEnd + anchor.getWidth();
                MotifAnchor anchor2 = new MotifAnchor(chr.getIndex(), previousEnd, currentEnd);
                translatedRegions.add(anchor2);
                previousEnd = currentEnd + ChromosomeHandler.CUSTOM_CHROMOSOME_BUFFER;
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
    public void initialize(Chromosome chr1, Chromosome chr2, HiCZoom zoom, ChromosomeHandler handler) {
        allRegionsForChr.clear();
        boundariesOfCustomChromosomeX.clear();
        boundariesOfCustomChromosomeY.clear();

        populateRegions(chr1, handler, boundariesOfCustomChromosomeX, zoom);
        if (chr1.getIndex() != chr2.getIndex()) {
            populateRegions(chr2, handler, boundariesOfCustomChromosomeY, zoom);
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

    private void populateRegions(Chromosome chr, ChromosomeHandler handler, List<Integer> boundaries, HiCZoom zoom) {
        int chrIndex = chr.getIndex();
        Pair<List<MotifAnchor>, List<MotifAnchor>> allRegionsInfo = getAllRegionsFromSubChromosomes(handler, chr);

        if (allRegionsInfo != null) {
            allRegionsForChr.put(chrIndex, allRegionsInfo);
            List<MotifAnchor> translatedRegions = allRegionsInfo.getSecond();
            for (MotifAnchor anchor : translatedRegions) {
                boundaries.add(anchor.getX2() / zoom.getBinSize());
            }
        }
    }

    public List<Pair<MotifAnchor, MotifAnchor>> getIntersectingFeatures(int index, int gx1, int gx2) {

        int idx1 = OneDimSearchUtils.indexedBinaryNearestSearch(
                allRegionsForChr.get(index).getSecond(), new MotifAnchor(index, gx1, gx1), true);
        int idx2 = OneDimSearchUtils.indexedBinaryNearestSearch(
                allRegionsForChr.get(index).getSecond(), new MotifAnchor(index, gx2, gx2), false);

        final List<Pair<MotifAnchor, MotifAnchor>> foundFeatures = new ArrayList<>();
        for (int i = idx1; i <= idx2; i++) {
            foundFeatures.add(new Pair<>(
                    allRegionsForChr.get(index).getFirst().get(i),
                    allRegionsForChr.get(index).getSecond().get(i)));
        }

        return foundFeatures;
    }

    public List<Pair<MotifAnchor, MotifAnchor>> getIntersectingFeatures(int index, int gx1) {
        int idx1 = OneDimSearchUtils.indexedBinaryNearestSearch(
                allRegionsForChr.get(index).getSecond(), new MotifAnchor(index, gx1, gx1), true);

        final List<Pair<MotifAnchor, MotifAnchor>> foundFeatures = new ArrayList<>();
        if (idx1 > 0) {
            foundFeatures.add(new Pair<>(
                    allRegionsForChr.get(index).getFirst().get(idx1),
                    allRegionsForChr.get(index).getSecond().get(idx1)));
        }

        return foundFeatures;
    }
}


