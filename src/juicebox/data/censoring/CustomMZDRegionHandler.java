/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
import juicebox.data.anchor.GenericLocus;
import juicebox.data.anchor.MotifAnchor;
import juicebox.data.basics.Chromosome;
import juicebox.windowui.HiCZoom;
import org.broad.igv.util.Pair;

import java.util.*;

public class CustomMZDRegionHandler {
    
    private final Map<String, Pair<List<GenericLocus>, List<GenericLocus>>> allRegionsForChr = new HashMap<>();
    private final List<Long> boundariesOfCustomChromosomeX = new ArrayList<>();
    private final List<Long> boundariesOfCustomChromosomeY = new ArrayList<>();
    
    private static Pair<List<GenericLocus>, List<GenericLocus>> getAllRegionsFromSubChromosomes(
            final ChromosomeHandler handler, Chromosome chr) {
        
        if (handler.isCustomChromosome(chr)) {
            final List<GenericLocus> allRegions = new ArrayList<>();
            final List<GenericLocus> translatedRegions = new ArrayList<>();
            
            handler.getListOfRegionsInCustomChromosome(chr.getIndex()).processLists(
                    new juicebox.data.feature.FeatureFunction<GenericLocus>() {
                        @Override
                        public void process(String chr, List<GenericLocus> featureList) {
                            allRegions.addAll(featureList);
                        }
                    });
            Collections.sort(allRegions);
            
            long previousEnd = 0;
            for (GenericLocus anchor : allRegions) {
                long currentEnd = previousEnd + anchor.getWidth();
                GenericLocus anchor2 = new GenericLocus(chr.getName(), previousEnd, currentEnd);
                translatedRegions.add(anchor2);
                previousEnd = currentEnd + ChromosomeHandler.CUSTOM_CHROMOSOME_BUFFER;
            }

            return new Pair<>(allRegions, translatedRegions);
        } else {
            // just a standard chromosome
            final List<GenericLocus> allRegions = new ArrayList<>();
            final List<GenericLocus> translatedRegions = new ArrayList<>();
            allRegions.add(new GenericLocus(chr.getName(), 0, (int) chr.getLength()));
            translatedRegions.add(new GenericLocus(chr.getName(), 0, (int) chr.getLength()));
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
    
    public List<Long> getBoundariesOfCustomChromosomeX() {
        return boundariesOfCustomChromosomeX;
    }
    
    public List<Long> getBoundariesOfCustomChromosomeY() {
        return boundariesOfCustomChromosomeY;
    }
    
    private void populateRegions(Chromosome chr, ChromosomeHandler handler, List<Long> boundaries, HiCZoom zoom) {
        String name = chr.getName();
        Pair<List<GenericLocus>, List<GenericLocus>> allRegionsInfo = getAllRegionsFromSubChromosomes(handler, chr);
        
        if (allRegionsInfo != null) {
            allRegionsForChr.put(name, allRegionsInfo);
            List<GenericLocus> translatedRegions = allRegionsInfo.getSecond();
            for (GenericLocus anchor : translatedRegions) {
                boundaries.add(anchor.getX2() / zoom.getBinSize());
            }
        }
    }
    
    public List<Pair<GenericLocus, GenericLocus>> getIntersectingFeatures(String name, long gx1, long gx2) {
        
        int idx1 = OneDimSearchUtils.indexedBinaryNearestSearch(
                allRegionsForChr.get(name).getSecond(), new MotifAnchor(name, gx1, gx1), true);
        int idx2 = OneDimSearchUtils.indexedBinaryNearestSearch(
                allRegionsForChr.get(name).getSecond(), new MotifAnchor(name, gx2, gx2), false);
        
        final List<Pair<GenericLocus, GenericLocus>> foundFeatures = new ArrayList<>();
        for (int i = idx1; i <= idx2; i++) {
            foundFeatures.add(new Pair<>(
                    allRegionsForChr.get(name).getFirst().get(i),
                    allRegionsForChr.get(name).getSecond().get(i)));
        }

        return foundFeatures;
    }

    public List<Pair<GenericLocus, GenericLocus>> getIntersectingFeatures(String name, int gx1) {
        int idx1 = OneDimSearchUtils.indexedBinaryNearestSearch(
                allRegionsForChr.get(name).getSecond(), new MotifAnchor(name, gx1, gx1), true);

        final List<Pair<GenericLocus, GenericLocus>> foundFeatures = new ArrayList<>();
        if (idx1 > 0) {
            foundFeatures.add(new Pair<>(
                    allRegionsForChr.get(name).getFirst().get(idx1),
                    allRegionsForChr.get(name).getSecond().get(idx1)));
        }

        return foundFeatures;
    }
}


