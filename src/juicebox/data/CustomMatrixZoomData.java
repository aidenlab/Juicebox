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

package juicebox.data;

import gnu.trove.procedure.TIntProcedure;
import juicebox.HiCGlobals;
import juicebox.data.anchor.MotifAnchor;
import juicebox.windowui.NormalizationType;
import net.sf.jsi.SpatialIndex;
import net.sf.jsi.rtree.RTree;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by muhammadsaadshamim on 7/21/17.
 */
public class CustomMatrixZoomData extends MatrixZoomData {

    private Map<String, MatrixZoomData> zoomDatasForDifferentRegions = new HashMap<>();

    private final Map<Integer, SpatialIndex> regionsRtree = new HashMap<>();
    private final Map<Integer, Pair<List<MotifAnchor>, List<MotifAnchor>>> allRegionsForChr = new HashMap<>();

    public CustomMatrixZoomData(Chromosome chr1, Chromosome chr2, ChromosomeHandler handler, String regionKey,
                                MatrixZoomData zd, DatasetReader reader) {
        super(chr1, chr2, zd.getZoom(), -1, -1,
                new int[0], new int[0], reader);
        expandAvailableZoomDatas(regionKey, zd);
        initializeRTree(handler);
    }

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

    public static Block modifyBlock(Block block, int binSize, int chr1Idx, int chr2Idx, RegionPair rp) {
        List<ContactRecord> alteredContacts = new ArrayList<>();
        for (ContactRecord record : block.getContactRecords()) {

            int newX = record.getBinX();
            if (newX >= rp.xRegion.getX1() / binSize && newX <= rp.xRegion.getX2() / binSize) {
                newX = rp.xTransRegion.getX1() + (newX - (rp.xRegion.getX1() / binSize));
            } else {
                continue;
            }

            int newY = record.getBinY();
            if (newY >= rp.yRegion.getX1() / binSize && newY <= rp.yRegion.getX2() / binSize) {
                newY = rp.yTransRegion.getX1() + (newY - (rp.yRegion.getX1() / binSize));
            } else {
                continue;
            }

            if (chr1Idx == chr2Idx && newY < newX) {
                alteredContacts.add(new ContactRecord(newY, newX, record.getCounts()));
            } else {
                alteredContacts.add(new ContactRecord(newX, newY, record.getCounts()));
            }
        }
        block = new Block(block.getNumber(), alteredContacts);
        return block;
    }

    public void expandAvailableZoomDatas(String regionKey, MatrixZoomData zd) {
        if (getZoom().equals(zd.getZoom())) {
            zoomDatasForDifferentRegions.put(regionKey, zd);
        }
    }

    @Override
    public List<Block> getNormalizedBlocksOverlapping(int binX1, int binY1, int binX2, int binY2, final NormalizationType norm) {
        int resolution = zoom.getBinSize();
        return addNormalizedBlocksToListByGenomeCoordinates(
                binX1 * resolution, binY1 * resolution, binX2 * resolution, binY2 * resolution, norm);
    }

    @Override
    public void printDescription() {
        System.out.println("Custom Chromosome: " + chr1.getName() + " - " + chr2.getName());
        System.out.println("unit: " + zoom.getUnit());
        System.out.println("binSize (bp): " + zoom.getBinSize());
    }

    private List<Block> addNormalizedBlocksToListByGenomeCoordinates(int gx1, int gy1, int gx2, int gy2,
                                                                     final NormalizationType no) {
        List<Block> blockList = new ArrayList<>();
        Map<MatrixZoomData, Set<Pair<Integer, RegionPair>>>
                blocksNumsToLoadForZd = new HashMap<>();
        // remember these are pseudo genome coordinates

        // x window
        net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(gx1, gx1, gx2, gx2);
        List<Pair<MotifAnchor, MotifAnchor>> xAxisRegions = getIntersectingFeatures(chr1.getIndex(), currentWindow);

        // y window
        currentWindow = new net.sf.jsi.Rectangle(gy1, gy1, gy2, gy2);
        List<Pair<MotifAnchor, MotifAnchor>> yAxisRegions = getIntersectingFeatures(chr2.getIndex(), currentWindow);

        for (Pair<MotifAnchor, MotifAnchor> xRegion : xAxisRegions) {
            for (Pair<MotifAnchor, MotifAnchor> yRegion : yAxisRegions) {

                Pair<MotifAnchor, MotifAnchor> xLocalRegion = xRegion;
                Pair<MotifAnchor, MotifAnchor> yLocalRegion = yRegion;

                int xI = xLocalRegion.getFirst().getChr();
                int yI = yLocalRegion.getFirst().getChr();
                MatrixZoomData zd = zoomDatasForDifferentRegions.get(Matrix.generateKey(xI, yI));

                if (zd.getChr1Idx() == xI && zd.getChr2Idx() == yI) {
                    // this is the current assumption
                } else if (zd.getChr2Idx() == xI && zd.getChr1Idx() == yI) {
                    // flip order
                    xLocalRegion = yRegion;
                    yLocalRegion = xRegion;
                } else {
                    System.err.println("Error in getting proper region...");
                    continue;
                }
                int[] originalGenomePosition = new int[]{
                        xLocalRegion.getFirst().getX1(), xLocalRegion.getFirst().getX2(),
                        yLocalRegion.getFirst().getX1(), yLocalRegion.getFirst().getX2()};

                List<Integer> tempBlockNumbers = zd.getBlockNumbersForRegionFromGenomePosition(originalGenomePosition);
                for (int blockNumber : tempBlockNumbers) {
                    if (!blocksNumsToLoadForZd.containsKey(zd)) {
                        blocksNumsToLoadForZd.put(zd, new HashSet<Pair<Integer, RegionPair>>());
                    }

                    if (blocksNumsToLoadForZd.get(zd).contains(blockNumber)) {
                        continue;
                    } else {
                        String key = getBlockKey(zd.getChr1Idx(), zd.getChr2Idx(), blockNumber, no);
                        Block b;
                        if (HiCGlobals.useCache && blockCache.containsKey(key)) {
                            b = blockCache.get(key);
                            blockList.add(b);
                        } else {
                            blocksNumsToLoadForZd.get(zd).add(new Pair<>(blockNumber,
                                    new RegionPair(zd.getChr1Idx(), xLocalRegion, zd.getChr2Idx(), yLocalRegion)));
                        }
                    }
                }
            }
        }

        // Actually load new blocks
        actuallyLoadGivenBlocks(blockList, no, blocksNumsToLoadForZd);

        return new ArrayList<>(new HashSet<>(blockList));
    }

    /**
     * not quite an override since inputs are different, but naming preserved as parent class
     *
     * @param blockList
     * @param no
     * @param blocksNumsToLoadForZd
     */
    private void actuallyLoadGivenBlocks(final List<Block> blockList, final NormalizationType no, Map<MatrixZoomData,
            Set<Pair<Integer, RegionPair>>> blocksNumsToLoadForZd) {
        final AtomicInteger errorCounter = new AtomicInteger();

        List<Thread> threads = new ArrayList<>();

        final int binSize = getBinSize();

        for (final MatrixZoomData zd : blocksNumsToLoadForZd.keySet()) {
            final int chr1Index = zd.getChr1Idx();
            final int chr2Index = zd.getChr2Idx();
            for (final Pair<Integer, RegionPair> blockNumberObj
                    : blocksNumsToLoadForZd.get(zd)) {
                Runnable loader = new Runnable() {
                    @Override
                    public void run() {
                        try {
                            String key = getBlockKey(chr1Index, chr2Index, blockNumberObj.getFirst(), no);
                            Block b = reader.readNormalizedBlock(blockNumberObj.getFirst(), zd, no);
                            if (b == null) {
                                b = new Block(blockNumberObj.getFirst());   // An empty block
                            } else {

                                b = modifyBlock(b, binSize, chr1Index, chr2Index, blockNumberObj.getSecond());
                            }

                            if (HiCGlobals.useCache) {
                                blockCache.put(key, b);
                            }
                            blockList.add(b);
                        } catch (IOException e) {
                            errorCounter.incrementAndGet();
                        }
                    }
                };

                Thread t = new Thread(loader);
                threads.add(t);
                t.start();
            }
        }

        // Wait for all threads to complete
        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException ignore) {
            }
        }
        if (errorCounter.get() > 0) {
            System.err.println(errorCounter.get() + " errors while reading blocks");
        }
    }

    private void initializeRTree(ChromosomeHandler handler) {
        regionsRtree.clear();
        allRegionsForChr.clear();

        populateRTreeWithRegions(chr1, handler);
        if (chr1.getIndex() != chr2.getIndex())
            populateRTreeWithRegions(chr2, handler);
    }

    private void populateRTreeWithRegions(Chromosome chr, ChromosomeHandler handler) {
        int chrIndex = chr.getIndex();
        Pair<List<MotifAnchor>, List<MotifAnchor>> allRegionsInfo = getAllRegionsFromSubChromosomes(handler, chr);

        if (allRegionsInfo != null) {
            allRegionsForChr.put(chrIndex, allRegionsInfo);
            SpatialIndex si = new RTree();
            si.init(null);
            List<MotifAnchor> translatedRegions = allRegionsInfo.getSecond();
            for (int i = 0; i < translatedRegions.size(); i++) {
                MotifAnchor anchor = translatedRegions.get(i);
                si.add(new net.sf.jsi.Rectangle((float) anchor.getX1(), (float) anchor.getX1(),
                        (float) anchor.getX2(), (float) anchor.getX2()), i);
            }
            regionsRtree.put(chrIndex, si);
        }
    }

    private List<Pair<MotifAnchor, MotifAnchor>> getIntersectingFeatures(final int chrIdx, net.sf.jsi.Rectangle selectionWindow) {
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

    public String getBlockKey(int xChrIndx, int yChrIndx, int blockNumber, NormalizationType no) {
        return getKey() + "_" + xChrIndx + "-" + yChrIndx + "_" + blockNumber + "_" + no;
    }

    private class RegionPair {

        int xI, yI;
        MotifAnchor xRegion;
        MotifAnchor xTransRegion;
        MotifAnchor yRegion;
        MotifAnchor yTransRegion;

        public RegionPair(int xI, Pair<MotifAnchor, MotifAnchor> xLocalRegion,
                          int yI, Pair<MotifAnchor, MotifAnchor> yLocalRegion) {
            this.xI = xI;
            this.yI = yI;
            this.xRegion = xLocalRegion.getFirst();
            this.xTransRegion = xLocalRegion.getSecond();
            this.yRegion = yLocalRegion.getFirst();
            this.yTransRegion = yLocalRegion.getSecond();
        }
    }
}
