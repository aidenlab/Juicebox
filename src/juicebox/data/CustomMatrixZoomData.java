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

    private boolean isImportant = false;

    public void expandAvailableZoomDatas(String regionKey, MatrixZoomData zd) {
        if (getZoom().equals(zd.getZoom())) {
            zoomDatasForDifferentRegions.put(regionKey, zd);
        }
    }

    public static Block modifyBlock(Block block, String key, MatrixZoomData zd, RegionPair rp) {
        int binSize = zd.getBinSize();
        int chr1Idx = zd.getChr1Idx();
        int chr2Idx = zd.getChr2Idx();

        List<ContactRecord> alteredContacts = new ArrayList<>();
        for (ContactRecord record : block.getContactRecords()) {

            int newX = record.getBinX() * binSize;
            if (newX >= rp.xRegion.getX1() && newX <= rp.xRegion.getX2()) {
                newX = rp.xTransRegion.getX1() + newX - rp.xRegion.getX1();
            } else {
                continue;
            }

            int newY = record.getBinY() * binSize;
            if (newY >= rp.yRegion.getX1() && newY <= rp.yRegion.getX2()) {
                newY = rp.yTransRegion.getX1() + newY - rp.yRegion.getX1();
            } else {
                continue;
            }
            int newBinX = newX / binSize;
            int newBinY = newY / binSize;

            if (chr1Idx == chr2Idx && newBinY < newBinX) {
                alteredContacts.add(new ContactRecord(newBinY, newBinX, record.getCounts()));
            } else {
                alteredContacts.add(new ContactRecord(newBinX, newBinY, record.getCounts()));
            }
        }
        //System.out.println("num orig records "+block.getContactRecords().size()+ " after alter "+alteredContacts.size()+" bnum "+block.getNumber());
        return new Block(block.getNumber(), alteredContacts, key);
    }

    @Override
    public List<Block> getNormalizedBlocksOverlapping(int binX1, int binY1, int binX2, int binY2,
                                                      final NormalizationType norm, boolean isImportant) {
        this.isImportant = isImportant;
        float resolution = zoom.getBinSize();
        //if(isImportant) System.out.println("zt12 "+resolution+" --x "+binX1+" "+binX2+" y "+binY1+" "+binY2);
        return addNormalizedBlocksToListByGenomeCoordinates(
                binX1 * resolution, binY1 * resolution, binX2 * resolution, binY2 * resolution, norm);
    }

    @Override
    public void printFullDescription() {
        System.out.println("Custom Chromosome: " + chr1.getName() + " - " + chr2.getName());
        System.out.println("unit: " + zoom.getUnit());
        System.out.println("binSize (bp): " + zoom.getBinSize());
    }

    private List<Block> addNormalizedBlocksToListByGenomeCoordinates(float gx1, float gy1, float gx2, float gy2,
                                                                     final NormalizationType no) {
        List<Block> blockList = new ArrayList<>();
        Map<MatrixZoomData, Set<Pair<Integer, RegionPair>>> blocksNumsToLoadForZd = new HashMap<>();
        // remember these are pseudo genome coordinates

        // x window
        net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(gx1, gx1, gx2, gx2);
        List<Pair<MotifAnchor, MotifAnchor>> xAxisRegions = getIntersectingFeatures(chr1.getIndex(), currentWindow);

        // y window
        currentWindow = new net.sf.jsi.Rectangle(gy1, gy1, gy2, gy2);
        List<Pair<MotifAnchor, MotifAnchor>> yAxisRegions = getIntersectingFeatures(chr2.getIndex(), currentWindow);

        if (isImportant) {
            //System.out.println("num x regions " + xAxisRegions.size()+ " num y regions " + yAxisRegions.size());
        }

        for (Pair<MotifAnchor, MotifAnchor> xRegion : xAxisRegions) {
            for (Pair<MotifAnchor, MotifAnchor> yRegion : yAxisRegions) {

                Pair<MotifAnchor, MotifAnchor> xLocalRegion = xRegion;
                Pair<MotifAnchor, MotifAnchor> yLocalRegion = yRegion;
                /*if(isImportant) {
                    System.out.println("xr " + xRegion.getFirst() + " yr " + yRegion.getFirst());
                    System.out.println("xr2 " + xRegion.getSecond() + " yr2 " + yRegion.getSecond());
                }*/
                int xI = xLocalRegion.getFirst().getChr();
                int yI = yLocalRegion.getFirst().getChr();

                int sortedXI = xI, sortedYI = yI;
                if (xI <= yI) {
                    // this is the current assumption
                    //if(isImportant) System.out.println("was already ok");
                } else {
                    // flip order
                    //if(isImportant) System.out.println("flipped order");
                    xLocalRegion = yRegion;
                    yLocalRegion = xRegion;
                    sortedXI = yI;
                    sortedYI = xI;
                }

                String keyI = Matrix.generateKey(sortedXI, sortedYI);
                MatrixZoomData zd = zoomDatasForDifferentRegions.get(keyI);
                //System.out.println("key is "+keyI+"  --- zd is "+zd);

                RegionPair rp = new RegionPair(zd.getChr1Idx(), xLocalRegion, zd.getChr2Idx(), yLocalRegion);

                int[] originalGenomePosition = new int[]{
                        xLocalRegion.getFirst().getX1(), xLocalRegion.getFirst().getX2(),
                        yLocalRegion.getFirst().getX1(), yLocalRegion.getFirst().getX2()};

                List<Integer> tempBlockNumbers = zd.getBlockNumbersForRegionFromGenomePosition(originalGenomePosition);
                for (int blockNumber : tempBlockNumbers) {
                    if (!blocksNumsToLoadForZd.containsKey(zd)) {
                        blocksNumsToLoadForZd.put(zd, new HashSet<Pair<Integer, RegionPair>>());
                    }

                    Pair<Integer, RegionPair> blockRegionPair = new Pair<>(blockNumber, rp);

                    if (blocksNumsToLoadForZd.get(zd).contains(blockRegionPair)) {
                        continue;
                    } else {
                        String key = zd.getBlockKey(blockNumber, no);
                        Block b;
                        if (HiCGlobals.useCache && blockCache.containsKey(key)) {
                            b = blockCache.get(key);
                            blockList.add(b);
                        } else {
                            blocksNumsToLoadForZd.get(zd).add(blockRegionPair);
                        }
                    }
                }
            }
        }
        // Actually load new blocks
        actuallyLoadGivenBlocks(blockList, no, blocksNumsToLoadForZd);
        //System.out.println("num blocks post "+blockList.size());

        return blockList;
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

        for (final MatrixZoomData zd : blocksNumsToLoadForZd.keySet()) {
            for (final Pair<Integer, RegionPair> blockNumberObj : blocksNumsToLoadForZd.get(zd)) {
                Runnable loader = new Runnable() {
                    @Override
                    public void run() {
                        try {
                            //TODO blocknums may cause incident down the road
                            String key = zd.getBlockKey(blockNumberObj.getFirst(), no);
                            Block b = reader.readNormalizedBlock(blockNumberObj.getFirst(), zd, no);
                            if (b == null) {
                                b = new Block(blockNumberObj.getFirst(), key);   // An empty block
                            } else {
                                b = modifyBlock(b, key, zd, blockNumberObj.getSecond());
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


    private final List<Integer> boundariesOfCustomChromosomeX = new ArrayList<>();
    private final List<Integer> boundariesOfCustomChromosomeY = new ArrayList<>();

    /**
     * @param handler
     */
    private void initializeRTree(ChromosomeHandler handler) {
        regionsRtree.clear();
        allRegionsForChr.clear();
        boundariesOfCustomChromosomeX.clear();
        boundariesOfCustomChromosomeY.clear();

        populateRTreeWithRegions(chr1, handler, boundariesOfCustomChromosomeX);
        if (chr1.getIndex() != chr2.getIndex()) {
            populateRTreeWithRegions(chr2, handler, boundariesOfCustomChromosomeY);
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

    private void populateRTreeWithRegions(Chromosome chr, ChromosomeHandler handler, List<Integer> boundaries) {
        int chrIndex = chr.getIndex();
        Pair<List<MotifAnchor>, List<MotifAnchor>> allRegionsInfo = getAllRegionsFromSubChromosomes(handler, chr);

        if (allRegionsInfo != null) {
            allRegionsForChr.put(chrIndex, allRegionsInfo);
            SpatialIndex si = new RTree();
            si.init(null);
            List<MotifAnchor> translatedRegions = allRegionsInfo.getSecond();
            for (int i = 0; i < translatedRegions.size(); i++) {
                MotifAnchor anchor = translatedRegions.get(i);
                if (isImportant) System.out.println("ztM11 " + anchor.toString());
                boundaries.add(anchor.getX2() / zoom.getBinSize());
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
