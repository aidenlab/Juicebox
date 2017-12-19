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

import juicebox.HiCGlobals;
import juicebox.data.anchor.MotifAnchor;
import juicebox.data.censoring.CustomMZDRegionHandler;
import juicebox.data.censoring.RegionPair;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;
import org.broad.igv.util.collections.LRUCache;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by muhammadsaadshamim on 7/21/17.
 */
public class CustomMatrixZoomData extends MatrixZoomData {

    private final Map<String, MatrixZoomData> zoomDatasForDifferentRegions = new HashMap<>();
    private final Map<MatrixZoomData, Map<RegionPair, LRUCache<String, Block>>> allBlockCaches = new HashMap<>();
    private final CustomMZDRegionHandler rTreeHandler = new CustomMZDRegionHandler();

    public CustomMatrixZoomData(Chromosome chr1, Chromosome chr2, ChromosomeHandler handler, String regionKey,
                                MatrixZoomData zd, DatasetReader reader) {
        super(chr1, chr2, zd.getZoom(), -1, -1,
                new int[0], new int[0], reader);
        expandAvailableZoomDatas(regionKey, zd);
        rTreeHandler.initialize(chr1, chr2, zoom, handler);
    }

    private boolean isImportant = false;

    public void expandAvailableZoomDatas(String regionKey, MatrixZoomData zd) {
        if (getZoom().equals(zd.getZoom())) {
            zoomDatasForDifferentRegions.put(regionKey, zd);
        }
    }

    private static Block modifyBlock(Block block, String key, MatrixZoomData zd, RegionPair rp) {
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
                alteredContacts.add(new ContactRecord(newBinY, newBinX, record.getBaseCounts(), RGBButton.Channel.RED));
            } else {
                alteredContacts.add(new ContactRecord(newBinX, newBinY, record.getBaseCounts(), RGBButton.Channel.RED));
            }
        }
        //System.out.println("num orig records "+block.getContactRecords().size()+ " after alter "+alteredContacts.size()+" bnum "+block.getNumber());
        return new Block(block.getNumber(), alteredContacts, key + rp.getDescription());
    }

    @Override
    public List<Block> getNormalizedBlocksOverlapping(int binX1, int binY1, int binX2, int binY2,
                                                      final NormalizationType norm, boolean isImportant) {
        this.isImportant = isImportant;
        float resolution = zoom.getBinSize();
        //if(isImportant) System.out.println("zt12 "+resolution+" --x "+binX1+" "+binX2+" y "+binY1+" "+binY2);
        int gx1 = (int) (binX1 * resolution);
        int gy1 = (int) (binY1 * resolution);
        int gx2 = (int) (binX2 * resolution);
        int gy2 = (int) (binY2 * resolution);

        return addNormalizedBlocksToListByGenomeCoordinates(gx1, gy1, gx2, gy2, norm);
    }

    @Override
    public void printFullDescription() {
        System.out.println("Custom Chromosome: " + chr1.getName() + " - " + chr2.getName());
        System.out.println("unit: " + zoom.getUnit());
        System.out.println("binSize (bp): " + zoom.getBinSize());
    }

    private List<Block> addNormalizedBlocksToListByGenomeCoordinates(int gx1, int gy1, int gx2, int gy2,
                                                                     final NormalizationType no) {
        List<Block> blockList = new ArrayList<>();
        Map<MatrixZoomData, Map<RegionPair, List<Integer>>> blocksNumsToLoadForZd = new HashMap<>();
        // remember these are pseudo genome coordinates

        // x window
        //net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(gx1, gx1, gx2, gx2);
        List<Pair<MotifAnchor, MotifAnchor>> xAxisRegions = rTreeHandler.getIntersectingFeatures(chr1.getIndex(), gx1, gx2);

        // y window
        //currentWindow = new net.sf.jsi.Rectangle(gy1, gy1, gy2, gy2);
        List<Pair<MotifAnchor, MotifAnchor>> yAxisRegions = rTreeHandler.getIntersectingFeatures(chr2.getIndex(), gy1, gy2);

        if (isImportant) {
            //System.out.println("num x regions " + xAxisRegions.size()+ " num y regions " + yAxisRegions.size());
        }

        if (xAxisRegions.size() < 1) {
            System.err.println("no x?");
        }
        if (yAxisRegions.size() < 1) {
            System.err.println("no y?");
        }


        for (Pair<MotifAnchor, MotifAnchor> xRegion : xAxisRegions) {
            for (Pair<MotifAnchor, MotifAnchor> yRegion : yAxisRegions) {

                RegionPair rp = RegionPair.generateRegionPair(xRegion, yRegion);
                MatrixZoomData zd = zoomDatasForDifferentRegions.get(Matrix.generateKey(rp.xI, rp.yI));

                if (!blocksNumsToLoadForZd.containsKey(zd)) {
                    blocksNumsToLoadForZd.put(zd, new HashMap<RegionPair, List<Integer>>());
                }

                if (!blocksNumsToLoadForZd.get(zd).containsKey(rp)) {
                    blocksNumsToLoadForZd.get(zd).put(rp, new ArrayList<Integer>());
                }

                int[] originalGenomePosition = rp.getOriginalGenomeRegion();

                List<Integer> tempBlockNumbers = zd.getBlockNumbersForRegionFromGenomePosition(originalGenomePosition);
                for (int blockNumber : tempBlockNumbers) {
                    String key = zd.getBlockKey(blockNumber, no);
                    if (HiCGlobals.useCache && allBlockCaches.containsKey(zd)
                            && allBlockCaches.get(zd).containsKey(rp) && allBlockCaches.get(zd).get(rp).containsKey(key)) {
                        blockList.add(allBlockCaches.get(zd).get(rp).get(key));
                    } else {
                        blocksNumsToLoadForZd.get(zd).get(rp).add(blockNumber);
                    }
                }
            }
        }
        // Actually load new blocks
        actuallyLoadGivenBlocks(blockList, no, blocksNumsToLoadForZd);
        //System.out.println("num blocks post "+blockList.size());

        if (blockList.size() < 1) {
            System.err.println("no blocks??");
        }

        return blockList;
    }

    /**
     * not quite an override since inputs are different, but naming preserved as parent class
     *
     * @param blockList
     * @param no
     * @param blocksNumsToLoadForZd
     */
    private void actuallyLoadGivenBlocks(final List<Block> blockList, final NormalizationType no,
                                         Map<MatrixZoomData, Map<RegionPair, List<Integer>>> blocksNumsToLoadForZd) {
        final AtomicInteger errorCounter = new AtomicInteger();
        ExecutorService service = Executors.newFixedThreadPool(1000);

        for (final MatrixZoomData zd : blocksNumsToLoadForZd.keySet()) {
            final Map<RegionPair, List<Integer>> blockNumberMap = blocksNumsToLoadForZd.get(zd);
            for (final RegionPair rp : blockNumberMap.keySet()) {
                Runnable loader = new Runnable() {
                    @Override
                    public void run() {
                        try {
                            for (final int blockNum : blockNumberMap.get(rp)) {
                                String key = zd.getBlockKey(blockNum, no);
                                Block b = reader.readNormalizedBlock(blockNum, zd, no);
                                if (b == null) {
                                    b = new Block(blockNum, key + rp.getDescription());   // An empty block
                                } else {
                                    b = modifyBlock(b, key, zd, rp);
                                }

                                if (HiCGlobals.useCache) {
                                    if (!allBlockCaches.containsKey(zd)) {
                                        allBlockCaches.put(zd, new HashMap<RegionPair, LRUCache<String, Block>>());
                                    }
                                    if (!allBlockCaches.get(zd).containsKey(rp)) {
                                        allBlockCaches.get(zd).put(rp, new LRUCache<String, Block>(50));
                                    }
                                    allBlockCaches.get(zd).get(rp).put(key, b);
                                }
                                blockList.add(b);
                            }
                        } catch (IOException e) {
                            errorCounter.incrementAndGet();
                        }
                    }
                };
                service.submit(loader);
            }
        }

        // done submitting all jobs
        service.shutdown();

        // wait for all to finish
        try {
            service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            //String.format("Pending tasks: %d", executor.getQueue().size())
            System.err.println("Error loading custom mzd data " + e.getLocalizedMessage());
            if (HiCGlobals.printVerboseComments) {
                e.printStackTrace();
            }
        }

        // error printing
        if (errorCounter.get() > 0) {
            System.err.println(errorCounter.get() + " errors while reading blocks");
        }
    }

    public List<Integer> getBoundariesOfCustomChromosomeX() {
        return rTreeHandler.getBoundariesOfCustomChromosomeX();
    }

    public List<Integer> getBoundariesOfCustomChromosomeY() {
        return rTreeHandler.getBoundariesOfCustomChromosomeY();
    }

    // TODO
    public double getExpected(int binX, int binY, ExpectedValueFunction df) {
        // x window
        int gx1 = binX * zoom.getBinSize();
        net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(gx1, gx1, gx1, gx1);
        List<Pair<MotifAnchor, MotifAnchor>> xRegions = rTreeHandler.getIntersectingFeatures(chr1.getIndex(), gx1);

        // y window
        int gy1 = binY * zoom.getBinSize();
        currentWindow = new net.sf.jsi.Rectangle(gy1, gy1, gy1, gy1);
        List<Pair<MotifAnchor, MotifAnchor>> yRegions = rTreeHandler.getIntersectingFeatures(chr2.getIndex(), gy1);

        RegionPair rp = RegionPair.generateRegionPair(xRegions.get(0), yRegions.get(0));
        MatrixZoomData zd = zoomDatasForDifferentRegions.get(Matrix.generateKey(rp.xI, rp.yI));

        return zd.getAverageCount();
        /*
        if(rp.xI == rp.yI){
            if (df != null) {
                int dist = Math.abs(binX - binY);
                return df.getExpectedValue(rp.xI, dist);
            }
        } else {
            return zd.getAverageCount();
        }

        return 0;
        */
    }


    public List<Pair<MotifAnchor, MotifAnchor>> getRTreeHandlerIntersectingFeatures(int chrIndex, int g1, int g2) {
        return rTreeHandler.getIntersectingFeatures(chrIndex, g1, g2);
    }
}
