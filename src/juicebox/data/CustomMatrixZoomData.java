/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
import juicebox.data.anchor.GenericLocus;
import juicebox.data.anchor.MotifAnchor;
import juicebox.data.basics.Chromosome;
import juicebox.data.censoring.CustomMZDRegionHandler;
import juicebox.data.censoring.RegionPair;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.util.Pair;
import org.broad.igv.util.collections.LRUCache;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by muhammadsaadshamim on 7/21/17.
 */
public class CustomMatrixZoomData extends MatrixZoomData {

    private final Map<String, MatrixZoomData> zoomDatasForDifferentRegions = new HashMap<>();
    private final Map<MatrixZoomData, Map<RegionPair, LRUCache<String, Block>>> allBlockCaches = new HashMap<>();
    private final CustomMZDRegionHandler rTreeHandler = new CustomMZDRegionHandler();
    private final ChromosomeHandler handler;

    public CustomMatrixZoomData(Chromosome chr1, Chromosome chr2, ChromosomeHandler handler,
                                HiCZoom zoom, DatasetReader reader) {
        super(chr1, chr2, zoom, -1, -1, new int[0], new int[0], reader);
        this.handler = handler;
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

            long newX = record.getBinX() * binSize;
            if (newX >= rp.xRegion.getX1() && newX <= rp.xRegion.getX2()) {
                newX = rp.xTransRegion.getX1() + newX - rp.xRegion.getX1();
			} else {
				continue;
			}
	
			long newY = record.getBinY() * binSize;
			if (newY >= rp.yRegion.getX1() && newY <= rp.yRegion.getX2()) {
				newY = rp.yTransRegion.getX1() + newY - rp.yRegion.getX1();
			} else {
				continue;
			}
			int newBinX = (int) (newX / binSize);
			int newBinY = (int) (newY / binSize);
	
			if (chr1Idx == chr2Idx && newBinY < newBinX) {
				alteredContacts.add(new ContactRecord(newBinY, newBinX, record.getCounts()));
			} else {
				alteredContacts.add(new ContactRecord(newBinX, newBinY, record.getCounts()));
			}
		}
        //System.out.println("num orig records "+block.getContactRecords().size()+ " after alter "+alteredContacts.size()+" bnum "+block.getNumber());
        return new Block(block.getNumber(), alteredContacts, key + rp.getDescription());
    }
	
	@Override
	public List<Block> getNormalizedBlocksOverlapping(long binX1, long binY1, long binX2, long binY2,
													  final NormalizationType norm, boolean isImportant, boolean fillUnderDiagonal) {
		this.isImportant = isImportant;
		int resolution = zoom.getBinSize();
		//if(isImportant) System.out.println("zt12 "+resolution+" --x "+binX1+" "+binX2+" y "+binY1+" "+binY2);
		long gx1 = (binX1 * resolution);
		long gy1 = (binY1 * resolution);
		long gx2 = (binX2 * resolution);
		long gy2 = (binY2 * resolution);
		
		return addNormalizedBlocksToListByGenomeCoordinates(gx1, gy1, gx2, gy2, norm);
	}

    @Override
    public void printFullDescription() {
        System.out.println("Custom Chromosome: " + chr1.getName() + " - " + chr2.getName());
        System.out.println("unit: " + zoom.getUnit());
        System.out.println("binSize (bp): " + zoom.getBinSize());
    }
	
	private List<Block> addNormalizedBlocksToListByGenomeCoordinates(long gx1, long gy1, long gx2, long gy2,
																	 final NormalizationType no) {
        List<Block> blockList = Collections.synchronizedList(new ArrayList<>());
        Map<MatrixZoomData, Map<RegionPair, List<Integer>>> blocksNumsToLoadForZd = new ConcurrentHashMap<>();
        // remember these are pseudo genome coordinates
        
        // x window
        //net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(gx1, gx1, gx2, gx2);
        List<Pair<GenericLocus, GenericLocus>> xAxisRegions = rTreeHandler.getIntersectingFeatures(chr1.getName(), gx1, gx2);
        
        // y window
        //currentWindow = new net.sf.jsi.Rectangle(gy1, gy1, gy2, gy2);
        List<Pair<GenericLocus, GenericLocus>> yAxisRegions = rTreeHandler.getIntersectingFeatures(chr2.getName(), gy1, gy2);

        if (isImportant) {
            if (HiCGlobals.printVerboseComments)
                System.out.println("num x regions " + xAxisRegions.size() + " num y regions " + yAxisRegions.size());
        }

        if (xAxisRegions.size() < 1) {
            System.err.println("no x?");
        }
        if (yAxisRegions.size() < 1) {
            System.err.println("no y?");
        }

        ExecutorService executor = HiCGlobals.newFixedThreadPool();
        // todo change to be by chromosome?
        for (Pair<GenericLocus, GenericLocus> xRegion : xAxisRegions) {
            for (Pair<GenericLocus, GenericLocus> yRegion : yAxisRegions) {
                Runnable worker = new Runnable() {
                    @Override
                    public void run() {
                        RegionPair rp = RegionPair.generateRegionPair(xRegion, yRegion, handler);
                        MatrixZoomData zd = zoomDatasForDifferentRegions.get(Matrix.generateKey(rp.xI, rp.yI));
                        if (zd == null || rp == null) return;

                        synchronized (blocksNumsToLoadForZd) {
                            if (!blocksNumsToLoadForZd.containsKey(zd)) {
                                blocksNumsToLoadForZd.put(zd, new HashMap<>());
                            }

                            if (!blocksNumsToLoadForZd.get(zd).containsKey(rp)) {
                                blocksNumsToLoadForZd.get(zd).put(rp, new ArrayList<>());
                            }
                        }
	
						// todo mss custom matrix zd doesn't have long support yet
                        List<Integer> tempBlockNumbers = zd.getBlockNumbersForRegionFromGenomePosition(rp.getOriginalGenomeRegion());
                        synchronized (blocksNumsToLoadForZd) {
                            for (int blockNumber : tempBlockNumbers) {
                                String key = zd.getBlockKey(blockNumber, no);
                                if (HiCGlobals.useCache
                                        && allBlockCaches.containsKey(zd)
                                        && allBlockCaches.get(zd).containsKey(rp)
                                        && allBlockCaches.get(zd).get(rp).containsKey(key)) {
                                    synchronized (blockList) {
                                        blockList.add(allBlockCaches.get(zd).get(rp).get(key));
                                    }
                                } else if (blocksNumsToLoadForZd.containsKey(zd) && blocksNumsToLoadForZd.get(zd).containsKey(rp)) {
                                    blocksNumsToLoadForZd.get(zd).get(rp).add(blockNumber);
                                } else {
                                    System.err.println("Something went wrong CZDErr3 " + zd.getDescription() +
                                            " rp " + rp.getDescription() + " block num " + blockNumber);
                                }
                            }
                        }
                    }
                };
                executor.execute(worker);
            }
        }
        executor.shutdown();

        // Wait until all threads finish
        while (!executor.isTerminated()) {
        }

        // Actually load new blocks
        actuallyLoadGivenBlocks(blockList, no, blocksNumsToLoadForZd);
        //System.out.println("num blocks post "+blockList.size());

        if (blockList.size() < 1) {
            if (HiCGlobals.printVerboseComments)
                System.err.println("no blocks?? for num x regions " + xAxisRegions.size() + " num y regions " + yAxisRegions.size());
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
        ExecutorService service = HiCGlobals.newFixedThreadPool();

        long[] timesPassed = new long[3];
        long overallTimeStart = System.currentTimeMillis();

        for (final MatrixZoomData zd : blocksNumsToLoadForZd.keySet()) {
            final Map<RegionPair, List<Integer>> blockNumberMap = blocksNumsToLoadForZd.get(zd);
            for (final RegionPair rp : blockNumberMap.keySet()) {
                Runnable loader = new Runnable() {
                    @Override
                    public void run() {
                        for (final int blockNum : blockNumberMap.get(rp)) {
                            try {
                                long time0 = System.currentTimeMillis();
                                String key = zd.getBlockKey(blockNum, no);
                                long time1 = System.currentTimeMillis();
                                Block b = reader.readNormalizedBlock(blockNum, zd, no);
                                long time2 = System.currentTimeMillis();
                                if (b == null) {
                                    b = new Block(blockNum, key + rp.getDescription());   // An empty block
                                } else {
                                    b = modifyBlock(b, key, zd, rp);
                                }
                                long time3 = System.currentTimeMillis();

                                if (HiCGlobals.useCache) {
                                    synchronized (allBlockCaches) {
                                        if (!allBlockCaches.containsKey(zd)) {
                                            allBlockCaches.put(zd, new HashMap<>());
                                        }
                                        if (!allBlockCaches.get(zd).containsKey(rp)) {
                                            allBlockCaches.get(zd).put(rp, new LRUCache<>(50));
                                        }
                                        allBlockCaches.get(zd).get(rp).put(key, b);
                                    }
                                }
                                blockList.add(b);

                                synchronized (timesPassed) {
                                    timesPassed[0] += time1 - time0;
                                    timesPassed[1] += time2 - time1;
                                    timesPassed[2] += time3 - time2;
                                }
                            } catch (IOException e) {
                                System.err.println("--e0 " + zd.getDescription() + " - " + rp.getDescription());
                                errorCounter.incrementAndGet();
                            }
                        }
                    }
                };
                service.submit(loader);
            }
        }

        // done submitting all jobs
        service.shutdown();

        // Wait until all threads finish
        while (!service.isTerminated()) {
        }

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

        long timeFinalOverall = System.currentTimeMillis();
        //System.out.println("Time taken in actuallyLoadGivenBlocks (seconds): " + timesPassed[0] / 1000.0 + " - " + timesPassed[1] / 1000.0 + " - " + timesPassed[2] / 1000.0);

        if (HiCGlobals.printVerboseComments) {
            System.out.println("Time taken overall breakdown (seconds): "
                    + DatasetReaderV2.globalTimeDiffThings[0] + " - "
                    + DatasetReaderV2.globalTimeDiffThings[1] + " - "
                    + DatasetReaderV2.globalTimeDiffThings[2] + " - "
                    + DatasetReaderV2.globalTimeDiffThings[3] + " - "
                    + DatasetReaderV2.globalTimeDiffThings[4]

            );
            System.out.println("Time taken overall (seconds): " + (overallTimeStart - timeFinalOverall) / 1000.0);
        }
        // error printing
        if (errorCounter.get() > 0) {
            System.err.println(errorCounter.get() + " errors while reading blocks");
        }
    }
	
	public List<Long> getBoundariesOfCustomChromosomeX() {
		return rTreeHandler.getBoundariesOfCustomChromosomeX();
	}
	
	public List<Long> getBoundariesOfCustomChromosomeY() {
		return rTreeHandler.getBoundariesOfCustomChromosomeY();
	}

    // TODO get Expected should be appropriately caculated in the custom regions
    public double getExpected(int binX, int binY, ExpectedValueFunction df) {
        // x window
        int gx1 = binX * zoom.getBinSize();
        net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(gx1, gx1, gx1, gx1);
        List<Pair<GenericLocus, GenericLocus>> xRegions = rTreeHandler.getIntersectingFeatures(chr1.getName(), gx1);

        // y window
        int gy1 = binY * zoom.getBinSize();
        currentWindow = new net.sf.jsi.Rectangle(gy1, gy1, gy1, gy1);
        List<Pair<GenericLocus, GenericLocus>> yRegions = rTreeHandler.getIntersectingFeatures(chr2.getName(), gy1);

        RegionPair rp = RegionPair.generateRegionPair(xRegions.get(0), yRegions.get(0), handler);
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

    @Override
    public double getAverageCount() {
        return 10; //todo
    }


    public List<Pair<GenericLocus, GenericLocus>> getRTreeHandlerIntersectingFeatures(String name, int g1, int g2) {
        return rTreeHandler.getIntersectingFeatures(name, g1, g2);
    }
}
