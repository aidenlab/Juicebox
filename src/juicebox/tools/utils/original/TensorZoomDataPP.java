/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2022 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.utils.original;

import htsjdk.tribble.util.LittleEndianInputStream;
import htsjdk.tribble.util.LittleEndianOutputStream;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.ContactRecord;
import juicebox.data.basics.Chromosome;
import juicebox.data.v9depth.V9Depth;
import juicebox.tools.utils.original.stats.PointTriple;
import juicebox.windowui.HiCZoom;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.broad.igv.tdf.BufferedByteWriter;
import org.broad.igv.util.collections.DownsampledDoubleArrayList;

import java.awt.*;
import java.io.*;
import java.util.List;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.zip.Deflater;

public class TensorZoomDataPP {

//    final boolean isFrag;
    final Set<Integer> blockNumbers;  // The only reason for this is to get a count
    final ConcurrentHashMap<Integer, Integer> blockNumRecords;
    final List<File> tmpFiles;
    final Map<Integer, Map<File, Long>> tmpFilesByBlockNumber;
//    private final Chromosome chr1;  // Redundant, but convenient    BinDatasetReader
//    private final Chromosome chr2;  // Redundant, but convenient
//    private final Chromosome chr3; // Redundant, but convenient
    private final Chromosome[] sortedChrs;
    private final int[] unsorted2Sorted;
    private final int[] sorted2Unsorted;
    private final int numSameChrs;
    private final int[] sameChrs; // 0 if not the same with any other chromsomes; 1 otherwise;
    private final int zoom;
    private final int binSize;              // bin size in bp
    private final int blockBinCountZ;        // block size in bins in x dimension
    private final int blockBinCountY;        // block size in bins in y dimension
    private final int blockBinCountX;        // block size in bins in z dimension
    private final int blockXCount;     // number of block columns
    private final int blockYCount;        // number of block rows
    private final int blockZCount;     // number of block depths
    private final LinkedHashMap<Integer, TensorBlockPP> blocks;
    private final int countThreshold;
    long blockIndexPosition;
    private double sum = 0;
    private double numRecords = 0;
    private double cellCount = 0;
    private double percent5;
    private double percent95;
    private int BLOCK_CAPACITY = 1000;
//    private final V9Depth v9Depth;

    /**
     * Representation of MatrixZoomData used for preprocessing
     *
     * @param chr1             index of first chromosome  (x-axis)
     * @param chr2             index of second chromosome
     * @param binSize          size of each grid bin in bp
     * @param zoom             integer zoom (resolution) level index.  TODO Is this needed?
     */
    TensorZoomDataPP(Chromosome chr1, Chromosome chr2, Chromosome chr3, int binSize, int blockXCount, int blockYCount,
                     int blockZCount, int zoom, int countThreshold) {
        this.tmpFiles = new ArrayList<>();
        this.tmpFilesByBlockNumber = new ConcurrentHashMap<>();
        this.blockNumbers = Collections.synchronizedSet(new HashSet<>(BLOCK_CAPACITY));
        this.blockNumRecords = new ConcurrentHashMap<>(BLOCK_CAPACITY);
        this.countThreshold = countThreshold;

        sameChrs = new int[3];

        long chr1Len = chr1.getLength();
        long chr2Len = chr2.getLength();
        long chr3Len = chr3.getLength();

        int count = 0;
        if (chr1Len == chr2Len ) {
            count += 1;
            sameChrs[0] = 1;
            sameChrs[1] = 1;
        }
        if (chr2Len == chr3Len) {
            count += 1;
            sameChrs[1] = 1;
            sameChrs[2] = 1;
        }
        if (chr1Len == chr3Len) {
            count += 1;
            sameChrs[0] = 1;
            sameChrs[2] = 1;
        }
        numSameChrs = count;

        if (numSameChrs == 3) {
            sortedChrs = new Chromosome[] {chr1, chr2, chr3};
            unsorted2Sorted = new int[] {0, 1, 2};
            sorted2Unsorted = new int[] {0, 1, 2};
        } else {
            sortedChrs = new Chromosome[3];
            unsorted2Sorted = new int[3];
            sorted2Unsorted = new int[3];
            // this.chr1 length >= this.chr2 length >= this.chr3 length
            if (chr1Len >= chr2Len && chr1Len >= chr3Len) {
                // input argument chr1 is the largest;
                sortedChrs[0] = chr1;
                unsorted2Sorted[0] = 0;
                sorted2Unsorted[0] = 0;
                if (chr2Len >= chr3Len) {
                    sortedChrs[1] = chr2;
                    unsorted2Sorted[1] = 1;
                    sorted2Unsorted[1] = 1;
                    sortedChrs[2] = chr3;
                    unsorted2Sorted[2] = 2;
                    sorted2Unsorted[2] = 2;
                } else {
                    sortedChrs[1] = chr3;
                    unsorted2Sorted[2] = 1;
                    sorted2Unsorted[1] = 2;
                    sortedChrs[2] = chr2;
                    unsorted2Sorted[1] = 2;
                    sorted2Unsorted[2] = 1;
                }
            } else if (chr2Len >= chr3Len) {
                // input argument chr2 is the largest
                sortedChrs[0] = chr2;
                unsorted2Sorted[1] = 0;
                sorted2Unsorted[0] = 1;
                if (chr1Len >= chr3Len) {
                    sortedChrs[1] = chr1;
                    unsorted2Sorted[0] = 1;
                    sorted2Unsorted[1] = 0;
                    sortedChrs[2] = chr3;
                    unsorted2Sorted[2] = 2;
                    sorted2Unsorted[2] = 2;
                } else {
                    sortedChrs[1] = chr3;
                    unsorted2Sorted[2] = 1;
                    sorted2Unsorted[1] = 2;
                    sortedChrs[2] = chr1;
                    unsorted2Sorted[0] = 2;
                    sorted2Unsorted[2] = 0;
                }
            } else {
                // input argument chr3 is the largest
                sortedChrs[0] = chr3;
                unsorted2Sorted[2] = 0;
                sorted2Unsorted[0] = 2;
                if (chr1Len >= chr2Len) {
                    sortedChrs[1] = chr1;
                    unsorted2Sorted[0] = 1;
                    sorted2Unsorted[1] = 0;
                    sortedChrs[2] = chr2;
                    unsorted2Sorted[1] = 2;
                    sorted2Unsorted[2] = 1;
                } else {
                    sortedChrs[1] = chr2;
                    unsorted2Sorted[1] = 1;
                    sorted2Unsorted[1] = 1;
                    sortedChrs[2] = chr1;
                    unsorted2Sorted[0] = 2;
                    sorted2Unsorted[2] = 0;
                }
            }
            int[] tmpSameChrs = new int[3];
            tmpSameChrs[0] = sameChrs[unsorted2Sorted[0]];
            tmpSameChrs[1] = sameChrs[unsorted2Sorted[1]];
            tmpSameChrs[2] = sameChrs[unsorted2Sorted[2]];
            sameChrs[0] = tmpSameChrs[0];
            sameChrs[1] = tmpSameChrs[1];
            sameChrs[2] = tmpSameChrs[2];
        }

        this.binSize = binSize;
        this.blockXCount = blockXCount;
        this.blockYCount = blockYCount;
        this.blockZCount = blockZCount;
        this.zoom = zoom;

        int nBinsX = (int) (sortedChrs[0].getLength() / binSize + 1);
        int nBinsY = (int) (sortedChrs[1].getLength() / binSize + 1);
        int nBinsZ = (int) (sortedChrs[2].getLength() / binSize + 1);
        this.blockBinCountX = nBinsX / this.blockXCount + 1;
        this.blockBinCountY = nBinsY / this.blockYCount + 1;
        this.blockBinCountZ = nBinsZ / this.blockZCount + 1;
        blocks = new LinkedHashMap<>(blockBinCountZ);
    }

    TensorZoomDataPP(Chromosome chr1, Chromosome chr2, Chromosome chr3, int binSize, int blockXCount, int blockYCount,
                     int blockZCount, int zoom, int countThreshold, int BLOCK_CAPACITY) {
        this.tmpFiles = new ArrayList<>();
        this.tmpFilesByBlockNumber = new ConcurrentHashMap<>();
        this.BLOCK_CAPACITY = BLOCK_CAPACITY;
        this.blockNumbers = Collections.synchronizedSet(new HashSet<>(BLOCK_CAPACITY));
        this.blockNumRecords = new ConcurrentHashMap<>(BLOCK_CAPACITY);
        this.countThreshold = countThreshold;

        sameChrs = new int[3];

        long chr1Len = chr1.getLength();
        long chr2Len = chr2.getLength();
        long chr3Len = chr3.getLength();

        int count = 0;
        if (chr1Len == chr2Len ) {
            count += 1;
            sameChrs[0] = 1;
            sameChrs[1] = 1;
        }
        if (chr2Len == chr3Len) {
            count += 1;
            sameChrs[1] = 1;
            sameChrs[2] = 1;
        }
        if (chr1Len == chr3Len) {
            count += 1;
            sameChrs[0] = 1;
            sameChrs[2] = 1;
        }
        numSameChrs = count;

        if (numSameChrs == 3) {
            sortedChrs = new Chromosome[] {chr1, chr2, chr3};
            unsorted2Sorted = new int[] {0, 1, 2};
            sorted2Unsorted = new int[] {0, 1, 2};
        } else {
            sortedChrs = new Chromosome[3];
            unsorted2Sorted = new int[3];
            sorted2Unsorted = new int[3];
            // this.chr1 length >= this.chr2 length >= this.chr3 length
            if (chr1Len >= chr2Len && chr1Len >= chr3Len) {
                // input argument chr1 is the largest;
                sortedChrs[0] = chr1;
                unsorted2Sorted[0] = 0;
                sorted2Unsorted[0] = 0;
                if (chr2Len >= chr3Len) {
                    sortedChrs[1] = chr2;
                    unsorted2Sorted[1] = 1;
                    sorted2Unsorted[1] = 1;
                    sortedChrs[2] = chr3;
                    unsorted2Sorted[2] = 2;
                    sorted2Unsorted[2] = 2;
                } else {
                    sortedChrs[1] = chr3;
                    unsorted2Sorted[2] = 1;
                    sorted2Unsorted[1] = 2;
                    sortedChrs[2] = chr2;
                    unsorted2Sorted[1] = 2;
                    sorted2Unsorted[2] = 1;
                }
            } else if (chr2Len >= chr3Len) {
                // input argument chr2 is the largest
                sortedChrs[0] = chr2;
                unsorted2Sorted[1] = 0;
                sorted2Unsorted[0] = 1;
                if (chr1Len >= chr3Len) {
                    sortedChrs[1] = chr1;
                    unsorted2Sorted[0] = 1;
                    sorted2Unsorted[1] = 0;
                    sortedChrs[2] = chr3;
                    unsorted2Sorted[2] = 2;
                    sorted2Unsorted[2] = 2;
                } else {
                    sortedChrs[1] = chr3;
                    unsorted2Sorted[2] = 1;
                    sorted2Unsorted[1] = 2;
                    sortedChrs[2] = chr1;
                    unsorted2Sorted[0] = 2;
                    sorted2Unsorted[2] = 0;
                }
            } else {
                // input argument chr3 is the largest
                sortedChrs[0] = chr3;
                unsorted2Sorted[2] = 0;
                sorted2Unsorted[0] = 2;
                if (chr1Len >= chr2Len) {
                    sortedChrs[1] = chr1;
                    unsorted2Sorted[0] = 1;
                    sorted2Unsorted[1] = 0;
                    sortedChrs[2] = chr2;
                    unsorted2Sorted[1] = 2;
                    sorted2Unsorted[2] = 1;
                } else {
                    sortedChrs[1] = chr2;
                    unsorted2Sorted[1] = 1;
                    sorted2Unsorted[1] = 1;
                    sortedChrs[2] = chr1;
                    unsorted2Sorted[0] = 2;
                    sorted2Unsorted[2] = 0;
                }
            }
            int[] tmpSameChrs = new int[3];
            tmpSameChrs[0] = sameChrs[unsorted2Sorted[0]];
            tmpSameChrs[1] = sameChrs[unsorted2Sorted[1]];
            tmpSameChrs[2] = sameChrs[unsorted2Sorted[2]];
            sameChrs[0] = tmpSameChrs[0];
            sameChrs[1] = tmpSameChrs[1];
            sameChrs[2] = tmpSameChrs[2];
        }

        this.binSize = binSize;
        this.blockXCount = blockXCount;
        this.blockYCount = blockYCount;
        this.blockZCount = blockZCount;
        this.zoom = zoom;

        int nBinsX = (int) (sortedChrs[0].getLength() / binSize + 1);
        int nBinsY = (int) (sortedChrs[1].getLength() / binSize + 1);
        int nBinsZ = (int) (sortedChrs[2].getLength() / binSize + 1);
        this.blockBinCountX = nBinsX / this.blockXCount + 1;
        this.blockBinCountY = nBinsY / this.blockYCount + 1;
        this.blockBinCountZ = nBinsZ / this.blockZCount + 1;
//        /*Considering the 75% load factor*/
//        Double depth75Loading = blockBinCountZ / 0.75;
        blocks = new LinkedHashMap<>(blockBinCountZ);
    }

    HiC.Unit getUnit() {
        return HiC.Unit.BP;
    }

    double getSum() {
        return sum;
    }

    double getOccupiedCellCount() {
        return cellCount;
    }

    double getPercent95() {
        return percent95;
    }

    double getPercent5() {
        return percent5;
    }


    int getBinSize() {
        return binSize;
    }

    Chromosome getUnsortedChr1() {
        return sortedChrs[unsorted2Sorted[0]];
    }

    Chromosome getUnsortedChr2() {
        return sortedChrs[unsorted2Sorted[1]];
    }

    Chromosome getUnsortedChr3() {
        return sortedChrs[unsorted2Sorted[2]];
    }

    Chromosome getSortedChr1() {
        return sortedChrs[0];
    }

    Chromosome getSortedChr2() {
        return sortedChrs[1];
    }

    Chromosome getSortedChr3() {
        return sortedChrs[2];
    }

    int getZoom() {
        return zoom;
    }

    int getBlockBinCountX() {
        return blockBinCountX;
    }

    int getBlockBinCountY() {
        return blockBinCountY;
    }

    int getBlockBinCountZ() {
        return blockBinCountZ;
    }

    int getBlockXCount() {
        return blockXCount;
    }

    int getBlockYCount() {
        return blockYCount;
    }

    int getBlockZCount() {
        return blockZCount;
    }


    Map<Integer, TensorBlockPP> getBlocks() {
        return blocks;
    }

    /**
     * Increment the count for the bin represented by the GENOMIC position (pos1, pos2, pos3)
     */
    void incrementCount(int pos1, int pos2, int pos3, float score, Map<String, ExpectedValueCalculation> expectedValueCalculations,
                        File tmpDir) throws IOException {

        sum += score;
        if (pos1 < 0 || pos2 < 0) return;
        if (pos3 < 0) return;

        int[] unsortedPosList = new int[] {pos1, pos2, pos3};

        int sortedPos1;
        int sortedPos2;
        int sortedPos3;

        // Case 1: if all three chromsomes are the same; just resort based on numeric order
        // unsorted2sorted mapping in this case is an identity map
        if (numSameChrs == 3) {
            Arrays.sort(unsortedPosList);
            sortedPos1 = unsortedPosList[0];
            sortedPos2 = unsortedPosList[1];
            sortedPos3 = unsortedPosList[2];
        } else {
            int[] sortedPosList = new int[3];
            for (int i = 0; i < unsortedPosList.length; i++) {
                sortedPosList[unsorted2Sorted[i]] = unsortedPosList[i];
            }
            // Case 2: if two same chrs, need to resort based on ascending order;
            if (numSameChrs == 2) {
                int tmp;
                if (sameChrs[0] == 1 && sameChrs[1] == 1) {
                    tmp = Math.min(sortedPosList[0], sortedPosList[1]);
                    sortedPosList[1] = Math.max(sortedPosList[0], sortedPosList[1]);
                    sortedPosList[0] = tmp;
                } else if (sameChrs[0] == 1 && sameChrs[2] == 1) {
                    tmp = Math.min(sortedPosList[0], sortedPosList[2]);
                    sortedPosList[2] = Math.max(sortedPosList[0], sortedPosList[2]);
                    sortedPosList[0] = tmp;
                } else if (sameChrs[1] == 1 && sameChrs[2] == 1) {
                    tmp = Math.min(sortedPosList[1], sortedPosList[2]);
                    sortedPosList[2] = Math.max(sortedPosList[1], sortedPosList[2]);
                    sortedPosList[1] = tmp;
                } else {
                    System.err.println("Error in counting sameChrs!\n");
                }
            }
            // if no two chromsomes are the same, no need to resort;
            sortedPos1 = sortedPosList[0];
            sortedPos2 = sortedPosList[1];
            sortedPos3 = sortedPosList[2];
        }

        int xBin = sortedPos1 / binSize;
        int yBin = sortedPos2 / binSize;
        int zBin = sortedPos3 / binSize;

        int blockX = xBin / blockBinCountX;
        int blockY = yBin / blockBinCountY;
        int blockZ = zBin / blockBinCountZ;
        int blockNumber = blockZ * (blockXCount * blockYCount) + blockY * blockXCount + blockX;

        TensorBlockPP block = blocks.get(blockNumber);
        if (block == null) {
            block = new TensorBlockPP(blockNumber);
            blocks.put(blockNumber, block);
        }
        block.incrementCount(xBin, yBin, zBin, score);

        // If too many blocks write to tmp directory
        if (blocks.size() > BLOCK_CAPACITY) {
            File tmpfile = tmpDir == null ? File.createTempFile("blocks", "bin") : File.createTempFile("blocks", "bin", tmpDir);
            //System.out.println(chr1.getName() + "-" + chr2.getName() + " Dumping blocks to " + tmpfile.getAbsolutePath());
            dumpBlocks(tmpfile);
            tmpFiles.add(tmpfile);
            tmpfile.deleteOnExit();
        }
    }

    /**
     * Dump the blocks calculated so far to a temporary file
     *
     * @param file File to write to
     * @throws IOException
     */
    private void dumpBlocks(File file) throws IOException {
        LittleEndianOutputStream los = null;
        try {
            if (HiCGlobals.printVerboseComments) {
                System.err.println("Used Memory prior to dumping blocks " + binSize);
                System.err.println(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());
            }
            // TODO Question: How to determine the file size here?
            // exactly 4MB; might need to adjust for performance gains;
            los = new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(file), 4194304));

            List<TensorBlockPP> blockList = new ArrayList<>(blocks.values());
            Collections.sort(blockList, new Comparator<TensorBlockPP>() {
                @Override
                public int compare(TensorBlockPP o1, TensorBlockPP o2) {
                    return o1.getNumber() - o2.getNumber();
                }
            });

            for (TensorBlockPP b : blockList) {

                // Remove from map
                blocks.remove(b.getNumber());

                int number = b.getNumber();
                blockNumbers.add(number);

                if (!blockNumRecords.containsKey(number)) {
                    blockNumRecords.put(number, b.getNumRecords());
                } else {
                    blockNumRecords.put(number, blockNumRecords.get(number)+b.getNumRecords());
                }
                numRecords += b.getNumRecords();

                if (tmpFilesByBlockNumber.get(number)==null) {
                    tmpFilesByBlockNumber.put(number, new ConcurrentHashMap<>());
                }
                tmpFilesByBlockNumber.get(number).put(file, los.getWrittenCount());

                los.writeInt(number);
                Map<PointTriple, ContactCount> records = b.getContactRecordMap();

                los.writeInt(records.size());
                for (Map.Entry<PointTriple, ContactCount> entry : records.entrySet()) {

                    PointTriple point = entry.getKey();
                    ContactCount count = entry.getValue();

                    los.writeInt(point.getFirst());
                    los.writeInt(point.getSecond());
                    los.writeInt(point.getThird());
                    los.writeFloat(count.getCounts());
                }
                b.clear();
            }

            blocks.clear();
            if (HiCGlobals.printVerboseComments) {
                System.err.println("Used Memory after dumping blocks " + binSize);
                System.err.println(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());
            }
        } finally {
            if (los != null) los.close();

        }
    }

    // Merge and write out blocks one at a time.
    protected List<IndexEntry> mergeAndWriteBlocks(LittleEndianOutputStream los, Deflater compressor) throws IOException {
        DownsampledDoubleArrayList sampledData = new DownsampledDoubleArrayList(10000, 10000);

        List<TensorBlockQueue> activeList = new ArrayList<>();

        // Initialize queues -- first whatever is left over in memory
        if (blocks.size() > 0) {
            TensorBlockQueue bqInMem = new TensorBlockQueueMem(blocks.values());
            activeList.add(bqInMem);
        }
        // Now from files
        for (File file : tmpFiles) {
            TensorBlockQueue bq = new TensorBlockQueueFB(file);
            if (bq.getBlock() != null) {
                activeList.add(bq);
            }
        }

        List<IndexEntry> indexEntries = new ArrayList<>();

        if (activeList.size() == 0) {
            throw new RuntimeException("No reads in Hi-C contact matrices. This could be because the MAPQ filter is set too high (-q) or because all reads map to the same fragment.");
        }

        // TODO: Question: need to look more carefully about the do-while loop here!
        do {
            activeList.sort(new Comparator<TensorBlockQueue>() {
                @Override
                public int compare(TensorBlockQueue o1, TensorBlockQueue o2) {
                    return o1.getBlock().getNumber() - o2.getBlock().getNumber();
                }
            });

            TensorBlockQueue topQueue = activeList.get(0);
            TensorBlockPP currentBlock = topQueue.getBlock();
            topQueue.advance();
            int num = currentBlock.getNumber();

            for (int i = 1; i < activeList.size(); i++) {
                TensorBlockQueue blockQueue = activeList.get(i);
                TensorBlockPP block = blockQueue.getBlock();
                if (block.getNumber() == num) {
                    currentBlock.merge(block);
                    blockQueue.advance();
                }
            }

            Iterator<TensorBlockQueue> iterator = activeList.iterator();
            while (iterator.hasNext()) {
                if (iterator.next().getBlock() == null) {
                    iterator.remove();
                }
            }

            // Output block
            long position = los.getWrittenCount();
            writeBlock(currentBlock, sampledData, los, compressor);
            long size = los.getWrittenCount() - position;

            indexEntries.add(new IndexEntry(num, position, (int) size));

        } while (activeList.size() > 0);


        for (File f : tmpFiles) {
            boolean result = f.delete();
            if (!result) {
                System.out.println("Error while deleting file");
            }
        }

        computeStats(sampledData);
        return indexEntries;
    }

    private void computeStats(DownsampledDoubleArrayList sampledData) {

        DescriptiveStatistics stats = new DescriptiveStatistics(sampledData.toArray());
        this.percent5 = stats.getPercentile(5);
        this.percent95 = stats.getPercentile(95);
    }

    /*Question: What's the purpose of parsing complete here?*/
    void parsingComplete() {
        // Add the block numbers still in memory
        for (TensorBlockPP block : blocks.values()) {
            int number = block.getNumber();
            blockNumbers.add(number);
            if (!blockNumRecords.containsKey(number)) {
                blockNumRecords.put(number, block.getNumRecords());
            } else {
                blockNumRecords.put(number, blockNumRecords.get(number)+block.getNumRecords());
            }
            numRecords += block.getNumRecords();
        }
        // TODO: question should we remove these blocks from the memory then?
    }

    /**
     * Note -- compressed
     *
     * @param block       Block to write
     * @param sampledData Array to hold a sample of the data (to compute statistics)
     * @throws IOException
     */
    protected void writeBlock(TensorBlockPP block, DownsampledDoubleArrayList sampledData, LittleEndianOutputStream los, Deflater compressor) throws IOException {

        final Map<PointTriple, ContactCount> records = block.getContactRecordMap();//   getContactRecords();
        // System.out.println("Write contact records : records count = " + records.size());

        // Count records first
        int nRecords;
        if (countThreshold > 0) {
            nRecords = 0;
            for (ContactCount rec : records.values()) {
                if (rec.getCounts() >= countThreshold) {
                    nRecords++;
                }
            }
        } else {
            nRecords = records.size();
        }
        // TODO: need to be more sure about the size for each record!
        BufferedByteWriter buffer = new BufferedByteWriter(nRecords * 16);
        buffer.putInt(nRecords);
        incrementCellCount(nRecords);

        // Find extents of occupied cells
        int binXOffset = Integer.MAX_VALUE;
        int binYOffset = Integer.MAX_VALUE;
        int binZOffset = Integer.MAX_VALUE;
        int binXMax = 0;
        int binYMax = 0;
        int binZMax = 0;
        for (Map.Entry<PointTriple, ContactCount> entry : records.entrySet()) {
            PointTriple point = entry.getKey();
            binXOffset = Math.min(binXOffset, point.getFirst());
            binYOffset = Math.min(binYOffset, point.getSecond());
            binZOffset = Math.min(binZOffset, point.getThird());
            binXMax = Math.max(binXMax, point.getFirst());
            binYMax = Math.max(binYMax, point.getSecond());
            binZMax = Math.max(binZMax, point.getThird());
        }

        // TODO: question for blockoffset, also only include occupied bins?
        buffer.putInt(binXOffset);
        buffer.putInt(binYOffset);
        buffer.putInt(binZOffset);

        // Sort keys in a slice-major order, and then a row-major order
        List<PointTriple> keys = new ArrayList<>(records.keySet());
        keys.sort(new Comparator<PointTriple>() {
            @Override
            public int compare(PointTriple o1, PointTriple o2) {
                if (o1.getThird() != o2.getThird()) {
                    return o1.getThird() - o2.getThird();
                } else {
                    if (o1.getSecond() != o2.getSecond()) {
                        return o1.getSecond() - o2.getSecond();
                    } else {
                        return o1.getFirst() - o2.getFirst();
                    }
                }
            }
        });

        PointTriple lastPoint = keys.get(keys.size() - 1);
        // TODO: usage of this short w here!
        final short w = (short) (binXMax - binXOffset + 1);
        final int w1 = binXMax - binXOffset + 1;
        final int w2 = binYMax - binYOffset + 1;
        final int w3 = binZMax - binZOffset + 1;

        boolean isInteger = true;
        float maxCounts = 0;

        // Could also store as a list of pairs (third coord, counts); second coord is wasted
        LinkedHashMap<Integer, LinkedHashMap<Integer, List<ContactRecord>>> slices = new LinkedHashMap<>();
        for (PointTriple point : keys) {
            final ContactCount contactCount = records.get(point);
            float counts = contactCount.getCounts();
            if (counts >= countThreshold) {
                isInteger = isInteger && (Math.floor(counts) == counts);
                maxCounts = Math.max(counts, maxCounts);

                final int px = point.getFirst() - binXOffset;
                final int py = point.getSecond() - binYOffset;
                final int pz = point.getThird() - binZOffset;
                LinkedHashMap<Integer, List<ContactRecord>> slice = slices.get(pz);
                List<ContactRecord> row;
                if (slice == null) {
                    // Create a new slice;
                    slice = new LinkedHashMap<>(w2);
                    // then add a new row in that slice!
                    row = new ArrayList<>(w1);
                    slice.put(py, row);
                } else {
                    row = slice.get(py);
                    // Create a new row
                    if (row == null) {
                        row = new ArrayList<>(w1);
                        slice.put(py, row);
                    }
                }
                row.add(new ContactRecord(px, py, counts));
            }
        }

        // Compute size for each representation and choose smallest
        boolean useShort = isInteger && (maxCounts < Short.MAX_VALUE);
        boolean useShortBinX = w1 < Short.MAX_VALUE;
        boolean useShortBinY = w2 < Short.MAX_VALUE;
        boolean useShortBinZ = w3 < Short.MAX_VALUE;
        int valueSize = useShort ? 2 : 4;

        int lorSize = 0;
        // TODO: could compute the required dense size later!
//        int nDensePts = (lastPoint.getSecond() - binYOffset) * w + (lastPoint.getFirst() - binXOffset) + 1;
//
//        int denseSize = nDensePts * valueSize;
        for (Map.Entry<Integer, LinkedHashMap<Integer, List<ContactRecord>>> entry : slices.entrySet()) {
            LinkedHashMap<Integer, List<ContactRecord>> rows = entry.getValue();
            for (List<ContactRecord> row : rows.values()) {
                // TODO: why is it only row.size() here? Why not 2x?
                lorSize += 4 + row.size() * valueSize;
            }
        }

        buffer.put((byte) (useShort ? 0 : 1));
        buffer.put((byte) (useShortBinX ? 0 : 1));
        buffer.put((byte) (useShortBinY ? 0 : 1));
        buffer.put((byte) (useShortBinZ ? 0 : 1));

        //dense calculation is incorrect for v9
        int denseSize = Integer.MAX_VALUE;

        if (lorSize < denseSize) {
            buffer.put((byte) 1);  // List of slices representation
            if (useShortBinZ) {
                buffer.putShort((short) slices.size()); // # of rows
            } else {
                buffer.putInt(slices.size());  // # of rows
            }

            for (Map.Entry<Integer, LinkedHashMap<Integer, List<ContactRecord>>> entry : slices.entrySet()) {
                int pz = entry.getKey();
                LinkedHashMap<Integer, List<ContactRecord>> rows = entry.getValue();
                if (useShortBinZ) {
                    buffer.putShort((short) pz); // Slice number
                } else {
                    buffer.putInt(pz); // Slice number
                }
                if (useShortBinY) {
                    buffer.putShort((short) rows.size()); // # of rows
                } else {
                    buffer.putInt(rows.size());  // # of rows
                }
                for (Map.Entry<Integer, List<ContactRecord>> entry2 : rows.entrySet()) {
                    int py = entry2.getKey();
                    List<ContactRecord> row = entry2.getValue();
                    if (useShortBinY) {
                        buffer.putShort((short) py);  // Row number
                    } else {
                        buffer.putInt(py); // Row number
                    }
                    if (useShortBinX) {
                        buffer.putShort((short) row.size());  // size of row
                    } else {
                        buffer.putInt(row.size()); // size of row
                    }

                    for (ContactRecord contactRecord : row) {
                        if (useShortBinX) {
                            buffer.putShort((short) (contactRecord.getBinX()));
                        } else {
                            buffer.putInt(contactRecord.getBinX());
                        }

                        final float counts = contactRecord.getCounts();
                        if (useShort) {
                            buffer.putShort((short) counts);
                        } else {
                            buffer.putFloat(counts);
                        }

                        synchronized (sampledData) {
                            sampledData.add(counts);
                        }
                        incrementSum(counts);
                    }

                }

            }
        } else {
            System.err.println("Error: triplets data not be able to stored in dense tensors\n");
            return;
        }

            byte[] bytes = buffer.getBytes();
            byte[] compressedBytes = compress(bytes, compressor);
            los.write(compressedBytes);
    }

    private synchronized void incrementSum(float counts) {
        sum += counts;
    }

    private synchronized void incrementCellCount(int nRecords) {
        cellCount += nRecords;
    }

    /**
     * todo should this be synchronized?
     *
     * @param data
     * @param compressor
     * @return
     */
    protected byte[] compress(byte[] data, Deflater compressor) {

        // Give the compressor the data to compress
        compressor.reset();
        compressor.setInput(data);
        compressor.finish();

        // Create an expandable byte array to hold the compressed data.
        // You cannot use an array that's the same size as the orginal because
        // there is no guarantee that the compressed data will be smaller than
        // the uncompressed data.
        ByteArrayOutputStream bos = new ByteArrayOutputStream(data.length);

        // Compress the data
        byte[] buf = new byte[1024];
        while (!compressor.finished()) {
            int count = compressor.deflate(buf);
            bos.write(buf, 0, count);
        }
        try {
            bos.close();
        } catch (IOException e) {
            System.err.println("Error clossing ByteArrayOutputStream");
            e.printStackTrace();
        }

        return bos.toByteArray();
    }
}
