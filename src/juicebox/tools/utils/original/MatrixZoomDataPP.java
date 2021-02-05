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

package juicebox.tools.utils.original;

import htsjdk.tribble.util.LittleEndianOutputStream;
import juicebox.HiC;
import juicebox.data.ContactRecord;
import juicebox.data.basics.Chromosome;
import juicebox.data.v9depth.V9Depth;
import juicebox.windowui.HiCZoom;
import org.apache.commons.math.stat.StatUtils;
import org.broad.igv.tdf.BufferedByteWriter;
import org.broad.igv.util.collections.DownsampledDoubleArrayList;

import java.awt.*;
import java.io.*;
import java.util.List;
import java.util.*;
import java.util.zip.Deflater;

public class MatrixZoomDataPP {

    final boolean isFrag;
    final Set<Integer> blockNumbers;  // The only reason for this is to get a count
    final List<File> tmpFiles;
    private final Chromosome chr1;  // Redundant, but convenient    BinDatasetReader
    private final Chromosome chr2;  // Redundant, but convenient
    private final int zoom;
    private final int binSize;              // bin size in bp
    private final int blockBinCount;        // block size in bins
    private final int blockColumnCount;     // number of block columns
    private final LinkedHashMap<Integer, BlockPP> blocks;
    private final int countThreshold;
    long blockIndexPosition;
    private double sum = 0;
    private double cellCount = 0;
    private double percent5;
    private double percent95;
    public static int BLOCK_CAPACITY = 1000;
    private final V9Depth v9Depth;

    /**
     * Representation of MatrixZoomData used for preprocessing
     *
     * @param chr1             index of first chromosome  (x-axis)
     * @param chr2             index of second chromosome
     * @param binSize          size of each grid bin in bp
     * @param blockColumnCount number of block columns
     * @param zoom             integer zoom (resolution) level index.  TODO Is this needed?
     */
    MatrixZoomDataPP(Chromosome chr1, Chromosome chr2, int binSize, int blockColumnCount, int zoom, boolean isFrag,
                     FragmentCalculation fragmentCalculation, int countThreshold, int v9BaseDepth) {
        this.tmpFiles = new ArrayList<>();
        this.blockNumbers = new HashSet<>(BLOCK_CAPACITY);
        this.countThreshold = countThreshold;

        this.sum = 0;
        this.chr1 = chr1;
        this.chr2 = chr2;
        this.binSize = binSize;
        this.blockColumnCount = blockColumnCount;
        this.zoom = zoom;
        this.isFrag = isFrag;

        // Get length in proper units
        Chromosome longChr = chr1.getLength() > chr2.getLength() ? chr1 : chr2;
        long len = isFrag ? fragmentCalculation.getNumberFragments(longChr.getName()) : longChr.getLength();
    
        int nBinsX = (int) (len / binSize + 1);

        blockBinCount = nBinsX / blockColumnCount + 1;
        blocks = new LinkedHashMap<>(blockColumnCount);
        v9Depth = V9Depth.setDepthMethod(v9BaseDepth, blockBinCount);
    }

    HiC.Unit getUnit() {
        return isFrag ? HiC.Unit.FRAG : HiC.Unit.BP;
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


    Chromosome getChr1() {
        return chr1;
    }


    Chromosome getChr2() {
        return chr2;
    }

    int getZoom() {
        return zoom;
    }

    int getBlockBinCount() {
        return blockBinCount;
    }

    int getBlockColumnCount() {
        return blockColumnCount;
    }

    Map<Integer, BlockPP> getBlocks() {
        return blocks;
    }

    /**
     * Increment the count for the bin represented by the GENOMIC position (pos1, pos2)
     */
    void incrementCount(int pos1, int pos2, float score, Map<String, ExpectedValueCalculation> expectedValueCalculations,
                        File tmpDir) throws IOException {

        sum += score;
        // Convert to proper units,  fragments or base-pairs

        if (pos1 < 0 || pos2 < 0) return;

        int xBin = pos1 / binSize;
        int yBin = pos2 / binSize;

        int blockNumber;

        // Intra chromosome -- we'll store lower diagonal only
        if (chr1.equals(chr2)) {
            int b1 = Math.min(xBin, yBin);
            int b2 = Math.max(xBin, yBin);
            xBin = b1;
            yBin = b2;

            if (b1 != b2) {
                sum += score;  // <= count for mirror cell.
            }

            if (expectedValueCalculations != null) {
                String evKey = (isFrag ? "FRAG_" : "BP_") + binSize;
                ExpectedValueCalculation ev = expectedValueCalculations.get(evKey);
                if (ev != null) {
                    ev.addDistance(chr1.getIndex(), xBin, yBin, score);
                }
            }

            //compute intra chromosomal block number (version 9 and up)
            int depth = v9Depth.getDepth(xBin, yBin);
            int positionAlongDiagonal = ((xBin + yBin) / 2 / blockBinCount);
            blockNumber = depth * blockColumnCount + positionAlongDiagonal;
        }
        else {
            // compute interchromosomal block number (version 9 and up, first block is zero)
            int blockCol = xBin / blockBinCount;
            int blockRow = yBin / blockBinCount;
            blockNumber = blockColumnCount * blockRow + blockCol;
        }


        BlockPP block = blocks.get(blockNumber);
        if (block == null) {

            block = new BlockPP(blockNumber);
            blocks.put(blockNumber, block);
        }
        block.incrementCount(xBin, yBin, score);

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
     * Increment the count for the bin represented by the CONTACT RECORD for a given ZOOM
     */
    public void  incrementCount(ContactRecord cr, Map<String, ExpectedValueCalculation> expectedValueCalculations,
                        File tmpDir, HiCZoom recordZoom) throws IOException {

        float score = cr.getCounts();
        sum += score;
        // Convert to proper units,  fragments or base-pairs

        if (cr.getBinX() < 0 || cr.getBinY() < 0) return;
        if (recordZoom.getBinSize() > binSize) return;
        //if (binSize % recordZoom.getBinSize() > 0) return;
        float rescale = (float) binSize / (float) recordZoom.getBinSize();
        float thresholdX;
        float thresholdY;
        Random generator = new Random(0);
        int xBin;
        int yBin;

        if ( binSize % recordZoom.getBinSize() == 0) {
            xBin = (int) ((float) cr.getBinX() / rescale);
            yBin = (int) ((float) cr.getBinY() / rescale);
        } else {
            if ((int) ((float) cr.getBinX() / rescale) == (int) ((float) cr.getBinX()+1 / rescale)) {
                xBin = (int) ((float) cr.getBinX() / rescale);
            } else {
                thresholdX = ((int) ((float) cr.getBinX() / rescale) + 1) * rescale - cr.getBinX();
                if (generator.nextFloat() < thresholdX) {
                    xBin = (int) ((float) cr.getBinX() / rescale);
                } else {
                    xBin = (int) ((float) cr.getBinX() / rescale) + 1;
                }
            }
            if ((int) ((float) cr.getBinY() / rescale) == (int) ((float) cr.getBinY()+1 / rescale)) {
                yBin = (int) ((float) cr.getBinY() / rescale);
            } else {
                thresholdY = ((int) ((float) cr.getBinY() / rescale) + 1) * rescale - cr.getBinY();
                if (generator.nextFloat() < thresholdY) {
                    yBin = (int) ((float) cr.getBinY() / rescale);
                } else {
                    yBin = (int) ((float) cr.getBinY() / rescale) + 1;
                }
            }
        }
        int blockNumber;

        // Intra chromosome -- we'll store lower diagonal only
        if (chr1.equals(chr2)) {
            int b1 = Math.min(xBin, yBin);
            int b2 = Math.max(xBin, yBin);
            xBin = b1;
            yBin = b2;

            if (b1 != b2) {
                sum += score;  // <= count for mirror cell.
            }

            if (expectedValueCalculations != null) {
                String evKey = (isFrag ? "FRAG_" : "BP_") + binSize;
                ExpectedValueCalculation ev = expectedValueCalculations.get(evKey);
                if (ev != null) {
                    ev.addDistance(chr1.getIndex(), xBin, yBin, score);
                }
            }

            //compute intra chromosomal block number (version 9 and up)
            int depth = v9Depth.getDepth(xBin, yBin);
            int positionAlongDiagonal = ((xBin + yBin) / 2 / blockBinCount);
            blockNumber = depth * blockColumnCount + positionAlongDiagonal;
        }
        else {
            // compute interchromosomal block number (version 9 and up, first block is zero)
            int blockCol = xBin / blockBinCount;
            int blockRow = yBin / blockBinCount;
            blockNumber = blockColumnCount * blockRow + blockCol;
        }


        BlockPP block = blocks.get(blockNumber);
        if (block == null) {

            block = new BlockPP(blockNumber);
            blocks.put(blockNumber, block);
        }
        block.incrementCount(xBin, yBin, score);

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
            los = new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(file), 4194304));

            List<BlockPP> blockList = new ArrayList<>(blocks.values());
            Collections.sort(blockList, new Comparator<BlockPP>() {
                @Override
                public int compare(BlockPP o1, BlockPP o2) {
                    return o1.getNumber() - o2.getNumber();
                }
            });

            for (BlockPP b : blockList) {

                // Remove from map
                blocks.remove(b.getNumber());

                int number = b.getNumber();
                blockNumbers.add(number);

                los.writeInt(number);
                Map<Point, ContactCount> records = b.getContactRecordMap();

                los.writeInt(records.size());
                for (Map.Entry<Point, ContactCount> entry : records.entrySet()) {

                    Point point = entry.getKey();
                    ContactCount count = entry.getValue();

                    los.writeInt(point.x);
                    los.writeInt(point.y);
                    los.writeFloat(count.getCounts());
                }
            }

            blocks.clear();

        } finally {
            if (los != null) los.close();

        }
    }


    // Merge and write out blocks one at a time.
    protected List<IndexEntry> mergeAndWriteBlocks(LittleEndianOutputStream los, Deflater compressor) throws IOException {
        DownsampledDoubleArrayList sampledData = new DownsampledDoubleArrayList(10000, 10000);

        List<BlockQueue> activeList = new ArrayList<>();

        // Initialize queues -- first whatever is left over in memory
        if (blocks.size() > 0) {
            BlockQueue bqInMem = new BlockQueueMem(blocks.values());
            activeList.add(bqInMem);
        }
        // Now from files
        for (File file : tmpFiles) {
            BlockQueue bq = new BlockQueueFB(file);
            if (bq.getBlock() != null) {
                activeList.add(bq);
            }
        }

        List<IndexEntry> indexEntries = new ArrayList<>();

        if (activeList.size() == 0) {
            throw new RuntimeException("No reads in Hi-C contact matrices. This could be because the MAPQ filter is set too high (-q) or because all reads map to the same fragment.");
        }

        do {
            activeList.sort(new Comparator<BlockQueue>() {
                @Override
                public int compare(BlockQueue o1, BlockQueue o2) {
                    return o1.getBlock().getNumber() - o2.getBlock().getNumber();
                }
            });

            BlockQueue topQueue = activeList.get(0);
            BlockPP currentBlock = topQueue.getBlock();
            topQueue.advance();
            int num = currentBlock.getNumber();


            for (int i = 1; i < activeList.size(); i++) {
                BlockQueue blockQueue = activeList.get(i);
                BlockPP block = blockQueue.getBlock();
                if (block.getNumber() == num) {
                    currentBlock.merge(block);
                    blockQueue.advance();
                }
            }

            Iterator<BlockQueue> iterator = activeList.iterator();
            while (iterator.hasNext()) {
                if (iterator.next().getBlock() == null) {
                    iterator.remove();
                }
            }

            // Output block
            long position = los.getWrittenCount();
            writeBlock(this, currentBlock, sampledData, los, compressor);
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

        double[] data = sampledData.toArray();
        this.percent5 = StatUtils.percentile(data, 5);
        this.percent95 = StatUtils.percentile(data, 95);

    }

    void parsingComplete() {
        // Add the block numbers still in memory
        for (BlockPP block : blocks.values()) {
            blockNumbers.add(block.getNumber());
        }
    }

    /**
     * used by multithreaded code
     *
     * @param otherMatrixZoom
     */
    void mergeMatrices(MatrixZoomDataPP otherMatrixZoom) {
        sum += otherMatrixZoom.sum;
        for (Map.Entry<Integer, BlockPP> otherBlock : otherMatrixZoom.blocks.entrySet()) {
            int blockNumber = otherBlock.getKey();
            BlockPP block = blocks.get(blockNumber);
            if (block == null) {
                blocks.put(blockNumber, otherBlock.getValue());
                blockNumbers.add(blockNumber);
            } else {
                block.merge(otherBlock.getValue());
            }
        }
    }

    /**
     * Note -- compressed
     *
     * @param zd          Matrix zoom data
     * @param block       Block to write
     * @param sampledData Array to hold a sample of the data (to compute statistics)
     * @throws IOException
     */
    protected void writeBlock(MatrixZoomDataPP zd, BlockPP block, DownsampledDoubleArrayList sampledData, LittleEndianOutputStream los, Deflater compressor) throws IOException {

        final Map<Point, ContactCount> records = block.getContactRecordMap();//   getContactRecords();

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
        BufferedByteWriter buffer = new BufferedByteWriter(nRecords * 12);
        buffer.putInt(nRecords);
        zd.cellCount += nRecords;


        // Find extents of occupied cells
        int binXOffset = Integer.MAX_VALUE;
        int binYOffset = Integer.MAX_VALUE;
        int binXMax = 0;
        int binYMax = 0;
        for (Map.Entry<Point, ContactCount> entry : records.entrySet()) {
            Point point = entry.getKey();
            binXOffset = Math.min(binXOffset, point.x);
            binYOffset = Math.min(binYOffset, point.y);
            binXMax = Math.max(binXMax, point.x);
            binYMax = Math.max(binYMax, point.y);
        }

        buffer.putInt(binXOffset);
        buffer.putInt(binYOffset);

        // Sort keys in row-major order
        List<Point> keys = new ArrayList<>(records.keySet());
        keys.sort(new Comparator<Point>() {
            @Override
            public int compare(Point o1, Point o2) {
                if (o1.y != o2.y) {
                    return o1.y - o2.y;
                } else {
                    return o1.x - o2.x;
                }
            }
        });
        Point lastPoint = keys.get(keys.size() - 1);
        final short w = (short) (binXMax - binXOffset + 1);
        final int w1 = binXMax - binXOffset + 1;
        final int w2 = binYMax - binYOffset + 1;

        boolean isInteger = true;
        float maxCounts = 0;

        LinkedHashMap<Integer, List<ContactRecord>> rows = new LinkedHashMap<>();
        for (Point point : keys) {
            final ContactCount contactCount = records.get(point);
            float counts = contactCount.getCounts();
            if (counts >= countThreshold) {

                isInteger = isInteger && (Math.floor(counts) == counts);
                maxCounts = Math.max(counts, maxCounts);

                final int px = point.x - binXOffset;
                final int py = point.y - binYOffset;
                List<ContactRecord> row = rows.get(py);
                if (row == null) {
                    row = new ArrayList<>(10);
                    rows.put(py, row);
                }
                row.add(new ContactRecord(px, py, counts));
            }
        }

        // Compute size for each representation and choose smallest
        boolean useShort = isInteger && (maxCounts < Short.MAX_VALUE);
        boolean useShortBinX = w1 < Short.MAX_VALUE;
        boolean useShortBinY = w2 < Short.MAX_VALUE;
        int valueSize = useShort ? 2 : 4;

        int lorSize = 0;
        int nDensePts = (lastPoint.y - binYOffset) * w + (lastPoint.x - binXOffset) + 1;

        int denseSize = nDensePts * valueSize;
        for (List<ContactRecord> row : rows.values()) {
            lorSize += 4 + row.size() * valueSize;
        }

        buffer.put((byte) (useShort ? 0 : 1));
        buffer.put((byte) (useShortBinX ? 0 : 1));
        buffer.put((byte) (useShortBinY ? 0 : 1));

        //dense calculation is incorrect for v9
        denseSize = Integer.MAX_VALUE;

        if (lorSize < denseSize) {

            buffer.put((byte) 1);  // List of rows representation

            if (useShortBinY) {
                buffer.putShort((short) rows.size()); // # of rows
            } else {
                buffer.putInt(rows.size());  // # of rows
            }

            for (Map.Entry<Integer, List<ContactRecord>> entry : rows.entrySet()) {

                int py = entry.getKey();
                List<ContactRecord> row = entry.getValue();
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

                    sampledData.add(counts);
                    zd.sum += counts;
                }
            }

        } else {
            buffer.put((byte) 2);  // Dense matrix


            buffer.putInt(nDensePts);
            buffer.putShort(w);

            int lastIdx = 0;
            for (Point p : keys) {

                int idx = (p.y - binYOffset) * w + (p.x - binXOffset);
                for (int i = lastIdx; i < idx; i++) {
                    // Filler value
                    if (useShort) {
                        buffer.putShort(Short.MIN_VALUE);
                    } else {
                        buffer.putFloat(Float.NaN);
                    }
                }
                float counts = records.get(p).getCounts();
                if (useShort) {
                    buffer.putShort((short) counts);
                } else {
                    buffer.putFloat(counts);
                }
                lastIdx = idx + 1;

                sampledData.add(counts);
                zd.sum += counts;
            }
        }


        byte[] bytes = buffer.getBytes();
        byte[] compressedBytes = compress(bytes, compressor);
        los.write(compressedBytes);

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
