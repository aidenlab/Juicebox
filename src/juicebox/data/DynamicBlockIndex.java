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

package juicebox.data;

import htsjdk.samtools.seekablestream.SeekableStream;
import htsjdk.tribble.util.LittleEndianInputStream;
import juicebox.tools.utils.original.IndexEntry;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.List;

public class DynamicBlockIndex extends BlockIndex {

    private final int maxBlocks;
    private final long minPosition, maxPosition;
    private Integer blockNumberRangeMin = null, blockNumberRangeMax = null;
    private Long mapFileBoundsMin = null, mapFileBoundsMax = null;
    private final SeekableStream stream;

    public DynamicBlockIndex(SeekableStream stream, int numBlocks, int maxBlocks, long minPosition) {
        super(numBlocks);
        this.stream = stream;
        this.maxBlocks = maxBlocks;
        this.minPosition = minPosition;
        maxPosition = minPosition + numBlocks * 16;
    }


    @Override
    public List<Integer> getBlockNumbers() {
        // cause break - should not be called for this type of indexing
        return null;
    }

    @Override
    public IndexEntry getBlock(int blockNumber) {
        if (blockNumber > maxBlocks) {
            return null;
        } else if (blockIndex.containsKey(blockNumber)) {
            return blockIndex.get(blockNumber);
        } else if (blockNumber == 0) {
            try {
                return searchForBlockIndexEntry(blockNumber, this.minPosition, this.minPosition + 16);
            } catch (Exception e) {
                return null;
            }
        } else {
            long minPosition = this.minPosition;
            long maxPosition = this.maxPosition;
            if (blockNumberRangeMin != null && mapFileBoundsMin != null) {
                if (blockNumber < blockNumberRangeMin) {
                    maxPosition = mapFileBoundsMin;
                } else if (blockNumber > blockNumberRangeMax) {
                    minPosition = mapFileBoundsMax;
                }
            }
            if (maxPosition - minPosition < 16) {
                return null;
            } else {
                try {
                    return searchForBlockIndexEntry(blockNumber, minPosition, maxPosition);
                } catch (Exception e) {
                    return null;
                }
            }
        }
    }


    // Search entry for blockNumber between file positions boundsMin and boundsMax
    // boundsMin is guaranteed to start at the beginning of an entry, boundsMax at the end
    private IndexEntry searchForBlockIndexEntry(int blockNumber, long boundsMin, long boundsMax) throws IOException {

        int chunkSize = 1600000;
        if (boundsMax - boundsMin < chunkSize) {

            synchronized (stream) {
                stream.seek(boundsMin);
                LittleEndianInputStream dis = new LittleEndianInputStream(new BufferedInputStream(stream));

                Integer firstBlockNumber = null;
                Integer lastBlockNumber = null;
                long pointer = boundsMin;

                while (pointer < boundsMax) {
                    int blockNumberFound = dis.readInt();
                    long filePosition = dis.readLong();
                    int blockSizeInBytes = dis.readInt();
                    blockIndex.put(blockNumberFound, new IndexEntry(filePosition, blockSizeInBytes));
                    if (firstBlockNumber == null) firstBlockNumber = blockNumberFound;
                    lastBlockNumber = blockNumberFound;
                    pointer += 16;
                }

                // recent memory
                mapFileBoundsMin = boundsMin;
                mapFileBoundsMax = boundsMax;
                blockNumberRangeMin = firstBlockNumber;
                blockNumberRangeMax = lastBlockNumber;
            }

            return blockIndex.get(blockNumber);
        }
        // Midpoint in units of 16 byte chunks
        int nEntries = (int) ((boundsMax - boundsMin) / 16);
        long positionToSeek = boundsMin + (long) Math.floor(nEntries / 2f) * 16;

        int blockNumberFound, blockSizeInBytes;
        long filePosition;
        synchronized (stream) {
            stream.seek(positionToSeek);
            byte[] buffer = new byte[16];
            stream.readFully(buffer);
            LittleEndianInputStream dis = new LittleEndianInputStream(new ByteArrayInputStream(buffer));

            blockNumberFound = dis.readInt();
            filePosition = dis.readLong();
            blockSizeInBytes = dis.readInt();
        }
        if (blockNumberFound == blockNumber) {
            blockIndex.put(blockNumberFound, new IndexEntry(filePosition, blockSizeInBytes));
            return blockIndex.get(blockNumber);
        } else if (blockNumber > blockNumberFound) {
            return searchForBlockIndexEntry(blockNumber, positionToSeek + 16, boundsMax);
        } else {
            return searchForBlockIndexEntry(blockNumber, boundsMin, positionToSeek);
        }
    }
}
