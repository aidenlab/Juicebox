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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.data.iterator;

import juicebox.HiCGlobals;
import juicebox.data.Block;
import juicebox.data.ContactRecord;
import juicebox.data.DatasetReader;
import juicebox.data.MatrixZoomData;
import juicebox.windowui.NormalizationHandler;
import org.broad.igv.util.collections.LRUCache;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

public /**
 * Class for iterating over the contact records
 */
class ContactRecordIterator implements Iterator<ContactRecord> {
    
    private final List<Integer> blockNumbers;
    private int blockIdx;
    private Iterator<ContactRecord> currentBlockIterator;
    private final DatasetReader reader;
    private final MatrixZoomData zd;
    private final LRUCache<String, Block> blockCache;
    
    /**
     * Initializes the iterator
     */
    ContactRecordIterator(DatasetReader reader, MatrixZoomData zd, LRUCache<String, Block> blockCache) {
        this.reader = reader;
        this.zd = zd;
        this.blockCache = blockCache;
        this.blockIdx = -1;
        this.blockNumbers = reader.getBlockNumbers(zd);
    }

    /**
     * Indicates whether or not there is another block waiting; checks current block
     * iterator and creates a new one if need be
     *
     * @return true if there is another block to be read
     */
    @Override
    public boolean hasNext() {

        if (currentBlockIterator != null && currentBlockIterator.hasNext()) {
            return true;
        } else {
            blockIdx++;
            if (blockNumbers != null && blockIdx < blockNumbers.size()) {
                try {
                    int blockNumber = blockNumbers.get(blockIdx);

                    // Optionally check the cache
                    String key = zd.getBlockKey(blockNumber, NormalizationHandler.NONE);
                    Block nextBlock;
                    if (HiCGlobals.useCache && blockCache.containsKey(key)) {
                        nextBlock = blockCache.get(key);
                    } else {
                        nextBlock = reader.readNormalizedBlock(blockNumber, zd, NormalizationHandler.NONE);
                    }
                    currentBlockIterator = nextBlock.getContactRecords().iterator();
                    return true;
                } catch (IOException e) {
                    System.err.println("Error fetching block " + e.getMessage());
                    return false;
                }
            }
        }

        return false;
    }

    /**
     * Returns the next contact record
     *
     * @return The next contact record
     */
    @Override
    public ContactRecord next() {
        return currentBlockIterator == null ? null : currentBlockIterator.next();
    }

    /**
     * Not supported
     */
    @Override
    public void remove() {
        //Not supported
        throw new RuntimeException("remove() is not supported");
    }
}
