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

package juicebox.data;

import juicebox.data.basics.Chromosome;
import juicebox.windowui.HiCZoom;
import org.broad.igv.util.collections.LRUCache;

import java.util.Iterator;
import java.util.List;

public class IteratorContainer {

    private final static int USE_ZD = 1, USE_LIST = 2, USE_GW = 3;
    private final LRUCache<String, Block> blockCache;
    private final DatasetReader reader;
    private final MatrixZoomData zd;
    private final int type;
    private final List<ContactRecord> readList;
    private final Dataset dataset;
    private final ChromosomeHandler handler;
    private final HiCZoom zoom;
    private final boolean includeIntra;
    private final long matrixSize;
    private long numberOfContactRecords = -1;

    public IteratorContainer(DatasetReader reader, MatrixZoomData zd, LRUCache<String, Block> blockCache) {
        type = USE_ZD;
        this.reader = reader;
        this.zd = zd;
        this.matrixSize = zd.getXGridAxis().getBinCount();
        this.blockCache = blockCache;
        this.readList = null;

        this.dataset = null;
        this.handler = null;
        this.zoom = null;
        this.includeIntra = false;
    }

    public IteratorContainer(List<ContactRecord> readList, long matrixSize) {
        type = USE_LIST;
        this.reader = null;
        this.zd = null;
        this.blockCache = null;
        this.readList = readList;
        this.matrixSize = matrixSize;

        this.dataset = null;
        this.handler = null;
        this.zoom = null;
        this.includeIntra = false;
    }

    public IteratorContainer(Dataset dataset, ChromosomeHandler handler,
                             HiCZoom zoom, boolean includeIntra) {
        type = USE_GW;
        this.reader = null;
        this.zd = null;
        this.blockCache = null;
        this.readList = null;

        this.dataset = dataset;
        this.handler = handler;
        this.zoom = zoom;
        this.includeIntra = includeIntra;

        long totalSize = 0;
        for (Chromosome c1 : handler.getChromosomeArrayWithoutAllByAll()) {
            totalSize += (c1.getLength() / zoom.getBinSize()) + 1;
        }
        this.matrixSize = totalSize;

    }

    public Iterator<ContactRecord> getNewContactRecordIterator() {
        if (type == USE_ZD) {
            if (reader != null && zd != null && blockCache != null) {
                return new ContactRecordIterator(reader, zd, blockCache);
            }
        }
        if (type == USE_LIST) {
            if (readList != null) {
                return readList.iterator();
            }
        }
        if (type == USE_GW) {
            if (dataset != null && handler != null && zoom != null) {
                return new GenomeWideIterator(dataset, handler, zoom, includeIntra);
            }
        }

        System.err.println("Null Contact Record Iterator");
        return null;
    }

    public long getNumberOfContactRecords() {
        if (numberOfContactRecords > 0) return numberOfContactRecords;

        numberOfContactRecords = 0;
        Iterator<ContactRecord> iterator = getNewContactRecordIterator();
        while (iterator.hasNext()) {
            iterator.next();
            numberOfContactRecords++;
        }

        return numberOfContactRecords;
    }

    public long getMatrixSize() {
        return matrixSize;
    }
}
