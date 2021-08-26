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

import juicebox.data.ChromosomeHandler;
import juicebox.data.ContactRecord;
import juicebox.data.Dataset;
import juicebox.data.basics.Chromosome;
import juicebox.data.basics.ListOfFloatArrays;
import juicebox.tools.dev.ParallelizedJuicerTools;
import juicebox.windowui.HiCZoom;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;


public class GWIteratorContainer extends IteratorContainer {

    private final Dataset dataset;
    private final ChromosomeHandler handler;
    private final HiCZoom zoom;
    private final boolean includeIntra;

    public GWIteratorContainer(Dataset dataset, ChromosomeHandler handler,
                               HiCZoom zoom, boolean includeIntra) {
        super(calculateMatrixSize(handler, zoom));
        this.dataset = dataset;
        this.handler = handler;
        this.zoom = zoom;
        this.includeIntra = includeIntra;

    }

    private static long calculateMatrixSize(ChromosomeHandler handler, HiCZoom zoom) {
        long totalSize = 0;
        for (Chromosome c1 : handler.getChromosomeArrayWithoutAllByAll()) {
            totalSize += (c1.getLength() / zoom.getBinSize()) + 1;
        }
        return totalSize;
    }

    @Override
    public Iterator<ContactRecord> getNewContactRecordIterator() {
        return new GenomeWideIterator(dataset, handler, zoom, includeIntra);
    }

    public List<Iterator<ContactRecord>> getAllFromFileContactRecordIterators() {
        return GenomeWideIterator.getAllFromFileIterators(dataset, handler, zoom, includeIntra);
    }

    @Override
    public ListOfFloatArrays sparseMultiply(ListOfFloatArrays vector, long vectorLength) {
        final ListOfFloatArrays totalSumVector = new ListOfFloatArrays(vectorLength);

        List<Iterator<ContactRecord>> allIterators = getAllFromFileContactRecordIterators();

        AtomicInteger index = new AtomicInteger(0);
        ParallelizedJuicerTools.launchParallelizedCode(numCPUMatrixThreads, () -> {
            int i = index.getAndIncrement();
            ListOfFloatArrays accumSumVector = new ListOfFloatArrays(vectorLength);
            while (i < allIterators.size()) {
                accumSumVector.addValuesFrom(ZDIteratorContainer.matrixVectorMultiplyOnIterator(
                        allIterators.get(i), vector, vectorLength));
                i = index.getAndIncrement();
            }
            synchronized (totalSumVector) {
                totalSumVector.addValuesFrom(accumSumVector);
            }
        });

        allIterators.clear();

        return totalSumVector;
    }

    @Override
    public void clear() {
        // null, doesn't need to clean anything
    }
}
