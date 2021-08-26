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
import juicebox.data.*;
import juicebox.tools.dev.ParallelizedJuicerTools;
import juicebox.windowui.HiCZoom;
import org.broad.igv.util.collections.LRUCache;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class ListOfListGenerator {
    private static final int MAX_LIMIT = Integer.MAX_VALUE - 10;

    public static IteratorContainer createFromZD(DatasetReader reader, MatrixZoomData matrixZoomData,
                                                 LRUCache<String, Block> blockCache) {
        IteratorContainer ic = new ZDIteratorContainer(reader, matrixZoomData, blockCache);
        return tryToCreateIteratorInRAM(ic);
    }

    public static IteratorContainer createForWholeGenome(Dataset dataset, ChromosomeHandler chromosomeHandler,
                                                         HiCZoom zoom, boolean includeIntraData) {
        IteratorContainer ic = new GWIteratorContainer(dataset, chromosomeHandler, zoom, includeIntraData);
        return tryToCreateIteratorInRAM(ic);
    }

    private static IteratorContainer tryToCreateIteratorInRAM(IteratorContainer ic0) {
        if (HiCGlobals.USE_ITERATOR_NOT_ALL_IN_RAM) {
            return ic0;
        }

        try {
            // we should count once to ensure this is reasonable to do so memory-wise
            boolean shouldFitInMemory = true;
            if (HiCGlobals.CHECK_RAM_USAGE) {
                shouldFitInMemory = checkMemory(ic0);
            }

            if (shouldFitInMemory) {
                List<List<ContactRecord>> allContactRecords = populateListOfLists(ic0);
                long numOfContactRecords = getTotalSize(allContactRecords);

                IteratorContainer newIC = new ListOfListIteratorContainer(allContactRecords,
                        ic0.getMatrixSize(),
                        numOfContactRecords);
                return newIC;
            }
        } catch (Exception e) {
            System.err.println(e.getLocalizedMessage());
            System.err.println("Will use default iterator");
        }

        return ic0;
    }

    private static List<List<ContactRecord>> populateListOfLists(IteratorContainer ic) {

        if (ic instanceof GWIteratorContainer) {
            List<Iterator<ContactRecord>> iterators = ((GWIteratorContainer) ic).getAllFromFileContactRecordIterators();
            List<List<ContactRecord>> allRecords = new ArrayList<>();

            AtomicInteger index = new AtomicInteger(0);
            ParallelizedJuicerTools.launchParallelizedCode(IteratorContainer.numCPUMatrixThreads, () -> {
                int i = index.getAndIncrement();
                List<List<ContactRecord>> recordsForThread = new ArrayList<>();
                while (i < iterators.size()) {
                    List<List<ContactRecord>> recordsForIter = populateListOfListsFromSingleIterator(iterators.get(i));
                    recordsForThread.addAll(recordsForIter);
                    i = index.getAndIncrement();
                }
                synchronized (allRecords) {
                    allRecords.addAll(recordsForThread);
                }
            });
            return allRecords;
        } else {
            return populateListOfListsFromSingleIterator(ic.getNewContactRecordIterator());
        }
    }

    private static List<List<ContactRecord>> populateListOfListsFromSingleIterator(Iterator<ContactRecord> iterator) {

        List<List<ContactRecord>> allRecords = new ArrayList<>();
        List<ContactRecord> tempList = new ArrayList<>();
        int counter = 0;
        while (iterator.hasNext()) {
            tempList.add(iterator.next());
            counter++;
            if (counter > MAX_LIMIT) {
                allRecords.add(tempList);
                tempList = new ArrayList<>();
                counter = 0;
            }
        }
        if (tempList.size() > 0) {
            allRecords.add(tempList);
        }
        return allRecords;
    }

    private static boolean checkMemory(IteratorContainer ic) {
        long ramForRowSums = ic.getMatrixSize() * 4;
        long ramForAllContactRecords = ic.getNumberOfContactRecords() * 12;
        return ramForRowSums + ramForAllContactRecords < Runtime.getRuntime().maxMemory();
    }

    private static long getTotalSize(List<List<ContactRecord>> allContactRecords) {
        long numOfContactRecords = 0;
        for (List<ContactRecord> records : allContactRecords) {
            numOfContactRecords += records.size();
        }
        return numOfContactRecords;
    }
}
