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

package juicebox.tools.utils.original.stats;

import juicebox.data.ChromosomeHandler;
import juicebox.tools.utils.original.Chunk;
import juicebox.tools.utils.original.FragmentCalculation;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class ParallelStatistics {

    private final static Object mergerLock = new Object();
    private final int numThreads;
    private final StatisticsContainer mergedContainer;
    private final AtomicInteger threadCounter = new AtomicInteger();
    private final List<Chunk> mndChunks;
    private final String siteFile;
    private final List<String> statsFiles;
    private final List<Integer> mapqThresholds;
    private final String ligationJunction;
    private final String inFile;
    private final ChromosomeHandler localHandler;
    private final FragmentCalculation fragmentCalculation;

    public ParallelStatistics(int numThreads, StatisticsContainer mergedContainer,
                              List<Chunk> mndChunks, String siteFile, List<String> statsFiles,
                              List<Integer> mapqThresholds, String ligationJunction, String inFile,
                              ChromosomeHandler localHandler, FragmentCalculation fragmentCalculation) {
        this.numThreads = numThreads;
        this.mergedContainer = mergedContainer;
        this.mndChunks = mndChunks;
        this.siteFile = siteFile;
        this.statsFiles = statsFiles;
        this.mapqThresholds = mapqThresholds;
        this.ligationJunction = ligationJunction;
        this.inFile = inFile;
        this.localHandler = localHandler;
        this.fragmentCalculation = fragmentCalculation;
    }

    public void launchThreads() {
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        for (int l = 0; l < numThreads; l++) {
            executor.execute(() -> runParallelizedStatistics(mergedContainer));
        }
        executor.shutdown();
        // Wait until all threads finish
        while (!executor.isTerminated()) {
        }
    }

    public void runParallelizedStatistics(final StatisticsContainer mergedContainer) {
        int currentCount = threadCounter.getAndIncrement();
        while (currentCount < mndChunks.size()) {
            Chunk chunk = mndChunks.get(currentCount);
            try {
                ParallelStatisticsWorker runner = new ParallelStatisticsWorker(siteFile, statsFiles, mapqThresholds,
                        ligationJunction, inFile, localHandler, fragmentCalculation);
                runner.infileStatistics(chunk);
                synchronized (mergerLock) {
                    mergedContainer.add(runner.getResultsContainer(), statsFiles.size());
                }
            } catch (Exception e2) {
                e2.printStackTrace();
            }
            currentCount = threadCounter.getAndIncrement();
        }
    }
}
