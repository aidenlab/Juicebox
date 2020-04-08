/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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
import juicebox.data.ChromosomeHandler;
import juicebox.tools.utils.original.Preprocessor;
import juicebox.data.HiCFileTools;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;


public class MultithreadedPreprocessor extends Preprocessor {
    private final Map<Integer, String> chromosomePairIndexes;
    private final Map<String, Integer> chromosomePairIndexesReverse;
    private final Map<Integer, Integer> chromosomePairIndex1;
    private final Map<Integer, Integer> chromosomePairIndex2;
    private int chromosomePairCounter = 0;
    private final Map<Integer, Integer> nonemptyChromosomePairs;
    private final Map<Integer, MatrixPP> wholeGenomeMatrixParts;
    private final Map<Integer, IndexEntry> localMatrixPositions;
    private final Map<Integer, Integer> matrixSizes;
    private LittleEndianOutputStream losWholeGenome;
    private LittleEndianOutputStream losFooter;
    private final Map<Integer, Map<Long, List<IndexEntry>>> chromosomePairBlockIndexes;
    protected static int numCPUThreads = 1;

    public MultithreadedPreprocessor(File outputFile, String genomeId, ChromosomeHandler chromosomeHandler, double hicFileScalingFactor) {
        super(outputFile, genomeId, chromosomeHandler, hicFileScalingFactor);
        this.localMatrixPositions = new ConcurrentHashMap<>();
        this.matrixSizes = new ConcurrentHashMap<>();
        this.wholeGenomeMatrixParts = new ConcurrentHashMap<>();
        this.nonemptyChromosomePairs = new ConcurrentHashMap<>();

        chromosomeIndexes = new ConcurrentHashMap<>(chromosomeHandler.size(), (float) 0.75, numCPUThreads);
        for (int i = 0; i < chromosomeHandler.size(); i++) {
            chromosomeIndexes.put(chromosomeHandler.getChromosomeFromIndex(i).getName(), i);
        }

        this.chromosomePairIndexes = new ConcurrentHashMap<>();
        this.chromosomePairIndexesReverse = new ConcurrentHashMap<>();
        this.chromosomePairIndex1 = new ConcurrentHashMap<>();
        this.chromosomePairIndex2 = new ConcurrentHashMap<>();
        String genomeWideName = chromosomeHandler.getChromosomeFromIndex(0).getName();
        String genomeWidePairName = genomeWideName + "_" + genomeWideName;
        chromosomePairIndexes.put(chromosomePairCounter, genomeWidePairName);
        chromosomePairIndexesReverse.put(genomeWidePairName, chromosomePairCounter);
        chromosomePairIndex1.put(chromosomePairCounter, 0);
        chromosomePairIndex2.put(chromosomePairCounter, 0);
        chromosomePairCounter++;
        for (int i = 1; i < chromosomeHandler.size(); i++) {
            for (int j = i; j < chromosomeHandler.size(); j++){
                String c1Name = chromosomeHandler.getChromosomeFromIndex(i).getName();
                String c2Name = chromosomeHandler.getChromosomeFromIndex(j).getName();
                String chromosomePairName = c1Name + "_" + c2Name;
                chromosomePairIndexes.put(chromosomePairCounter, chromosomePairName);
                chromosomePairIndexesReverse.put(chromosomePairName,chromosomePairCounter);
                chromosomePairIndex1.put(chromosomePairCounter, i);
                chromosomePairIndex2.put(chromosomePairCounter, j);
                chromosomePairCounter++;
            }
        }

        this.chromosomePairBlockIndexes = new ConcurrentHashMap<>(chromosomePairCounter, (float) 0.75, numCPUThreads);
    }

    public void setNumCPUThreads(int numCPUThreads) {
        MultithreadedPreprocessor.numCPUThreads = numCPUThreads;
    }


}
