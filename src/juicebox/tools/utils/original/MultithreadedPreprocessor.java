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

package juicebox.tools.utils.original;

import htsjdk.tribble.util.LittleEndianOutputStream;
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.tools.utils.original.mnditerator.AlignmentPair;
import juicebox.tools.utils.original.mnditerator.AsciiPairIterator;
import juicebox.tools.utils.original.mnditerator.PairIterator;
import juicebox.windowui.NormalizationHandler;
import org.broad.igv.util.Pair;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.Deflater;


public class MultithreadedPreprocessor extends Preprocessor {
    public static final String CAT_SCRIPT = "_cat_outputs.sh";
    private final Map<Integer, String> chromosomePairIndexes = new ConcurrentHashMap<>();
    private final Map<String, Integer> chromosomePairIndexesReverse = new ConcurrentHashMap<>();
    private final Map<Integer, Integer> chromosomePairIndex1 = new ConcurrentHashMap<>();
    private final Map<Integer, Integer> chromosomePairIndex2 = new ConcurrentHashMap<>();
    private int chromosomePairCounter = 0;
    private final Map<Integer, Integer> nonemptyChromosomePairs = new ConcurrentHashMap<>();
    private final Map<Integer, Map<Integer, MatrixPP>> wholeGenomeMatrixParts = new ConcurrentHashMap<>();
    private final Map<String, IndexEntry> localMatrixPositions = new ConcurrentHashMap<>();
    private final Map<Integer, Long> matrixSizes = new ConcurrentHashMap<>();
    private final Map<Integer, Map<Long, List<IndexEntry>>> chromosomePairBlockIndexes;
    protected static int numCPUThreads = 1;
    private final Map<Integer, Map<String, ExpectedValueCalculation>> allLocalExpectedValueCalculations;
    protected static Map<Integer, List<Chunk>> mndIndex = null;
    private final AtomicInteger chunkCounter = new AtomicInteger(0);
    private int totalChunks = 0;
    private int totalChrPairToWrite = 0;
    private final AtomicInteger totalChrPairsWritten = new AtomicInteger(0);
    private final ConcurrentHashMap<Integer, AtomicInteger> completedChunksPerChrPair = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, Integer> numChunksPerChrPair = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, AtomicInteger> chrPairCompleted = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, AtomicInteger> chrPairAvailableThreads = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, Integer> chrPairBlockCapacities = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, Integer> chunkCounterToChrPairMap = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, Integer> chunkCounterToChrChunkMap = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, Map<Integer, Pair<Pair<Integer, Integer>, MatrixPP>>> threadSpecificChrPairMatrices = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, MatrixPP> finalChrMatrices = new ConcurrentHashMap<>();

    public MultithreadedPreprocessor(File outputFile, String genomeId, ChromosomeHandler chromosomeHandler,
                                     double hicFileScalingFactor, int numCPUThreads, String mndIndexFile) throws IOException {
        super(outputFile, genomeId, chromosomeHandler, hicFileScalingFactor);
        MultithreadedPreprocessor.numCPUThreads = numCPUThreads;
        chromosomeIndexes = MTIndexHandler.populateChromosomeIndexes(chromosomeHandler, numCPUThreads);
        chromosomePairCounter = MTIndexHandler.populateChromosomePairIndexes(chromosomeHandler,
                chromosomePairIndexes, chromosomePairIndexesReverse,
                chromosomePairIndex1, chromosomePairIndex2);
        setMndIndex(mndIndexFile, chromosomePairIndexes);
        this.chromosomePairBlockIndexes = new ConcurrentHashMap<>(chromosomePairCounter, (float) 0.75, numCPUThreads);
        this.allLocalExpectedValueCalculations = new ConcurrentHashMap<>(numCPUThreads, (float) 0.75, numCPUThreads);
    }

    public void setMndIndex(String mndIndexFile, Map<Integer, String> chromosomePairIndexes) throws IOException {
        if (mndIndexFile != null && mndIndexFile.length() > 1) {
            mndIndex = MTIndexHandler.readMndIndex(mndIndexFile, chromosomePairIndexes);
        } else {
            throw new IOException("No mndIndex provided");
        }
    }

    @Override
    public void preprocess(final String inputFile, String ignore1, String ignore2, Map<Integer,
            List<Chunk>> ignore3) throws IOException {
        super.preprocess(inputFile, outputFile + "_header", outputFile + "_footer", mndIndex);

        try {
            PrintWriter finalOutput = new PrintWriter(outputFile + CAT_SCRIPT);
            StringBuilder catOutputLine = new StringBuilder();
            StringBuilder removeLine = new StringBuilder();
            catOutputLine.append("cat ").append(outputFile + "_header");
            removeLine.append("rm ").append(outputFile + "_header");
            for (int i = 0; i < chromosomePairCounter; i++) {
                if ((nonemptyChromosomePairs.containsKey(i) && chromosomePairBlockIndexes.containsKey(i) && mndIndex.containsKey(i)) || i == 0) {
                    catOutputLine.append(" ").append(outputFile).append("_").append(chromosomePairIndexes.get(i));
                    removeLine.append(" ").append(outputFile).append("_").append(chromosomePairIndexes.get(i));
                    if (i > 0) {
                        int numOfNeededThreads = chrPairAvailableThreads.get(i).get();
                        if (numOfNeededThreads > 1) {
                            for (int j = 1; j <= numOfNeededThreads * numResolutions; j++) {
                                catOutputLine.append(" ").append(outputFile).append("_").append(chromosomePairIndexes.get(i)).append("_").append(j);
                                removeLine.append(" ").append(outputFile).append("_").append(chromosomePairIndexes.get(i)).append("_").append(j);
                            }
                        }
                    }
                }
            }
            catOutputLine.append(" ").append(outputFile + "_footer").append(" > ").append(outputFile).append("\n");
            removeLine.append(" ").append(outputFile + "_footer\n");
            finalOutput.println(catOutputLine.toString());
            finalOutput.println(removeLine.toString());
            finalOutput.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Unable to write to catOutputs.sh");
            System.exit(70);
        }
    }

    private int getGenomicPosition(int chr, int pos, ChromosomeHandler localChromosomeHandler) {
        long len = 0;
        for (int i = 1; i < chr; i++) {
            len += localChromosomeHandler.getChromosomeFromIndex(i).getLength();
        }
        len += pos;

        return (int) (len / 1000);

    }

    private Pair<Pair<Integer,Integer>, MatrixPP> processIndividualMatrixChunk(String inputFile, int chunkNumber,
                                                                               int currentChrPair, Set<String> syncWrittenMatrices, Map<String, ExpectedValueCalculation>
                                                                                       localExpectedValueCalculations, int threadNum) throws IOException {


        MatrixPP wholeGenomeMatrix = getInitialGenomeWideMatrixPP(chromosomeHandler);
        int i = chunkNumber;
        int chunksProcessed = 0;

        String currentMatrixName = null;
        int currentPairIndex = -1;

        int currentChr1 = -1;
        int currentChr2 = -1;
        MatrixPP currentMatrix = null;
        String currentMatrixKey = null;

        while (i < totalChunks) {
            int chrPair = chunkCounterToChrPairMap.get(i);
            if (chrPair != currentChrPair) {
                break;
            }
            int chrChunk = chunkCounterToChrChunkMap.get(i);
            List<Chunk> chunkPositions = mndIndex.get(chrPair);
            PairIterator iter = null;
            if (mndIndex == null) {
                System.err.println("No index for merged nodups file.");
                System.exit(67);
            } else {
                iter = new AsciiPairIterator(inputFile, chromosomeIndexes, chunkPositions.get(chrChunk),
                        chromosomeHandler);
            }
            while (iter.hasNext()) {
                AlignmentPair pair = iter.next();
                // skip pairs that mapped to contigs
                if (!pair.isContigPair()) {
                    if (shouldSkipContact(pair)) continue;
                    // Flip pair if needed so chr1 < chr2
                    int chr1, chr2, bp1, bp2, frag1, frag2;
                    if (pair.getChr1() < pair.getChr2()) {
                        bp1 = pair.getPos1();
                        bp2 = pair.getPos2();
                        frag1 = pair.getFrag1();
                        frag2 = pair.getFrag2();
                        chr1 = pair.getChr1();
                        chr2 = pair.getChr2();
                    } else {
                        bp1 = pair.getPos2();
                        bp2 = pair.getPos1();
                        frag1 = pair.getFrag2();
                        frag2 = pair.getFrag1();
                        chr1 = pair.getChr2();
                        chr2 = pair.getChr1();
                    }

                    bp1 = ensureFitInChromosomeBounds(bp1, chr1);
                    bp2 = ensureFitInChromosomeBounds(bp2, chr2);

                    // Randomize position within fragment site
                    if (allowPositionsRandomization && fragmentCalculation != null) {
                        Pair<Integer, Integer> newBPos12 = getRandomizedPositions(chr1, chr2, frag1, frag2, bp1, bp2);
                        bp1 = newBPos12.getFirst();
                        bp2 = newBPos12.getSecond();
                    }
                    // only increment if not intraFragment and passes the mapq threshold
                    if (!(currentChr1 == chr1 && currentChr2 == chr2)) {

                        // Start the next matrix
                        currentChr1 = chr1;
                        currentChr2 = chr2;
                        currentMatrixKey = currentChr1 + "_" + currentChr2;

                        currentMatrixName = chromosomeHandler.getChromosomeFromIndex(chr1).getName() + "-" + chromosomeHandler.getChromosomeFromIndex(chr2).getName();
                        currentPairIndex = chromosomePairIndexesReverse.get(currentMatrixName);

                        if (currentPairIndex != currentChrPair) {
                            break;
                        }

                        if (syncWrittenMatrices.contains(currentMatrixKey)) {
                            System.err.println("Error: the chromosome combination " + currentMatrixKey + " appears in multiple blocks");
                            if (outputFile != null) outputFile.deleteOnExit();
                            System.exit(58);
                        }
                        currentMatrix = new MatrixPP(currentChr1, currentChr2, chromosomeHandler, bpBinSizes, fragmentCalculation, fragBinSizes, countThreshold, v9DepthBase, chrPairBlockCapacities.get(currentChrPair));
                    }
                    currentMatrix.incrementCount(bp1, bp2, frag1, frag2, pair.getScore(), localExpectedValueCalculations, tmpDir);

                    int pos1 = getGenomicPosition(chr1, bp1, chromosomeHandler);
                    int pos2 = getGenomicPosition(chr2, bp2, chromosomeHandler);
                    wholeGenomeMatrix.incrementCount(pos1, pos2, pos1, pos2, pair.getScore(), localExpectedValueCalculations, tmpDir);

                }
            }

            if (iter != null) iter.close();
            chunksProcessed++;
            i = chunkCounter.getAndIncrement();
        }
        if (currentMatrix != null) {
            currentMatrix.parsingComplete();
            //LittleEndianOutputStream[] localLos = {new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(outputFile + "_" + chromosomePairIndexes.get(currentPairIndex)), HiCGlobals.bufferSize))};
            //writeMatrix(currentMatrix, localLos, getDefaultCompressor(), localMatrixPositions, currentPairIndex, true);
        }
        wholeGenomeMatrixParts.get(currentChrPair).put(threadNum, wholeGenomeMatrix);
        return new Pair<>(new Pair<>(i, chunksProcessed), currentMatrix);

    }

    @Override
    protected void writeBody(String inputFile, Map<Integer, List<Chunk>> mndIndex) throws IOException {

        Set<String> syncWrittenMatrices = Collections.synchronizedSet(new HashSet<>());
        final AtomicInteger freeThreads = new AtomicInteger(numCPUThreads);



        for (int chrPair = 1; chrPair < chromosomePairCounter; chrPair++) {
            if (mndIndex.containsKey(chrPair)) {
                int numOfChunks = mndIndex.get(chrPair).size();
                totalChrPairToWrite++;
                completedChunksPerChrPair.put(chrPair, new AtomicInteger(0));
                numChunksPerChrPair.put(chrPair, numOfChunks);
                chrPairCompleted.put(chrPair, new AtomicInteger(0));
                chrPairAvailableThreads.put(chrPair, new AtomicInteger(0));
                chrPairBlockCapacities.put(chrPair, BLOCK_CAPACITY/Math.min(numCPUThreads,numOfChunks));
                threadSpecificChrPairMatrices.put(chrPair, new ConcurrentHashMap<>());
                wholeGenomeMatrixParts.put(chrPair, new ConcurrentHashMap<>());
                for (int i=0; i<numOfChunks; i++) {
                    int currentChunk = totalChunks;
                    chunkCounterToChrPairMap.put(currentChunk, chrPair);
                    chunkCounterToChrChunkMap.put(currentChunk, i);
                    totalChunks++;
                }
            }
        }

        ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);
        for (int i = 1; i < numCPUThreads; i++) {
            int threadNum = i;
            Runnable worker = new Runnable() {
                @Override
                public void run() {
                    try {
                        int currentChunk = chunkCounter.getAndIncrement();
                        Map<String, ExpectedValueCalculation> localExpectedValueCalculations = null;
                        if (expectedVectorFile == null) {
                            localExpectedValueCalculations = new LinkedHashMap<>();
                            for (int bBinSize : bpBinSizes) {
                                ExpectedValueCalculation calc = new ExpectedValueCalculation(chromosomeHandler, bBinSize, null, NormalizationHandler.NONE);
                                String key = "BP_" + bBinSize;
                                localExpectedValueCalculations.put(key, calc);
                            }
                            if (fragmentCalculation != null) {
                                // Create map of chr name -> # of fragments
                                Map<String, int[]> sitesMap = fragmentCalculation.getSitesMap();
                                Map<String, Integer> fragmentCountMap = new HashMap<>();
                                for (Map.Entry<String, int[]> entry : sitesMap.entrySet()) {
                                    int fragCount = entry.getValue().length + 1;
                                    String chr = entry.getKey();
                                    fragmentCountMap.put(chr, fragCount);
                                }

                                for (int fBinSize : fragBinSizes) {
                                    ExpectedValueCalculation calc = new ExpectedValueCalculation(chromosomeHandler, fBinSize, fragmentCountMap, NormalizationHandler.NONE);
                                    String key = "FRAG_" + fBinSize;
                                    localExpectedValueCalculations.put(key, calc);
                                }

                            }
                        }
                        while (currentChunk < totalChunks) {
                            int currentChrPair = chunkCounterToChrPairMap.get(currentChunk);
                            threadSpecificChrPairMatrices.get(currentChrPair).put(threadNum, processIndividualMatrixChunk(inputFile, currentChunk, currentChrPair, syncWrittenMatrices, localExpectedValueCalculations, threadNum));
                            synchronized(finalChrMatrices) {
                                if (!finalChrMatrices.containsKey(currentChrPair)) {
                                    int currentChr1 = chromosomePairIndex1.get(currentChrPair);
                                    int currentChr2 = chromosomePairIndex2.get(currentChrPair);
                                    finalChrMatrices.put(currentChrPair, new MatrixPP(currentChr1, currentChr2, chromosomeHandler, bpBinSizes, fragmentCalculation, fragBinSizes, countThreshold, v9DepthBase, chrPairBlockCapacities.get(currentChrPair)));
                                }
                                synchronized(finalChrMatrices.get(currentChrPair)) {
                                    finalChrMatrices.get(currentChrPair).mergeMatrices(threadSpecificChrPairMatrices.get(currentChrPair).get(threadNum).getSecond());
                                }
                            }

                            for (int completedChunks = 0; completedChunks < threadSpecificChrPairMatrices.get(currentChrPair).get(threadNum).getFirst().getSecond(); completedChunks++) {
                                completedChunksPerChrPair.get(currentChrPair).getAndIncrement();
                            }
                            //System.err.println(currentChrPair + " " + threadSpecificChrPairMatrices.get(currentChrPair).get(threadNum).getFirst().getSecond() + " " + Duration.between(A,B).toMillis() + " " + Duration.between(B,C).toMillis() + " " + completedChunksPerChrPair.get(currentChrPair).get());
                            currentChunk = threadSpecificChrPairMatrices.get(currentChrPair).get(threadNum).getFirst().getFirst();
                            int currentAvailableThreads = chrPairAvailableThreads.get(currentChrPair).incrementAndGet();
                            if (completedChunksPerChrPair.get(currentChrPair).get() == numChunksPerChrPair.get(currentChrPair)) {
                                WriteIndividualMatrix(currentChrPair, currentAvailableThreads);
                                finalChrMatrices.remove(currentChrPair);
                                threadSpecificChrPairMatrices.remove(currentChrPair);
                                chrPairCompleted.get(currentChrPair).getAndIncrement();
                                //System.err.println(currentChrPair + " " + Duration.between(D,E).toMillis());
                            }
                            while (chrPairCompleted.get(currentChrPair).get() == 0) {
                                try {
                                    Thread.sleep(1000);
                                } catch (InterruptedException e) {
                                    System.err.println(e.getLocalizedMessage());
                                }
                            }

                        }
                        allLocalExpectedValueCalculations.put(threadNum, localExpectedValueCalculations);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            };
            executor.execute(worker);
        }
        executor.shutdown();

        // Wait until all threads finish
        while (!executor.isTerminated()) {
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                System.err.println(e.getLocalizedMessage());
            }
        }


        if (expectedVectorFile == null) {
            for (int i = 0; i < numCPUThreads; i++) {
                if (allLocalExpectedValueCalculations.get(i) != null) {
                    for (Map.Entry<String, ExpectedValueCalculation> entry : allLocalExpectedValueCalculations.get(i).entrySet()) {
                        expectedValueCalculations.get(entry.getKey()).merge(entry.getValue());
                    }
                }
            }
        }

        MatrixPP wholeGenomeMatrix = getInitialGenomeWideMatrixPP(chromosomeHandler);

        for (int i = 1; i < chromosomePairCounter; i++) {
            if (nonemptyChromosomePairs.containsKey(i)) {
                if (wholeGenomeMatrixParts.containsKey(i)) {
                    for (Map.Entry<Integer, MatrixPP> entry : wholeGenomeMatrixParts.get(i).entrySet()) {
                        wholeGenomeMatrix.mergeMatrices(entry.getValue());
                    }
                }
            }
        }

        // just making this more readable
        FileOutputStream tempFOS = new FileOutputStream(outputFile + "_" + chromosomePairIndexes.get(0));
        LittleEndianOutputStream tempLOS = new LittleEndianOutputStream(new BufferedOutputStream(tempFOS, HiCGlobals.bufferSize));
        LittleEndianOutputStream[] localLos = {tempLOS};
        writeMatrix(wholeGenomeMatrix, localLos, getDefaultCompressor(), localMatrixPositions, 0, true);
        nonemptyChromosomePairs.put(0, 1);

        long currentPosition = losArray[0].getWrittenCount();
        long nextMatrixPosition = 0;
        String currentMatrixKey = null;

        for (int i = 0; i < chromosomePairCounter; i++) {
            if (nonemptyChromosomePairs.containsKey(i) && chromosomePairBlockIndexes.containsKey(i)) {
                for (Map.Entry<Long, List<IndexEntry>> entry : chromosomePairBlockIndexes.get(i).entrySet()) {
                    updateIndexPositions(entry.getValue(), null, false,
                            new File(outputFile + "_" + chromosomePairIndexes.get(i)),
                            currentPosition, entry.getKey());
                }
                nextMatrixPosition = localMatrixPositions.get("" + i).position + currentPosition;
                currentMatrixKey = chromosomePairIndex1.get(i) + "_" + chromosomePairIndex2.get(i);
                matrixPositions.put(currentMatrixKey, new IndexEntry(nextMatrixPosition, localMatrixPositions.get("" + i).size));
                currentPosition += matrixSizes.get(i);
            }
        }

        masterIndexPosition = currentPosition;


    }

    void WriteIndividualMatrix(Integer chromosomePair, int numOfNeededThreads) throws IOException {
        int chr1 = chromosomePairIndex1.get(chromosomePair);
        int chr2 = chromosomePairIndex2.get(chromosomePair);
        if (includedChromosomes != null) {
            String c1Name = chromosomeHandler.getChromosomeFromIndex(chr1).getName();
            String c2Name = chromosomeHandler.getChromosomeFromIndex(chr2).getName();
            if (includedChromosomes.contains(c1Name) || includedChromosomes.contains(c2Name)) {
                nonemptyChromosomePairs.put(chromosomePair, 1);
            }
        } else {
            nonemptyChromosomePairs.put(chromosomePair, 1);
        }

        LittleEndianOutputStream[] localLos;
        if (numOfNeededThreads == 1) {
            localLos = new LittleEndianOutputStream[1];
            localLos[0] = new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(outputFile + "_" + chromosomePairIndexes.get(chromosomePair)), HiCGlobals.bufferSize));
        } else {
            localLos = new LittleEndianOutputStream[(numOfNeededThreads * numResolutions) + 1];
            localLos[0] = new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(outputFile + "_" + chromosomePairIndexes.get(chromosomePair)), HiCGlobals.bufferSize));
            for (int i = 1; i <= numOfNeededThreads * numResolutions; i++) {
                localLos[i] = new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(outputFile + "_" + chromosomePairIndexes.get(chromosomePair) + "_" + i), HiCGlobals.bufferSize));
            }
        }

        writeMatrix(finalChrMatrices.get(chromosomePair), localLos, getDefaultCompressor(), localMatrixPositions, chromosomePair, true);

    }

    @Override
    // MatrixPP matrix, LittleEndianOutputStream los, Deflater compressor
    protected Pair<Map<Long, List<IndexEntry>>, Long> writeMatrix(MatrixPP matrix, LittleEndianOutputStream[] localLos,
                                                                  Deflater localCompressor, Map<String, IndexEntry> localMatrixPositions,
                                                                  int chromosomePairIndex, boolean doMultiThreadedBehavior) throws IOException {

        Pair<Map<Long, List<IndexEntry>>, Long> localBlockIndexes = super.writeMatrix(matrix, localLos, localCompressor,
                localMatrixPositions, chromosomePairIndex, true);

        chromosomePairBlockIndexes.put(chromosomePairIndex, localBlockIndexes.getFirst());
        long size = 0 - localBlockIndexes.getSecond();
        for (int i = 0; i < localLos.length; i++) {
            size += localLos[i].getWrittenCount();
            localLos[i].close();
        }
        matrixSizes.put(chromosomePairIndex, size);


        //System.out.print(".");

        return localBlockIndexes;
    }
}
