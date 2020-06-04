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
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import org.broad.igv.util.Pair;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.Deflater;


public class MultithreadedPreprocessor extends Preprocessor {
    private final Map<Integer, String> chromosomePairIndexes = new ConcurrentHashMap<>();
    private final Map<String, Integer> chromosomePairIndexesReverse = new ConcurrentHashMap<>();
    private final Map<Integer, Integer> chromosomePairIndex1 = new ConcurrentHashMap<>();
    private final Map<Integer, Integer> chromosomePairIndex2 = new ConcurrentHashMap<>();
    private int chromosomePairCounter = 0;
    private final Map<Integer, Integer> nonemptyChromosomePairs = new ConcurrentHashMap<>();
    private final Map<Integer, MatrixPP> wholeGenomeMatrixParts = new ConcurrentHashMap<>();
    private final Map<String, IndexEntry> localMatrixPositions = new ConcurrentHashMap<>();
    private final Map<Integer, Long> matrixSizes = new ConcurrentHashMap<>();
    private final Map<Integer, Map<Long, List<IndexEntry>>> chromosomePairBlockIndexes;
    protected static int numCPUThreads = 1;
    protected static Map<Integer, Long> mndIndex = null;

    public MultithreadedPreprocessor(File outputFile, String genomeId, ChromosomeHandler chromosomeHandler, double hicFileScalingFactor) {
        super(outputFile, genomeId, chromosomeHandler, hicFileScalingFactor);

        chromosomeIndexes = new ConcurrentHashMap<>(chromosomeHandler.size(), (float) 0.75, numCPUThreads);
        for (int i = 0; i < chromosomeHandler.size(); i++) {
            chromosomeIndexes.put(chromosomeHandler.getChromosomeFromIndex(i).getName(), i);
        }

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

    public void setMndIndex(String mndIndexFile) {
        if (mndIndexFile != null && mndIndexFile.length() > 1) {
            mndIndex = readMndIndex(mndIndexFile);
        }
    }

    @Override
    public void preprocess(final String inputFile, String ignore1, String ignore2, Map<Integer, Long> ignore3) throws IOException {
        super.preprocess(inputFile, outputFile + "_header", outputFile + "_footer", mndIndex);

        try {
            PrintWriter finalOutput = new PrintWriter(outputFile + "catOutputs.sh");
            StringBuilder catOutputLine = new StringBuilder();
            StringBuilder removeLine = new StringBuilder();
            catOutputLine.append("cat ").append(outputFile + "_header");
            removeLine.append("rm ").append(outputFile + "_header");
            for (int i = 0; i < chromosomePairCounter; i++) {
                if ((nonemptyChromosomePairs.containsKey(i) && chromosomePairBlockIndexes.containsKey(i)) || i == 0) {
                    catOutputLine.append(" ").append(outputFile).append("_").append(chromosomePairIndexes.get(i));
                    removeLine.append(" ").append(outputFile).append("_").append(chromosomePairIndexes.get(i));
                }
            }
            catOutputLine.append(" ").append(outputFile + "_footer").append(" > ").append(outputFile).append("\n");
            removeLine.append(" ").append(outputFile + "_footer\n");
            finalOutput.println(catOutputLine.toString());
            finalOutput.println(removeLine.toString());
            finalOutput.close();
        } catch (Exception e) {
            System.err.println("Unable to write to catOutputs.sh");
            System.exit(70);
        }
    }

    private Map<Integer, Long> readMndIndex(String mndIndexFile) {
        FileInputStream is = null;
        Map<String, Long> tempIndex = new HashMap<>();
        Map<Integer, Long> mndIndex = new ConcurrentHashMap<>();
        try {
            is = new FileInputStream(mndIndexFile);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is), HiCGlobals.bufferSize);
            String nextLine;
            while ((nextLine = reader.readLine()) != null) {
                String[] nextEntry = nextLine.split(",");
                if (nextEntry.length > 2 || nextEntry.length < 2) {
                    System.err.println("Improperly formatted merged nodups index");
                    System.exit(70);
                } else {
                    tempIndex.put(nextEntry[0], Long.parseLong(nextEntry[1]));
                }
            }
        } catch (Exception e) {
            System.err.println("Unable to read merged nodups index");
            System.exit(70);
        }
        for (Map.Entry<Integer,String> entry : chromosomePairIndexes.entrySet()) {
            String reverseName = entry.getValue().split("_")[1] + "_" + entry.getValue().split("_")[0];
            if (tempIndex.containsKey(entry.getValue())) {
                mndIndex.put(entry.getKey(), tempIndex.get(entry.getValue()));
            } else if (tempIndex.containsKey(reverseName)) {
                mndIndex.put(entry.getKey(), tempIndex.get(reverseName));
            }
        }
        return mndIndex;
    }

    private int getGenomicPosition(int chr, int pos, ChromosomeHandler localChromosomeHandler) {
        long len = 0;
        for (int i = 1; i < chr; i++) {
            len += localChromosomeHandler.getChromosomeFromIndex(i).getLength();
        }
        len += pos;

        return (int) (len / 1000);

    }

    private void writeBodySingleChromosomePair(String inputFile, String splitInputFile, int givenChromosomePairIndex, Set<String> syncWrittenMatrices, ChromosomeHandler
            localChromosomeHandler, Long mndIndexPosition) throws IOException {

        MatrixPP wholeGenomeMatrix = getInitialGenomeWideMatrixPP(localChromosomeHandler);

        PairIterator iter;
        if (mndIndex == null) {
            iter = (inputFile.endsWith(".bin")) ?
                    new BinPairIterator(splitInputFile) :
                    new AsciiPairIterator(splitInputFile, chromosomeIndexes, chromosomeHandler);
        } else {
            iter = new AsciiPairIterator(inputFile, chromosomeIndexes, mndIndexPosition, chromosomeHandler);
        }


        String currentMatrixName = null;
        int currentPairIndex = -1;

        int currentChr1 = -1;
        int currentChr2 = -1;
        MatrixPP currentMatrix = null;
        String currentMatrixKey = null;

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


                // Randomize position within fragment site
                if (allowPositionsRandomization && fragmentCalculation != null) {
                    Pair<Integer, Integer> newBPos12 = getRandomizedPositions(chr1, chr2, frag1, frag2, bp1, bp2);
                    bp1 = newBPos12.getFirst();
                    bp2 = newBPos12.getSecond();
                }
                // only increment if not intraFragment and passes the mapq threshold
                if (!(currentChr1 == chr1 && currentChr2 == chr2)) {
                    // Starting a new matrix
                    if (currentMatrix != null) {
                        currentMatrix.parsingComplete();
                        LittleEndianOutputStream[] localLos = {new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(outputFile + "_" + chromosomePairIndexes.get(currentPairIndex)), HiCGlobals.bufferSize))};
                        writeMatrix(currentMatrix, localLos, getDefaultCompressor(), localMatrixPositions, currentPairIndex, true);
                        syncWrittenMatrices.add(currentMatrixKey);
                        currentMatrix = null;
                        System.gc();
                        break;
                        //System.out.println("Available memory: " + RuntimeUtils.getAvailableMemory());
                    }

                    // Start the next matrix
                    currentChr1 = chr1;
                    currentChr2 = chr2;
                    currentMatrixKey = currentChr1 + "_" + currentChr2;

                    currentMatrixName = localChromosomeHandler.getChromosomeFromIndex(chr1).getName() + "_" + localChromosomeHandler.getChromosomeFromIndex(chr2).getName();
                    currentPairIndex = chromosomePairIndexesReverse.get(currentMatrixName);

                    if (currentPairIndex != givenChromosomePairIndex) {
                        break;
                    }

                    if (syncWrittenMatrices.contains(currentMatrixKey)) {
                        System.err.println("Error: the chromosome combination " + currentMatrixKey + " appears in multiple blocks");
                        if (outputFile != null) outputFile.deleteOnExit();
                        System.exit(58);
                    }
                    currentMatrix = new MatrixPP(currentChr1, currentChr2, chromosomeHandler, bpBinSizes, fragmentCalculation, fragBinSizes, countThreshold);
                }
                currentMatrix.incrementCount(bp1, bp2, frag1, frag2, pair.getScore(), expectedValueCalculations, tmpDir);

                int pos1 = getGenomicPosition(chr1, bp1, localChromosomeHandler);
                int pos2 = getGenomicPosition(chr2, bp2, localChromosomeHandler);
                wholeGenomeMatrix.incrementCount(pos1, pos2, pos1, pos2, pair.getScore(), expectedValueCalculations, tmpDir);

            }
        }

        if (currentMatrix != null) {
            currentMatrix.parsingComplete();
            LittleEndianOutputStream[] localLos = {new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(outputFile + "_" + chromosomePairIndexes.get(currentPairIndex)), HiCGlobals.bufferSize))};
            writeMatrix(currentMatrix, localLos, getDefaultCompressor(), localMatrixPositions, currentPairIndex, true);
        }


        if (iter != null) iter.close();

        wholeGenomeMatrixParts.put(currentPairIndex, wholeGenomeMatrix);

    }

    @Override
    protected void writeBody(String inputFile, Map<Integer, Long> mndIndex) throws IOException {

        Set<String> syncWrittenMatrices = Collections.synchronizedSet(new HashSet<>());
        final AtomicInteger chromosomePair = new AtomicInteger(1);

        ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);
        for (int l = 0; l < numCPUThreads; l++) {
            final int threadNum = l;
            Runnable worker = new Runnable() {
                @Override
                public void run() {
                    runIndividualMatrixCode(chromosomePair, inputFile, syncWrittenMatrices, threadNum, mndIndex);
                }
            };
            executor.execute(worker);
        }
        executor.shutdown();
        // Wait until all threads finish
        while (!executor.isTerminated()) {
        }

        MatrixPP wholeGenomeMatrix = getInitialGenomeWideMatrixPP(chromosomeHandler);

        for (int i = 1; i < chromosomePairCounter; i++) {
            if (nonemptyChromosomePairs.containsKey(i)) {
                if (wholeGenomeMatrixParts.containsKey(i)) {
                    wholeGenomeMatrix.mergeMatrices(wholeGenomeMatrixParts.get(i));
                }
            }
        }

        LittleEndianOutputStream[] localLos = {new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(outputFile + "_" + chromosomePairIndexes.get(0)), HiCGlobals.bufferSize))};
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

    void runIndividualMatrixCode(AtomicInteger chromosomePair, String inputFile, Set<String> syncWrittenMatrices, int threadNum,
                                 Map<Integer,Long> mndIndex) {
        int i = chromosomePair.getAndIncrement();

        int localChromosomePairCounter = chromosomePairCounter;

        while (i < localChromosomePairCounter) {
            Long mndIndexPosition = (long) 0;
            if (mndIndex != null) {
                if (!mndIndex.containsKey(i)) {
                    System.out.println("No index position for " + chromosomePairIndexes.get(i));
                    i = chromosomePair.getAndIncrement();
                    continue;
                } else {
                    mndIndexPosition = mndIndex.get(i);
                    try {
                        writeBodySingleChromosomePair(inputFile, null, i, syncWrittenMatrices, chromosomeHandler, mndIndexPosition);
                    } catch (Exception e2) {
                        e2.printStackTrace();
                    }
                }
            } else {
                // split method; deprecate?
                boolean isGZ = inputFile.endsWith(".gz");
                String chrInputFile = inputFile.replaceAll(".gz", "") + "_" + chromosomePairIndexes.get(i);
                String chrInputFile2 = inputFile + "_" + chromosomeHandler.getChromosomeFromIndex(
                        chromosomePairIndex2.get(i)).getName() + "_" + chromosomeHandler.getChromosomeFromIndex(
                        chromosomePairIndex1.get(i)).getName();
                if (isGZ) {
                    chrInputFile = chrInputFile + ".gz";
                    chrInputFile2 = chrInputFile2 + ".gz";
                }

                try {
                    writeBodySingleChromosomePair(inputFile, chrInputFile, i, syncWrittenMatrices, chromosomeHandler, mndIndexPosition);
                } catch (Exception e) {
                    try {
                        writeBodySingleChromosomePair(inputFile, chrInputFile2, i, syncWrittenMatrices, chromosomeHandler, mndIndexPosition);
                    } catch (Exception e2) {
                        System.err.println("Unable to open " + inputFile + "_" + chromosomePairIndexes.get(i));
                    }
                }
            }
            int chr1 = chromosomePairIndex1.get(i);
            int chr2 = chromosomePairIndex2.get(i);
            if (includedChromosomes != null) {
                String c1Name = chromosomeHandler.getChromosomeFromIndex(chr1).getName();
                String c2Name = chromosomeHandler.getChromosomeFromIndex(chr2).getName();
                if (includedChromosomes.contains(c1Name) || includedChromosomes.contains(c2Name)) {
                    nonemptyChromosomePairs.put(i, 1);
                }
            } else {
                nonemptyChromosomePairs.put(i, 1);
            }

            i = chromosomePair.getAndIncrement();
        }
    }
    @Override
    // MatrixPP matrix, LittleEndianOutputStream los, Deflater compressor
    protected Pair<Map<Long, List<IndexEntry>>, Long> writeMatrix(MatrixPP matrix, LittleEndianOutputStream[] localLos,
                                                                  Deflater localCompressor, Map<String, IndexEntry> localMatrixPositions, int chromosomePairIndex, boolean doMultiThreadedBehavior) throws IOException {

        Pair<Map<Long, List<IndexEntry>>, Long> localBlockIndexes = super.writeMatrix(matrix, localLos, localCompressor, localMatrixPositions, chromosomePairIndex, true);

        chromosomePairBlockIndexes.put(chromosomePairIndex, localBlockIndexes.getFirst());
        long size = localLos[0].getWrittenCount() - localBlockIndexes.getSecond();
        matrixSizes.put(chromosomePairIndex, size);
        localLos[0].close();

        System.out.print(".");

        return localBlockIndexes;
    }
}
