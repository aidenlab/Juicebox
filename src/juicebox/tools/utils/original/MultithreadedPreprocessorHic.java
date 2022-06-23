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
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.tools.utils.original.mnditerator.AlignmentPair;
import juicebox.tools.utils.original.mnditerator.AsciiPairIterator;
import juicebox.tools.utils.original.mnditerator.PairIterator;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import org.broad.igv.util.Pair;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.Deflater;


public class MultithreadedPreprocessorHic extends Preprocessor {
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
    private Dataset ds;
    private long currentPosition;

    public MultithreadedPreprocessorHic(File outputFile, String genomeId, ChromosomeHandler chromosomeHandler,
                                        double hicFileScalingFactor, int numCPUThreads) {
        super(outputFile, genomeId, chromosomeHandler, hicFileScalingFactor);
        MultithreadedPreprocessorHic.numCPUThreads = numCPUThreads;
        chromosomeIndexes = MTIndexHandler.populateChromosomeIndexes(chromosomeHandler, numCPUThreads);
        chromosomePairCounter = MTIndexHandler.populateChromosomePairIndexes(chromosomeHandler,
                chromosomePairIndexes, chromosomePairIndexesReverse,
                chromosomePairIndex1, chromosomePairIndex2);
        this.chromosomePairBlockIndexes = new ConcurrentHashMap<>(chromosomePairCounter, (float) 0.75, numCPUThreads);
        this.allLocalExpectedValueCalculations = new ConcurrentHashMap<>(numCPUThreads, (float) 0.75, numCPUThreads);
    }

    @Override
    public void preprocess(final String inputFile, String ignore1, String ignore2, Map<Integer,
            List<Chunk>> ignore3) throws IOException {
        List<String> summedHiCFiles = Arrays.asList(inputFile.split("\\+"));
        ds = HiCFileTools.extractDatasetForCLT(summedHiCFiles, true);
        super.preprocess(inputFile, outputFile + "_header", outputFile + "_footer", mndIndex);

        try {
            PrintWriter finalOutput = new PrintWriter(outputFile + CAT_SCRIPT);
            StringBuilder catOutputLine = new StringBuilder();
            StringBuilder removeLine = new StringBuilder();
            catOutputLine.append("cat ").append(outputFile + "_header").append(" >> ").append(outputFile).append("\n");
            removeLine.append("rm ").append(outputFile + "_header\n");
            for (int i = 0; i < chromosomePairCounter; i++) {
                if ((nonemptyChromosomePairs.containsKey(i) && chromosomePairBlockIndexes.containsKey(i)) || i == 0) {
                    catOutputLine.append("cat ").append(outputFile).append("_").append(chromosomePairIndexes.get(i)).append(" >> ").append(outputFile).append("\n");
                    removeLine.append("rm ").append(outputFile).append("_").append(chromosomePairIndexes.get(i)).append("\n");
                    if (i > 0) {
                        int numOfNeededThreads = numCPUThreads;
                        if (numOfNeededThreads > 1) {
                            for (int j = 1; j <= numOfNeededThreads * numResolutions; j++) {
                                catOutputLine.append("cat ").append(outputFile).append("_").append(chromosomePairIndexes.get(i)).append("_").append(j).append(" >> ").append(outputFile).append("\n");
                                removeLine.append("rm ").append(outputFile).append("_").append(chromosomePairIndexes.get(i)).append("_").append(j).append("\n");
                            }
                        }
                    }
                }
            }
            catOutputLine.append("cat ").append(outputFile + "_footer").append(" >> ").append(outputFile).append("\n");
            removeLine.append("rm ").append(outputFile + "_footer\n");
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



    @Override
    protected void writeBody(String inputFile, Map<Integer, List<Chunk>> mndIndex) throws IOException {

        currentPosition = losArray[0].getWrittenCount();
        Set<String> syncWrittenMatrices = Collections.synchronizedSet(new HashSet<>());
        final AtomicInteger freeThreads = new AtomicInteger(numCPUThreads);

        Map<String, Integer> fragmentCountMap = null;
        boolean calculateExpecteds = false;
        if (expectedVectorFile == null) {
            if (fragmentCalculation != null) {
                // Create map of chr name -> # of fragments
                Map<String, int[]> sitesMap = fragmentCalculation.getSitesMap();
                fragmentCountMap = new HashMap<>();
                for (Map.Entry<String, int[]> entry : sitesMap.entrySet()) {
                    int fragCount = entry.getValue().length + 1;
                    String chr = entry.getKey();
                    fragmentCountMap.put(chr, fragCount);
                }
            }
            calculateExpecteds = true;
        }

        WriteIndividualMatrix(0,1, false, fragmentCountMap);


        for (int chrPair = 1; chrPair < chromosomePairCounter; chrPair++) {
            WriteIndividualMatrix(chrPair, numCPUThreads, calculateExpecteds, fragmentCountMap);
            totalChrPairToWrite++;
        }

        masterIndexPosition = currentPosition;


    }

    void WriteIndividualMatrix(Integer chromosomePair, int numOfNeededThreads, boolean calculateExpecteds, Map<String, Integer> fragmentCountMap) throws IOException {
        int chr1 = chromosomePairIndex1.get(chromosomePair);
        int chr2 = chromosomePairIndex2.get(chromosomePair);

        Matrix combinedMatrix = ds.getMatrix(chromosomeHandler.getChromosomeFromIndex(chr1), chromosomeHandler.getChromosomeFromIndex(chr2));

        if (includedChromosomes != null && combinedMatrix!=null) {
            String c1Name = chromosomeHandler.getChromosomeFromIndex(chr1).getName();
            String c2Name = chromosomeHandler.getChromosomeFromIndex(chr2).getName();
            if (includedChromosomes.contains(c1Name) || includedChromosomes.contains(c2Name)) {
                nonemptyChromosomePairs.put(chromosomePair, 1);
            }
        } else {
            nonemptyChromosomePairs.put(chromosomePair, 1);
        }
        if (combinedMatrix!=null && nonemptyChromosomePairs.containsKey(chromosomePair)) {
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


            LittleEndianOutputStream los = localLos[0];
            long position = los.getWrittenCount();

            los.writeInt(combinedMatrix.getChr1Idx());
            los.writeInt(combinedMatrix.getChr2Idx());

            int numResolutions = 0;
            for (int i = 0; i < bpBinSizes.length; i++) {
                MatrixZoomData zd = combinedMatrix.getZoomData(new HiCZoom(HiC.Unit.BP, bpBinSizes[i]));
                if (zd != null) {
                    numResolutions += 1;
                }
                if (chromosomePair == 0) {
                    break;
                }
            }
            for (int i = 0; i < fragBinSizes.length; i++) {
                MatrixZoomData zd = combinedMatrix.getZoomData(new HiCZoom(HiC.Unit.FRAG, fragBinSizes[i]));
                if (chromosomePair > 0) {
                    if (zd != null) {
                        numResolutions += 1;
                    }
                }
            }
            los.writeInt(numResolutions);

            for (int i = 0; i < numResolutions; i++) {
                MatrixZoomData zd;
                if (i < bpBinSizes.length) {
                    zd = combinedMatrix.getZoomData(new HiCZoom(HiC.Unit.BP, bpBinSizes[i]));
                } else {
                    zd = combinedMatrix.getZoomData(new HiCZoom(HiC.Unit.FRAG, fragBinSizes[i - bpBinSizes.length]));
                }
                if (zd != null)
                    if (i < bpBinSizes.length) {
                        writeZoomHeader(zd, los, i);
                    } else {
                        writeZoomHeader(zd, los, i - bpBinSizes.length);
                    }

            }

            long size = los.getWrittenCount() - position;
            matrixPositions.put(combinedMatrix.getKey(), new IndexEntry(currentPosition + position, (int) size));

            final Map<Long, List<IndexEntry>> localBlockIndexes = new ConcurrentHashMap<>();

            for (int i = 0; i < numResolutions; i++) {
                MatrixZoomData zd;
                if (i < bpBinSizes.length) {
                    zd = combinedMatrix.getZoomData(new HiCZoom(HiC.Unit.BP, bpBinSizes[i]));
                } else {
                    zd = combinedMatrix.getZoomData(new HiCZoom(HiC.Unit.FRAG, fragBinSizes[i - bpBinSizes.length]));
                }
                if (zd != null) {
                    Pair<List<IndexEntry>, ExpectedValueCalculation> zdOutput = null;
                    if (localLos.length > 1) {
                        zdOutput = zd.mergeAndWriteBlocks(localLos, compressor, i, numResolutions, calculateExpecteds, fragmentCountMap, chromosomeHandler, subsampleFraction, randomSubsampleGenerator);
                    } else {
                        zdOutput = zd.mergeAndWriteBlocks(localLos[0], compressor, calculateExpecteds, fragmentCountMap, chromosomeHandler, subsampleFraction, randomSubsampleGenerator);
                    }
                    localBlockIndexes.put(zd.blockIndexPosition, zdOutput.getFirst());
                    if (calculateExpecteds) {
                        ExpectedValueCalculation tmpCalc = zdOutput.getSecond();
                        String key;
                        if (zd.getZoom().getUnit() == HiC.Unit.BP) {
                            key = "BP_" + zd.getZoom().getBinSize();
                        } else {
                            key = "FRAG_" + zd.getZoom().getBinSize();
                        }
                        expectedValueCalculations.get(key).merge(tmpCalc);
                        tmpCalc = null;
                    }
                }
            }



            long matrixSize = 0;
            for (int i = 0; i < localLos.length; i++) {
                matrixSize += localLos[i].getWrittenCount();
                localLos[i].close();
            }

            chromosomePairBlockIndexes.put(chromosomePair, localBlockIndexes);
            for (Map.Entry<Long, List<IndexEntry>> entry : localBlockIndexes.entrySet()) {
                updateIndexPositions(entry.getValue(), null, false,
                        new File(outputFile + "_" + chromosomePairIndexes.get(chromosomePair)),
                        currentPosition, entry.getKey());
            }

            matrixSizes.put(chromosomePair, matrixSize);
            currentPosition += matrixSize;
            System.out.print(".");
        }
    }

    protected void writeZoomHeader(MatrixZoomData zd, LittleEndianOutputStream los, int zoom) throws IOException {
        int numberOfBlocks = zd.getBlockNumbers().size();
        los.writeString(zd.getZoom().getUnit().toString());  // Unit
        los.writeInt(zoom);     // zoom index,  lowest res is zero
        los.writeFloat((float) zd.getSumCount());      // sum
        los.writeFloat((float) 0);
        los.writeFloat((float) 0);
        los.writeFloat((float) 0);
        los.writeInt(zd.getBinSize());
        los.writeInt(zd.getBlockBinCount());
        los.writeInt(zd.getBlockColumnCount());
        los.writeInt(numberOfBlocks);

        zd.blockIndexPosition = los.getWrittenCount();

        // Placeholder for block index
        for (int i = 0; i < numberOfBlocks; i++) {
            los.writeInt(0);
            los.writeLong(0L);
            los.writeInt(0);
        }
    }

}
