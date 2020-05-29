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
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.HiCFileTools;
import juicebox.windowui.NormalizationHandler;
import org.broad.igv.tdf.BufferedByteWriter;

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
    private final Map<Integer, IndexEntry> localMatrixPositions = new ConcurrentHashMap<>();
    private final Map<Integer, Integer> matrixSizes = new ConcurrentHashMap<>();
    private LittleEndianOutputStream losWholeGenome;
    private LittleEndianOutputStream losFooter;
    private final Map<Integer, Map<Long, List<IndexEntry>>> chromosomePairBlockIndexes;
    protected static int numCPUThreads = 1;
    private final Map<Integer, Map<String, ExpectedValueCalculation>> allLocalExpectedValueCalculations;
    protected static String mndIndexFile;

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
        this.allLocalExpectedValueCalculations = new ConcurrentHashMap<>(chromosomePairCounter, (float) 0.75, numCPUThreads);
    }

    public void setNumCPUThreads(int numCPUThreads) {
        MultithreadedPreprocessor.numCPUThreads = numCPUThreads;
    }

    public void setMndIndex(String mndIndexFile) {
        MultithreadedPreprocessor.mndIndexFile = mndIndexFile;
    }

    @Override
    public void preprocess(final String inputFile) throws IOException {
        File file = new File(inputFile);
        Map<Integer, Long> mndIndex = new ConcurrentHashMap<>();

        if (mndIndexFile != null) {
            mndIndex = readMndIndex(mndIndexFile);
        }

        if (!file.exists() || file.length() == 0) {
            System.err.println(inputFile + " does not exist or does not contain any reads.");
            System.exit(57);
        }

        try {
            StringBuilder stats = null;
            StringBuilder graphs = null;
            StringBuilder hicFileScaling = new StringBuilder().append(hicFileScalingFactor);
            if (fragmentFileName != null) {
                try {
                    fragmentCalculation = FragmentCalculation.readFragments(fragmentFileName, chromosomeHandler);
                } catch (Exception e) {
                    System.err.println("Warning: Unable to process fragment file. Pre will continue without fragment file.");
                    fragmentCalculation = null;
                }
            } else {
                System.out.println("Not including fragment map");
            }

            if (allowPositionsRandomization) {
                if (randomizeFragMapFiles != null) {
                    fragmentCalculationsForRandomization = new ArrayList<>();
                    for (String fragmentFileName : randomizeFragMapFiles) {
                        try {
                            FragmentCalculation fragmentCalculation = FragmentCalculation.readFragments(fragmentFileName, chromosomeHandler);
                            fragmentCalculationsForRandomization.add(fragmentCalculation);
                            System.out.println(String.format("added %s", fragmentFileName));
                        } catch (Exception e) {
                            System.err.println(String.format("Warning: Unable to process fragment file %s. Randomization will continue without fragment file %s.", fragmentFileName, fragmentFileName));
                        }
                    }
                } else {
                    System.out.println("Using default fragment map for randomization");
                }

            } else if (randomizeFragMapFiles != null) {
                System.err.println("Position randomizer seed not set, disregarding map options");
            }

            if (statsFileName != null) {
                FileInputStream is = null;
                try {
                    is = new FileInputStream(statsFileName);
                    BufferedReader reader = new BufferedReader(new InputStreamReader(is), HiCGlobals.bufferSize);
                    stats = new StringBuilder();
                    String nextLine;
                    while ((nextLine = reader.readLine()) != null) {
                        stats.append(nextLine).append("\n");
                    }
                } catch (IOException e) {
                    System.err.println("Error while reading stats file: " + e);
                    stats = null;
                } finally {
                    if (is != null) {
                        is.close();
                    }
                }

            }
            if (graphFileName != null) {
                FileInputStream is = null;
                try {
                    is = new FileInputStream(graphFileName);
                    BufferedReader reader = new BufferedReader(new InputStreamReader(is), HiCGlobals.bufferSize);
                    graphs = new StringBuilder();
                    String nextLine;
                    while ((nextLine = reader.readLine()) != null) {
                        graphs.append(nextLine).append("\n");
                    }
                } catch (IOException e) {
                    System.err.println("Error while reading graphs file: " + e);
                    graphs = null;
                } finally {
                    if (is != null) {
                        is.close();
                    }
                }
            }

            if (expectedVectorFile == null) {
                expectedValueCalculations = new LinkedHashMap<>();
                for (int bBinSize : bpBinSizes) {
                    ExpectedValueCalculation calc = new ExpectedValueCalculation(chromosomeHandler, bBinSize, null, NormalizationHandler.NONE);
                    String key = "BP_" + bBinSize;
                    expectedValueCalculations.put(key, calc);
                }
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

                if (expectedVectorFile == null) {
                    for (int fBinSize : fragBinSizes) {
                        ExpectedValueCalculation calc = new ExpectedValueCalculation(chromosomeHandler, fBinSize, fragmentCountMap, NormalizationHandler.NONE);
                        String key = "FRAG_" + fBinSize;
                        expectedValueCalculations.put(key, calc);
                    }
                }
            }

            try {
                los = new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(outputFile+"_header"), HiCGlobals.bufferSize));
                //losContainer = new ConcurrentHashMap<>(chromosomePairCounter, (float) 0.75, numCPUThreads);
                //for (int i = 0; i < chromosomePairCounter; i++) {
                //    losContainer.put(i, new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(outputFile+"_"+chromosomePairIndexes.get(i)), HiCGlobals.bufferSize)));
                //}
                losFooter = new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(outputFile+"_footer"), HiCGlobals.bufferSize));
            } catch (Exception e) {
                System.err.println("Unable to write to " + outputFile);
                System.exit(70);
            }

            System.out.println("Start preprocess");

            System.out.println("Writing header");

            writeHeader(stats, graphs, hicFileScaling);

            System.out.println("Writing body");
            writeBody(inputFile, mndIndex);

            if (expectedVectorFile == null) {
                for (int i = 0; i < numCPUThreads; i++) {
                    for (Map.Entry<String, ExpectedValueCalculation> entry : allLocalExpectedValueCalculations.get(i).entrySet()) {
                        expectedValueCalculations.get(entry.getKey()).merge(entry.getValue());
                    }
                }
            }

            System.out.println();
            System.out.println("Writing footer");
            writeFooter();


        } finally {
            if (los != null)
                los.close();
        }

        updateMasterIndex();

        try {
            PrintWriter finalOutput = new PrintWriter("catOutputs.sh");
            StringBuilder outputLine = new StringBuilder();
            outputLine.append("cat ").append(outputFile + "_header");
            for (int i = 0; i < chromosomePairCounter; i++) {
                if (nonemptyChromosomePairs.containsKey(i) || i == 0) {
                    outputLine.append(" ").append(outputFile).append("_").append(chromosomePairIndexes.get(i));
                }
            }
            outputLine.append(" ").append(outputFile + "_footer").append("\n");
            finalOutput.println(outputLine.toString());
            finalOutput.close();
        } catch (Exception e) {
            System.err.println("Unable to write to catOutputs.sh");
            System.exit(70);
        }
        System.out.println("\nFinished preprocess");
    }

    private Map<Integer,Long> readMndIndex(String mndIndexFile) throws IOException {
        FileInputStream is = null;
        Map<String,Long> tempIndex = new HashMap<>();
        Map<Integer,Long> mndIndex = new ConcurrentHashMap<>();
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

    private void writeBodySingleChromosomePair(String inputFile, String chrInputFile, Set<String> syncWrittenMatrices, ChromosomeHandler
            localChromosomeHandler, Set<String> localIncludedChromosomes, Map<String, ExpectedValueCalculation>
            localExpectedValueCalculations, Long mndIndexPosition) throws IOException {

        MatrixPP wholeGenomeMatrix;
        // NOTE: always true that c1 <= c2
        int genomeLength = localChromosomeHandler.getChromosomeFromIndex(0).getLength();  // <= whole genome in KB
        int binSize = genomeLength / 500;
        if (binSize == 0) binSize = 1;
        int nBinsX = genomeLength / binSize + 1;
        int nBlockColumns = nBinsX / BLOCK_SIZE + 1;
        wholeGenomeMatrix = new MatrixPP(0, 0, binSize, nBlockColumns);


        PairIterator iter;
        if (mndIndexFile==null) {
            iter = (inputFile.endsWith(".bin")) ?
                    new BinPairIterator(chrInputFile) :
                    new AsciiPairIterator(chrInputFile, chromosomeIndexes, chromosomeHandler);
        } else {
            iter = new AsciiPairIterator(inputFile, chromosomeIndexes, mndIndexPosition);

        }


        int currentChr1 = -1;
        int currentChr2 = -1;
        MatrixPP currentMatrix = null;

        String currentMatrixKey = null;
        String currentMatrixName = null;
        int currentPairIndex = -1;

        // randomization error/ambiguity stats
        int noMapFoundCount = 0;
        int mapDifferentCount = 0;


        while (iter.hasNext()) {
            AlignmentPair pair = iter.next();
            // skip pairs that mapped to contigs
            if (!pair.isContigPair()) {
                // Flip pair if needed so chr1 < chr2
                int chr1, chr2, bp1, bp2, frag1, frag2, mapq;
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
                mapq = Math.min(pair.getMapq1(), pair.getMapq2());
                int pos1, pos2;
                // Filters
                if (diagonalsOnly && chr1 != chr2) continue;
                if (localIncludedChromosomes != null && chr1 != 0) {
                    String c1Name = localChromosomeHandler.getChromosomeFromIndex(chr1).getName();
                    String c2Name = localChromosomeHandler.getChromosomeFromIndex(chr2).getName();
                    if (!(localIncludedChromosomes.contains(c1Name) || localIncludedChromosomes.contains(c2Name))) {
                        continue;
                    }
                }
                if (alignmentFilter != null && !alignmentsAreEqual(calculateAlignment(pair), alignmentFilter)) {
                    continue;
                }

                // Randomize position within fragment site
                if (fragmentCalculation != null && allowPositionsRandomization) {
                    FragmentCalculation fragMapToUse;
                    if (fragmentCalculationsForRandomization != null) {
                        FragmentCalculation fragMap1 = findFragMap(fragmentCalculationsForRandomization, localChromosomeHandler.getChromosomeFromIndex(chr1).getName(), bp1, frag1);
                        FragmentCalculation fragMap2 = findFragMap(fragmentCalculationsForRandomization, localChromosomeHandler.getChromosomeFromIndex(chr2).getName(), bp2, frag2);

                        if (fragMap1 == null && fragMap2 == null) {
                            noMapFoundCount += 1;
                            continue;
                        } else if (fragMap1 != null && fragMap2 != null && fragMap1 != fragMap2) {
                            mapDifferentCount += 1;
                            continue;
                        }

                        if (fragMap1 != null) {
                            fragMapToUse = fragMap1;
                        } else {
                            fragMapToUse = fragMap2;
                        }

                    } else {
                        // use default map
                        fragMapToUse = fragmentCalculation;
                    }

                    bp1 = randomizePos(fragMapToUse, localChromosomeHandler.getChromosomeFromIndex(chr1).getName(), frag1);
                    bp2 = randomizePos(fragMapToUse, localChromosomeHandler.getChromosomeFromIndex(chr2).getName(), frag2);
                }
                // only increment if not intraFragment and passes the mapq threshold
                if (mapq < mapqThreshold || (throwOutIntraFrag && chr1 == chr2 && frag1 == frag2)) continue;
                if (!(currentChr1 == chr1 && currentChr2 == chr2)) {
                    // Starting a new matrix
                    if (currentMatrix != null) {
                        currentMatrix.parsingComplete();
                        writeMatrixIndividualFile(currentMatrix, currentPairIndex);
                        syncWrittenMatrices.add(currentMatrixKey);
                        currentMatrix = null;
                        //System.gc();
                        break;
                        //System.out.println("Available memory: " + RuntimeUtils.getAvailableMemory());
                    }

                    // Start the next matrix
                    currentChr1 = chr1;
                    currentChr2 = chr2;
                    currentMatrixKey = currentChr1 + "_" + currentChr2;
                    currentMatrixName = localChromosomeHandler.getChromosomeFromIndex(chr1).getName() + "_" + localChromosomeHandler.getChromosomeFromIndex(chr2).getName();
                    currentPairIndex = chromosomePairIndexesReverse.get(currentMatrixName);

                    if (syncWrittenMatrices.contains(currentMatrixKey)) {
                        System.err.println("Error: the chromosome combination " + currentMatrixKey + " appears in multiple blocks");
                        if (outputFile != null) outputFile.deleteOnExit();
                        System.exit(58);
                    }
                    currentMatrix = new MatrixPP(currentChr1, currentChr2);
                }
                currentMatrix.incrementCount(bp1, bp2, frag1, frag2, pair.getScore(), localExpectedValueCalculations);
                pos1 = getGenomicPosition(chr1, bp1, localChromosomeHandler);
                pos2 = getGenomicPosition(chr2, bp2, localChromosomeHandler);
                wholeGenomeMatrix.incrementCount(pos1, pos2, pos1, pos2, pair.getScore(), localExpectedValueCalculations);


            }
        }

        if (fragmentCalculation != null && allowPositionsRandomization) {
            System.out.println(String.format("Randomization errors encountered: %d no map found, " +
                    "%d two different maps found", noMapFoundCount, mapDifferentCount));
        }

        if (currentMatrix != null) {
            currentMatrix.parsingComplete();
            writeMatrixIndividualFile(currentMatrix, currentPairIndex);
        }


        if (iter != null) iter.close();
        wholeGenomeMatrixParts.put(currentPairIndex, wholeGenomeMatrix);

    }
    private void writeBody(String inputFile, Map<Integer, Long> mndIndex) throws IOException {

        final int numPairsPerThread = (chromosomePairCounter - 1) / numCPUThreads;

        HashSet<String> writtenMatrices = new HashSet<>();
        Set<String> syncWrittenMatrices = Collections.synchronizedSet(writtenMatrices);
        final AtomicInteger chromosomePair = new AtomicInteger(1);

        ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);
        for (int l = 0; l < numCPUThreads; l++) {
            final int threadNum = l;

            Runnable worker = new Runnable() {
                @Override
                public void run() {
                    runIndividualMatrixCode(chromosomePair, inputFile, syncWrittenMatrices, threadNum, mndIndex);
                }
                //}
            };
            executor.execute(worker);
        }

        executor.shutdown();

        // Wait until all threads finish
        while (!executor.isTerminated()) {
        }


        int genomeLength = chromosomeHandler.getChromosomeFromIndex(0).getLength();  // <= whole genome in KB
        int binSize = genomeLength / 500;
        if (binSize == 0) binSize = 1;
        int nBinsX = genomeLength / binSize + 1;
        int nBlockColumns = nBinsX / BLOCK_SIZE + 1;
        MatrixPP wholeGenomeMatrix = new MatrixPP(0, 0, binSize, nBlockColumns);

        for (int i = 1; i < chromosomePairCounter; i++) {
            if (nonemptyChromosomePairs.containsKey(i)) {
                wholeGenomeMatrix.mergeMatrices(wholeGenomeMatrixParts.get(i));
            }
        }

        writeMatrixIndividualFile(wholeGenomeMatrix, 0);
        nonemptyChromosomePairs.put(0,1);

        long currentPosition = los.getWrittenCount();
        long nextMatrixPosition = 0;
        String currentMatrixKey = null;

        for (int i = 0; i < chromosomePairCounter; i++) {
            if (nonemptyChromosomePairs.containsKey(i)) {
                for (Map.Entry<Long, List<IndexEntry>> entry : chromosomePairBlockIndexes.get(i).entrySet()) {
                    updateIndexPositionsIndividualFile(entry.getKey(), entry.getValue(), i, currentPosition);
                }
                nextMatrixPosition = localMatrixPositions.get(i).position + currentPosition;
                currentMatrixKey = chromosomePairIndex1.get(i) + "_" + chromosomePairIndex2.get(i);
                matrixPositions.put(currentMatrixKey, new IndexEntry(nextMatrixPosition, localMatrixPositions.get(i).size));
                currentPosition += matrixSizes.get(i);

                //System.out.println(chromosomePairIndexes.get(i)+" "+matrixSizes.get(i));
            }
        }

        masterIndexPosition = currentPosition;
    }
    void runIndividualMatrixCode(AtomicInteger chromosomePair, String inputFile, Set<String> syncWrittenMatrices, int threadNum,
                                 Map<Integer,Long> mndIndex) {
        int i = chromosomePair.getAndIncrement();
        HashMap<Integer, String> localChromosomePairIndexes = new HashMap<Integer, String>(chromosomePairIndexes);
        HashMap<Integer, Integer> localChromosomePairIndex1 = new HashMap<Integer, Integer>(chromosomePairIndex1);
        HashMap<Integer, Integer> localChromosomePairIndex2 = new HashMap<Integer, Integer>(chromosomePairIndex2);
        HashSet<String> localIncludedChromosomes = null;
        if (includedChromosomes != null) {
            localIncludedChromosomes = new HashSet<String>(includedChromosomes);
        }
        ChromosomeHandler localChromosomeHandler = HiCFileTools.loadChromosomes(genomeId);
        int localChromosomePairCounter = chromosomePairCounter;
        Map<String, ExpectedValueCalculation> localExpectedValueCalculations = null;
        if (expectedVectorFile == null) {
            localExpectedValueCalculations = new LinkedHashMap<>();
            for (int bBinSize : bpBinSizes) {
                ExpectedValueCalculation calc = new ExpectedValueCalculation(localChromosomeHandler, bBinSize, null, NormalizationHandler.NONE);
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
                    ExpectedValueCalculation calc = new ExpectedValueCalculation(localChromosomeHandler, fBinSize, fragmentCountMap, NormalizationHandler.NONE);
                    String key = "FRAG_" + fBinSize;
                    localExpectedValueCalculations.put(key, calc);
                }

            }
        }
        while (i < localChromosomePairCounter) {
            Long mndIndexPosition = (long) 0;
            if (mndIndexFile!=null) {
                if (!mndIndex.containsKey(i)) {
                    System.out.println("No index position for " + localChromosomePairIndexes.get(i));
                    continue;
                } else {
                    mndIndexPosition = mndIndex.get(i);
                }
            }
            String chrInputFile;
            String chrInputFile2;
            if (inputFile.endsWith(".gz")) {
                chrInputFile = inputFile.replaceAll(".gz", "") + "_" + localChromosomePairIndexes.get(i) + ".gz";
                chrInputFile2 = inputFile.replaceAll(".gz", "") + "_" + localChromosomeHandler.getChromosomeFromIndex(
                        localChromosomePairIndex2.get(i)).getName() + "_" + localChromosomeHandler.getChromosomeFromIndex(
                        localChromosomePairIndex1.get(i)).getName() + ".gz";
            } else {
                chrInputFile = inputFile + "_" + localChromosomePairIndexes.get(i);
                chrInputFile2 = inputFile + "_" + localChromosomeHandler.getChromosomeFromIndex(
                        localChromosomePairIndex2.get(i)).getName() + "_" + localChromosomeHandler.getChromosomeFromIndex(
                        localChromosomePairIndex1.get(i)).getName();
            }
            try {
                writeBodySingleChromosomePair(inputFile, chrInputFile,
                        syncWrittenMatrices, localChromosomeHandler, localIncludedChromosomes,
                        localExpectedValueCalculations, mndIndexPosition);
                int chr1 = localChromosomePairIndex1.get(i);
                int chr2 = localChromosomePairIndex2.get(i);
                if (localIncludedChromosomes != null) {
                    String c1Name = localChromosomeHandler.getChromosomeFromIndex(chr1).getName();
                    String c2Name = localChromosomeHandler.getChromosomeFromIndex(chr2).getName();
                    if (localIncludedChromosomes.contains(c1Name) || localIncludedChromosomes.contains(c2Name)) {
                        nonemptyChromosomePairs.put(i, 1);
                    }
                } else {
                    nonemptyChromosomePairs.put(i, 1);
                }
            } catch (Exception e) {
                try {
                    writeBodySingleChromosomePair(inputFile, chrInputFile2, syncWrittenMatrices, localChromosomeHandler,
                            localIncludedChromosomes, localExpectedValueCalculations, mndIndexPosition);
                    int chr1 = localChromosomePairIndex1.get(i);
                    int chr2 = localChromosomePairIndex2.get(i);
                    if (localIncludedChromosomes != null) {
                        String c1Name = localChromosomeHandler.getChromosomeFromIndex(chr1).getName();
                        String c2Name = localChromosomeHandler.getChromosomeFromIndex(chr2).getName();
                        if (localIncludedChromosomes.contains(c1Name) || localIncludedChromosomes.contains(c2Name)) {
                            nonemptyChromosomePairs.put(i, 1);
                        }
                    } else {
                        nonemptyChromosomePairs.put(i, 1);
                    }
                } catch (Exception e2) {
                    System.err.println("Unable to open " + inputFile + "_" + localChromosomePairIndexes.get(i));
                    //System.exit(70);
                }
            }
            i = chromosomePair.getAndIncrement();
        }
        allLocalExpectedValueCalculations.put(threadNum, localExpectedValueCalculations);
    }

    void updateIndexPositionsIndividualFile(Long blockIndexPosition, List<IndexEntry> blockIndex, int chromosomePairIndex, long currentPosition) throws IOException {

        // Temporarily close output stream.  Remember position
        //long losPos = losContainer.get(chromosomePairIndex).getWrittenCount();
        //losContainer.get(chromosomePairIndex).close();

        RandomAccessFile raf = null;
        try {
            raf = new RandomAccessFile(outputFile+"_"+chromosomePairIndexes.get(chromosomePairIndex), "rw");

            // Block indices
            long pos = blockIndexPosition;
            raf.getChannel().position(pos);

            // Write as little endian
            BufferedByteWriter buffer = new BufferedByteWriter();
            for (IndexEntry aBlockIndex : blockIndex) {
                buffer.putInt(aBlockIndex.id);
                buffer.putLong((aBlockIndex.position + currentPosition));
                buffer.putInt(aBlockIndex.size);
            }
            raf.write(buffer.getBytes());

        } finally {

            if (raf != null) raf.close();

            // Restore
            //FileOutputStream fos = new FileOutputStream(outputFile+"_"+chromosomePairIndexes.get(chromosomePairIndex), true);
            //fos.getChannel().position(losPos);
            //losContainer.put(chromosomePairIndex, new LittleEndianOutputStream(new BufferedOutputStream(fos, HiCGlobals.bufferSize)));
            //losContainer.get(chromosomePairIndex).setWrittenCount(losPos);

        }
    }

    @Override
    protected void updateMasterIndex() throws IOException {
        RandomAccessFile raf = null;
        try {
            raf = new RandomAccessFile(outputFile+"_header", "rw");

            // Master index
            raf.getChannel().position(masterIndexPositionPosition);
            BufferedByteWriter buffer = new BufferedByteWriter();
            buffer.putLong(masterIndexPosition);
            raf.write(buffer.getBytes());

        } finally {
            if (raf != null) raf.close();
        }
    }


    /* todo
    private void updateNormVectorIndexInfo() throws IOException {
        RandomAccessFile raf = null;
        try {
            raf = new RandomAccessFile(outputFile, "rw");

            // NVI index
            raf.getChannel().position(normVectorIndexPosition);
            BufferedByteWriter buffer = new BufferedByteWriter();
            generateZeroPaddedString
            buffer.putNullTerminatedString(normVectorIndex);
            raf.write(buffer.getBytes());


            // NVI length
            raf.getChannel().position(normVectorLengthPosition);
            buffer = new BufferedByteWriter();
            buffer.putNullTerminatedString(normVectorLength);
            raf.write(buffer.getBytes());

        } finally {
            if (raf != null) raf.close();
        }
    }
    */

    @Override
    protected void writeFooter() throws IOException {

        // Index
        BufferedByteWriter buffer = new BufferedByteWriter();
        buffer.putInt(matrixPositions.size());
        for (Map.Entry<String, IndexEntry> entry : matrixPositions.entrySet()) {
            buffer.putNullTerminatedString(entry.getKey());
            buffer.putLong(entry.getValue().position);
            buffer.putInt(entry.getValue().size);
        }


        // Vectors  (Expected values,  other).
        /***  NEVA ***/
        if (expectedVectorFile == null) {
            buffer.putInt(expectedValueCalculations.size());
            for (Map.Entry<String, ExpectedValueCalculation> entry : expectedValueCalculations.entrySet()) {
                ExpectedValueCalculation ev = entry.getValue();

                ev.computeDensity();

                int binSize = ev.getGridSize();
                HiC.Unit unit = ev.isFrag ? HiC.Unit.FRAG : HiC.Unit.BP;

                buffer.putNullTerminatedString(unit.toString());
                buffer.putInt(binSize);

                // The density values
                double[] expectedValues = ev.getDensityAvg();
                buffer.putInt(expectedValues.length);
                for (double expectedValue : expectedValues) {
                    buffer.putDouble(expectedValue);
                }

                // Map of chromosome index -> normalization factor
                Map<Integer, Double> normalizationFactors = ev.getChrScaleFactors();
                buffer.putInt(normalizationFactors.size());
                for (Map.Entry<Integer, Double> normFactor : normalizationFactors.entrySet()) {
                    buffer.putInt(normFactor.getKey());
                    buffer.putDouble(normFactor.getValue());
                    //System.out.println(normFactor.getKey() + "  " + normFactor.getValue());
                }
            }
        }
        else {
            // read in expected vector file. to get # of resolutions, might have to read twice.

            int count=0;
            try (Reader reader = new FileReader(expectedVectorFile);
                 BufferedReader bufferedReader = new BufferedReader(reader)) {

                String line;
                while ((line = bufferedReader.readLine()) != null) {
                    if (line.startsWith("fixedStep"))
                        count++;
                    if (line.startsWith("variableStep")) {
                        System.err.println("Expected vector file must be in wiggle fixedStep format");
                        System.exit(19);
                    }
                }
            }
            buffer.putInt(count);
            try (Reader reader = new FileReader(expectedVectorFile);
                 BufferedReader bufferedReader = new BufferedReader(reader)) {

                String line;
                while ((line = bufferedReader.readLine()) != null) {
                    if (line.startsWith("fixedStep")) {
                        String[] words = line.split("\\s+");
                        for (String str:words){
                            if (str.contains("chrom")){
                                String[] chrs = str.split("=");

                            }
                        }
                    }
                    // parse linef ixedStep  chrom=chrN
                    //start=position  step=stepInterval
                }
            }
        }

        byte[] bytes = buffer.getBytes();
        losFooter.writeInt(bytes.length);
        losFooter.write(bytes);
        losFooter.close();
    }

    private void writeMatrixIndividualFile(MatrixPP matrix, int chromosomePairIndex) throws IOException {
        Deflater localCompressor = new Deflater();
        localCompressor.setLevel(Deflater.DEFAULT_COMPRESSION);

        LittleEndianOutputStream localLos = new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(outputFile+"_"+chromosomePairIndexes.get(chromosomePairIndex)), HiCGlobals.bufferSize));
        long position = localLos.getWrittenCount();

        localLos.writeInt(matrix.getChr1Idx());
        localLos.writeInt(matrix.getChr2Idx());

        int numResolutions = 0;

        for (MatrixZoomDataPP zd : matrix.getZoomData()) {
            if (zd != null) {
                numResolutions++;
            }
        }
        localLos.writeInt(numResolutions);

        for (MatrixZoomDataPP zd : matrix.getZoomData()) {
            if (zd != null)
                writeZoomHeaderIndividualFile(zd, localLos);
        }

        int size = (int) (localLos.getWrittenCount() - position);
        localMatrixPositions.put(chromosomePairIndex, new IndexEntry(position, size));

        final Map<Long, List<IndexEntry>> localBlockIndexes = new ConcurrentHashMap<>();

        for (MatrixZoomDataPP zd : matrix.getZoomData()) {
            if (zd != null) {
                List<IndexEntry> blockIndex = zd.mergeAndWriteBlocks(localLos, localCompressor);
                localBlockIndexes.put(zd.blockIndexPosition, blockIndex);
            }
        }

        chromosomePairBlockIndexes.put(chromosomePairIndex, localBlockIndexes);
        size = (int) (localLos.getWrittenCount() - position);
        matrixSizes.put(chromosomePairIndex, size);
        localLos.close();

        System.out.print(".");

    }

    private void writeZoomHeaderIndividualFile(MatrixZoomDataPP zd, LittleEndianOutputStream localLos) throws IOException {

        int numberOfBlocks = zd.blockNumbers.size();
        localLos.writeString(zd.getUnit().toString());  // Unit
        localLos.writeInt(zd.getZoom());     // zoom index,  lowest res is zero
        localLos.writeFloat((float) zd.getSum());      // sum
        localLos.writeFloat((float) zd.getOccupiedCellCount());
        localLos.writeFloat((float) zd.getPercent5());
        localLos.writeFloat((float) zd.getPercent95());
        localLos.writeInt(zd.getBinSize());
        localLos.writeInt(zd.getBlockBinCount());
        localLos.writeInt(zd.getBlockColumnCount());
        localLos.writeInt(numberOfBlocks);

        zd.blockIndexPosition = localLos.getWrittenCount();

        // Placeholder for block index
        for (int i = 0; i < numberOfBlocks; i++) {
            localLos.writeInt(0);
            localLos.writeLong(0L);
            localLos.writeInt(0);
        }

    }
}
