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

package juicebox.tools.clt.juicer;

import com.google.common.primitives.Ints;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.data.basics.Chromosome;
import juicebox.data.basics.ListOfFloatArrays;
import juicebox.tools.clt.CommandLineParserForJuicer;
import juicebox.tools.clt.JuicerCLT;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class Generate1DTracks extends JuicerCLT {
    private File outputDirectory;
    private Dataset ds;
    private ChromosomeHandler chromosomeHandler;
    private Set<String> tracksToGenerate = new HashSet<>();
    private Set<String> includedChromosomes;
    private int chromosomePairCounter = 0;
    private Map<Integer, Chromosome[]> chromosomePairs = new ConcurrentHashMap<>();
    private Map<String, PrintWriter> outputFiles = new ConcurrentHashMap<>();

    //defaults
    private double minDist = 100;
    private double maxDist = 1000;
    private double window = 2000;
    private int[] resolutions = new int[]{1000};
    private boolean includeDistCorrection = false;

    public Generate1DTracks() {
        super("generateTracks <trackTypes> <hicFile(s)> <SaveFolder>");
        HiCGlobals.useCache = false;
    }

    public static String getBasicUsage() {
        return "generateTracks <trackTypes> <hicFile(s)> <SaveFolder>";
    }

    public void setIncludedChromosomes(List<String> includedChromosomes) {
        this.includedChromosomes = Collections.synchronizedSet(new HashSet<>());
        if (includedChromosomes != null && includedChromosomes.size() > 0) {
            for (String name : includedChromosomes) {
                this.includedChromosomes.add(chromosomeHandler.cleanUpName(name));
            }
        } else {
            for (Chromosome chr : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
                this.includedChromosomes.add(chr.getName());
            }
        }
    }

    @Override
    protected void readJuicerArguments(String[] args, CommandLineParserForJuicer juicerParser) {
        if (args.length != 4) {
            printUsageAndExit();
        }

        outputDirectory = HiCFileTools.createValidDirectory(args[3]);

        List<String> inputTrackTypes = Arrays.asList(args[1].split(","));
        for (int i = 0; i < inputTrackTypes.size(); i++) {
            if (!isAllowedTrack(inputTrackTypes.get(i))) {
                System.out.println(inputTrackTypes.get(i));
                System.out.println("track type must be one of: interBias, gradient, insulation, insulationEdge");
                System.exit(0);
            } else {
                tracksToGenerate.add(inputTrackTypes.get(i));
                tracksToGenerate.addAll(addNecessaryTracks(inputTrackTypes.get(i)));
            }

        }

        List<String> summedHiCFiles = Arrays.asList(args[2].split("\\+"));
        ds = HiCFileTools.extractDatasetForCLT(summedHiCFiles, true);
        chromosomeHandler = ds.getChromosomeHandler();

        setIncludedChromosomes(givenChromosomes);

        NormalizationType preferredNorm = juicerParser.getNormalizationTypeOption(ds.getNormalizationHandler());
        if (preferredNorm != null)
            norm = preferredNorm;

        double potentialMinDist = juicerParser.getAPAMinVal();
        if (potentialMinDist > -1)
            minDist = potentialMinDist;

        double potentialMaxDist = juicerParser.getAPAMaxVal();
        if (potentialMaxDist > -1)
            maxDist = potentialMaxDist;

        int potentialWindow = juicerParser.getAPAWindowSizeOption();
        if (potentialWindow > 0)
            window = potentialWindow;

        List<String> possibleResolutions = juicerParser.getMultipleResolutionOptions();
        if (possibleResolutions != null) {
            List<Integer> intResolutions = new ArrayList<>();
            for (String res : possibleResolutions) {
                intResolutions.add(Integer.parseInt(res));
            }
            resolutions = Ints.toArray(intResolutions);
        }

        numCPUThreads = juicerParser.getNumThreads();

    }

    @Override
    public void run() {
        for (final int resolution : HiCFileTools.filterResolutions(ds.getBpZooms(), resolutions)) {
            HiCZoom zoom = new HiCZoom(HiC.Unit.BP, resolution);
            initializeOutputFiles(resolution);
            if (tracksToGenerate.contains("interBias")) {
                matricesToProcess(true);
                runGlobal(zoom, resolution);
            } else {
                matricesToProcess(false);
                runLocal(zoom, resolution);
            }
            closeOutputFiles();
        }
    }

    private void runGlobal(HiCZoom zoom, int resolution) {
        Map<String, List<ListOfFloatArrays>> interBiasVectors = new ConcurrentHashMap<>();
        for (int chrPair = 1; chrPair < chromosomePairCounter; chrPair++) {
            Chromosome chr1 = chromosomePairs.get(chrPair)[0];
            if (!interBiasVectors.containsKey(chr1.getName()) && includedChromosomes.contains(chr1.getName())) {
                List<ListOfFloatArrays> chromosomeList = Collections.synchronizedList(new ArrayList<>());
                for (int i = 0; i<2; i++) {
                    chromosomeList.add(new ListOfFloatArrays(chr1.getLength()/resolution + 1));
                }
                interBiasVectors.put(chr1.getName(), chromosomeList);
            }
            Chromosome chr2 = chromosomePairs.get(chrPair)[1];
            if (!interBiasVectors.containsKey(chr2.getName()) && includedChromosomes.contains(chr2.getName())) {
                List<ListOfFloatArrays> chromosomeList = Collections.synchronizedList(new ArrayList<>());
                for (int i = 0; i<2; i++) {
                    chromosomeList.add(new ListOfFloatArrays(chr2.getLength()/resolution + 1));
                }
                interBiasVectors.put(chr2.getName(), chromosomeList);
            }

            boolean intrachromosomal;
            Map<String, ListOfFloatArrays> intrachromTracks, shiftIntrachromTracks;
            if (chr1.equals(chr2)) {
                intrachromTracks = new ConcurrentHashMap<>();
                shiftIntrachromTracks = new ConcurrentHashMap<>();
                for (String trackType : tracksToGenerate) {
                    if (isIntrachromosomalType(trackType)) {
                        intrachromTracks.put(trackType, new ListOfFloatArrays(chr1.getLength()/resolution +1));
                        shiftIntrachromTracks.put(trackType, new ListOfFloatArrays(chr1.getLength()/resolution +1));
                    }
                }
                intrachromosomal = true;
            } else {
                intrachromosomal = false;
                intrachromTracks = null;
                shiftIntrachromTracks = null;
            }

            Set<String> nonWindowLimitedOperations = new HashSet<>();
            Set<String> windowLimitedOperations = new HashSet<>();
            for (String trackType : tracksToGenerate) {
                if (!intrachromosomal && !isIntrachromosomalType(trackType)) {
                    nonWindowLimitedOperations.add(trackType);
                }
                if (intrachromosomal) {
                    if (!isWindowLimitedType(trackType)) {
                        nonWindowLimitedOperations.add(trackType);
                    } else {
                        windowLimitedOperations.add(trackType);
                    }
                }
            }

            MatrixZoomData zd;
            zd = HiCFileTools.getMatrixZoomData(ds, chr1, chr2, zoom);
            List<Integer> blockNumbers = zd.getBlockNumbers();

            final AtomicInteger currentBlock = new AtomicInteger(0);
            ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);

            for (int i = 0; i < numCPUThreads; i++) {
                Runnable worker = new Runnable() {
                    @Override
                    public void run() {
                        int blockIndex = currentBlock.getAndIncrement();
                        while (blockIndex < blockNumbers.size()) {
                            int blockToGet = blockNumbers.get(blockIndex);
                            try {
                                List<ContactRecord> blockRecords = zd.getContactRecordsForBlock(blockToGet, norm);
                                incrementCount(chr1, chr2, intrachromosomal, intrachromTracks, shiftIntrachromTracks, interBiasVectors, blockRecords, nonWindowLimitedOperations, windowLimitedOperations);
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                            blockIndex = currentBlock.getAndIncrement();
                        }
                    }
                };
                executor.execute(worker);
            }
            executor.shutdown();

            // Wait until all threads finish
            while (!executor.isTerminated()) {
            }

            postProcessIntrachromosomalVectors(chr1, resolution, intrachromosomal, intrachromTracks, shiftIntrachromTracks);

        }

        postProcessInterchromosomalVectors(resolution, interBiasVectors);

    }

    private void runLocal(HiCZoom zoom, int resolution) {
        Map<String, List<ListOfFloatArrays>> interBiasVectors = null;
        for (int chrPair = 1; chrPair < chromosomePairCounter; chrPair++) {
            Chromosome chr1 = chromosomePairs.get(chrPair)[0];
            Chromosome chr2 = chromosomePairs.get(chrPair)[1];
            MatrixZoomData zd;
            zd = HiCFileTools.getMatrixZoomData(ds, chr1, chr2, zoom);
            boolean intrachromosomal;
            Map<String, ListOfFloatArrays> intrachromTracks, shiftIntrachromTracks;
            if (chr1.equals(chr2)) {
                intrachromTracks = new ConcurrentHashMap<>();
                shiftIntrachromTracks = new ConcurrentHashMap<>();
                for (String trackType : tracksToGenerate) {
                    if (isIntrachromosomalType(trackType)) {
                        intrachromTracks.put(trackType, new ListOfFloatArrays(chr1.getLength()/resolution +1));
                        shiftIntrachromTracks.put(trackType, new ListOfFloatArrays(chr1.getLength()/resolution +1));
                    }
                }
                intrachromosomal = true;
            } else {
                intrachromosomal = false;
                intrachromTracks = null;
                shiftIntrachromTracks = null;
            }

            Set<String> nonWindowLimitedOperations = new HashSet<>();
            Set<String> windowLimitedOperations = new HashSet<>();
            for (String trackType : tracksToGenerate) {
                if (!intrachromosomal && !isIntrachromosomalType(trackType)) {
                    nonWindowLimitedOperations.add(trackType);
                }
                if (intrachromosomal) {
                    if (!isWindowLimitedType(trackType)) {
                        nonWindowLimitedOperations.add(trackType);
                    } else {
                        windowLimitedOperations.add(trackType);
                    }
                }
            }

            List<Integer> blockNumbers = zd.getBlockNumbers();
            System.out.println(blockNumbers.size());

            final AtomicInteger currentBlock = new AtomicInteger(0);
            ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);
            System.out.println("Starting data load for: " + chr1.getName() + "-" + chr2.getName());
            for (int i = 0; i < numCPUThreads; i++) {
                Runnable worker = new Runnable() {
                    @Override
                    public void run() {
                        Map<String, ListOfFloatArrays> threadIntrachromTracks = new ConcurrentHashMap<>();
                        Map<String, ListOfFloatArrays> threadShiftIntrachromTracks = new ConcurrentHashMap<>();
                        for (String trackType : tracksToGenerate) {
                            if (isIntrachromosomalType(trackType) && !isPostProcessedType(trackType)) {
                                threadIntrachromTracks.put(trackType, new ListOfFloatArrays(chr1.getLength()/resolution +1));
                                threadShiftIntrachromTracks.put(trackType, new ListOfFloatArrays(chr1.getLength()/resolution +1));
                            }
                        }
                        int blockIndex = currentBlock.getAndIncrement();
                        while (blockIndex < blockNumbers.size()) {
                            int blockToGet = blockNumbers.get(blockIndex);
                            int[] blockDists = zd.getBlockMinMaxDist(blockToGet);
                            if ((blockDists[0] >= minDist && blockDists[0] <= maxDist+1) || (blockDists[1] >= minDist && blockDists[1] <= maxDist+1) || (blockDists[0] <= minDist && blockDists[1] >= maxDist+1)) {
                                try {
                                    List<ContactRecord> blockRecords = zd.getContactRecordsForBlock(blockToGet, norm);
                                    incrementCount(chr1, chr2, intrachromosomal, threadIntrachromTracks, threadShiftIntrachromTracks, interBiasVectors, blockRecords, nonWindowLimitedOperations, windowLimitedOperations);
                                } catch (Exception e) {
                                    e.printStackTrace();
                                }
                            }
                            blockIndex = currentBlock.getAndIncrement();
                        }
                        for (String trackType : tracksToGenerate) {
                            if (isIntrachromosomalType(trackType) && !isPostProcessedType(trackType)) {
                                synchronized(intrachromTracks.get(trackType)) {
                                    intrachromTracks.get(trackType).addValuesFrom(threadIntrachromTracks.get(trackType));
                                }
                                synchronized(shiftIntrachromTracks.get(trackType)) {
                                    shiftIntrachromTracks.get(trackType).addValuesFrom(threadShiftIntrachromTracks.get(trackType));
                                }
                            }
                        }
                    }
                };
                executor.execute(worker);
            }
            executor.shutdown();

            // Wait until all threads finish
            while (!executor.isTerminated()) {
            }
            System.out.println("Finished sums for: " + chr1.getName() + "-" + chr2.getName());

            postProcessIntrachromosomalVectors(chr1, resolution, intrachromosomal, intrachromTracks, shiftIntrachromTracks);
            System.out.println("Finished postprocessing for: " + chr1.getName() + "-" + chr2.getName());

        }
    }

    private boolean isAllowedTrack(String input) {
        return input.equals("interBias") || input.equals("upGradient") || input.equals("downGradient") || input.equals("insulation") || input.equals("insulationEdge");
    }

    private boolean isIntrachromosomalType(String input) {
        return input.equals("upGradient") || input.equals("downGradient") || input.equals("insulation") || input.equals("insulationEdge");
    }

    private boolean isWindowLimitedType(String input) {
        return input.equals("upGradient") || input.equals("downGradient") || input.equals("insulation") || input.equals("insulationEdge");
    }

    private boolean isPostProcessedType(String input) {
        return input.equals("insulationEdge");
    }

    private Set<String> addNecessaryTracks(String input) {
        Set<String> necessaryTracks = new HashSet<>();
        necessaryTracks.add(input);
        if (input.equals("insulationEdge")) {
            necessaryTracks.add("upGradient");
            necessaryTracks.add("downGradient");
            necessaryTracks.add("insulation");
        }

        return necessaryTracks;
    }

    private void matricesToProcess(boolean includeInter) {
        int pairCounter = 1;
        for (Chromosome chr1 : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            for (Chromosome chr2 : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
                Chromosome[] chromosomePair = {chr1, chr2};
                if (chr1.equals(chr2) || (includeInter && chr1.getIndex() < chr2.getIndex())) {
                    if (includedChromosomes.contains(chr1.getName()) || includedChromosomes.contains(chr2.getName())) {
                        chromosomePairs.put(pairCounter, chromosomePair);
                        pairCounter++;
                    }
                }

            }
        }
        this.chromosomePairCounter = pairCounter;
    }

    private void initializeOutputFiles(int res) {
        for (String trackType : tracksToGenerate) {
            try {
                outputFiles.put(trackType, new PrintWriter(new FileOutputStream(new File(outputDirectory, trackType + "_" + res + ".bdg").getAbsolutePath())));
                if (trackType.equals("interBias")) {
                    outputFiles.put("intraCoverage", new PrintWriter(new FileOutputStream(new File(outputDirectory, "intraCoverage_" + res + ".bdg").getAbsolutePath())));
                    outputFiles.put("interCoverage", new PrintWriter(new FileOutputStream(new File(outputDirectory, "interCoverage_" + res + ".bdg").getAbsolutePath())));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void closeOutputFiles() {
        for (PrintWriter pw : outputFiles.values()) {
            pw.close();
        }
    }

    private void incrementCount(Chromosome chr1, Chromosome chr2, boolean intrachromosomal,
                                Map<String, ListOfFloatArrays> intrachromVectors, Map<String, ListOfFloatArrays> shiftIntrachromVectors, Map<String, List<ListOfFloatArrays>> interBiasVectors,
                                List<ContactRecord> contactRecords, Set<String> nonWindowLimitedOperations, Set<String> windowLimitedOperations) {

        for (ContactRecord rec : contactRecords) {
            long binX = rec.getBinX();
            long binY = rec.getBinY();
            long dist = Math.abs(binX-binY);
            float val = rec.getCounts();

            for (String operation : nonWindowLimitedOperations) {
                process(chr1, chr2, intrachromosomal, intrachromVectors, shiftIntrachromVectors, interBiasVectors, binX, binY, val, dist, operation);
            }


            if (dist >= minDist && dist < maxDist + 1) {
                for (String operation : windowLimitedOperations) {
                    process(chr1, chr2, intrachromosomal, intrachromVectors, shiftIntrachromVectors, interBiasVectors, binX, binY, val, dist, operation);
                }
            }
        }
    }

    private void process(Chromosome chr1, Chromosome chr2, boolean intrachromosomal,
                         Map<String, ListOfFloatArrays> intrachromVectors, Map<String, ListOfFloatArrays> shiftIntrachromVectors, Map<String, List<ListOfFloatArrays>> interBiasVectors,
                         long binX, long binY, float val, long dist, String operation) {

        if (operation.equals("interBias")) {
            computeInterBias(chr1, chr2, intrachromosomal, interBiasVectors, binX, binY, val);
        }
        if (operation.equals("upGradient")) {
            computeUpGradient(intrachromVectors, shiftIntrachromVectors, binX, binY, val, dist);
        }
        if (operation.equals("downGradient")) {
            computeDownGradient(intrachromVectors, shiftIntrachromVectors, binX, binY, val, dist);
        }
        if (operation.equals("insulation")) {
            computeInsulation(intrachromVectors, shiftIntrachromVectors, binX, binY, val, dist);
        }
    }

    private void computeInterBias(Chromosome chr1, Chromosome chr2, boolean intrachromosomal, Map<String, List<ListOfFloatArrays>> interBiasVectors,
                                  long binX, long binY, float val) {

        if (includedChromosomes.contains(chr1.getName())) {
            synchronized(interBiasVectors.get(chr1.getName())) {
                if (intrachromosomal) {
                    synchronized(interBiasVectors.get(chr1.getName()).get(0)) {
                        interBiasVectors.get(chr1.getName()).get(0).addTo(binX, val);
                        interBiasVectors.get(chr1.getName()).get(0).addTo(binY, val);
                    }
                } else {
                    synchronized(interBiasVectors.get(chr1.getName()).get(1)) {
                        interBiasVectors.get(chr1.getName()).get(1).addTo(binX, val);
                    }
                }
            }
        }
        if (!intrachromosomal && includedChromosomes.contains(chr2.getName())) {
            synchronized (interBiasVectors.get(chr2.getName())) {
                synchronized(interBiasVectors.get(chr2.getName()).get(1)) {
                    interBiasVectors.get(chr2.getName()).get(1).addTo(binY, val);
                }
            }

        }
    }

    private void computeUpGradient(Map<String, ListOfFloatArrays> intrachromVectors, Map<String, ListOfFloatArrays> shiftIntrachromVectors,
                                   long binX, long binY, float val, long dist) {

        if (dist < maxDist) { intrachromVectors.get("upGradient").addTo(binY, val);}
        if (dist > minDist) { shiftIntrachromVectors.get("upGradient").addTo(binY, val);}


    }

    private void computeDownGradient(Map<String, ListOfFloatArrays> intrachromVectors, Map<String, ListOfFloatArrays> shiftIntrachromVectors,
                                     long binX, long binY, float val, long dist) {

        if (dist < maxDist) { intrachromVectors.get("downGradient").addTo(binX, val);}
        if (dist > minDist) { shiftIntrachromVectors.get("downGradient").addTo(binX, val);}

    }

    private void computeInsulation(Map<String, ListOfFloatArrays> intrachromVectors, Map<String, ListOfFloatArrays> shiftIntrachromVectors,
                                   long binX, long binY, float val, long dist) {
        long bound1 = Math.min(binX, binY) + 1;
        long bound2 = Math.max(binX, binY) - 1;
        for (long i = bound1; i <= bound2; i++) {

            intrachromVectors.get("insulation").addTo(i, val);

        }
    }

    private float computeDistCorrection(long dist) {
        float correction = 0;
        if (includeDistCorrection) {

        }
        return correction;
    }

    private void postProcessIntrachromosomalVectors(Chromosome chr, int res, boolean intrachromosomal,
                                                    Map<String, ListOfFloatArrays> intrachromVectors, Map<String, ListOfFloatArrays> shiftIntrachromVectors) {
        Set<String> operations = new HashSet<>();
        for (String trackType : tracksToGenerate) {
            if (intrachromosomal && isIntrachromosomalType(trackType)) {
                operations.add(trackType);
            }
        }
        for (String operation : operations) {
            if (operation.equals("upGradient")) {
                postProcessUpGradient(chr, res, intrachromVectors, shiftIntrachromVectors);
            }
            if (operation.equals("downGradient")) {
                postProcessDownGradient(chr, res, intrachromVectors, shiftIntrachromVectors);
            }
            if (operation.equals("insulation")) {
                postProcessInsulation(chr, res, intrachromVectors);
            }
            if (operation.equals("insulationEdge")) {
                postProcessInsulationEdge(chr, res, intrachromVectors);
            }
        }
    }

    private void postProcessUpGradient(Chromosome chr, int res, Map<String, ListOfFloatArrays> intrachromVectors, Map<String, ListOfFloatArrays> shiftIntrachromVectors) {
        ListOfFloatArrays rawSums = intrachromVectors.get("upGradient");
        float rawSumAverage = rawSums.getAverage();
        ListOfFloatArrays shiftRawSums = shiftIntrachromVectors.get("upGradient");
        float shiftRawSumAverage = shiftRawSums.getAverage();

        float previousPPgrad = 1;
        PrintWriter pw = outputFiles.get("upGradient");
        String chrName = chr.getName();
        long arrayLength = rawSums.getLength();

        List<Float> ratios = new ArrayList<>();

        for (long i = (long) maxDist + 1; i < arrayLength; i++) {
            if (shiftRawSums.get(i) > 0.1 * shiftRawSumAverage && rawSums.get(i-1) > 0.1 * rawSumAverage) {
                ratios.add(shiftRawSums.get(i) / rawSums.get(i - 1));
            }
        }

        Collections.sort(ratios);
        float correctionFactor = 1 - ratios.get(ratios.size()/2);
        System.out.println(correctionFactor);

        for (long i = 0; i <= (long) maxDist; i++) {
            pw.println(chrName + "\t" + i*res + "\t" + (i+1)*res + "\t" + 1);
        }
        for (long i = (long) maxDist + 1; i < arrayLength; i++) {
            float ppGrad;
            if (shiftRawSums.get(i) <= 0.1 * shiftRawSumAverage || rawSums.get(i-1) <= 0.1 * rawSumAverage) {
                //ppGrad = previousPPgrad;
                ppGrad = 0;
            } else {
                //ppGrad = 1 / ((rawSums.get(i - 1) / shiftRawSums.get(i) / previousPPgrad) - correctionFactor);
                ppGrad = shiftRawSums.get(i) / rawSums.get(i - 1);
            }
            pw.println(chrName + "\t" + i*res + "\t" + (i+1)*res + "\t" + ppGrad);
            previousPPgrad = ppGrad;
        }
    }

    private void postProcessDownGradient(Chromosome chr, int res, Map<String, ListOfFloatArrays> intrachromVectors, Map<String, ListOfFloatArrays> shiftIntrachromVectors) {
        ListOfFloatArrays rawSums = intrachromVectors.get("downGradient");
        float rawSumAverage = rawSums.getAverage();
        ListOfFloatArrays shiftRawSums = shiftIntrachromVectors.get("downGradient");
        float shiftRawSumAverage = shiftRawSums.getAverage();
        float previousPPgrad = 1;
        PrintWriter pw = outputFiles.get("downGradient");
        String chrName = chr.getName();
        long arrayLength = rawSums.getLength();

        List<Float> ratios = new ArrayList<>();

        for (long i = arrayLength - (long) maxDist - 2; i >= 0; i--) {
            if (shiftRawSums.get(i) > 0.1 * shiftRawSumAverage && rawSums.get(i-1) > 0.1 * rawSumAverage) {
                ratios.add(shiftRawSums.get(i) / rawSums.get(i + 1));
            }
        }

        Collections.sort(ratios);
        float correctionFactor = 1 - ratios.get(ratios.size()/2);
        System.out.println(correctionFactor);


        for (long i = arrayLength - 1; i >= arrayLength - (long) maxDist - 1; i--) {
            pw.println(chrName + "\t" + i*res + "\t" + (i+1)*res + "\t" + 1);
        }
        for (long i = arrayLength - (long) maxDist - 2; i >= 0; i--) {
            float ppGrad;
            if (shiftRawSums.get(i) <= 0.1 * shiftRawSumAverage || rawSums.get(i+1) <= 0.1 * rawSumAverage) {
                ppGrad = previousPPgrad;
            } else {
                ppGrad = 1 / ((rawSums.get(i+1) / shiftRawSums.get(i) / previousPPgrad) - correctionFactor);
            }
            pw.println(chrName + "\t" + i*res + "\t" + (i+1)*res + "\t" + ppGrad);
            previousPPgrad = ppGrad;
        }
    }

    private void postProcessInsulation(Chromosome chr, int res, Map<String, ListOfFloatArrays> intrachromVectors) {
        ListOfFloatArrays rawSums = intrachromVectors.get("insulation");
        float insulationAverage = rawSums.getAverage();
        PrintWriter pw = outputFiles.get("insulation");
        String chrName = chr.getName();
        long arrayLength = rawSums.getLength();

        for (long i = 0; i < arrayLength; i++) {
            pw.println(chrName + "\t" + i*res + "\t" + (i+1)*res + "\t" + rawSums.get(i)/insulationAverage);
        }
    }

    private void postProcessInsulationEdge(Chromosome chr, int res, Map<String, ListOfFloatArrays> intrachromVectors) {
        ListOfFloatArrays rawSumsI = intrachromVectors.get("insulation");
        float Icorrection = rawSumsI.getAverage() / 100;
        ListOfFloatArrays rawSumsU = intrachromVectors.get("upGradient");
        float Ucorrection = rawSumsU.getAverage() / 100;
        ListOfFloatArrays rawSumsD = intrachromVectors.get("downGradient");
        float Dcorrection = rawSumsD.getAverage() / 100;
        long arrayLength = rawSumsI.getLength();
        ListOfFloatArrays ppSums = new ListOfFloatArrays(arrayLength);


        for (long i = 0; i <= (long) maxDist; i++) {
            ppSums.set(i, 1);
        }
        for (long i = (long) maxDist+1; i <= arrayLength - (long) maxDist - 2; i++) {
            ppSums.set(i, (rawSumsI.get(i)+Icorrection)/(rawSumsU.get(i)+rawSumsD.get(i)+Ucorrection+Dcorrection));
        }
        for (long i = arrayLength - (long) maxDist - 1; i < arrayLength; i++) {
            ppSums.set(i, 1);
        }

        float insulationEdgeAverage = ppSums.getAverage();

        PrintWriter pw = outputFiles.get("insulationEdge");
        String chrName = chr.getName();

        for (long i = 0; i < arrayLength; i++) {
            pw.println(chrName + "\t" + i*res + "\t" + (i+1)*res + "\t" + ppSums.get(i)/insulationEdgeAverage);
        }

    }

    private void postProcessInterchromosomalVectors(int res, Map<String, List<ListOfFloatArrays>> interBiasVectors) {
        PrintWriter pwIntra = outputFiles.get("intraCoverage");
        PrintWriter pwInter = outputFiles.get("interCoverage");
        PrintWriter pwBias = outputFiles.get("interBias");

        for (String chrName : interBiasVectors.keySet()) {
            ListOfFloatArrays intraCoverage = interBiasVectors.get(chrName).get(0);
            float intraAverage = intraCoverage.getAverage();
            float intraCorrection = intraAverage / 100;
            ListOfFloatArrays interCoverage = interBiasVectors.get(chrName).get(1);
            float interAverage = interCoverage.getAverage();
            float interCorrection = interAverage / 100;
            long arrayLength = intraCoverage.getLength();


            for (long i = 0; i < arrayLength; i++) {
                pwIntra.println(chrName + "\t" + i*res + "\t" + (i+1)*res + "\t" + intraCoverage.get(i)/intraAverage);
                pwInter.println(chrName + "\t" + i*res + "\t" + (i+1)*res + "\t" + interCoverage.get(i)/interAverage);
                pwBias.println(chrName + "\t" + i*res + "\t" + (i+1)*res + "\t" + ((intraCoverage.get(i)+intraCorrection)/intraAverage)/((interCoverage.get(i)+interCorrection)/interAverage));
            }
        }
    }
}
