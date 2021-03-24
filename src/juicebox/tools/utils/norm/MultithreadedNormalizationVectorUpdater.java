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

package juicebox.tools.utils.norm;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.data.basics.Chromosome;
import juicebox.data.basics.ListOfFloatArrays;
import juicebox.tools.utils.original.ExpectedValueCalculation;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.broad.igv.tdf.BufferedByteWriter;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class MultithreadedNormalizationVectorUpdater extends NormalizationVectorUpdater {
    protected Set<Chromosome> synckrBPFailedChromosomes = Collections.synchronizedSet(krBPFailedChromosomes);
    protected Set<Chromosome> synckrFragFailedChromosomes = Collections.synchronizedSet(krFragFailedChromosomes);
    protected Set<Chromosome> syncmmbaBPFailedChromosomes = Collections.synchronizedSet(mmbaBPFailedChromosomes);
    protected Set<Chromosome> syncmmbaFragFailedChromosomes = Collections.synchronizedSet(mmbaFragFailedChromosomes);

    protected Map<HiCZoom, Set<Chromosome>> zoomSpecifickrBPFailedChromsomes = new ConcurrentHashMap<>();
    protected Map<HiCZoom, Set<Chromosome>> zoomSpecifickrFragFailedChromsomes = new ConcurrentHashMap<>();
    protected Map<HiCZoom, Set<Chromosome>> zoomSpecificmmbaBPFailedChromsomes = new ConcurrentHashMap<>();
    protected Map<HiCZoom, Set<Chromosome>> zoomSpecificmmbaFragFailedChromsomes = new ConcurrentHashMap<>();


    protected static int numCPUThreads = 1;

    public MultithreadedNormalizationVectorUpdater(int numCPUThreads) {
        MultithreadedNormalizationVectorUpdater.numCPUThreads = numCPUThreads;
    }

    @Override
    public void updateHicFile(String path, List<NormalizationType> normalizationsToBuild,
                              Map<NormalizationType, Integer> resolutionsToBuildTo, int genomeWideLowestResolutionAllowed, boolean noFrag) throws IOException {

        int minResolution = Integer.MAX_VALUE;
        for (Map.Entry<NormalizationType, Integer> entry : resolutionsToBuildTo.entrySet()) {
            if (entry.getValue() < minResolution) {
                minResolution = entry.getValue();
            }
        }

        DatasetReaderV2 reader = new DatasetReaderV2(path);
        Dataset ds = reader.read();
        HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());

        ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();
        Map<String, Integer> fragCountMap = ds.getFragmentCounts();
        List<HiCZoom> resolutions = ds.getAllPossibleResolutions();

        reEvaluateWhichIntraNormsToBuild(normalizationsToBuild);

        normVectorBuffers.add(new BufferedByteWriter());
        for (HiCZoom zoom : resolutions) {
            if (zoom.getBinSize() < minResolution) {
                System.out.println("skipping zoom" + zoom);
                continue;
            }
            if (noFrag && zoom.getUnit() == HiC.Unit.FRAG) continue;

            // compute genome-wide normalizations
            if (zoom.getUnit() == HiC.Unit.BP && zoom.getBinSize() >= genomeWideLowestResolutionAllowed) {
                GenomeWideNormalizationVectorUpdater.updateHicFileForGWfromPreAddNormOnly(ds, zoom, normalizationsToBuild, resolutionsToBuildTo,
                        normVectorIndices, normVectorBuffers, expectedValueCalculations);
            }

            System.out.println();
            System.out.print("Calculating norms for zoom " + zoom);

            Map<String, Integer> fcm = zoom.getUnit() == HiC.Unit.FRAG ? fragCountMap : null;

            ExpectedValueCalculation evVC = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.VC);
            ExpectedValueCalculation evVCSqrt = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.VC_SQRT);
            ExpectedValueCalculation evKR = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.KR);
            ExpectedValueCalculation evSCALE = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.SCALE);

            Map<Integer, Double> withinZoomVCSumFactors = new ConcurrentHashMap<>();
            Map<Integer, Double> withinZoomVCSQRTSumFactors = new ConcurrentHashMap<>();
            Map<Integer, Double> withinZoomKRSumFactors = new ConcurrentHashMap<>();
            Map<Integer, Double> withinZoomSCALESumFactors = new ConcurrentHashMap<>();
            Map<Integer, ListOfFloatArrays> withinZoomVCVectors = new ConcurrentHashMap<>();
            Map<Integer, ListOfFloatArrays> withinZoomVCSQRTVectors = new ConcurrentHashMap<>();
            Map<Integer, ListOfFloatArrays> withinZoomKRVectors = new ConcurrentHashMap<>();
            Map<Integer, ListOfFloatArrays> withinZoomSCALEVectors = new ConcurrentHashMap<>();

            final AtomicInteger chromosomeIndex = new AtomicInteger(0);

            Set<Chromosome> withinZoomSynckrBPFailedChromosomes = Collections.synchronizedSet(new HashSet<>());
            Set<Chromosome> withinZoomSynckrFragFailedChromosomes = Collections.synchronizedSet(new HashSet<>());
            Set<Chromosome> withinZoomSyncmmbaBPFailedChromosomes = Collections.synchronizedSet(new HashSet<>());
            Set<Chromosome> withinZoomSyncmmbaFragFailedChromosomes = Collections.synchronizedSet(new HashSet<>());

            Map<Integer, MatrixZoomData> allChrZoomData = new ConcurrentHashMap<>();

            ExecutorService executor = Executors.newFixedThreadPool(numCPUThreads);
            for (int l = 0; l < numCPUThreads; l++) {
                final int threadNum = l;

                Runnable worker = new Runnable() {
                    @Override
                    public void run() {
                        try {
                            DatasetReaderV2 localReader;
                            Dataset localds;
                            ChromosomeHandler localChromosomeHandler;
                            synchronized(this){
                                localReader = new DatasetReaderV2(path);
                                localds = localReader.read();
                                localChromosomeHandler = localds.getChromosomeHandler();
                            }
                            runIndividualChromosomeCode(chromosomeIndex, localds, localChromosomeHandler, zoom, resolutionsToBuildTo,
                                    withinZoomVCSumFactors, withinZoomVCSQRTSumFactors, withinZoomKRSumFactors, withinZoomSCALESumFactors,
                                    withinZoomVCVectors, withinZoomVCSQRTVectors, withinZoomKRVectors, withinZoomSCALEVectors,
                                    withinZoomSynckrBPFailedChromosomes, withinZoomSynckrFragFailedChromosomes, withinZoomSyncmmbaBPFailedChromosomes,
                                    withinZoomSyncmmbaFragFailedChromosomes, allChrZoomData, threadNum);
                        } catch (IOException e) {
                            System.err.println("Error: " + e);
                        }
                    }
                    //}
                };
                executor.execute(worker);
            }

            executor.shutdown();

            // Wait until all threads finish
            while (!executor.isTerminated()) {
            }


            // Loop through chromosomes
            for (Chromosome chr : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {

                if (allChrZoomData.get(chr.getIndex()) == null) continue;


                if (weShouldBuildVC && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.VC)) {
                    if (withinZoomVCSumFactors.get(chr.getIndex())!=null&&withinZoomVCVectors.get(chr.getIndex())!=null) {
                        updateExpectedValueCalculationForChr(chr.getIndex(), withinZoomVCSumFactors.get(chr.getIndex()),
                                withinZoomVCVectors.get(chr.getIndex()), NormalizationHandler.VC,
                                zoom, allChrZoomData.get(chr.getIndex()), evVC, normVectorBuffers, normVectorIndices);
                    }
                }
                if (weShouldBuildVCSqrt && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.VC_SQRT)) {
                    if (withinZoomVCSQRTSumFactors.get(chr.getIndex())!=null&&withinZoomVCSQRTVectors.get(chr.getIndex())!=null) {
                        updateExpectedValueCalculationForChr(chr.getIndex(), withinZoomVCSQRTSumFactors.get(chr.getIndex()),
                                withinZoomVCSQRTVectors.get(chr.getIndex()), NormalizationHandler.VC_SQRT,
                                zoom, allChrZoomData.get(chr.getIndex()), evVCSqrt, normVectorBuffers, normVectorIndices);
                    }
                }

                // KR normalization
                if (weShouldBuildKR && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.KR)) {
                    Set<Chromosome> withinZoomFailureSetKR = zoom.getUnit() == HiC.Unit.FRAG ? withinZoomSynckrFragFailedChromosomes : withinZoomSynckrBPFailedChromosomes;
                    if (!withinZoomFailureSetKR.contains(chr)&&withinZoomKRSumFactors.get(chr.getIndex())!=null&&withinZoomKRVectors.get(chr.getIndex())!=null) {
                        updateExpectedValueCalculationForChr(chr.getIndex(), withinZoomKRSumFactors.get(chr.getIndex()),
                                withinZoomKRVectors.get(chr.getIndex()), NormalizationHandler.KR,
                                zoom, allChrZoomData.get(chr.getIndex()), evKR, normVectorBuffers, normVectorIndices);
                    }
                }

                // Fast scaling normalization
                if (weShouldBuildScale && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.SCALE)) {
                    Set<Chromosome> withinZoomFailureSetMMBA = zoom.getUnit() == HiC.Unit.FRAG ? withinZoomSyncmmbaFragFailedChromosomes : withinZoomSyncmmbaBPFailedChromosomes;
                    if (!withinZoomFailureSetMMBA.contains(chr)&&withinZoomSCALESumFactors.get(chr.getIndex())!=null&&withinZoomSCALEVectors.get(chr.getIndex())!=null) {
                        updateExpectedValueCalculationForChr(chr.getIndex(), withinZoomSCALESumFactors.get(chr.getIndex()),
                                withinZoomSCALEVectors.get(chr.getIndex()), NormalizationHandler.SCALE,
                                zoom, allChrZoomData.get(chr.getIndex()), evSCALE, normVectorBuffers, normVectorIndices);
                    }
                }
            }

            if (weShouldBuildVC && evVC.hasData() && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.VC)) {
                expectedValueCalculations.add(evVC);
            }
            if (weShouldBuildVCSqrt && evVCSqrt.hasData() && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.VC_SQRT)) {
                expectedValueCalculations.add(evVCSqrt);
            }
            if (weShouldBuildKR && evKR.hasData() && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.KR)) {
                expectedValueCalculations.add(evKR);
            }
            if (weShouldBuildScale && evSCALE.hasData() && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.SCALE)) {
                expectedValueCalculations.add(evSCALE);
            }


        }
        writeNormsToUpdateFile(reader, path, true, expectedValueCalculations, null, normVectorIndices,
                normVectorBuffers, "Finished writing norms");

    }

    protected static void updateExpectedValueCalculationForChr(final int chrIdx, double factor, ListOfFloatArrays vec, NormalizationType type, HiCZoom zoom, MatrixZoomData zd,
                                                               ExpectedValueCalculation ev, List<BufferedByteWriter> normVectorBuffers, List<NormalizationVectorIndexEntry> normVectorIndex) throws IOException {
        vec.multiplyEverythingBy(factor);

        updateNormVectorIndexWithVector(normVectorIndex, normVectorBuffers, vec, chrIdx, type, zoom);

        ev.addDistancesFromIterator(chrIdx, zd.getIteratorContainer(), vec);

    }

    protected void buildVCOrVCSQRT(boolean weShouldBuildVC, boolean weShouldBuildVCSqrt, Chromosome chr,
                                   NormalizationCalculations nc, HiCZoom zoom, Map<Integer, Double> withinZoomVCSumFactors, Map<Integer, ListOfFloatArrays> withinZoomVCVectors,
                                   Map<Integer, Double> withinZoomVCSQRTSumFactors, Map<Integer, ListOfFloatArrays> withinZoomVCSQRTVectors) throws IOException {
        final int chrIdx = chr.getIndex();
        long currentTime = System.currentTimeMillis();
        ListOfFloatArrays vc = nc.computeVC();
        if (weShouldBuildVC) {
            withinZoomVCSumFactors.put(chrIdx, nc.getSumFactor(vc));
            withinZoomVCVectors.put(chrIdx, vc);
        }

        if (weShouldBuildVCSqrt) {
            ListOfFloatArrays vcSqrt = new ListOfFloatArrays(vc.getLength());
            for (long i = 0; i < vc.getLength(); i++) {
                vcSqrt.set(i, (float) Math.sqrt(vc.get(i)));
            }

            withinZoomVCSQRTSumFactors.put(chrIdx, nc.getSumFactor(vcSqrt));
            withinZoomVCSQRTVectors.put(chrIdx, vcSqrt);
        }
        printNormTiming("VC and VC_SQRT", chr, zoom, currentTime);
    }

    protected void buildKR(Chromosome chr, NormalizationCalculations nc, HiCZoom zoom,
                           Map<Integer, Double> withinZoomKRSumFactors,
                           Map<Integer, ListOfFloatArrays> withinZoomKRVectors,
                           Set<Chromosome> withinZoomSynckrBPFailedChromosomes,
                           Set<Chromosome> withinZoomSynckrFragFailedChromosomes) throws IOException {
        Set<Chromosome> failureSetKR = zoom.getUnit() == HiC.Unit.FRAG ? synckrFragFailedChromosomes : synckrBPFailedChromosomes;
        Set<Chromosome> withinZoomFailureSetKR = zoom.getUnit() == HiC.Unit.FRAG ? withinZoomSynckrFragFailedChromosomes : withinZoomSynckrBPFailedChromosomes;
        final int chrIdx = chr.getIndex();

        long currentTime = System.currentTimeMillis();
        if (!failureSetKR.contains(chr)) {
            ListOfFloatArrays kr = nc.computeKR();
            if (kr == null) {
                failureSetKR.add(chr);
                withinZoomFailureSetKR.add(chr);
                printNormTiming("FAILED KR", chr, zoom, currentTime);
            } else {
                withinZoomKRSumFactors.put(chrIdx, nc.getSumFactor(kr));
                withinZoomKRVectors.put(chrIdx, kr);
                printNormTiming("KR", chr, zoom, currentTime);
            }
        } else {
            withinZoomFailureSetKR.add(chr);
        }
    }

    protected void buildScale(Chromosome chr, NormalizationCalculations nc, HiCZoom zoom,
                              Map<Integer, Double> withinZoomSCALESumFactors,
                              Map<Integer, ListOfFloatArrays> withinZoomSCALEVectors,
                              Set<Chromosome> withinZoomSyncmmbaBPFailedChromosomes,
                              Set<Chromosome> withinZoomSyncmmbaFragFailedChromosomes) throws IOException {
        Set<Chromosome> failureSetMMBA = zoom.getUnit() == HiC.Unit.FRAG ? syncmmbaFragFailedChromosomes : syncmmbaBPFailedChromosomes;
        Set<Chromosome> withinZoomFailureSetMMBA = zoom.getUnit() == HiC.Unit.FRAG ? withinZoomSyncmmbaFragFailedChromosomes : withinZoomSyncmmbaBPFailedChromosomes;
        final int chrIdx = chr.getIndex();
        long currentTime = System.currentTimeMillis();

        if (!failureSetMMBA.contains(chr)) {
            ListOfFloatArrays mmba = nc.computeMMBA();
            if (mmba == null) {
                failureSetMMBA.add(chr);
                withinZoomFailureSetMMBA.add(chr);
                printNormTiming("FAILED SCALE", chr, zoom, currentTime);
            } else {
                withinZoomSCALESumFactors.put(chrIdx, nc.getSumFactor(mmba));
                withinZoomSCALEVectors.put(chrIdx, mmba);
                printNormTiming("SCALE", chr, zoom, currentTime);
            }
        } else {
            withinZoomFailureSetMMBA.add(chr);
        }
    }

    protected void runIndividualChromosomeCode(AtomicInteger chromosomeIndex,
                                               Dataset ds, ChromosomeHandler chromosomeHandler, HiCZoom zoom, Map<NormalizationType, Integer> resolutionsToBuildTo,
                                               Map<Integer, Double> withinZoomVCSumFactors, Map<Integer, Double> withinZoomVCSQRTSumFactors,
                                               Map<Integer, Double> withinZoomKRSumFactors, Map<Integer, Double> withinZoomSCALESumFactors,
                                               Map<Integer, ListOfFloatArrays> withinZoomVCVectors, Map<Integer, ListOfFloatArrays> withinZoomVCSQRTVectors,
                                               Map<Integer, ListOfFloatArrays> withinZoomKRVectors, Map<Integer, ListOfFloatArrays> withinZoomSCALEVectors,
                                               Set<Chromosome> withinZoomSynckrBPFailedChromosomes, Set<Chromosome> withinZoomSynckrFragFailedChromosomes,
                                               Set<Chromosome> withinZoomSyncmmbaBPFailedChromosomes, Set<Chromosome> withinZoomSyncmmbaFragFailedChromosomes,
                                               Map<Integer, MatrixZoomData> allChrZoomData, int threadnum) throws IOException {

        int i = chromosomeIndex.getAndIncrement();
        while (i < chromosomeHandler.getChromosomeArrayWithoutAllByAll().length) {
            Chromosome chr = chromosomeHandler.getChromosomeArrayWithoutAllByAll()[i];
            MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chr, chr, zoom);

            if (zd == null) {
                i = chromosomeIndex.getAndIncrement();
                continue;
            }
            NormalizationCalculations nc = new NormalizationCalculations(zd.getIteratorContainer());
            if (!nc.isEnoughMemory()) {
                System.err.println("Not enough memory, skipping " + chr);
                i = chromosomeIndex.getAndIncrement();
                continue;
            }
            allChrZoomData.put(chr.getIndex(), zd);

            if (weShouldBuildVC || weShouldBuildVCSqrt) {

                buildVCOrVCSQRT(weShouldBuildVC && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.VC),
                        weShouldBuildVCSqrt && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.VC_SQRT),
                        chr, nc, zoom, withinZoomVCSumFactors, withinZoomVCVectors, withinZoomVCSQRTSumFactors, withinZoomVCSQRTVectors);
            }
            if (weShouldBuildKR && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.KR)) {
                buildKR(chr, nc, zoom, withinZoomKRSumFactors, withinZoomKRVectors, withinZoomSynckrBPFailedChromosomes,
                        withinZoomSynckrFragFailedChromosomes);
            }
            if (weShouldBuildScale && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.SCALE)) {
                buildScale(chr, nc, zoom, withinZoomSCALESumFactors, withinZoomSCALEVectors, withinZoomSyncmmbaBPFailedChromosomes,
                        withinZoomSyncmmbaFragFailedChromosomes);
            }
            i = chromosomeIndex.getAndIncrement();
        }
    }
}
