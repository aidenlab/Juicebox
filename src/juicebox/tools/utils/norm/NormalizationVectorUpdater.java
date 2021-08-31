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

/**
 * Update an existing hic file with new normalization vectors (included expected value vectors)
 *
 * @author jrobinso
 * @since 2/8/13
 */
public class NormalizationVectorUpdater extends NormVectorUpdater {

    protected List<BufferedByteWriter> normVectorBuffers = new ArrayList<>();
    protected List<NormalizationVectorIndexEntry> normVectorIndices = new ArrayList<>();
    protected List<ExpectedValueCalculation> expectedValueCalculations = new ArrayList<>();

    // Keep track of chromosomes that fail to converge, so we don't try them at higher resolutions.
    protected Set<Chromosome> krBPFailedChromosomes = new HashSet<>();
    protected Set<Chromosome> krFragFailedChromosomes = new HashSet<>();
    protected Set<Chromosome> mmbaBPFailedChromosomes = new HashSet<>();
    protected Set<Chromosome> mmbaFragFailedChromosomes = new HashSet<>();

    // norms to build; gets overwritten
    protected boolean weShouldBuildVC = true;
    protected boolean weShouldBuildVCSqrt = true;
    protected boolean weShouldBuildKR = true;
    protected boolean weShouldBuildScale = true;

    protected static void printNormTiming(String norm, Chromosome chr, HiCZoom zoom, long currentTime) {
        if (HiCGlobals.printVerboseComments) {
            System.out.println(norm + " normalization of " + chr + " at " + zoom + " took " + (System.currentTimeMillis() - currentTime) + " milliseconds");
        }
    }

    protected static void updateExpectedValueCalculationForChr(final int chrIdx, NormalizationCalculations nc, ListOfFloatArrays vec, NormalizationType type, HiCZoom zoom, MatrixZoomData zd,
                                                               ExpectedValueCalculation ev, List<BufferedByteWriter> normVectorBuffers, List<NormalizationVectorIndexEntry> normVectorIndex) throws IOException {
        double factor = nc.getSumFactor(vec);
        vec.multiplyEverythingBy(factor);

        updateNormVectorIndexWithVector(normVectorIndex, normVectorBuffers, vec, chrIdx, type, zoom);

        ev.addDistancesFromIterator(chrIdx, zd.getIteratorContainer(), vec);
    }

    protected void reEvaluateWhichIntraNormsToBuild(List<NormalizationType> normalizationsToBuild) {
        weShouldBuildVC = normalizationsToBuild.contains(NormalizationHandler.VC);
        weShouldBuildVCSqrt = normalizationsToBuild.contains(NormalizationHandler.VC_SQRT);
        weShouldBuildKR = normalizationsToBuild.contains(NormalizationHandler.KR);
        weShouldBuildScale = normalizationsToBuild.contains(NormalizationHandler.SCALE);
    }

    protected void buildVCOrVCSQRT(boolean weShouldBuildVC, boolean weShouldBuildVCSqrt, Chromosome chr,
                                   NormalizationCalculations nc, HiCZoom zoom, MatrixZoomData zd, ExpectedValueCalculation evVC,
                                   ExpectedValueCalculation evVCSqrt) throws IOException {
        final int chrIdx = chr.getIndex();
        ListOfFloatArrays vc = nc.computeVC();

        ListOfFloatArrays vcSqrt = new ListOfFloatArrays(vc.getLength());
        if (weShouldBuildVCSqrt) {
            for (int i = 0; i < vc.getLength(); i++) {
                vcSqrt.set(i, (float) Math.sqrt(vc.get(i)));
            }
        }
        if (weShouldBuildVC) {
            updateExpectedValueCalculationForChr(chrIdx, nc, vc, NormalizationHandler.VC, zoom, zd, evVC, normVectorBuffers, normVectorIndices);
        }
        if (weShouldBuildVCSqrt) {
            updateExpectedValueCalculationForChr(chrIdx, nc, vcSqrt, NormalizationHandler.VC_SQRT, zoom, zd, evVCSqrt, normVectorBuffers, normVectorIndices);
        }
    }

    protected void buildKR(Chromosome chr, NormalizationCalculations nc, HiCZoom zoom, MatrixZoomData zd, ExpectedValueCalculation evKR) throws IOException {
        Set<Chromosome> failureSetKR = zoom.getUnit() == HiC.Unit.FRAG ? krFragFailedChromosomes : krBPFailedChromosomes;
        final int chrIdx = chr.getIndex();

        long currentTime = System.currentTimeMillis();
        if (!failureSetKR.contains(chr)) {
            ListOfFloatArrays kr = nc.computeKR();
            if (kr == null) {
                failureSetKR.add(chr);
                printNormTiming("FAILED KR", chr, zoom, currentTime);
            } else {
                updateExpectedValueCalculationForChr(chrIdx, nc, kr, NormalizationHandler.KR, zoom, zd, evKR, normVectorBuffers, normVectorIndices);
                printNormTiming("KR", chr, zoom, currentTime);
            }
        }
    }

    public void updateHicFile(String path, List<NormalizationType> normalizationsToBuild,
                              Map<NormalizationType, Integer> resolutionsToBuildTo, int genomeWideLowestResolutionAllowed, boolean noFrag) throws IOException {

        //System.out.println("test: using old norm code");
        int minResolution = Integer.MAX_VALUE;
        for (Map.Entry<NormalizationType, Integer> entry : resolutionsToBuildTo.entrySet()) {
            if (entry.getValue() < minResolution) {
                minResolution = entry.getValue();
            }
        }

        DatasetReaderV2 reader = new DatasetReaderV2(path);
        Dataset ds = reader.read();
        HiCGlobals.verifySupportedHiCFileWritingVersion(reader.getVersion());

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

            System.out.println();
            System.out.print("Calculating norms for zoom " + zoom);

            // compute genome-wide normalizations
            if (zoom.getUnit() == HiC.Unit.BP && zoom.getBinSize() >= genomeWideLowestResolutionAllowed) {
                GenomeWideNormalizationVectorUpdater.updateHicFileForGWfromPreAddNormOnly(ds, zoom, normalizationsToBuild, resolutionsToBuildTo,
                        normVectorIndices, normVectorBuffers, expectedValueCalculations);
            }

            ds.clearCache(true, zoom);

            //System.out.println("genomewide normalization: " + Duration.between(A,B).toMillis());

            Map<String, Integer> fcm = zoom.getUnit() == HiC.Unit.FRAG ? fragCountMap : null;

            ExpectedValueCalculation evVC = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.VC);
            ExpectedValueCalculation evVCSqrt = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.VC_SQRT);
            ExpectedValueCalculation evKR = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.KR);
            ExpectedValueCalculation evSCALE = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.SCALE);

            // Loop through chromosomes
            for (Chromosome chr : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {

                MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chr, chr, zoom);
                if (zd == null) continue;

                if (HiCGlobals.printVerboseComments) {
                    System.out.println("Now Doing " + chr.getName());
                }

                NormalizationCalculations nc = new NormalizationCalculations(zd.getIteratorContainer());
                if (!nc.isEnoughMemory()) {
                    System.err.println("Not enough memory, skipping " + chr);
                    continue;
                }

                if (weShouldBuildVC || weShouldBuildVCSqrt) {
                    buildVCOrVCSQRT(weShouldBuildVC && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.VC),
                            weShouldBuildVCSqrt && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.VC_SQRT),
                            chr, nc, zoom, zd, evVC, evVCSqrt);
                }

                // KR normalization
                if (weShouldBuildKR && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.KR)) {
                    buildKR(chr, nc, zoom, zd, evKR);
                }

                // Fast scaling normalization
                if (weShouldBuildScale && zoom.getBinSize() >= resolutionsToBuildTo.get(NormalizationHandler.SCALE)) {
                    buildScale(chr, nc, zoom, zd, evSCALE);
                }

                zd.clearCache(false);
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

            ds.clearCache(false, zoom);
        }
        writeNormsToUpdateFile(reader, path, true, expectedValueCalculations, null, normVectorIndices,
                normVectorBuffers, "Finished writing norms");
    }
    
    protected void buildScale(Chromosome chr, NormalizationCalculations nc, HiCZoom zoom, MatrixZoomData zd, ExpectedValueCalculation evSCALE) throws IOException {
        Set<Chromosome> failureSetMMBA = zoom.getUnit() == HiC.Unit.FRAG ? mmbaFragFailedChromosomes : mmbaBPFailedChromosomes;
        final int chrIdx = chr.getIndex();
        long currentTime = System.currentTimeMillis();
        
        if (!failureSetMMBA.contains(chr)) {
            ListOfFloatArrays mmba = nc.computeMMBA();
            
            if (mmba == null) {
                failureSetMMBA.add(chr);
                printNormTiming("FAILED SCALE", chr, zoom, currentTime);
            } else {
                updateExpectedValueCalculationForChr(chrIdx, nc, mmba, NormalizationHandler.SCALE, zoom, zd, evSCALE, normVectorBuffers, normVectorIndices);
                printNormTiming("SCALE", chr, zoom, currentTime);
            }
        }
    }
}
