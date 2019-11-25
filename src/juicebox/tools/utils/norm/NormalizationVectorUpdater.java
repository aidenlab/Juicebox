/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.norm;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.tools.utils.original.ExpectedValueCalculation;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;
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

    private BufferedByteWriter normVectorBuffer = new BufferedByteWriter();
    private List<NormalizationVectorIndexEntry> normVectorIndices = new ArrayList<>();
    private List<ExpectedValueCalculation> expectedValueCalculations = new ArrayList<>();

    // Keep track of chromosomes that fail to converge, so we don't try them at higher resolutions.
    private Set<Chromosome> krBPFailedChromosomes = new HashSet<>();
    private Set<Chromosome> krFragFailedChromosomes = new HashSet<>();
    private Set<Chromosome> mmbaBPFailedChromosomes = new HashSet<>();
    private Set<Chromosome> mmbaFragFailedChromosomes = new HashSet<>();

    // norms to build; gets overwritten
    private boolean weShouldBuildVC = true;
    private boolean weShouldBuildVCSqrt = true;
    private boolean weShouldBuildKR = true;
    private boolean weShouldBuildScale = true;

    private static void printNormTiming(String norm, Chromosome chr, HiCZoom zoom, long currentTime) {
        if (HiCGlobals.printVerboseComments) {
            System.out.println(norm + " normalization of " + chr + " at " + zoom + " took " + (System.currentTimeMillis() - currentTime) + " milliseconds");
        }
    }

    public void updateHicFile(String path, List<NormalizationType> normalizationsToBuild,
                              int genomeWideLowestResolutionAllowed, boolean noFrag) throws IOException {
        DatasetReaderV2 reader = new DatasetReaderV2(path);
        Dataset ds = reader.read();
        HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());

        ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();
        Map<String, Integer> fragCountMap = ds.getFragmentCounts();
        List<HiCZoom> resolutions = ds.getAllPossibleResolutions();

        reEvaluateWhichIntraNormsToBuild(normalizationsToBuild);

        for (HiCZoom zoom : resolutions) {
            if (noFrag && zoom.getUnit() == HiC.Unit.FRAG) continue;

            // compute genome-wide normalizations
            if (zoom.getUnit() == HiC.Unit.BP && zoom.getBinSize() >= genomeWideLowestResolutionAllowed) {
                GenomeWideNormalizationVectorUpdater.updateHicFileForGWfromPreAddNormOnly(ds, zoom, normalizationsToBuild,
                        normVectorIndices, normVectorBuffer, expectedValueCalculations);
            }

            System.out.println();
            System.out.print("Calculating norms for zoom " + zoom);

            Map<String, Integer> fcm = zoom.getUnit() == HiC.Unit.FRAG ? fragCountMap : null;

            ExpectedValueCalculation evVC = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.VC);
            ExpectedValueCalculation evVCSqrt = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.VC_SQRT);
            ExpectedValueCalculation evKR = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.KR);
            ExpectedValueCalculation evSCALE = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.SCALE);

            // Loop through chromosomes
            for (Chromosome chr : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
                MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chr, chr, zoom);
                if (zd == null) continue;

                NormalizationCalculations nc = new NormalizationCalculations(zd);
                if (!nc.isEnoughMemory()) {
                    System.err.println("Not enough memory, skipping " + chr);
                    continue;
                }

                if (weShouldBuildVC || weShouldBuildVCSqrt) {
                    buildVCOrVCSQRT(weShouldBuildVC, weShouldBuildVCSqrt, chr, nc, zoom, zd, evVC, evVCSqrt);
                }

                // KR normalization
                if (weShouldBuildKR) {
                    buildKR(chr, nc, zoom, zd, evKR);
                }

                // Fast scaling normalization
                if (weShouldBuildScale) {
                    buildScale(chr, nc, zoom, zd, evSCALE);
                }
            }

            if (weShouldBuildVC && evVC.hasData()) {
                expectedValueCalculations.add(evVC);
            }
            if (weShouldBuildVCSqrt && evVCSqrt.hasData()) {
                expectedValueCalculations.add(evVCSqrt);
            }
            if (weShouldBuildKR && evKR.hasData()) {
                if (evKR.hasData()) {
                    expectedValueCalculations.add(evKR);
                }
            }
            if (weShouldBuildScale && evSCALE.hasData()) {
                expectedValueCalculations.add(evSCALE);
            }
        }
        writeNormsToUpdateFile(reader, path, true, expectedValueCalculations, null, normVectorIndices,
                normVectorBuffer, "Finished writing norms");
    }

    private void reEvaluateWhichIntraNormsToBuild(List<NormalizationType> normalizationsToBuild) {
        weShouldBuildVC = normalizationsToBuild.contains(NormalizationHandler.VC);
        weShouldBuildVCSqrt = normalizationsToBuild.contains(NormalizationHandler.VC_SQRT);
        weShouldBuildKR = normalizationsToBuild.contains(NormalizationHandler.KR);
        weShouldBuildScale = normalizationsToBuild.contains(NormalizationHandler.SCALE);
    }

    private void buildVCOrVCSQRT(boolean weShouldBuildVC, boolean weShouldBuildVCSqrt, Chromosome chr,
                                 NormalizationCalculations nc, HiCZoom zoom, MatrixZoomData zd, ExpectedValueCalculation evVC,
                                 ExpectedValueCalculation evVCSqrt) throws IOException {
        final int chrIdx = chr.getIndex();
        long currentTime = System.currentTimeMillis();
        double[] vc = nc.computeVC();
        if (weShouldBuildVC) {
            updateExpectedValueCalculationForChr(chrIdx, nc, vc, NormalizationHandler.VC, zoom, zd, evVC, normVectorBuffer, normVectorIndices);
        }

        if (weShouldBuildVCSqrt) {
            double[] vcSqrt = new double[vc.length];
            for (int i = 0; i < vc.length; i++) {
                vcSqrt[i] = Math.sqrt(vc[i]);
            }
            updateExpectedValueCalculationForChr(chrIdx, nc, vcSqrt, NormalizationHandler.VC_SQRT, zoom, zd, evVCSqrt, normVectorBuffer, normVectorIndices);
        }
        printNormTiming("VC and VC_SQRT", chr, zoom, currentTime);
    }

    private void buildKR(Chromosome chr, NormalizationCalculations nc, HiCZoom zoom, MatrixZoomData zd, ExpectedValueCalculation evKR) throws IOException {
        Set<Chromosome> failureSetKR = zoom.getUnit() == HiC.Unit.FRAG ? krFragFailedChromosomes : krBPFailedChromosomes;
        final int chrIdx = chr.getIndex();

        long currentTime = System.currentTimeMillis();
        if (!failureSetKR.contains(chr)) {
            double[] kr = nc.computeKR();
            if (kr == null) {
                failureSetKR.add(chr);
                printNormTiming("FAILED KR", chr, zoom, currentTime);
            } else {
                updateExpectedValueCalculationForChr(chrIdx, nc, kr, NormalizationHandler.KR, zoom, zd, evKR, normVectorBuffer, normVectorIndices);
                printNormTiming("KR", chr, zoom, currentTime);
            }
        }
    }

    private static void updateExpectedValueCalculationForChr(final int chrIdx, NormalizationCalculations nc, double[] vec, NormalizationType type, HiCZoom zoom, MatrixZoomData zd,
                                                             ExpectedValueCalculation ev, BufferedByteWriter normVectorBuffer, List<NormalizationVectorIndexEntry> normVectorIndex) throws IOException {
        double factor = nc.getSumFactor(vec);
        for (int i = 0; i < vec.length; i++) {
            vec[i] = vec[i] * factor;
        }

        updateNormVectorIndexWithVector(normVectorIndex, normVectorBuffer, vec, chrIdx, type, zoom);

        ev.addDistancesFromIterator(chrIdx, zd.getNewContactRecordIterator(), vec);
    }

    private void buildScale(Chromosome chr, NormalizationCalculations nc, HiCZoom zoom, MatrixZoomData zd, ExpectedValueCalculation evSCALE) throws IOException {
        Set<Chromosome> failureSetMMBA = zoom.getUnit() == HiC.Unit.FRAG ? mmbaFragFailedChromosomes : mmbaBPFailedChromosomes;
        final int chrIdx = chr.getIndex();
        long currentTime = System.currentTimeMillis();

        if (!failureSetMMBA.contains(chr)) {
            double[] mmba = nc.computeMMBA();
            if (mmba == null) {
                failureSetMMBA.add(chr);
                printNormTiming("FAILED SCALE", chr, zoom, currentTime);
            } else {
                updateExpectedValueCalculationForChr(chrIdx, nc, mmba, NormalizationHandler.SCALE, zoom, zd, evSCALE, normVectorBuffer, normVectorIndices);
                printNormTiming("SCALE", chr, zoom, currentTime);
            }
        }
    }
}
