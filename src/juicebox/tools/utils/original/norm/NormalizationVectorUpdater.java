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

package juicebox.tools.utils.original.norm;

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

    public static void updateHicFile(String path, int genomeWideResolution, boolean noFrag) throws IOException {
        DatasetReaderV2 reader = new DatasetReaderV2(path);
        Dataset ds = reader.read();
        HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());

        ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();

        // chr -> frag count map.  Needed for expected value calculations
        Map<String, Integer> fragCountMap = ds.getFragmentCounts();

        List<HiCZoom> resolutions = ds.getAllPossibleResolutions();

        // Keep track of chromosomes that fail to converge, so we don't try them at higher resolutions.
        Set<Chromosome> krBPFailedChromosomes = new HashSet<>();
        Set<Chromosome> krFragFailedChromosomes = new HashSet<>();
        Set<Chromosome> mmbaBPFailedChromosomes = new HashSet<>();
        Set<Chromosome> mmbaFragFailedChromosomes = new HashSet<>();

        BufferedByteWriter normVectorBuffer = new BufferedByteWriter();
        List<NormalizationVectorIndexEntry> normVectorIndices = new ArrayList<>();
        List<ExpectedValueCalculation> expectedValueCalculations = new ArrayList<>();


        // Loop through resolutions
        for (HiCZoom zoom : resolutions) {

            // compute genome-wide normalizaton but just for scaling
            if (zoom.getUnit() == HiC.Unit.BP) {// genomeWideResolution > 0 &&  && zoom.getBinSize() >= genomeWideResolution) {
                GenomeWideNormalizationVectorUpdater.updateHicFileForGWfromPreOnly(ds, zoom, normVectorIndices, normVectorBuffer, expectedValueCalculations);
            }
            System.out.println();
            System.out.print("Calculating norms for zoom " + zoom);

            // Integer is either limit on genome wide resolution or limit on what fragment resolution to calculate
            if (noFrag && zoom.getUnit() == HiC.Unit.FRAG) continue;

            Set<Chromosome> failureSetKR = zoom.getUnit() == HiC.Unit.FRAG ? krFragFailedChromosomes : krBPFailedChromosomes;
            Set<Chromosome> failureSetMMBA = zoom.getUnit() == HiC.Unit.FRAG ? mmbaFragFailedChromosomes : mmbaBPFailedChromosomes;

            Map<String, Integer> fcm = zoom.getUnit() == HiC.Unit.FRAG ? fragCountMap : null;

            ExpectedValueCalculation evVC = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.VC);
            ExpectedValueCalculation evVCSqrt = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.VC_SQRT);
            ExpectedValueCalculation evKR = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.KR);
            ExpectedValueCalculation evMMBA = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationHandler.SCALE);


            // Loop through chromosomes
            for (Chromosome chr : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
                Matrix matrix = ds.getMatrix(chr, chr);

                if (matrix == null) continue;
                MatrixZoomData zd = matrix.getZoomData(zoom);

                NormalizationCalculations nc = new NormalizationCalculations(zd);

                if (!nc.isEnoughMemory()) {
                    System.err.println("Not enough memory, skipping " + chr);
                    continue;
                }
                long currentTime = System.currentTimeMillis();


                double[] vc = nc.computeVC();
                double[] vcSqrt = new double[vc.length];
                for (int i = 0; i < vc.length; i++) {
                    vcSqrt[i] = Math.sqrt(vc[i]);
                }

                final int chrIdx = chr.getIndex();

                updateExpectedValueCalculationForChr(chrIdx, nc, vc, NormalizationHandler.VC, zoom, zd, evVC, normVectorBuffer, normVectorIndices);
                updateExpectedValueCalculationForChr(chrIdx, nc, vcSqrt, NormalizationHandler.VC_SQRT, zoom, zd, evVCSqrt, normVectorBuffer, normVectorIndices);

                if (HiCGlobals.printVerboseComments) {
                    System.out.println("\nVC normalization of " + chr + " at " + zoom + " took " + (System.currentTimeMillis() - currentTime) + " milliseconds");
                }
                currentTime = System.currentTimeMillis();

                // KR normalization
                if (!failureSetKR.contains(chr)) {
                    double[] kr = nc.computeKR();
                    if (kr == null) {
                        failureSetKR.add(chr);
                    } else {
                        updateExpectedValueCalculationForChr(chrIdx, nc, kr, NormalizationHandler.KR, zoom, zd, evKR, normVectorBuffer, normVectorIndices);
                    }
                    if (HiCGlobals.printVerboseComments) {
                        System.out.println("KR normalization of " + chr + " at " + zoom + " took " + (System.currentTimeMillis() - currentTime) + " milliseconds");
                    }
                }

                currentTime = System.currentTimeMillis();
                // Fast scaling normalization
                if (!failureSetMMBA.contains(chr)) {
                    double[] mmba = nc.computeMMBA();
                    if (mmba == null) {
                        failureSetMMBA.add(chr);
                    } else {
                        updateExpectedValueCalculationForChr(chrIdx, nc, mmba, NormalizationHandler.SCALE, zoom, zd, evMMBA, normVectorBuffer, normVectorIndices);
                    }
                    if (HiCGlobals.printVerboseComments) {
                        System.out.println("Fast scaling normalization of " + chr + " at " + zoom + " took " + (System.currentTimeMillis() - currentTime) + " milliseconds");
                    }
                }
            }

            if (evVC.hasData()) {
                expectedValueCalculations.add(evVC);
            }
            if (evVCSqrt.hasData()) {
                expectedValueCalculations.add(evVCSqrt);
            }
            if (evKR.hasData()) {
                expectedValueCalculations.add(evKR);
            }
            if (evMMBA.hasData()) {
                expectedValueCalculations.add(evMMBA);
            }
        }
        writeNormsToUpdateFile(reader, path, true, expectedValueCalculations, null, normVectorIndices,
                normVectorBuffer, "Finished writing norms");
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
}
