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

import juicebox.HiCGlobals;
import juicebox.data.*;
import juicebox.data.basics.Chromosome;
import juicebox.data.basics.ListOfFloatArrays;
import juicebox.data.iterator.IteratorContainer;
import juicebox.data.iterator.ListOfListGenerator;
import juicebox.tools.utils.original.ExpectedValueCalculation;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.broad.igv.tdf.BufferedByteWriter;
import org.broad.igv.util.Pair;

import java.io.IOException;
import java.util.*;

public class GenomeWideNormalizationVectorUpdater extends NormVectorUpdater {
    // todo remove
    /*
    public static void addGWNorm(String path, int genomeWideResolution) throws IOException {
        DatasetReaderV2 reader = new DatasetReaderV2(path);
        Dataset ds = reader.read();
        HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());

        List<HiCZoom> resolutions = new ArrayList<>();
        resolutions.addAll(ds.getBpZooms());
        resolutions.addAll(ds.getFragZooms());

        List<BufferedByteWriter> normVectorBuffers = new ArrayList<>();
        List<NormalizationVectorIndexEntry> normVectorIndex = new ArrayList<>();
        Map<String, ExpectedValueFunction> expectedValueFunctionMap = ds.getExpectedValueFunctionMap();

        for (Iterator<Map.Entry<String, ExpectedValueFunction>> it = expectedValueFunctionMap.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<String, ExpectedValueFunction> entry = it.next();
            if (entry.getKey().contains("NONE")) {
                it.remove();
            }
        }

        // Loop through resolutions
        for (HiCZoom zoom : resolutions) {

            // compute genome-wide normalization
            // TODO make this dependent on memory, do as much as possible
            if (genomeWideResolution >= 10000 && zoom.getUnit() == HiC.Unit.BP && zoom.getBinSize() >= genomeWideResolution) {
                for (NormalizationType normType : NormalizationHandler.getAllGWNormTypes(false)) {

                    Pair<Map<Chromosome, NormalizationVector>, ExpectedValueCalculation> wgVectors = getWGVectors(ds, zoom, normType);

                    if (wgVectors != null) {
                        Map<Chromosome, NormalizationVector> nvMap = wgVectors.getFirst();
                        for (Chromosome chromosome : nvMap.keySet()) {
                            updateNormVectorIndexWithVector(normVectorIndex, normVectorBuffers, nvMap.get(chromosome).getData().convertToFloats(), chromosome.getIndex(), normType, zoom);
                        }
                        ExpectedValueCalculation calculation = wgVectors.getSecond();
                        String key = ExpectedValueFunctionImpl.getKey(zoom, normType);
                        expectedValueFunctionMap.put(key, calculation.getExpectedValueFunction());
                    }
                }
            }
            System.out.println();
            System.out.print("Calculating norms for zoom " + zoom);

            // Integer is either limit on genome wide resolution or limit on what fragment resolution to calculate
            // if (genomeWideResolution == 0 && zoom.getUnit() == HiC.Unit.FRAG) continue;
            // if (genomeWideResolution < 10000 && zoom.getUnit() == HiC.Unit.FRAG && zoom.getBinSize() <= genomeWideResolution) continue;


            // Loop through chromosomes
            for (Chromosome chr : ds.getChromosomeHandler().getChromosomeArrayWithoutAllByAll()) {
                Matrix matrix = ds.getMatrix(chr, chr);
                if (matrix == null) continue;

                for (NormalizationType normType : NormalizationHandler.getAllNormTypes()) {
                    NormalizationVector vector = ds.getNormalizationVector(chr.getIndex(), zoom, normType);
                    if (vector != null) {
                        updateNormVectorIndexWithVector(normVectorIndex, normVectorBuffers, vector.getData().convertToFloats(), chr.getIndex(), normType, zoom);
                    }
                }
            }
        }

        int version = reader.getVersion();
        long filePosition = reader.getNormFilePosition();
        long nviHeaderPosition = reader.getNviHeaderPosition();

        System.out.println();
        NormalizationVectorUpdater.update(path, version, filePosition, expectedValueFunctionMap, normVectorIndex,
                normVectorBuffers, nviHeaderPosition);
        System.out.println("Finished normalization");
    }
    */


    public static void updateHicFileForGWfromPreAddNormOnly(Dataset ds, HiCZoom zoom, List<NormalizationType> normalizationsToBuild,
                                                            Map<NormalizationType, Integer> resolutionsToBuildTo, List<NormalizationVectorIndexEntry> normVectorIndices,
                                                            List<BufferedByteWriter> normVectorBuffers, List<ExpectedValueCalculation> expectedValueCalculations) throws IOException {
        for (NormalizationType normType : normalizationsToBuild) {
            if (NormalizationHandler.isGenomeWideNorm(normType)) {
                if (zoom.getBinSize() >= resolutionsToBuildTo.get(normType)) {

                    if (HiCGlobals.printVerboseComments) {
                        System.out.println("Now Doing " + normType.getLabel());
                    }
                    long currentTime = System.currentTimeMillis();
                    Pair<Map<Chromosome, NormalizationVector>, ExpectedValueCalculation> wgVectors = getWGVectors(ds, zoom, normType);
                    if (HiCGlobals.printVerboseComments) {
                        System.out.println("\n" + normType.getLabel() + " normalization genome wide at " + zoom + " took " + (System.currentTimeMillis() - currentTime) + " milliseconds");
                    }

                    if (wgVectors != null) {
                        Map<Chromosome, NormalizationVector> nvMap = wgVectors.getFirst();
                        List<Chromosome> chromosomes = new ArrayList<>(nvMap.keySet());
                        Collections.sort(chromosomes, Comparator.comparingInt(Chromosome::getIndex));
                        for (Chromosome chromosome : chromosomes) {
                            updateNormVectorIndexWithVector(normVectorIndices, normVectorBuffers,
                                    nvMap.get(chromosome).getData().convertToFloats(), chromosome.getIndex(),
                                    normType, zoom);
                        }

                        expectedValueCalculations.add(wgVectors.getSecond());
                    }
                }
            }
        }
    }

    /**
     * Compute the whole-genome normalization and expected value vectors and return as a pair (normalization vector first)
     */

    private static Pair<Map<Chromosome, NormalizationVector>, ExpectedValueCalculation> getWGVectors(Dataset dataset,
                                                                                                     HiCZoom zoom,
                                                                                                     NormalizationType norm) {
        boolean includeIntraData = NormalizationHandler.isGenomeWideNormIntra(norm); // default INTER type
        final ChromosomeHandler chromosomeHandler = dataset.getChromosomeHandler();
        final int resolution = zoom.getBinSize();
        final IteratorContainer ic = ListOfListGenerator.createForWholeGenome(dataset, chromosomeHandler, zoom,
                includeIntraData);

        NormalizationCalculations calculations = new NormalizationCalculations(ic);
        ListOfFloatArrays vector = calculations.getNorm(norm);
        if (vector == null) {
            return null;
        }

        ExpectedValueCalculation expectedValueCalculation = new ExpectedValueCalculation(chromosomeHandler, resolution, null, norm);
        int addY = 0;
        // Loop through chromosomes
        for (Chromosome chr : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            MatrixZoomData zd = HiCFileTools.getMatrixZoomData(dataset, chr, chr, zoom);
            if (zd == null) continue;
            final int chrIdx = chr.getIndex();

            Iterator<ContactRecord> iterator = zd.getFromFileIteratorContainer().getNewContactRecordIterator();
            while (iterator.hasNext()) {
                ContactRecord cr = iterator.next();
                int x = cr.getBinX();
                int y = cr.getBinY();
                final float vx = vector.get(x + addY);
                final float vy = vector.get(y + addY);
                if (isValidNormValue(vx) && isValidNormValue(vy)) {
                    double value = cr.getCounts() / (vx * vy);
                    expectedValueCalculation.addDistance(chrIdx, x, y, value);
                }
            }
            addY += chr.getLength() / resolution + 1;
        }

        // Split normalization vector by chromosome
        Map<Chromosome, NormalizationVector> normVectorMap =
                NormalizationTools.parCreateNormVectorMap(chromosomeHandler, resolution, vector, norm, zoom);

        ic.clear();

        return new Pair<>(normVectorMap, expectedValueCalculation);
    }
}
