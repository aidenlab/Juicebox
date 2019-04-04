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
import org.broad.igv.util.Pair;

import java.io.IOException;
import java.util.*;

public class GenomeWideNormalizationVectorUpdater extends NormVectorUpdater {
    public static void addGWNorm(String path, int genomeWideResolution) throws IOException {
        DatasetReaderV2 reader = new DatasetReaderV2(path);
        Dataset ds = reader.read();
        HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());

        List<HiCZoom> resolutions = new ArrayList<>();
        resolutions.addAll(ds.getBpZooms());
        resolutions.addAll(ds.getFragZooms());


        BufferedByteWriter normVectorBuffer = new BufferedByteWriter();
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
                            updateNormVectorIndexWithVector(normVectorIndex, normVectorBuffer, nvMap.get(chromosome).getData(), chromosome.getIndex(), normType, zoom);
                        }
                        ExpectedValueCalculation calculation = wgVectors.getSecond();
                        String key = ExpectedValueFunctionImpl.getKey(zoom, normType);
                        expectedValueFunctionMap.put(key, calculation.getExpectedValueFunction());
                    }
                }

            }
            System.out.println();
            System.out.print("Calculating norms for zoom " + zoom);

            /*
            // Integer is either limit on genome wide resolution or limit on what fragment resolution to calculate
            if (genomeWideResolution == 0 && zoom.getUnit() == HiC.Unit.FRAG) continue;
            if (genomeWideResolution < 10000 && zoom.getUnit() == HiC.Unit.FRAG && zoom.getBinSize() <= genomeWideResolution) continue;
            */

            // Loop through chromosomes
            for (Chromosome chr : ds.getChromosomeHandler().getChromosomeArrayWithoutAllByAll()) {
                Matrix matrix = ds.getMatrix(chr, chr);

                if (matrix == null) continue;

                for (NormalizationType normType : NormalizationHandler.getAllNormTypes()) {
                    NormalizationVector vector = ds.getNormalizationVector(chr.getIndex(), zoom, normType);
                    if (vector != null) {
                        updateNormVectorIndexWithVector(normVectorIndex, normVectorBuffer, vector.getData(), chr.getIndex(), normType, zoom);
                    }
                }
            }
        }


        int version = reader.getVersion();
        long filePosition = reader.getNormFilePosition();
        reader.close();
        System.out.println();
        NormalizationVectorUpdater.update(path, version, filePosition, expectedValueFunctionMap, normVectorIndex,
                normVectorBuffer.getBytes());
        System.out.println("Finished normalization");
    }

    /**
     * Compute the whole-genome normalization and expected value vectors and return as a pair (normalization vector first)
     */

    private static Pair<Map<Chromosome, NormalizationVector>, ExpectedValueCalculation> getWGVectors(Dataset dataset,
                                                                                                     HiCZoom zoom,
                                                                                                     NormalizationType norm) {

        boolean includeIntra = false;
        if (NormalizationHandler.isGenomeWideNorm(norm)) {
            includeIntra = true;
        }
        final ChromosomeHandler chromosomeHandler = dataset.getChromosomeHandler();
        final int resolution = zoom.getBinSize();
        final ArrayList<ContactRecord> recordArrayList = createWholeGenomeRecords(dataset, chromosomeHandler, zoom, includeIntra);

        int totalSize = 0;
        for (Chromosome c1 : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            totalSize += c1.getLength() / resolution + 1;
        }


        NormalizationCalculations calculations = new NormalizationCalculations(recordArrayList, totalSize);
        double[] vector = calculations.getNorm(norm);

        if (vector == null) {
            return null;
        }

        ExpectedValueCalculation expectedValueCalculation = new ExpectedValueCalculation(chromosomeHandler, resolution, null, norm);
        int addY = 0;
        // Loop through chromosomes
        for (Chromosome chr : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            final int chrIdx = chr.getIndex();
            Matrix matrix = dataset.getMatrix(chr, chr);

            if (matrix == null) continue;
            MatrixZoomData zd = matrix.getZoomData(zoom);
            Iterator<ContactRecord> iter = zd.getNewContactRecordIterator();
            while (iter.hasNext()) {
                ContactRecord cr = iter.next();
                int x = cr.getBinX();
                int y = cr.getBinY();
                final double vx = vector[x + addY];
                final double vy = vector[y + addY];
                if (isValidNormValue(vx) && isValidNormValue(vy)) {
                    double value = cr.getCounts() / (vx * vy);
                    expectedValueCalculation.addDistance(chrIdx, x, y, value);
                }
            }

            addY += chr.getLength() / resolution + 1;
        }

        // Split normalization vector by chromosome
        Map<Chromosome, NormalizationVector> normVectorMap = new LinkedHashMap<>();
        int location1 = 0;
        for (Chromosome c1 : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            int chrBinned = c1.getLength() / resolution + 1;
            double[] chrNV = new double[chrBinned];
            for (int i = 0; i < chrNV.length; i++) {
                chrNV[i] = vector[location1];
                location1++;
            }
            normVectorMap.put(c1, new NormalizationVector(norm, c1.getIndex(), zoom.getUnit(), resolution, chrNV));
        }

        return new Pair<>(normVectorMap, expectedValueCalculation);
    }

    public static ArrayList<ContactRecord> createWholeGenomeRecords(Dataset dataset, ChromosomeHandler handler,
                                                                    HiCZoom zoom, boolean includeIntra) {
        ArrayList<ContactRecord> recordArrayList = new ArrayList<>();
        int addX = 0;
        int addY = 0;
        for (Chromosome c1 : handler.getChromosomeArrayWithoutAllByAll()) {
            for (Chromosome c2 : handler.getChromosomeArrayWithoutAllByAll()) {
                if (c1.getIndex() < c2.getIndex() || (c1.equals(c2) && includeIntra)) {
                    Matrix matrix = dataset.getMatrix(c1, c2);
                    if (matrix != null) {
                        MatrixZoomData zd = matrix.getZoomData(zoom);
                        if (zd != null) {
                            Iterator<ContactRecord> iter = zd.getNewContactRecordIterator();
                            while (iter.hasNext()) {
                                ContactRecord cr = iter.next();
                                int binX = cr.getBinX() + addX;
                                int binY = cr.getBinY() + addY;
                                recordArrayList.add(new ContactRecord(binX, binY, cr.getCounts()));
                            }
                        }
                    }
                }
                addY += c2.getLength() / zoom.getBinSize() + 1;
            }
            addX += c1.getLength() / zoom.getBinSize() + 1;
            addY = 0;
        }
        return recordArrayList;
    }

    public static void updateHicFileForGWfromPreOnly(Dataset ds, HiCZoom zoom, List<NormalizationVectorIndexEntry> normVectorIndices,
                                                     BufferedByteWriter normVectorBuffer, List<ExpectedValueCalculation> expectedValueCalculations) throws IOException {

        for (NormalizationType normType : NormalizationHandler.getAllGWNormTypes(true)) {

            Pair<Map<Chromosome, NormalizationVector>, ExpectedValueCalculation> wgVectors = getWGVectors(ds, zoom, normType);

            if (wgVectors != null) {
                Map<Chromosome, NormalizationVector> nvMap = wgVectors.getFirst();
                for (Chromosome chromosome : nvMap.keySet()) {

                    NormalizationVector nv = nvMap.get(chromosome);

                    int position = normVectorBuffer.bytesWritten();
                    putArrayValuesIntoBuffer(normVectorBuffer, nv.getData());

                    int sizeInBytes = normVectorBuffer.bytesWritten() - position;
                    normVectorIndices.add(new NormalizationVectorIndexEntry(
                            normType.toString(), chromosome.getIndex(), zoom.getUnit().toString(), zoom.getBinSize(), position, sizeInBytes));
                }

                expectedValueCalculations.add(wgVectors.getSecond());
            }
        }
    }
}
