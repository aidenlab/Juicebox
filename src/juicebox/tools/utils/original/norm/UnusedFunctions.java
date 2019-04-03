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

import juicebox.data.*;
import juicebox.tools.utils.original.Preprocessor;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.tdf.BufferedByteWriter;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

class UnusedFunctions {

    /**
     * Main method is for internal testing and should not be used in general
     *
     * @param args If one argument, call updater code; if two, call dump index code
     * @throws IOException In case of error while reading or writing
     */
    public static void internalTest(String[] args) throws IOException {

        String path = args[0];

        if (args.length == 2) {
            CustomNormVectorFileHandler.updateHicFile(path, args[1]);
        } else {
            NormalizationVectorUpdater.updateHicFile(path, -100, false);
        }

     /*   if (args.length > 1) {
            dumpNormalizationVectorIndex(path, args[1]);
        }
        else updateHicFile(path);
       */
    }

    static void writeNormSums(
            List<Chromosome> chromosomes, Dataset ds, List<HiCZoom> zooms, Map<String, NormalizationVector> normVectors,
            BufferedByteWriter buffer) throws IOException {


        List<NormalizedSum> sums = new ArrayList<>();
        // Conventions:  chromosomes[0] == Chr_ALL.  Other  chromosomes in increasing order
        for (int i = 1; i < chromosomes.size(); i++) {
            Chromosome chr1 = chromosomes.get(i);

            // Start at i+1, don't need this for intra
            for (int j = i; j < chromosomes.size(); j++) {
                // Normalized sums (used to compute averages for expected values)
                Chromosome chr2 = chromosomes.get(j);

                for (HiCZoom zoom : zooms) {

                    MatrixZoomData zd2 = ds.getMatrix(chr1, chr2).getZoomData(zoom);
                    Iterator<ContactRecord> iter2 = zd2.getNewContactRecordIterator();

                    getNormalizedSumForNormalizationType(sums, iter2, normVectors, NormalizationHandler.VC, chr1, chr2, zoom);
                    getNormalizedSumForNormalizationType(sums, iter2, normVectors, NormalizationHandler.VC_SQRT, chr1, chr2, zoom);
                    getNormalizedSumForNormalizationType(sums, iter2, normVectors, NormalizationHandler.KR, chr1, chr2, zoom);
                    getNormalizedSumForNormalizationType(sums, iter2, normVectors, NormalizationHandler.SCALE, chr1, chr2, zoom);
                }
            }
        }

        buffer.putInt(sums.size());
        for (NormalizedSum sum : sums) {
            buffer.putNullTerminatedString(sum.type);
            buffer.putInt(sum.chr1Idx);
            buffer.putInt(sum.chr2Idx);
            buffer.putNullTerminatedString(sum.unit);
            buffer.putInt(sum.resolution);
            buffer.putDouble(sum.value);
        }

    }

    private static void getNormalizedSumForNormalizationType(List<NormalizedSum> sums, Iterator<ContactRecord> iter2, Map<String, NormalizationVector> normVectors, NormalizationType vc, Chromosome chr1, Chromosome chr2, HiCZoom zoom) {

        String key1 = NormalizationVector.getKey(NormalizationHandler.VC, chr1.getIndex(), zoom.getUnit().toString(), zoom.getBinSize());
        NormalizationVector vector1 = normVectors.get(key1);
        double[] vec1 = vector1.getData();

        String key2 = NormalizationVector.getKey(NormalizationHandler.VC, chr2.getIndex(), zoom.getUnit().toString(), zoom.getBinSize());
        NormalizationVector vector2 = normVectors.get(key2);
        double[] vec2 = vector2.getData();

        double vecSum = 0;

        if (vec1 == null || vec2 == null) return;

        while (iter2.hasNext()) {
            ContactRecord cr = iter2.next();
            int x = cr.getBinX();
            int y = cr.getBinY();

            if (!Double.isNaN(vec1[x]) && !Double.isNaN(vec2[y]) && vec1[x] > 0 && vec2[y] > 0) {
                // want total sum of matrix, not just upper triangle
                if (x == y) {
                    vecSum += cr.getCounts() / (vec1[x] * vec2[y]);
                } else {
                    vecSum += 2 * cr.getCounts() / (vec1[x] * vec2[y]);
                }
            }
        }

        if (vecSum > 0) {
            sums.add(new NormalizedSum(NormalizationHandler.VC.getLabel(), chr1.getIndex(), chr2.getIndex(), zoom.getUnit().toString(), zoom.getBinSize(), vecSum));
        }
    }

    static private void dumpNormalizationVectorIndex(String path, String outputFile, NormalizationHandler normalizationHandler) throws IOException {
        DatasetReaderV2 reader = new DatasetReaderV2(path);
        reader.read();
        try (RandomAccessFile raf = new RandomAccessFile(outputFile, "rw")) {

            BufferedByteWriter buffer = new BufferedByteWriter();

            // header: magic string HICNORM; version number 1; path
            String HIC_NORM = "HICNORM";
            buffer.putNullTerminatedString(HIC_NORM);
            buffer.putInt(1);
            buffer.putNullTerminatedString(path);

            Map<String, Preprocessor.IndexEntry> normVectorMap = reader.getNormVectorIndex();

            List<NormalizationVectorIndexEntry> normList = new ArrayList<>();

            for (Map.Entry<String, Preprocessor.IndexEntry> entry : normVectorMap.entrySet()) {
                String[] parts = entry.getKey().split("_");
                String strType;
                int chrIdx;
                String unit;
                int resolution;

                if (parts.length != 4) {
                    NormalizationType type = normalizationHandler.getNormTypeFromString(parts[0] + "_" + parts[1]);
                    strType = type.toString();
                    chrIdx = Integer.valueOf(parts[2]);
                    unit = parts[3];
                    resolution = Integer.valueOf(parts[4]);
                } else {
                    strType = parts[0];
                    chrIdx = Integer.valueOf(parts[1]);
                    unit = parts[2];
                    resolution = Integer.valueOf(parts[3]);
                }
                NormalizationVectorIndexEntry newEntry = new NormalizationVectorIndexEntry(strType, chrIdx, unit, resolution, entry.getValue().position, entry.getValue().size);
                normList.add(newEntry);
            }

            NormalizationVectorUpdater.writeNormIndex(buffer, normList);
            raf.write(buffer.getBytes());
        }
    }
}
