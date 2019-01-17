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
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.tdf.BufferedByteWriter;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.GZIPInputStream;

public class CustomNormVectorFileHandler extends NormVectorUpdater {

    public static void updateHicFile(String path, String vectorPath) throws IOException {
        DatasetReaderV2 reader = new DatasetReaderV2(path);
        Dataset ds = reader.read();
        HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());

        ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();
        Map<String, NormalizationVector> normVectors = readVectorFile(vectorPath, chromosomeHandler);

        // chr -> frag count map.  Needed for expected value calculations
        Map<String, Integer> fragCountMap = ds.getFragmentCounts();

        List<HiCZoom> resolutions = ds.getAllPossibleResolutions();

        BufferedByteWriter normVectorBuffer = new BufferedByteWriter();
        List<NormalizationVectorIndexEntry> normVectorIndices = new ArrayList<>();
        Map<String, ExpectedValueFunction> expectedValueFunctionMap = ds.getExpectedValueFunctionMap();

        for (Iterator<Map.Entry<String, ExpectedValueFunction>> it = expectedValueFunctionMap.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<String, ExpectedValueFunction> entry = it.next();
            if (entry.getKey().contains("NONE")) {
                it.remove();
            }
        }

        // Loop through resolutions
        for (HiCZoom zoom : resolutions) {

            Map<String, Integer> fcm = zoom.getUnit() == HiC.Unit.FRAG ? fragCountMap : null;
            ExpectedValueCalculation evLoaded = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationType.LOADED);

            // Loop through chromosomes
            for (Chromosome chr : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {

                Matrix matrix = ds.getMatrix(chr, chr);

                if (matrix == null) continue;
                MatrixZoomData zd = matrix.getZoomData(zoom);

                // Get existing norm vectors so we don't lose them
                for (NormalizationType type : NormalizationType.values()) {
                    NormalizationVector existingNorm = ds.getNormalizationVector(chr.getIndex(), zoom, type);
                    if (existingNorm != null) {

                        //System.out.println(type.getLabel() + " normalization for chromosome " + chr.getName() + " at " + zoom.getBinSize() + " " + zoom.getUnit() + " resolution already exists.  Keeping this normalization vector.");

                        int position = normVectorBuffer.bytesWritten();
                        putArrayValuesIntoBuffer(normVectorBuffer, existingNorm.getData());
                        int sizeInBytes = normVectorBuffer.bytesWritten() - position;
                        normVectorIndices.add(new NormalizationVectorIndexEntry(
                                type.toString(), chr.getIndex(), zoom.getUnit().toString(), zoom.getBinSize(), position, sizeInBytes));
                    }
                }

                handleLoadedVector(chr.getIndex(), zoom, normVectors, normVectorBuffer, normVectorIndices, zd, evLoaded);
            }

            String key = zoom.getUnit().toString() + zoom.getBinSize() + "_" + NormalizationType.LOADED;
            expectedValueFunctionMap.put(key, evLoaded.getExpectedValueFunction());
        }

        writeNormsToUpdateFile(reader, path, false, null, expectedValueFunctionMap, normVectorIndices,
                normVectorBuffer, "Finished adding normalizations.");
    }

    private static void handleLoadedVector(final int chrIndx, HiCZoom zoom, Map<String, NormalizationVector> normVectors,
                                           BufferedByteWriter normVectorBuffer, List<NormalizationVectorIndexEntry> normVectorIndex,
                                           MatrixZoomData zd, ExpectedValueCalculation evLoaded) throws IOException {

        String key = NormalizationType.LOADED + "_" + chrIndx + "_" + zoom.getUnit() + "_" + zoom.getBinSize();
        NormalizationVector vector = normVectors.get(key);
        if (vector == null) return;
        // Write loaded norm
        int position = normVectorBuffer.bytesWritten();
        putArrayValuesIntoBuffer(normVectorBuffer, vector.getData());

        int sizeInBytes = normVectorBuffer.bytesWritten() - position;
        normVectorIndex.add(new NormalizationVectorIndexEntry(
                NormalizationType.LOADED.toString(), chrIndx, zoom.getUnit().toString(), zoom.getBinSize(), position, sizeInBytes));

        // Calculate the expected values
        Iterator<ContactRecord> iter = zd.contactRecordIterator();

        while (iter.hasNext()) {
            ContactRecord cr = iter.next();
            int x = cr.getBinX();
            int y = cr.getBinY();
            final float counts = cr.getCounts();
            if (isValidNormValue(vector.getData()[x]) & isValidNormValue(vector.getData()[y])) {
                double value = counts / (vector.getData()[x] * vector.getData()[y]);
                evLoaded.addDistance(chrIndx, x, y, value);
            }

        }
    }

    protected static Map<String, NormalizationVector> readVectorFile(String fname, ChromosomeHandler chromosomeHandler) throws IOException {
        BufferedReader vectorReader;
        if (fname.endsWith(".gz")) {
            InputStream fileStream = new FileInputStream(fname);
            InputStream gzipStream = new GZIPInputStream(fileStream);
            Reader decoder = new InputStreamReader(gzipStream, StandardCharsets.UTF_8);
            vectorReader = new BufferedReader(decoder, 4194304);
        } else {
            //this.reader = org.broad.igv.util.ParsingUtils.openBufferedReader(path);
            vectorReader = new BufferedReader(new InputStreamReader(new FileInputStream(fname)), HiCGlobals.bufferSize);
        }

        Map<String, NormalizationVector> normVectors = new HashMap<>();
        Chromosome chr = null;
        int resolution = -1;
        HiC.Unit unit = null;

        String nextLine = vectorReader.readLine();
        while (nextLine != null) {
            // Header: vector  type  chr1    2048000 BP
            if (nextLine.startsWith("vector")) {
                String[] tokens = nextLine.split("\\s+");
                chr = chromosomeHandler.getChromosomeFromName(tokens[2]);
                if (chr == null) {
                    System.err.println("Skipping " + tokens[2] + " which isn't in dataset");
                    nextLine = vectorReader.readLine();
                    // List<Double> data = new ArrayList<Double>();
                    while (nextLine != null && !(nextLine.startsWith("vector"))) {
                        nextLine = vectorReader.readLine();
                    }
                    continue;
                }
                String normType;   // we're going to ignore this for now; need to have a way to add to enums.

                // TODO: the normalization type should be read in; but need to modify other code for that
                normType = tokens[1];     // this is ignored
                resolution = Integer.valueOf(tokens[3]);
                unit = HiC.Unit.valueOf(tokens[4]);
            }
            if (chr != null) {
                System.out.println("Adding normalization for chromosome " + chr.getName() + " at " + resolution + " " + unit + " resolution.");


                // Now do work on loaded norm vector
                // Create the new vector by looping through the loaded vector file line by line
                int size = chr.getLength() / resolution + 1;
                double[] data = new double[size];
                int i = 0;
                nextLine = vectorReader.readLine();
                // List<Double> data = new ArrayList<Double>();
                while (nextLine != null && !(nextLine.startsWith("vector"))) {
                    if (nextLine.toLowerCase().equals("nan") || nextLine.equals(".")) {
                        data[i] = Double.NaN;
                    } else data[i] = Double.valueOf(nextLine);
                    i++;
                    if (i > size) {
                        throw new IOException("More values than resolution would indicate");
                    }
                    nextLine = vectorReader.readLine();
                }
                NormalizationVector vector = new NormalizationVector(NormalizationType.LOADED, chr.getIndex(), unit, resolution, data);
                normVectors.put(vector.getKey(), vector);
            } else {
                System.err.println("Chromosome null"); // this shouldn't happen due to continue above
            }
        }

        return normVectors;
    }
}
