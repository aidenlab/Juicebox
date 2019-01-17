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

        List<HiCZoom> resolutions = new ArrayList<>();
        resolutions.addAll(ds.getBpZooms());
        resolutions.addAll(ds.getFragZooms());

        // Keep track of chromosomes that fail to converge, so we don't try them at higher resolutions.
        Set<Chromosome> krBPFailedChromosomes = new HashSet<>();
        Set<Chromosome> krFragFailedChromosomes = new HashSet<>();

        BufferedByteWriter normVectorBuffer = new BufferedByteWriter();
        List<NormalizationVectorIndexEntry> normVectorIndices = new ArrayList<>();
        List<ExpectedValueCalculation> expectedValueCalculations = new ArrayList<>();


        // Loop through resolutions
        for (HiCZoom zoom : resolutions) {

            // Optionally compute genome-wide normalizaton
            if (genomeWideResolution > 0 && zoom.getUnit() == HiC.Unit.BP && zoom.getBinSize() >= genomeWideResolution) {
                GenomeWideNormalizationVectorUpdater.updateHicFileForGW(ds, zoom, normVectorIndices, normVectorBuffer, expectedValueCalculations);
            }
            System.out.println();
            System.out.print("Calculating norms for zoom " + zoom);

            // Integer is either limit on genome wide resolution or limit on what fragment resolution to calculate
            if (noFrag && zoom.getUnit() == HiC.Unit.FRAG) continue;

            Set<Chromosome> failureSet = zoom.getUnit() == HiC.Unit.FRAG ? krFragFailedChromosomes : krBPFailedChromosomes;

            Map<String, Integer> fcm = zoom.getUnit() == HiC.Unit.FRAG ? fragCountMap : null;

            ExpectedValueCalculation evVC = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationType.VC);
            ExpectedValueCalculation evVCSqrt = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationType.VC_SQRT);
            ExpectedValueCalculation evKR = new ExpectedValueCalculation(chromosomeHandler, zoom.getBinSize(), fcm, NormalizationType.KR);

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

                updateExpectedValueCalculationForChr(chrIdx, nc, vc, NormalizationType.VC, zoom, zd, evVC, normVectorBuffer, normVectorIndices);
                updateExpectedValueCalculationForChr(chrIdx, nc, vcSqrt, NormalizationType.VC_SQRT, zoom, zd, evVCSqrt, normVectorBuffer, normVectorIndices);

                if (HiCGlobals.printVerboseComments) {
                    System.out.println("\nVC normalization of " + chr + " at " + zoom + " took " + (System.currentTimeMillis() - currentTime) + " milliseconds");
                }
                currentTime = System.currentTimeMillis();
                // KR normalization


                if (!failureSet.contains(chr)) {

                    double[] kr = nc.computeKR();
                    if (kr == null) {
                        failureSet.add(chr);
                    } else {

                        updateExpectedValueCalculationForChr(chrIdx, nc, kr, NormalizationType.KR, zoom, zd, evKR, normVectorBuffer, normVectorIndices);

                    }
                    if (HiCGlobals.printVerboseComments) {
                        System.out.println("KR normalization of " + chr + " at " + zoom + " took " + (System.currentTimeMillis() - currentTime) + " milliseconds");
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
        }

        int version = reader.getVersion();
        long filePosition = reader.getNormFilePosition();
        reader.close();
        System.out.println();
        update(path, version, filePosition, expectedValueCalculations, normVectorIndices,
                normVectorBuffer.getBytes());
        System.out.println("Finished writing norms");
    }


    private static void updateExpectedValueCalculationForChr(final int chrIdx, NormalizationCalculations nc, double[] vec, NormalizationType type, HiCZoom zoom, MatrixZoomData zd,
                                                             ExpectedValueCalculation ev, BufferedByteWriter normVectorBuffer, List<NormalizationVectorIndexEntry> normVectorIndex) throws IOException {
        double factor = nc.getSumFactor(vec);
        for (int i = 0; i < vec.length; i++) {
            vec[i] = vec[i] * factor;
        }

        updateNormVectorIndexWithVector(normVectorIndex, normVectorBuffer, vec, chrIdx, type, zoom);

        Iterator<ContactRecord> iter = zd.contactRecordIterator();
        // TODO: this is inefficient, we have all of the contact records when we leave normcalculations, should do this there if possible
        while (iter.hasNext()) {
            ContactRecord cr = iter.next();
            int x = cr.getBinX();
            int y = cr.getBinY();
            final float counts = cr.getCounts();
            if (isValidNormValue(vec[x]) & isValidNormValue(vec[y])) {
                double value = counts / (vec[x] * vec[y]);
                ev.addDistance(chrIdx, x, y, value);
            }
        }
    }

    private static Map<String,NormalizationVector> readVectorFile(String fname, ChromosomeHandler chromosomeHandler) throws IOException {
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
            }
            else {
                System.err.println("Chromosome null"); // this shouldn't happen due to continue above
            }
        }

        return normVectors;
    }

    public static void updateHicFile(String path, String vectorPath) throws IOException {
        DatasetReaderV2 reader = new DatasetReaderV2(path);
        Dataset ds = reader.read();
        HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());

        ChromosomeHandler chromosomeHandler = ds.getChromosomeHandler();
        Map<String, NormalizationVector> normVectors =  readVectorFile(vectorPath, chromosomeHandler);

        // chr -> frag count map.  Needed for expected value calculations
        Map<String, Integer> fragCountMap = ds.getFragmentCounts();

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
                        normVectorIndex.add(new NormalizationVectorIndexEntry(
                                type.toString(), chr.getIndex(), zoom.getUnit().toString(), zoom.getBinSize(), position, sizeInBytes));
                    }
                }

                handleLoadedVector(chr.getIndex(), zoom, normVectors, normVectorBuffer, normVectorIndex, zd, evLoaded);
            }

            String key = zoom.getUnit().toString() + zoom.getBinSize() + "_" + NormalizationType.LOADED;
            expectedValueFunctionMap.put(key, evLoaded.getExpectedValueFunction());
        }

        int version = reader.getVersion();
        long filePosition = reader.getNormFilePosition();
        reader.close();
        System.out.println();
        update(path, version, filePosition, expectedValueFunctionMap, normVectorIndex,
                normVectorBuffer.getBytes());
        System.out.println("Finished adding normalizations.");
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


    private static void writeNormsToBuffer(BufferedByteWriter buffer, RandomAccessFile raf, List<NormalizationVectorIndexEntry> normVectorIndex, byte[] normVectorBuffer) throws IOException {
        System.out.println("Writing norms");
        // Get the size of the index in bytes, to compute an offset for the actual entries.
        buffer = new BufferedByteWriter();
        writeNormIndex(buffer, normVectorIndex);
        long normVectorStartPosition = raf.getChannel().position() + buffer.bytesWritten();

        // Update index entries
        for (NormalizationVectorIndexEntry entry : normVectorIndex) {
            entry.position += normVectorStartPosition;
        }

        // Now write for real
        buffer = new BufferedByteWriter();
        writeNormIndex(buffer, normVectorIndex);
        raf.write(buffer.getBytes());

        // Finally the norm vectors
        raf.write(normVectorBuffer);
    }


    private static void writeExpectedToBuffer(RandomAccessFile raf, BufferedByteWriter buffer, long filePosition) throws IOException {
        byte[] evBytes = buffer.getBytes();
        raf.getChannel().position(filePosition);
        raf.write(evBytes);
    }

    private static void handleVersionSix(RandomAccessFile raf, int version) throws IOException {
        if (version < 6) {
            // Update version
            // Master index
            raf.getChannel().position(4);
            BufferedByteWriter buffer = new BufferedByteWriter();
            buffer.putInt(6);
            raf.write(buffer.getBytes());
        }
    }

    /**
     * Compute the size of the index in bytes.  This is needed to set offsets for the actual index entries.  The
     * easiest way to do this is to write it to a buffer and check the size
     *
     * @param buffer Buffer to write to
     * @param normVectorIndex  Normalization index to write
     */
    public static void writeNormIndex(BufferedByteWriter buffer, List<NormalizationVectorIndexEntry> normVectorIndex) throws IOException {
        buffer.putInt(normVectorIndex.size());
        for (NormalizationVectorIndexEntry entry : normVectorIndex) {
            buffer.putNullTerminatedString(entry.type);
            buffer.putInt(entry.chrIdx);
            buffer.putNullTerminatedString(entry.unit);
            buffer.putInt(entry.resolution);
            buffer.putLong(entry.position);
            buffer.putInt(entry.sizeInBytes);
        }
    }

    private static void update(String hicfile, int version, final long filePosition, List<ExpectedValueCalculation> expectedValueCalculations,
                               List<NormalizationVectorIndexEntry> normVectorIndex, byte[] normVectorBuffer) throws IOException {

        try (RandomAccessFile raf = new RandomAccessFile(hicfile, "rw")) {
            handleVersionSix(raf, version);
            BufferedByteWriter buffer = new BufferedByteWriter();
            System.out.println("Writing expected");
            writeExpectedValues(buffer, expectedValueCalculations);
            writeExpectedToBuffer(raf, buffer, filePosition);
            writeNormsToBuffer(buffer, raf, normVectorIndex, normVectorBuffer);
        }
    }

    static void update(String hicfile, int version, final long filePosition, Map<String, ExpectedValueFunction> expectedValueFunctionMap,
                       List<NormalizationVectorIndexEntry> normVectorIndex, byte[] normVectorBuffer) throws IOException {

        try (RandomAccessFile raf = new RandomAccessFile(hicfile, "rw")) {
            handleVersionSix(raf, version);
            BufferedByteWriter buffer = new BufferedByteWriter();
            System.out.println("Writing expected");
            writeExpectedValues(buffer, expectedValueFunctionMap);
            writeExpectedToBuffer(raf, buffer, filePosition);
            writeNormsToBuffer(buffer, raf, normVectorIndex, normVectorBuffer);
        }
    }


    private static void writeExpectedValues(BufferedByteWriter buffer, List<ExpectedValueCalculation> expectedValueCalculations) throws IOException {

        buffer.putInt(expectedValueCalculations.size());
        for (ExpectedValueCalculation ev : expectedValueCalculations) {
            ev.computeDensity();
            buffer.putNullTerminatedString(ev.getType().toString());

            HiC.Unit unit = ev.isFrag ? HiC.Unit.FRAG : HiC.Unit.BP;
            buffer.putNullTerminatedString(unit.toString());

            buffer.putInt(ev.getGridSize());
            putArrayValuesIntoBuffer(buffer, ev.getDensityAvg());
            putMapValuesIntoBuffer(buffer, ev.getChrScaleFactors());
        }
    }

    private static void writeExpectedValues(BufferedByteWriter buffer, Map<String, ExpectedValueFunction> expectedValueFunctionMap) throws IOException {
        buffer.putInt(expectedValueFunctionMap.size());
        for (ExpectedValueFunction function : expectedValueFunctionMap.values()) {
            buffer.putNullTerminatedString(function.getNormalizationType().toString());
            buffer.putNullTerminatedString(function.getUnit().toString());

            buffer.putInt(function.getBinSize());
            putArrayValuesIntoBuffer(buffer, function.getExpectedValues());
            putMapValuesIntoBuffer(buffer, ((ExpectedValueFunctionImpl) function).getNormFactors());
        }
    }
}
