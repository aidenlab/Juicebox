/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
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
import org.broad.igv.util.Pair;

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
public class NormalizationVectorUpdater {

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
        List<NormalizationVectorIndexEntry> normVectorIndex = new ArrayList<>();
        List<ExpectedValueCalculation> expectedValueCalculations = new ArrayList<>();


        // Loop through resolutions
        for (HiCZoom zoom : resolutions) {

            // Optionally compute genome-wide normalizaton
            if (genomeWideResolution > 0 && zoom.getUnit() == HiC.Unit.BP && zoom.getBinSize() >= genomeWideResolution) {

                // do all four genome-wide normalizations
                NormalizationType[] types = {NormalizationType.GW_KR, NormalizationType.GW_VC,
                        NormalizationType.INTER_KR, NormalizationType.INTER_VC};

                for (NormalizationType normType : types) {

                    Pair<Map<Chromosome, NormalizationVector>, ExpectedValueCalculation> wgVectors = getWGVectors(ds, zoom, normType);

                    Map<Chromosome, NormalizationVector> nvMap = wgVectors.getFirst();
                    for (Chromosome chromosome : nvMap.keySet()) {

                        NormalizationVector nv = nvMap.get(chromosome);


                        int position = normVectorBuffer.bytesWritten();
                        writeNormalizationVector(normVectorBuffer, nv.getData());

                        int sizeInBytes = normVectorBuffer.bytesWritten() - position;
                        normVectorIndex.add(new NormalizationVectorIndexEntry(
                                normType.toString(), chromosome.getIndex(), zoom.getUnit().toString(), zoom.getBinSize(), position, sizeInBytes));
                    }

                    expectedValueCalculations.add(wgVectors.getSecond());
                }

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

                updateExpectedValueCalculationForChr(chrIdx, nc, vc, NormalizationType.VC, zoom, zd, evVC, normVectorBuffer, normVectorIndex);
                updateExpectedValueCalculationForChr(chrIdx, nc, vcSqrt, NormalizationType.VC_SQRT, zoom, zd, evVCSqrt, normVectorBuffer, normVectorIndex);

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

                        updateExpectedValueCalculationForChr(chrIdx, nc, kr, NormalizationType.KR, zoom, zd, evKR, normVectorBuffer, normVectorIndex);

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
        update(path, version, filePosition, expectedValueCalculations, normVectorIndex,
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

    private static void updateNormVectorIndexWithVector(List<NormalizationVectorIndexEntry> normVectorIndex, BufferedByteWriter normVectorBuffer, double[] vec,
                                                        int chrIdx, NormalizationType type, HiCZoom zoom) throws IOException {
        int position = normVectorBuffer.bytesWritten();
        writeNormalizationVector(normVectorBuffer, vec);
        int sizeInBytes = normVectorBuffer.bytesWritten() - position;
        normVectorIndex.add(new NormalizationVectorIndexEntry(type.toString(), chrIdx, zoom.getUnit().toString(), zoom.getBinSize(), position, sizeInBytes));

    }


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

                // do all four genome-wide normalizations
                NormalizationType[] types = {NormalizationType.GW_KR, NormalizationType.GW_VC,
                        NormalizationType.INTER_KR, NormalizationType.INTER_VC};

                for (NormalizationType normType : types) {

                    Pair<Map<Chromosome, NormalizationVector>, ExpectedValueCalculation> wgVectors = getWGVectors(ds, zoom, normType);

                    Map<Chromosome, NormalizationVector> nvMap = wgVectors.getFirst();
                    for (Chromosome chromosome : nvMap.keySet()) {

                        NormalizationVector nv = nvMap.get(chromosome);

                        int position = normVectorBuffer.bytesWritten();
                        writeNormalizationVector(normVectorBuffer, nv.getData());

                        int sizeInBytes = normVectorBuffer.bytesWritten() - position;
                        normVectorIndex.add(new NormalizationVectorIndexEntry(
                                normType.toString(), chromosome.getIndex(), zoom.getUnit().toString(), zoom.getBinSize(), position, sizeInBytes));
                    }
                    ExpectedValueCalculation calculation = wgVectors.getSecond();
                    String key = "BP_" + zoom.getBinSize() + "_" + normType;
                    expectedValueFunctionMap.put(key, calculation.getExpectedValueFunction());
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


                NormalizationVector vc = ds.getNormalizationVector(chr.getIndex(), zoom, NormalizationType.VC);
                NormalizationVector vcSqrt = ds.getNormalizationVector(chr.getIndex(), zoom, NormalizationType.VC_SQRT);
                NormalizationVector kr = ds.getNormalizationVector(chr.getIndex(), zoom, NormalizationType.KR);

                int position = normVectorBuffer.bytesWritten();
                writeNormalizationVector(normVectorBuffer, vc.getData());
                int sizeInBytes = normVectorBuffer.bytesWritten() - position;
                normVectorIndex.add(new NormalizationVectorIndexEntry(
                        NormalizationType.VC.toString(), chr.getIndex(), zoom.getUnit().toString(), zoom.getBinSize(), position, sizeInBytes));

                position = normVectorBuffer.bytesWritten();
                writeNormalizationVector(normVectorBuffer, vcSqrt.getData());
                sizeInBytes = normVectorBuffer.bytesWritten() - position;
                normVectorIndex.add(new NormalizationVectorIndexEntry(
                        NormalizationType.VC_SQRT.toString(), chr.getIndex(), zoom.getUnit().toString(), zoom.getBinSize(), position, sizeInBytes));

                // KR normalization
                if (kr != null) {
                    position = normVectorBuffer.bytesWritten();
                    writeNormalizationVector(normVectorBuffer, kr.getData());
                    sizeInBytes = normVectorBuffer.bytesWritten() - position;
                    final NormalizationVectorIndexEntry normalizationVectorIndexEntry1 =
                            new NormalizationVectorIndexEntry(NormalizationType.KR.toString(), chr.getIndex(), zoom.getUnit().toString(),
                                    zoom.getBinSize(), position, sizeInBytes);
                    normVectorIndex.add(normalizationVectorIndexEntry1);
                }
            }
        }


        int version = reader.getVersion();
        long filePosition = reader.getNormFilePosition();
        reader.close();
        System.out.println();
        update(path, version, filePosition, expectedValueFunctionMap, normVectorIndex,
                normVectorBuffer.getBytes());
        System.out.println("Finished normalization");
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
                        writeNormalizationVector(normVectorBuffer, existingNorm.getData());
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
        writeNormalizationVector(normVectorBuffer, vector.getData());

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


    private static void update(String hicfile, int version, final long filePosition, List<ExpectedValueCalculation> expectedValueCalculations,
                               List<NormalizationVectorIndexEntry> normVectorIndex, byte[] normVectorBuffer) throws IOException {

        try (RandomAccessFile raf = new RandomAccessFile(hicfile, "rw")) {

            if (version < 6) {
                // Update version
                // Master index
                raf.getChannel().position(4);
                BufferedByteWriter buffer = new BufferedByteWriter();
                buffer.putInt(6);
                raf.write(buffer.getBytes());
            }

            BufferedByteWriter buffer = new BufferedByteWriter();
            System.out.println("Writing expected");
            writeExpectedValues(buffer, expectedValueCalculations);

            byte[] evBytes = buffer.getBytes();
            raf.getChannel().position(filePosition);
            raf.write(evBytes);

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
    }

    private static void update(String hicfile,
                               int version,
                               final long filePosition,
                               Map<String, ExpectedValueFunction> expectedValueFunctionMap,
                               List<NormalizationVectorIndexEntry> normVectorIndex,
                               byte[] normVectorBuffer) throws IOException {

        try (RandomAccessFile raf = new RandomAccessFile(hicfile, "rw")) {

            if (version < 6) {
                // Update version
                // Master index
                raf.getChannel().position(4);
                BufferedByteWriter buffer = new BufferedByteWriter();
                buffer.putInt(6);
                raf.write(buffer.getBytes());
            }

            BufferedByteWriter buffer = new BufferedByteWriter();
            System.out.println("Writing expected");
            writeExpectedValues(buffer, expectedValueFunctionMap);

            byte[] evBytes = buffer.getBytes();
            raf.getChannel().position(filePosition);
            raf.write(evBytes);

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


    private static void writeExpectedValues(BufferedByteWriter buffer, List<ExpectedValueCalculation> expectedValueCalculations) throws IOException {

        buffer.putInt(expectedValueCalculations.size());
        for (ExpectedValueCalculation ev : expectedValueCalculations) {

            ev.computeDensity();

            buffer.putNullTerminatedString(ev.getType().toString());

            int binSize = ev.getGridSize();
            HiC.Unit unit = ev.isFrag ? HiC.Unit.FRAG : HiC.Unit.BP;

            buffer.putNullTerminatedString(unit.toString());
            buffer.putInt(binSize);

            // The density values
            double[] expectedValues = ev.getDensityAvg();
            buffer.putInt(expectedValues.length);
            for (double d : expectedValues) {
                buffer.putDouble(d);
            }

            // Map of chromosome index -> normalization factor
            Map<Integer, Double> normalizationFactors = ev.getChrScaleFactors();
            buffer.putInt(normalizationFactors.size());
            for (Map.Entry<Integer, Double> normFactor : normalizationFactors.entrySet()) {
                buffer.putInt(normFactor.getKey());
                buffer.putDouble(normFactor.getValue());
            }
        }
    }

    private static void writeExpectedValues(BufferedByteWriter buffer, Map<String, ExpectedValueFunction> expectedValueFunctionMap) throws IOException {

        buffer.putInt(expectedValueFunctionMap.size());

        for (ExpectedValueFunction function : expectedValueFunctionMap.values()) {
            buffer.putNullTerminatedString(function.getNormalizationType().toString());
            buffer.putNullTerminatedString(function.getUnit().toString());
            buffer.putInt(function.getBinSize());
            double[] expectedValues = function.getExpectedValues();
            buffer.putInt(expectedValues.length);
            for (double d : expectedValues) {
                buffer.putDouble(d);
            }
            Map<Integer, Double> normalizationFactors = ((ExpectedValueFunctionImpl) function).getNormFactors();
            buffer.putInt(normalizationFactors.size());
            for (Map.Entry<Integer, Double> normFactor : normalizationFactors.entrySet()) {
                buffer.putInt(normFactor.getKey());
                buffer.putDouble(normFactor.getValue());
            }
        }
    }


    private static void writeNormalizationVector(BufferedByteWriter buffer, double[] values) throws IOException {
        buffer.putInt(values.length);
        for (double value : values) buffer.putDouble(value);
    }


    /**
     * Compute the whole-genome normalization and expected value vectors and return as a pair (normalization vector first)
     */

    private static Pair<Map<Chromosome, NormalizationVector>, ExpectedValueCalculation> getWGVectors(Dataset dataset,
                                                                                                     HiCZoom zoom,
                                                                                                     NormalizationType norm) {

        boolean includeIntra = false;
        if (norm == NormalizationType.GW_KR || norm == NormalizationType.GW_VC) {
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


        ExpectedValueCalculation expectedValueCalculation = new ExpectedValueCalculation(chromosomeHandler, resolution, null, norm);
        int addY = 0;
        // Loop through chromosomes
        for (Chromosome chr : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
            final int chrIdx = chr.getIndex();
            Matrix matrix = dataset.getMatrix(chr, chr);

            if (matrix == null) continue;
            MatrixZoomData zd = matrix.getZoomData(zoom);
            Iterator<ContactRecord> iter = zd.contactRecordIterator();
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

    private static boolean isValidNormValue(double v) {
        return v > 0 && !Double.isNaN(v);
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
                            Iterator<ContactRecord> iter = zd.contactRecordIterator();
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


}
