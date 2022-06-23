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
import juicebox.data.DatasetReaderV2;
import juicebox.data.ExpectedValueFunction;
import juicebox.data.ExpectedValueFunctionImpl;
import juicebox.data.basics.ListOfDoubleArrays;
import juicebox.data.basics.ListOfFloatArrays;
import juicebox.tools.utils.original.ExpectedValueCalculation;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.tdf.BufferedByteWriter;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class NormVectorUpdater {

    static void updateNormVectorIndexWithVector(List<NormalizationVectorIndexEntry> normVectorIndex, List<BufferedByteWriter> normVectorBufferList, ListOfFloatArrays vec,
                                                int chrIdx, NormalizationType type, HiCZoom zoom) throws IOException {
        long position = 0;
        for (int i=0; i < normVectorBufferList.size(); i++) {
            position += normVectorBufferList.get(i).bytesWritten();
        }

        putFloatArraysIntoBufferList(normVectorBufferList, vec.getValues());

        long newPos = 0;
        for (int i=0; i < normVectorBufferList.size(); i++) {
            newPos += normVectorBufferList.get(i).bytesWritten();
        }
        int sizeInBytes = (int) (newPos - position);
        normVectorIndex.add(new NormalizationVectorIndexEntry(type.toString(), chrIdx, zoom.getUnit().toString(), zoom.getBinSize(), position, sizeInBytes));
    }

    public static boolean isValidNormValue(float v) {
        return v > 0 && !Float.isNaN(v);
    }

    static void putFloatArraysIntoBufferList(List<BufferedByteWriter> bufferList, List<float[]> arrays) throws IOException {

        BufferedByteWriter buffer = getBufferWithEnoughSpace(bufferList, 8);
        long vectorLength = 0;
        for (float[] array : arrays) {
            vectorLength += array.length;
        }
        buffer.putLong(vectorLength);

        for (float[] array : arrays) {
            buffer = getBufferWithEnoughSpace(bufferList, 4 * array.length);
            for (float val : array) {
                buffer.putFloat(val);
            }
        }
    }

    private static void putMapValuesIntoBuffer(List<BufferedByteWriter> bufferList, Map<Integer, Double> hashmap) throws IOException {
        int bytesNeeded = 4 + (8 * hashmap.size());
        BufferedByteWriter buffer = getBufferWithEnoughSpace(bufferList, bytesNeeded);

        buffer.putInt(hashmap.size());

        List<Integer> keys = new ArrayList<>(hashmap.keySet());
        Collections.sort(keys);

        for (Integer key : keys) {
            buffer.putInt(key);
            buffer.putFloat(hashmap.get(key).floatValue());
        }
    }

    private static void writeExpectedToBuffer(RandomAccessFile raf, List<BufferedByteWriter> expectedBuffers, long filePosition) throws IOException {
        raf.getChannel().position(filePosition);
        for (int i = 0; i < expectedBuffers.size(); i++) {
            byte[] evBytes = expectedBuffers.get(i).getBytes();
            raf.write(evBytes);
        }
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
     * @param buffer          Buffer to write to
     * @param normVectorIndex Normalization index to write
     */
    static void writeNormIndex(BufferedByteWriter buffer, List<NormalizationVectorIndexEntry> normVectorIndex) throws IOException {
        buffer.putInt(normVectorIndex.size());
        for (NormalizationVectorIndexEntry entry : normVectorIndex) {
            buffer.putNullTerminatedString(entry.type);
            buffer.putInt(entry.chrIdx);
            buffer.putNullTerminatedString(entry.unit);
            buffer.putInt(entry.resolution);
            buffer.putLong(entry.position);
            buffer.putLong(entry.sizeInBytes);
        }
    }

    static void writeNormsToUpdateFile(DatasetReaderV2 reader, String path, boolean useCalcNotFunc,
                                       List<ExpectedValueCalculation> expectedValueCalculations,
                                       Map<String, ExpectedValueFunction> expectedValueFunctionMap,
                                       List<NormalizationVectorIndexEntry> normVectorIndices,
                                       List<BufferedByteWriter> normVectorBuffers, String message) throws IOException {
        int version = reader.getVersion();
        long filePosition = reader.getNormFilePosition();
        long nviHeaderPosition = reader.getNviHeaderPosition();


        try (RandomAccessFile raf = new RandomAccessFile(path, "rw")) {
            handleVersionSix(raf, version);
            List<BufferedByteWriter> expectedBuffers = new ArrayList<>();
            expectedBuffers.add(new BufferedByteWriter());

            if (useCalcNotFunc) {
                writeExpectedValues(expectedBuffers, expectedValueCalculations);
            } else {
                writeExpectedValues(expectedBuffers, expectedValueFunctionMap);
            }

            writeExpectedToBuffer(raf, expectedBuffers, filePosition);
            writeNormsToBuffer(raf, normVectorIndices, normVectorBuffers, nviHeaderPosition);
        }

        System.out.println(message);
    }

    private static void writeExpectedValues(List<BufferedByteWriter> expectedBuffers, List<ExpectedValueCalculation> expectedValueCalculations) throws IOException {
        BufferedByteWriter buffer = getBufferWithEnoughSpace(expectedBuffers, 4);
        buffer.putInt(expectedValueCalculations.size());

        for (ExpectedValueCalculation ev : expectedValueCalculations) {
            ev.computeDensity();
            HiC.Unit unit = ev.isFrag ? HiC.Unit.FRAG : HiC.Unit.BP;
            appendExpectedValuesToBuffer(expectedBuffers, ev.getType(),
                    unit, ev.getGridSize(), ev.getDensityAvg(),
                    ev.getChrScaleFactors());
        }
    }

    private static void writeExpectedValues(List<BufferedByteWriter> expectedBuffers, Map<String, ExpectedValueFunction> expectedValueFunctionMap) throws IOException {

        BufferedByteWriter buffer = getBufferWithEnoughSpace(expectedBuffers, 4);
        buffer.putInt(expectedValueFunctionMap.size());

        for (ExpectedValueFunction function : expectedValueFunctionMap.values()) {
            appendExpectedValuesToBuffer(expectedBuffers, function.getNormalizationType(),
                    function.getUnit(), function.getBinSize(),
                    function.getExpectedValuesNoNormalization(),
                    ((ExpectedValueFunctionImpl) function).getNormFactors());
        }
    }

    private static void appendExpectedValuesToBuffer(List<BufferedByteWriter> expectedBuffers,
                                                     NormalizationType normalizationType,
                                                     HiC.Unit unit, int binSize,
                                                     ListOfDoubleArrays expectedValuesNoNormalization,
                                                     Map<Integer, Double> normFactors) throws IOException {

        int bytesNeeded = normalizationType.toString().length() + 1;
        bytesNeeded += unit.toString().length() + 1;
        bytesNeeded += 4;

        BufferedByteWriter buffer = getBufferWithEnoughSpace(expectedBuffers, bytesNeeded);
        buffer.putNullTerminatedString(normalizationType.toString());
        buffer.putNullTerminatedString(unit.toString());
        buffer.putInt(binSize);

        putFloatArraysIntoBufferList(expectedBuffers,
                expectedValuesNoNormalization.convertToFloats().getValues());

        putMapValuesIntoBuffer(expectedBuffers, normFactors);
    }

    private static BufferedByteWriter getBufferWithEnoughSpace(List<BufferedByteWriter> expectedBuffers, int bytesNeeded) {
        if (expectedBuffers.size()==0) {
            expectedBuffers.add(new BufferedByteWriter());
        }

        BufferedByteWriter buffer = expectedBuffers.get(expectedBuffers.size() - 1);
        int freeBytes = Integer.MAX_VALUE - 10 - buffer.bytesWritten();

        if (bytesNeeded >= freeBytes) {
            expectedBuffers.add(new BufferedByteWriter());
            return expectedBuffers.get(expectedBuffers.size() - 1);
        }
        return buffer;
    }

    private static void writeNormsToBuffer(RandomAccessFile raf, List<NormalizationVectorIndexEntry> normVectorIndex,
                                           List<BufferedByteWriter> normVectorBuffers, long nviHeaderPosition) throws IOException {
        // Get the size of the index in bytes, to compute an offset for the actual entries.
        BufferedByteWriter buffer = new BufferedByteWriter();
        writeNormIndex(buffer, normVectorIndex);
        long normVectorStartPosition = raf.getChannel().position() + buffer.bytesWritten();
        long size = buffer.bytesWritten();
        long NVI = normVectorStartPosition - size;
        // write NVI, size
        raf.getChannel().position(nviHeaderPosition);

        BufferedByteWriter headerBuffer = new BufferedByteWriter();
        headerBuffer.putLong(NVI);
        headerBuffer.putLong(size);
        raf.write(headerBuffer.getBytes());

        // reset pointer to where we were
        raf.getChannel().position(NVI);

        // Update index entries
        for (NormalizationVectorIndexEntry entry : normVectorIndex) {
            entry.position += normVectorStartPosition;
        }

        // Now write for real
        buffer = new BufferedByteWriter();
        writeNormIndex(buffer, normVectorIndex);
        raf.write(buffer.getBytes());
        // Finally the norm vectors
        for (int i = 0; i < normVectorBuffers.size(); i++) {
            raf.write(normVectorBuffers.get(i).getBytes());
        }
    }
}
