/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.original;

import htsjdk.tribble.util.LittleEndianOutputStream;
import org.broad.igv.feature.Chromosome;

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Utility class for converting an asscii "pair" file to a compact binary format.  Can greatly speed up calculation
 * of "hic" file
 * Input:   D0J8AACXX120130:6:1101:1003:8700/1 15 61559113 0 D0J8AACXX120130:6:1101:1003:8700/2 15 61559309 16
 * Output:  [chr1 index][pos1][chr2 index][pos 2]    (int, int, int, int)/
 *
 * @author Jim Robinson
 * @date 4/7/12
 */
public class AsciiToBinConverter {

    /**
     * @param inputPath
     * @param outputFile
     */
    public static void convert(String inputPath, String outputFile, List<Chromosome> chromosomes) throws IOException {

        Map<String, Integer> chromosomeOrdinals = new HashMap<String, Integer>();
        for (Chromosome c : chromosomes) {
            chromosomeOrdinals.put(c.getName(), c.getIndex());
        }

        AsciiPairIterator iter = null;
        BufferedOutputStream bos = null;
        try {
            bos = new BufferedOutputStream(new FileOutputStream(outputFile));
            LittleEndianOutputStream les = new LittleEndianOutputStream(bos);
            iter = new AsciiPairIterator(inputPath, chromosomeOrdinals
            );

            while (iter.hasNext()) {
                AlignmentPair pair = iter.next();
                les.writeBoolean(pair.getStrand1());
                les.writeInt(pair.getChr1());
                les.writeInt(pair.getPos1());
                les.writeInt(pair.getFrag1());
                les.writeBoolean(pair.getStrand2());
                les.writeInt(pair.getChr2());
                les.writeInt(pair.getPos2());
                les.writeInt(pair.getFrag2());
            }
            les.flush();
            bos.flush();
        } finally {
            if (iter != null) iter.close();
            if (bos != null) bos.close();

        }
    }

    public static void convertBack(String inputPath, String outputFile) throws IOException {
        PrintWriter pw = null;
        try {
            File f = new File(outputFile);
            FileWriter fw = new FileWriter(f);
            pw = new PrintWriter(fw);
            Map<String, Integer> chromosomeIndexMap = null;  // TODO
            BinPairIterator iter = new BinPairIterator(inputPath, chromosomeIndexMap);
            while (iter.hasNext()) {
                AlignmentPair pair = iter.next();
                pw.println(pair);
            }
        } finally {
            if (pw != null) {
                pw.close();
            }
        }
    }
}
