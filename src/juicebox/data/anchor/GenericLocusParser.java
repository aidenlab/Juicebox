/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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

package juicebox.data.anchor;

import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.basics.Chromosome;
import juicebox.data.feature.GenomeWideList;
import org.broad.igv.Globals;
import org.broad.igv.util.ParsingUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class GenericLocusParser {
    /**
     * @param handler
     * @param bedFilePath
     * @return List of motif anchors from the provided bed file
     */
    public static GenomeWideList<GenericLocus> loadFromBEDFile(ChromosomeHandler handler, String bedFilePath) {
        List<GenericLocus> anchors = new ArrayList<>();

        try {
            //BufferedReader br = ParsingUtils.openBufferedReader(bedFilePath);
            BufferedReader br = new BufferedReader(new InputStreamReader(ParsingUtils.openInputStream(bedFilePath)), HiCGlobals.bufferSize);
            anchors.addAll(parseBEDFile(br, handler));
        } catch (IOException ec) {
            ec.printStackTrace();
        }

        return new GenomeWideList<>(handler, anchors);
    }

    /**
     * Methods for handling BED Files
     */

    /**
     * Helper function for actually parsing BED file
     * Ignores any attributes beyond the third column (i.e. just chr and positions are read)
     *
     * @param bufferedReader
     * @param handler
     * @return list of motifs
     * @throws IOException
     */
    private static List<GenericLocus> parseBEDFile(BufferedReader bufferedReader, ChromosomeHandler handler) throws IOException {
        Set<GenericLocus> anchors = new HashSet<>();
        String nextLine;

        int errorCount = 0;
        while ((nextLine = bufferedReader.readLine()) != null) {
            String[] tokens = Globals.tabPattern.split(nextLine);

            String chr1Name;
            int start1, end1;

            if (tokens[0].startsWith("chr") && tokens.length > 2) {
                // valid line
                chr1Name = tokens[0];
                start1 = Integer.parseInt(tokens[1]);
                end1 = Integer.parseInt(tokens[2]);


                Chromosome chr = handler.getChromosomeFromName(chr1Name);
                if (chr == null) {
                    if (errorCount < 10) {
                        System.out.println("Skipping line: " + nextLine);
                    } else if (errorCount == 10) {
                        System.err.println("Maximum error count exceeded.  Further errors will not be logged");
                    }

                    errorCount++;
                    continue;
                }

                anchors.add(new GenericLocus(chr.getName(), start1, end1));
            }
        }
        if (anchors.size() < 1) System.err.println("BED File empty - file may have problems or error was encountered");
        bufferedReader.close();
        return new ArrayList<>(anchors);
    }
}
