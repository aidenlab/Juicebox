/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2014 Broad Institute, Aiden Lab
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

import org.broad.igv.bbfile.BBFileReader;
import org.broad.igv.bbfile.BigWigIterator;
import org.broad.igv.bbfile.WigItem;

import java.io.IOException;

/**
 * Some utils for querying bigwig files
 *
 * @author jrobinso
 *         Date: 5/28/13
 *         Time: 3:27 PM
 */
public class BigWigUtils {

    public static void computeBins(String path, int windowSize) throws IOException {

        BBFileReader reader = new BBFileReader(path);
        for (String chr : reader.getChromosomeNames()) {
            computeBins(reader, chr, 0, Integer.MAX_VALUE, windowSize);
        }

    }


    public static void computeBins(String path, String chr, int start, int end, int windowSize) throws IOException {

        BBFileReader reader = new BBFileReader(path);
        boolean found = false;
        String errString = "";
        for (String chr1 : reader.getChromosomeNames()) {
            if (chr.equals(chr1)) found = true;
            errString += "\"" + chr1 + "\" ";
        }
        if (!found) {
            System.err.println("Chromosome \"" + chr + "\" not found in " + path);
            System.err.println("The chromosomes in " + path + " are " + errString);
            return;
        }
        computeBins(reader, chr, start, end, windowSize);

    }

    /**
     * Private method, does the actual work
     *
     * @param reader
     * @param chr
     * @param start
     * @param end
     * @param windowSize
     */
    private static void computeBins(BBFileReader reader, String chr, int start, int end, int windowSize) {
        BigWigIterator iter = reader.getBigWigIterator(chr, start, chr, end, false);
        double sum = 0;
        int nPts = 0;
        double max = 0;
        int currentBin = 0;

        while (iter.hasNext()) {
            WigItem datum = iter.next();
            int dPosition = (datum.getStartBase() + datum.getEndBase()) / 2;


            if (dPosition > (currentBin + 1) * windowSize) {
                // Output previous window
                // if (nPts > 0) {

                double mean = sum / nPts;
                int wStart = windowSize * currentBin;
                int wEnd = wStart + windowSize;
                // unclear why, but sometimes datum.getChromosome() is null
                //System.out.println(datum.getChromosome() + "\t" + wStart + "\t" + wEnd + "\t" + mean + "\t" + max);
                System.out.println(chr + "\t" + wStart + "\t" + wEnd + "\t" + mean + "\t" + max);

                currentBin++;

                // deal with empty bins
                while (currentBin < dPosition / windowSize) {
                    wStart = windowSize * currentBin;
                    wEnd = wStart + windowSize;
                    mean = 0;
                    max = 0;
                    System.out.println(chr + "\t" + wStart + "\t" + wEnd + "\t" + mean + "\t" + max);
                    // System.out.println(datum.getChromosome() + "\t" + wStart + "\t" + wEnd + "\t" + mean + "\t" + max);
                    currentBin++;
                }
                // Start new window
                currentBin = dPosition / windowSize;
                sum = 0;
                nPts = 0;
                max = 0;
            }

            sum += datum.getWigValue();
            max = Math.max(max, datum.getWigValue());
            nPts++;

        }
    }


    /**
     * Example usage
     * <p/>
     * First argument (required): path, either file or URL.
     * http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeUwTfbs/wgEncodeUwTfbsGm12878CtcfStdRawRep1.bigWig
     * /Users/jrobinso/projects/hic/data/wgEncodeUwTfbsGm12878CtcfStdRawRep1.bigWig
     * Second argument (required): windowSize in base pairs
     * 25000
     * Third argument (optional): chromosome
     * Fourth argument (optional): start base -- if supplied end base must also be supplied
     * Firth argument (optional):  end base
     */
    // Example use
    public static void main(String[] args) throws Exception {

        if (args.length < 2) {
            System.err.println("Chromosome and window size are required");
            System.exit(0);
        }

        String path = args[0];
        int windowSize = Integer.parseInt(args[1]);
        if (args.length == 2) {
            computeBins(path, windowSize);
        } else {
            String chr = args[2];
            if (args.length == 3) {
                computeBins(path, chr, 0, Integer.MAX_VALUE, windowSize);
            } else {
                int start = Integer.parseInt(args[3]) - 1;  // Convert to "zero" based coords
                int end = Integer.parseInt(args[4]);
                computeBins(path, chr, start, end, windowSize);
            }

        }


    }

}
