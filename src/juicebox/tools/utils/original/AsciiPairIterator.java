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

package juicebox.tools.utils.original;


import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.tools.clt.JuiceboxCLT;
import org.broad.igv.util.ParsingUtils;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.GZIPInputStream;

/**
 * @author Jim Robinson
 * @since 9/24/11
 */
public class AsciiPairIterator implements PairIterator {

    /**
     * A map of chromosome name -> chromosome string.  A private "intern" pool.  The java "intern" pool stores string
     * in perm space, which is rather limited and can cause us to run out of memory.
     */
    private final Map<String, String> stringInternPool = new HashMap<>();
    // Map of name -> index
    private Map<String, Integer> chromosomeOrdinals;
    private AlignmentPair nextPair = null;
    private BufferedReader reader;
    private Format format = null;
    private int dcicFragIndex1 = -1;
    private int dcicFragIndex2 = -1;
    private int dcicMapqIndex1 = -1;
    private int dcicMapqIndex2 = -1;
    //CharMatcher.anyOf(";,.")

    public AsciiPairIterator(String path, Map<String, Integer> chromosomeOrdinals) throws IOException {
        if (path.endsWith(".gz")) {
            InputStream fileStream = new FileInputStream(path);
            InputStream gzipStream = new GZIPInputStream(fileStream);
            Reader decoder = new InputStreamReader(gzipStream, StandardCharsets.UTF_8);
            this.reader = new BufferedReader(decoder, 4194304);
        } else {
            //this.reader = org.broad.igv.util.ParsingUtils.openBufferedReader(path);
            this.reader = new BufferedReader(new InputStreamReader(ParsingUtils.openInputStream(path)), HiCGlobals.bufferSize);
        }
        this.chromosomeOrdinals = chromosomeOrdinals;
        advance();
    }

    /**
     * Read the next record
     * <p/>
     * Short form:
     * str1 chr1 pos1 frag1 str2 chr2 pos2 frag2
     * 0 15 61559113 0 16 15 61559309 16
     * 16 10 26641879 16 0 9 12797549 0
     * <p/>
     * Short with score:
     * str1 chr1 pos1 frag1 str2 chr2 pos2 frag2 score
     * score is the count for this location (instead of 1)
     * <p/>
     * Medium form:
     * readname str1 chr1 pos1 frag1 str2 chr2 pos2 frag2 mapq1 mapq2
     * <p/>
     * Long form:
     * str1 chr1 pos1 frag1 str2 chr2 pos2 frag2 mapq1 cigar1 seq1 mapq2 cigar2 seq2 rname1 rname2
     * <p/>
     * DCIC form:
     * First 7 fields reserved:
     * readID, chr1, pos1, chr2, pos2, strand1, strand2
     * Optionally, readID and strands can be blank (‘.’) : DCIC provides both readID and strands.
     * Positions are 5’end of reads.
     * Optional columns follow, ignored by us
     */
    private void advance() {

        try {
            String nextLine;
            if ((nextLine = reader.readLine()) != null) {
                //String[] tokens = Globals.singleTabMultiSpacePattern.split(nextLine);
                String[] tokens = JuiceboxCLT.splitToList(nextLine);

                int nTokens = tokens.length;

                if (nextLine.startsWith("#")) {
                    // header line, skip; DCIC files MUST have header
                    format = Format.DCIC;
                    nextPair = new AlignmentPair(true);
                    if (nextLine.contains("column")) {
                        for (int i = 0; i < tokens.length; i++) {
                            if (tokens[i].contains("frag1")) {
                                dcicFragIndex1 = i-1;
                            }
                            if (tokens[i].contains("frag2")) {
                                dcicFragIndex2 = i-1;
                            }
                            if (tokens[i].contains("mapq1")) {
                                dcicMapqIndex1 = i-1;
                            }
                            if (tokens[i].contains("mapq2")) {
                                dcicMapqIndex2 = i-1;
                            }
                        }
                    }
                    return;
                }

                if (format == null) {
                    switch (nTokens) {
                        case 8:
                            format = Format.SHORT;
                            break;
                        case 9:
                            format = Format.SHORT_WITH_SCORE;
                            break;
                        case 16:
                            format = Format.LONG;
                            break;
                        case 11:
                            format = Format.MEDIUM;
                            break;
                        default:
                            throw new IOException("Unexpected column count.  Check file format");
                    }
                }
                switch (format) {
                    case MEDIUM: {
                        String chrom1 = ChromosomeHandler.cleanUpName(getInternedString(tokens[2]));
                        String chrom2 = ChromosomeHandler.cleanUpName(getInternedString(tokens[6]));
                        // some contigs will not be present in the chrom.sizes file
                        if (chromosomeOrdinals.containsKey(chrom1) && chromosomeOrdinals.containsKey(chrom2)) {
                            int chr1 = chromosomeOrdinals.get(chrom1);
                            int chr2 = chromosomeOrdinals.get(chrom2);
                            int pos1 = Integer.parseInt(tokens[3]);
                            int pos2 = Integer.parseInt(tokens[7]);
                            int frag1 = Integer.parseInt(tokens[4]);
                            int frag2 = Integer.parseInt(tokens[8]);
                            int mapq1 = Integer.parseInt(tokens[9]);
                            int mapq2 = Integer.parseInt(tokens[10]);

                            boolean strand1 = Integer.parseInt(tokens[1]) == 0;
                            boolean strand2 = Integer.parseInt(tokens[5]) == 0;
                            nextPair = new AlignmentPair(strand1, chr1, pos1, frag1, mapq1, strand2, chr2, pos2, frag2, mapq2);
                        } else {
                            nextPair = new AlignmentPair(); // sets dummy values, sets isContigPair
                        }

                        break;
                    }
                    case DCIC: {
                        String chrom1 = ChromosomeHandler.cleanUpName(getInternedString(tokens[1]));
                        String chrom2 = ChromosomeHandler.cleanUpName(getInternedString(tokens[3]));
                        if (chromosomeOrdinals.containsKey(chrom1) && chromosomeOrdinals.containsKey(chrom2)) {
                            int chr1 = chromosomeOrdinals.get(chrom1);
                            int chr2 = chromosomeOrdinals.get(chrom2);
                            int pos1 = Integer.parseInt(tokens[2]);
                            int pos2 = Integer.parseInt(tokens[4]);
                            boolean strand1 = tokens[5].equals("+");
                            boolean strand2 = tokens[6].equals("+");
                            int frag1 = 0;
                            int frag2 = 1;
                            if (dcicFragIndex1 != -1 && dcicFragIndex2 != -1) {
                                frag1 = Integer.parseInt(tokens[dcicFragIndex1]);
                                frag2 = Integer.parseInt(tokens[dcicFragIndex2]);
                            }
                            int mapq1 = 1000;
                            int mapq2 = 1000;
                            if (dcicMapqIndex1 != -1 && dcicMapqIndex2 != -1) {
                                mapq1 = Integer.parseInt(tokens[dcicMapqIndex1]);
                                mapq2 = Integer.parseInt(tokens[dcicMapqIndex2]);
                            }
                            nextPair = new AlignmentPair(strand1, chr1, pos1, frag1, mapq1, strand2, chr2, pos2, frag2, mapq2);

                        } else {
                            nextPair = new AlignmentPair(); // sets dummy values, sets isContigPair
                        }
                        break;
                    }
                    default: {
                        // this should be strand, chromosome, position, fragment.

                        String chrom1 = ChromosomeHandler.cleanUpName(getInternedString(tokens[1]));
                        String chrom2 = ChromosomeHandler.cleanUpName(getInternedString(tokens[5]));
                        // some contigs will not be present in the chrom.sizes file
                        if (chromosomeOrdinals.containsKey(chrom1) && chromosomeOrdinals.containsKey(chrom2)) {
                            int chr1 = chromosomeOrdinals.get(chrom1);
                            int chr2 = chromosomeOrdinals.get(chrom2);
                            int pos1 = Integer.parseInt(tokens[2]);
                            int pos2 = Integer.parseInt(tokens[6]);
                            int frag1 = Integer.parseInt(tokens[3]);
                            int frag2 = Integer.parseInt(tokens[7]);
                            int mapq1 = 1000;
                            int mapq2 = 1000;

                            if (format == Format.LONG) {
                                mapq1 = Integer.parseInt(tokens[8]);
                                mapq2 = Integer.parseInt(tokens[11]);
                            }
                            boolean strand1 = Integer.parseInt(tokens[0]) == 0;
                            boolean strand2 = Integer.parseInt(tokens[4]) == 0;
                            nextPair = new AlignmentPair(strand1, chr1, pos1, frag1, mapq1, strand2, chr2, pos2, frag2, mapq2);
                            if (format == Format.SHORT_WITH_SCORE) {
                                nextPair.setScore(Float.parseFloat(tokens[8]));
                            }
                        } else {
                            nextPair = new AlignmentPair(); // sets dummy values, sets isContigPair
                        }
                        break;
                    }
                }
                return;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        nextPair = null;

    }

    /**
     * Replace "aString" with a stored equivalent object, if it exists.  If it does not store it.  The purpose
     * of this class is to avoid running out of memory storing zillions of equivalent string.
     *
     * @param aString
     * @return
     */
    private String getInternedString(String aString) {
        String s = stringInternPool.get(aString);
        if (s == null) {
            //noinspection RedundantStringConstructorCall
            s = new String(aString); // The "new" will break any dependency on larger strings if this is a "substring"
            stringInternPool.put(aString, s);
        }
        return s;
    }

    public boolean hasNext() {
        return nextPair != null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public AlignmentPair next() {
        AlignmentPair p = nextPair;
        advance();
        return p;

    }

    public void remove() {
        // Not implemented
    }

    public void close() {
        try {
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }
    }

    enum Format {SHORT, LONG, MEDIUM, SHORT_WITH_SCORE, DCIC}

}
