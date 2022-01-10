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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.data.anchor;

import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.HiCFileTools;
import juicebox.data.basics.Chromosome;
import juicebox.data.feature.FeatureFilter;
import juicebox.data.feature.GenomeWideList;
import org.broad.igv.Globals;
import org.broad.igv.util.ParsingUtils;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.zip.GZIPInputStream;

/**
 * Created by muhammadsaadshamim on 10/26/15.
 * <p/>
 * Parses the .txt file of motifs created by FIMO
 * <p/>
 * FIMO -
 */
public class MotifAnchorParser {

    public static GenomeWideList<MotifAnchor> loadMotifsFromGenomeID(String genomeID, FeatureFilter<MotifAnchor> anchorFilter) {
        return loadGlobalMotifs("", genomeID, anchorFilter, MotifLocation.VIA_ID);
    }

    public static GenomeWideList<MotifAnchor> loadMotifsFromLocalFile(String path, String genomeID, FeatureFilter<MotifAnchor> anchorFilter) {
        return loadGlobalMotifs(path, genomeID, anchorFilter, MotifLocation.LOCAL);
    }

    public static GenomeWideList<MotifAnchor> loadMotifsFromURL(String path, String genomeID, FeatureFilter<MotifAnchor> anchorFilter) {
        return loadGlobalMotifs(path, genomeID, anchorFilter, MotifLocation.URL);
    }

    private static GenomeWideList<MotifAnchor> loadGlobalMotifs(String path, String genomeID,
                                                                FeatureFilter<MotifAnchor> anchorFilter,
                                                                MotifLocation motifLocation) {
        InputStream is = null;
        BufferedReader reader = null;
        GenomeWideList<MotifAnchor> newAnchorList;

        try {
            // locate file from appropriate source and creat input stream
            is = ParsingUtils.openInputStream(extractProperMotifFilePath(genomeID, path, motifLocation));
            reader = new BufferedReader(new InputStreamReader(is), HiCGlobals.bufferSize);
        } catch (Exception e) {
            System.err.println("Unable to create input stream for global motifs " + motifLocation);
            System.exit(49);
        } finally {
            ChromosomeHandler handler = HiCFileTools.loadChromosomes(genomeID);

            Set<MotifAnchor> anchors = new HashSet<>();

            try {
                if (reader != null) {
                    anchors.addAll(parseGlobalMotifFile(reader, handler));
                }
            } catch (Exception e3) {
                //e3.printStackTrace();
                System.err.println("Unable to parse motif file");
                System.exit(50);
            }

            if (is != null) {
                try {
                    is.close();
                } catch (Exception e4) {
                    System.err.println("Error closing file stream for motif file");
                    //e4.printStackTrace();
                }
            }
            newAnchorList = new GenomeWideList<>(handler, new ArrayList<>(anchors));
        }

        if (anchorFilter != null)
            newAnchorList.filterLists(anchorFilter);

        return newAnchorList;
    }

    private static String extractProperMotifFilePath(String genomeID, String path, MotifLocation motifLocation) {
        String filePath = path;
        try {
            if (motifLocation == MotifLocation.VIA_ID) {
                String newURL = "https://hicfiles.s3.amazonaws.com/internal/motifs/" + genomeID + ".motifs.txt";
                filePath = downloadFromUrl(new URL(newURL), "motifs");
            } else if (motifLocation == MotifLocation.URL) {
                filePath = downloadFromUrl(new URL(path), "motifs");
            }
        } catch (Exception e) {
            System.err.println("Unable to find proper file via " + motifLocation);
            System.exit(51);
        }
        return filePath;
    }

    private static GenomeWideList<MotifAnchor> parseMotifFile(String path, ChromosomeHandler handler,
                                                              FeatureFilter<MotifAnchor> anchorFilter) {
        List<MotifAnchor> anchors = new ArrayList<>();

        try {
            //BufferedReader br = ParsingUtils.openBufferedReader(path);
            BufferedReader br = new BufferedReader(new InputStreamReader(ParsingUtils.openInputStream(path)), HiCGlobals.bufferSize);
            anchors.addAll(parseGlobalMotifFile(br, handler));
        } catch (IOException ec) {
            ec.printStackTrace();
        }

        GenomeWideList<MotifAnchor> newAnchorList = new GenomeWideList<>(handler, anchors);
        if (anchorFilter != null)
            newAnchorList.filterLists(anchorFilter);

        return newAnchorList;
    }

    /**
     * Parses a motif file (assumes FIMO output format)
     *
     * @param bufferedReader
     * @param handler
     * @return list of motifs and their attributes (score, sequence, etc)
     * @throws IOException
     */
    private static List<MotifAnchor> parseGlobalMotifFile(BufferedReader bufferedReader, ChromosomeHandler handler) throws IOException {
        Set<MotifAnchor> anchors = new HashSet<>();
        String nextLine;
        // skip header
        bufferedReader.readLine();

        //String[] headers = Globals.tabPattern.split(nextLine);

        //pattern_name	sequence_name	start	    stop	    strand	score	p-value	    q-value	    matched sequence
        //CTCF_REN	    chr12	        121937865	121937884	+	    28.6735	5.86e-13	0.00113	    GCGGCCACCAGGGGGCGCCC

        //OR IF FILTERED LIST
        //CTCF_M1_FLIPPED	chr10:100420000-100425000	4400	4413	-	18.2569	4.1e-07	0.00811	TCCAGTAGATGGCG

        int errorCount = 0;
        while ((nextLine = bufferedReader.readLine()) != null) {
            String[] tokens = Globals.tabPattern.split(nextLine);
            if (tokens.length != 9) {
                String text = "Improperly formatted FIMO output file: \npattern_name\tsequence_name\t" +
                        "start\tstop\tstrand\tscore\tp-value\tq-value\tmatched_sequence";
                System.err.println(text);
                System.exit(52);
            }

            boolean strand;
            String chr1Name, sequence;
            int start1, end1;
            double score, pValue, qValue;

            try {
                if (tokens[1].contains(":")) {// behavior when filtered list provided
                    String[] splitToken = tokens[1].split(":");
                    String[] indices = splitToken[1].split("-");

                    chr1Name = splitToken[0];
                    start1 = Integer.parseInt(indices[0]) + Integer.parseInt(indices[2]);
                    end1 = Integer.parseInt(indices[1]) + Integer.parseInt(indices[3]);
                } else {//default format all tabs
                    chr1Name = tokens[1];
                    start1 = Integer.parseInt(tokens[2]);
                    end1 = Integer.parseInt(tokens[3]);
                }

                strand = tokens[4].contains("+");
                score = Double.parseDouble(tokens[5]);
                pValue = Double.parseDouble(tokens[6]);
                qValue = Double.parseDouble(tokens[7]);
                sequence = tokens[8];

                Chromosome chr = handler.getChromosomeFromName(chr1Name);
                if (chr == null) {
                    if (HiCGlobals.printVerboseComments) {
                        if (errorCount < 10) {
                            System.out.println("Skipping line: " + nextLine);
                        } else if (errorCount == 10) {
                            System.err.println("Maximum error count exceeded.  Further errors will not be logged");
                        }
                    }

                    errorCount++;
                    continue;
                }

                MotifAnchor anchor = new MotifAnchor(chr.getName(), start1, end1);
                anchor.setFIMOAttributes(score, pValue, qValue, strand, sequence);

                anchors.add(anchor);

            } catch (Exception e) {
                String text = "Improperly formatted FIMO output file: \npattern_name\tsequence_name\t" +
                        "start\tstop\tstrand\tscore\tp-value\tq-value\tmatched_sequence";
                System.err.println(text);
                System.exit(53);
            }
        }
        bufferedReader.close();
        return new ArrayList<>(anchors);
    }

    /**
     * @param handler
     * @param bedFilePath
     * @return List of motif anchors from the provided bed file
     */
    public static GenomeWideList<MotifAnchor> loadFromBEDFile(ChromosomeHandler handler, String bedFilePath) {
        List<MotifAnchor> anchors = new ArrayList<>();

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
    private static List<MotifAnchor> parseBEDFile(BufferedReader bufferedReader, ChromosomeHandler handler) throws IOException {
        Set<MotifAnchor> anchors = new HashSet<>();
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

                anchors.add(new MotifAnchor(chr.getName(), start1, end1));
            }
        }
        if (anchors.size() < 1) System.err.println("BED File empty - file may have problems or error was encountered");
        bufferedReader.close();
        return new ArrayList<>(anchors);
    }

    /**
     * http://kamwo.me/java-download-file-from-url-to-temp-directory/
     *
     * @param url
     * @param localFilename
     * @return
     * @throws IOException
     */
    public static String downloadFromUrl(URL url, String localFilename) throws IOException {
        InputStream is = null;
        FileOutputStream fos = null;

        String tempDir = System.getProperty("java.io.tmpdir");
        File outputFile = new File(tempDir, localFilename);

        try {
            URLConnection urlConn = url.openConnection();
            is = urlConn.getInputStream();
            fos = new FileOutputStream(outputFile);

            byte[] buffer = new byte[HiCGlobals.bufferSize];
            int length;

            // read from source and write into local file
            while ((length = is.read(buffer)) > 0) {
                fos.write(buffer, 0, length);
            }
            return outputFile.getAbsolutePath();
        } finally {
            try {
                if (is != null) {
                    is.close();
                }
            } finally {
                if (fos != null) {
                    fos.close();
                }
            }
        }
    }


    public static String uncompressFromGzip(String compressedFile, String decompressedFile) throws IOException {

        InputStream fileIn = null;
        GZIPInputStream gZIPInputStream = null;
        FileOutputStream fileOutputStream = null;

        String tempDir = System.getProperty("java.io.tmpdir");
        File outputFile = new File(tempDir, decompressedFile);


        try {

            byte[] buffer = new byte[HiCGlobals.bufferSize];


            fileIn = new FileInputStream(compressedFile);
            gZIPInputStream = new GZIPInputStream(fileIn);
            fileOutputStream = new FileOutputStream(outputFile);

            int bytes_read;

            while ((bytes_read = gZIPInputStream.read(buffer)) > 0) {

                fileOutputStream.write(buffer, 0, bytes_read);
            }

            gZIPInputStream.close();
            fileOutputStream.close();

            System.out.println("The file was decompressed successfully!");
            return outputFile.getAbsolutePath();
        } finally {
            try {
                if (fileIn != null) {
                    fileIn.close();
                }
                if (gZIPInputStream != null) {
                    gZIPInputStream.close();
                }
            } finally {
                if (fileOutputStream != null) {
                    fileOutputStream.close();
                }
            }
        }
    }

    public enum MotifLocation {VIA_ID, URL, LOCAL}
}
