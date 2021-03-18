/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.data;

import juicebox.HiCGlobals;
import juicebox.data.basics.Chromosome;
import juicebox.data.basics.ListOfDoubleArrays;
import juicebox.tools.chrom.sizes.ChromosomeSizes;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Created by muhammadsaadshamim on 5/12/15.
 */
public class HiCFileTools {

    public static Dataset extractDatasetForCLT(List<String> files, boolean allowPrinting) {
        Dataset dataset = null;
        try {
            DatasetReader reader = null;
            if (files.size() == 1) {
                if (allowPrinting)
                    System.out.println("Reading file: " + files.get(0));
                String magicString = DatasetReaderFactory.getMagicString(files.get(0));
                if (magicString.equals("HIC")) {
                    reader = new DatasetReaderV2(files.get(0));
                } else {
                    System.err.println("This version of HIC is no longer supported");
                    System.exit(32);
                }
                dataset = reader.read();

            } else {
                if (allowPrinting)
                    System.out.println("Reading summed files: " + files);
                reader = DatasetReaderFactory.getReader(files);
                if (reader == null) {
                    System.err.println("Error while reading files");
                    System.exit(33);
                } else {
                    dataset = reader.read();
                }
            }
            HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());
        } catch (Exception e) {
            System.err.println("Could not read hic file: " + e.getMessage());
            System.exit(34);
            //e.printStackTrace();
        }
        return dataset;
    }

    public static DatasetReader extractDatasetReaderForCLT(List<String> files, boolean allowPrinting) {
        DatasetReader reader = null;
        try {
            if (files.size() == 1) {
                if (allowPrinting)
                    System.out.println("Reading file: " + files.get(0));
                String magicString = DatasetReaderFactory.getMagicString(files.get(0));
                if (magicString.equals("HIC")) {
                    reader = new DatasetReaderV2(files.get(0));
                } else {
                    System.err.println("This version of HIC is no longer supported");
                    System.exit(32);
                }


            } else {
                if (allowPrinting)
                    System.out.println("Reading summed files: " + files);
                reader = DatasetReaderFactory.getReader(files);
                if (reader == null) {
                    System.err.println("Error while reading files");
                    System.exit(33);
                }
            }

        } catch (Exception e) {
            System.err.println("Could not read hic file: " + e.getMessage());
            System.exit(34);
            //e.printStackTrace();
        }
        return reader;
    }

    /**
     * Load the list of chromosomes based on given genome id or file
     *
     * @param idOrFile string
     * @return list of chromosomes
     */
    public static ChromosomeHandler loadChromosomes(String idOrFile) {

        InputStream is = null;

        try {
            // Note: to get this to work, had to edit Intellij settings
            // so that "?*.sizes" are considered sources to be copied to class path
            is = ChromosomeSizes.class.getResourceAsStream(idOrFile + ".chrom.sizes");

            if (is == null) {
                // Not an ID,  see if its a file
                File file = new File(idOrFile);

                try {
                    if (file.exists()) {
                        is = new FileInputStream(file);
                    } else {
                        System.err.println("Could not find chromosome sizes file for: " + idOrFile);
                        System.exit(35);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            List<Chromosome> chromosomes = new ArrayList<>();
            chromosomes.add(0, null);   // Index 0 reserved for "whole genome" pseudo-chromosome

            Pattern pattern = Pattern.compile("\\s+");
            BufferedReader reader = new BufferedReader(new InputStreamReader(is), HiCGlobals.bufferSize);
            String nextLine;
            int idx = 1;

            try {
                while ((nextLine = reader.readLine()) != null) {
                    String[] tokens = pattern.split(nextLine);
                    if (tokens.length == 2) {
                        String name = tokens[0];
                        int length = Integer.parseInt(tokens[1]);
                        chromosomes.add(idx, new Chromosome(idx, name, length));
                        idx++;
                    } else {
                        System.out.println("Skipping " + nextLine);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

            // "pseudo-chromosome" All taken care of by by chromosome handler
            return new ChromosomeHandler(chromosomes, idOrFile, false);
        } finally {
            if (is != null) {
                try {
                    is.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * Given an array of possible resolutions, returns the actual resolutions available in the dataset
     *
     * @param availableZooms
     * @param resolutions
     * @return finalResolutions Set
     */
    public static List<Integer> filterResolutions(List<HiCZoom> availableZooms, int[] resolutions) {

        TreeSet<Integer> resSet = new TreeSet<>();
        for (HiCZoom zoom : availableZooms) {
            resSet.add(zoom.getBinSize());
        }

        List<Integer> finalResolutions = new ArrayList<>();
        for (int res : resolutions) {
            finalResolutions.add(closestValue(res, resSet));
        }

        return finalResolutions;
    }

    private static int closestValue(int val, TreeSet<Integer> valSet) {
        int floorVal;
        try {
            // sometimes no lower value is available and throws NPE
            floorVal = valSet.floor(val);
        } catch (Exception e) {
            return valSet.ceiling(val);
        }
        int ceilVal;
        try {
            // sometimes no higher value is available and throws NPE
            ceilVal = valSet.ceiling(val);
        } catch (Exception e) {
            return floorVal;
        }

        if (Math.abs(ceilVal - val) < Math.abs(val - floorVal))
            return ceilVal;

        return floorVal;
    }


    public static ChromosomeHandler getChromosomeSetIntersection(ChromosomeHandler handler1, ChromosomeHandler handler2) {
        return handler1.getIntersectionWith(handler2);
    }

    public static Set<HiCZoom> getZoomSetIntersection(Collection<HiCZoom> collection1, Collection<HiCZoom> collection2) {
        Set<HiCZoom> set1 = new HashSet<>(collection1);
        Set<HiCZoom> set2 = new HashSet<>(collection2);

        boolean set1IsLarger = set1.size() > set2.size();
        Set<HiCZoom> cloneSet = new HashSet<>(set1IsLarger ? set2 : set1);
        cloneSet.retainAll(set1IsLarger ? set1 : set2);
        return cloneSet;
    }

    /**
     * For each given chromosome name, find its equivalent Chromosome object
     *
     * @param chromosomesSpecified by strings
     * @param handler as Chromosome objects
     * @return the specified Chromosomes corresponding to the given strings
     */
    public static ChromosomeHandler stringToChromosomes(List<String> chromosomesSpecified,
                                                        ChromosomeHandler handler) {
        List<Chromosome> chromosomes = new ArrayList<>();
        chromosomes.add(0, null);

        for (String strKey : chromosomesSpecified) {
            boolean chrFound = false;
            for (Chromosome chrKey : handler.getChromosomeArray()) {
                if (equivalentChromosome(strKey, chrKey)) {
                    chromosomes.add(chrKey);
                    chrFound = true;
                    break;
                }
            }
            if (!chrFound) {
                System.err.println("Chromosome " + strKey + " not found");
            }
        }
        return new ChromosomeHandler(chromosomes, handler.getGenomeID(), false);
    }

    /**
     * Evaluates whether the same chromosome is being referenced by the token
     *
     * @param token
     * @param chr
     * @return
     */
    public static boolean equivalentChromosome(String token, Chromosome chr) {
        String token2 = token.toLowerCase().replaceAll("chr", "");
        String chrName = chr.getName().toLowerCase().replaceAll("chr", "");
        return token2.equals(chrName);
    }

    public static PrintWriter openWriter(File file) {
        try {
            file.createNewFile();
            file.setWritable(true);
            return new PrintWriter(new BufferedWriter(new FileWriter(file)), true);
        } catch (IOException e) {
            System.out.println("I/O error opening file.");
            System.exit(37);
        }
        return null;
    }

    public static RealMatrix extractLocalBoundedRegion(MatrixZoomData zd, int limStart, int limEnd, int n,
                                                       NormalizationType normalizationType, boolean fillUnderDiagonal) throws IOException {
        return extractLocalBoundedRegion(zd, limStart, limEnd, limStart, limEnd, n, n, normalizationType, fillUnderDiagonal);
    }

    public static RealMatrix extractLocalBoundedRegion(MatrixZoomData zd, int binXStart, int binYStart, int numRows, int numCols,
                                                       NormalizationType normalizationType, boolean fillUnderDiagonal) throws IOException {
        return extractLocalBoundedRegion(zd, binXStart, binXStart + numRows, binYStart, binYStart + numCols, numRows, numCols, normalizationType, fillUnderDiagonal);
    }
    
    /**
     * Extracts matrix from hic file for a specified region.
     * By default, only the top right part of the matrix is returned if the matrix is on the diagonal.
     *
     * @return section of the matrix
     */
    public static RealMatrix extractLocalBoundedRegion(MatrixZoomData zd, long binXStart, long binXEnd,
                                                       long binYStart, long binYEnd, int numRows, int numCols,
                                                       NormalizationType normalizationType, boolean fillUnderDiagonal) throws IOException {
        
        // numRows/numCols is just to ensure a set size in case bounds are approximate
        // left upper corner is reference for 0,0
        List<Block> blocks = getAllRegionBlocks(zd, binXStart, binXEnd, binYStart, binYEnd, normalizationType, fillUnderDiagonal);
        
        RealMatrix data = MatrixTools.cleanArray2DMatrix(numRows, numCols);
        
        if (blocks.size() > 0) {
            for (Block b : blocks) {
                if (b != null) {
                    for (ContactRecord rec : b.getContactRecords()) {
    
                        // only called for small regions - should not exceed int
                        int relativeX = (int) (rec.getBinX() - binXStart);
                        int relativeY = (int) (rec.getBinY() - binYStart);
    
                        if (relativeX >= 0 && relativeX < numRows) {
                            if (relativeY >= 0 && relativeY < numCols) {
                                data.addToEntry(relativeX, relativeY, rec.getCounts());
                            }
                        }
    
                        if (fillUnderDiagonal) {
                            relativeX = (int) (rec.getBinY() - binXStart);
                            relativeY = (int) (rec.getBinX() - binYStart);
        
                            if (relativeX >= 0 && relativeX < numRows) {
                                if (relativeY >= 0 && relativeY < numCols) {
                                    data.addToEntry(relativeX, relativeY, rec.getCounts());
                                }
                            }
                        }
                    }
                }
            }
        }
        // force cleanup
        blocks = null;
        //System.gc();
        
        return data;
    }

    public static RealMatrix extractLocalBoundedExpectedRegion(ExpectedValueFunction df, org.broad.igv.feature.Chromosome chr, int binXStart,
                                                               int binYStart, int numRows, int numCols) throws IOException {

        // numRows/numCols is just to ensure a set size in case bounds are approximate
        // left upper corner is reference for 0,0
        // List<Block> blocks = getAllRegionBlocks(zd, binXStart, binXEnd, binYStart, binYEnd, normalizationType, fillUnderDiagonal);

        RealMatrix data = MatrixTools.cleanArray2DMatrix(numRows, numCols);
        for (int relativeX = 0; relativeX < numRows; relativeX++) {
            for (int relativeY = 0; relativeY < numRows; relativeY++) {
                int dist = Math.abs((binXStart - binYStart) + (relativeX - relativeY));
                double expected = df.getExpectedValue(chr.getIndex(), dist);
                data.addToEntry(relativeX, relativeY, expected);
            }
        }

        return data;
    }

    public static RealMatrix extractLocalRowSums(MatrixZoomData zd, long binXStart, long binXEnd,
                                                 long chrStart, long chrEnd, int numRows,
                                                 NormalizationType normalizationType, boolean fillUnderDiagonal) throws IOException {

        // numRows/numCols is just to ensure a set size in case bounds are approximate
        // left upper corner is reference for 0,0
        List<Block> blocks = getAllRegionBlocks(zd, binXStart, binXEnd, chrStart, chrEnd, normalizationType, fillUnderDiagonal);


        RealMatrix data = MatrixTools.cleanArray2DMatrix(numRows, 1);

        if (blocks.size() > 0) {
            for (Block b : blocks) {
                if (b != null) {
                    for (ContactRecord rec : b.getContactRecords()) {
                        // only called for small regions - should not exceed int
                        int relativeX = (int) (rec.getBinX() - binXStart);
                        int relativeY = (int) (rec.getBinY() - binXStart);

                        if (relativeX >= 0 && relativeX < numRows) {
                            if (!Float.isNaN(rec.getCounts())) {
                                data.addToEntry(relativeX, 0, rec.getCounts());
                            }
                        } else if (relativeY >= 0 && relativeY < numRows) {
                            if (!Float.isNaN(rec.getCounts())) {
                                data.addToEntry(relativeY, 0, rec.getCounts());
                            }
                        }

                    }
                }
            }
        }

        // force cleanup
        blocks = null;
        //System.gc();

        //System.out.println("individual row sum: " + MatrixTools.sum(data.getData()));
        return data;
    }

    public static List<Block> getAllRegionBlocks(MatrixZoomData zd, long binXStart, long binXEnd,
                                                 long binYStart, long binYEnd,
                                                 NormalizationType normalizationType, boolean fillUnderDiagonal) throws IOException {
        
        List<Block> blocks = Collections.synchronizedList(new ArrayList<>());
        
        int numDataReadingErrors = 0;
        
        try {
            blocks.addAll(zd.getNormalizedBlocksOverlapping(binXStart, binYStart, binXEnd, binYEnd, normalizationType, false, fillUnderDiagonal));
        } catch (Exception e) {
            triggerNormError(normalizationType);
            if (HiCGlobals.printVerboseComments) {
                System.err.println("You do not have " + normalizationType + " normalized maps available for this resolution/region:");
                System.err.println("x1 " + binXStart + " x2 " + binXEnd + " y1 " + binYStart + " y2 " + binYEnd + " res " + zd.getBinSize());
                System.err.println("Map is likely too sparse or a different normalization/resolution should be chosen.");
                e.printStackTrace();
                System.exit(38);
            }
        }

        if (HiCGlobals.printVerboseComments && numDataReadingErrors > 0) {
            //System.err.println(numDataReadingErrors + " errors while reading data from region. Map is likely too sparse");
            triggerNormError(normalizationType);
        }

        return blocks;
    }
    
    public static ListOfDoubleArrays extractChromosomeExpectedVector(Dataset ds, int index, HiCZoom zoom, NormalizationType normalization) {
        ExpectedValueFunction expectedValueFunction = ds.getExpectedValues(zoom, normalization);
        long n = expectedValueFunction.getLength();
        
        ListOfDoubleArrays expectedVector = new ListOfDoubleArrays(n);
        for (long i = 0; i < n; i++) {
            expectedVector.set(i, expectedValueFunction.getExpectedValue(index, i));
        }
        return expectedVector;
    }


    public static void triggerNormError(NormalizationType normalizationType) throws IOException {
        System.err.println();
        System.err.println("You do not have " + normalizationType + " normalized maps available for this resolution/region.");
        System.err.println("Region is likely too sparse/does not exist, or a different normalization/resolution should be chosen.");
        throw new IOException("Norm could not be found");
    }


    /**
     * @param directoryPath
     * @return valid directory for unix/windows or exits with error code
     */
    public static File createValidDirectory(String directoryPath) {
        File outputDirectory = new File(directoryPath);
        if (!outputDirectory.exists() || !outputDirectory.isDirectory()) {
            if (!outputDirectory.mkdir()) {
                System.err.println("Couldn't create output directory " + directoryPath);
                System.exit(40);
            }
        }
        return outputDirectory;
    }

    public static String getTruncatedText(String text, int maxLengthEntryName) {
        String truncatedName = text;
        if (truncatedName.length() > maxLengthEntryName) {
            truncatedName = text.substring(0, maxLengthEntryName / 2 - 1);
            truncatedName += "...";
            truncatedName += text.substring(text.length() - maxLengthEntryName / 2);
        }
        return truncatedName;
    }

    public static boolean isDropboxURL(String url) {
        return url.contains("dropbox.com");
    }

    public static String cleanUpDropboxURL(String url) {
        return url.replace("?dl=0", "")
                .replace("://www.dropbox.com", "://dl.dropboxusercontent.com");
    }

    public static MatrixZoomData getMatrixZoomData(Dataset ds, Chromosome chrom1, Chromosome chrom2, HiCZoom zoom) {
        Matrix matrix = ds.getMatrix(chrom1, chrom2);
        if (matrix == null || zoom == null) return null;
        return matrix.getZoomData(zoom);
    }

    public static MatrixZoomData getMatrixZoomData(Dataset ds, Chromosome chrom1, Chromosome chrom2, int resolution) {
        return getMatrixZoomData(ds, chrom1, chrom2, ds.getZoomForBPResolution(resolution));
    }
}
