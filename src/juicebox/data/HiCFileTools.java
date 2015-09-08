/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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
import juicebox.tools.chrom.sizes.ChromosomeSizes;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Created by muhammadsaadshamim on 5/12/15.
 */
public class HiCFileTools {

    /**
     * Load chromosomes from given ID or file name.
     *
     * @param idOrFile Genome ID or file name where chromosome lengths written
     * @return Chromosome lengths
     * @throws java.io.IOException if chromosome length file not found
     */

    private static final String tempPath = System.getProperty("user.dir");

    public static Dataset extractDatasetForCLT(List<String> files, boolean allowPrinting) {
        Dataset dataset = null;
        try {
            DatasetReader reader = null;
            if (files.size() == 1) {
                if (allowPrinting)
                    System.out.println("Reading file: " + files.get(0));
                String magicString = DatasetReaderV2.getMagicString(files.get(0));
                if (magicString.equals("HIC")) {
                    reader = new DatasetReaderV2(files.get(0));
                } else {
                    System.err.println("This version of HIC is no longer supported");
                    System.exit(-1);
                }
                dataset = reader.read();

            } else {
                if (allowPrinting)
                    System.out.println("Reading summed files: " + files);
                reader = DatasetReaderFactory.getReader(files);
                if (reader == null) {
                    System.err.println("Error while reading files");
                    System.exit(-1);
                } else {
                    dataset = reader.read();
                }
            }
            HiCGlobals.verifySupportedHiCFileVersion(reader.getVersion());
        } catch (Exception e) {
            System.err.println("Could not read hic file: " + e.getMessage());
            System.exit(-6);
            //e.printStackTrace();
        }
        return dataset;
    }

    public static List<Chromosome> loadChromosomes(String idOrFile) {

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
                        System.exit(-3);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            List<Chromosome> chromosomes = new ArrayList<Chromosome>();
            chromosomes.add(0, null);   // Index 0 reserved for "whole genome" pseudo-chromosome

            Pattern pattern = Pattern.compile("\t");
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            String nextLine;
            long genomeLength = 0;
            int idx = 1;

            try {
                while ((nextLine = reader.readLine()) != null) {
                    String[] tokens = pattern.split(nextLine);
                    if (tokens.length == 2) {
                        String name = tokens[0];
                        int length = Integer.parseInt(tokens[1]);
                        genomeLength += length;
                        chromosomes.add(idx, new Chromosome(idx, name, length));
                        idx++;
                    } else {
                        System.out.println("Skipping " + nextLine);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

            // Add the "pseudo-chromosome" All, representing the whole genome.  Units are in kilo-bases
            chromosomes.set(0, new Chromosome(0, "All", (int) (genomeLength / 1000)));


            return chromosomes;
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
     * @param ds
     * @param resolutions
     * @return finalResolutions Set
     */
    public static Set<Integer> filterResolutions(Dataset ds, int[] resolutions) {

        TreeSet<Integer> resSet = new TreeSet<Integer>();
        for (HiCZoom zoom : ds.getBpZooms()) {
            resSet.add(zoom.getBinSize());
        }

        Set<Integer> finalResolutions = new HashSet<Integer>();
        for (int res : resolutions) {
            finalResolutions.add(closestValue(res, resSet));
        }

        return finalResolutions;
    }

    private static int closestValue(int val, TreeSet<Integer> valSet) {
        int floorVal = valSet.floor(val);
        int ceilVal = valSet.ceiling(val);

        if (ceilVal - val < val - floorVal)
            return ceilVal;

        return floorVal;
    }


    /**
     * Set intersection
     * <p/>
     * http://stackoverflow.com/questions/7574311/efficiently-compute-intersection-of-two-sets-in-java
     *
     * @param set1
     * @param set2
     * @return
     */
    public static Set<Chromosome> getSetIntersection(Set<Chromosome> set1, Set<Chromosome> set2) {
        boolean set1IsLarger = set1.size() > set2.size();
        Set<Chromosome> cloneSet = new HashSet<Chromosome>(set1IsLarger ? set2 : set1);
        cloneSet.retainAll(set1IsLarger ? set1 : set2);
        return cloneSet;
    }

    public static Set<Chromosome> stringToChromosomes(Set<String> chromosomesSpecified,
                                                      List<Chromosome> referenceChromosomes) {

        Set<Chromosome> chrKeys = new HashSet<Chromosome>(referenceChromosomes);
        Set<String> strKeys = new HashSet<String>(chromosomesSpecified);
        Set<Chromosome> convertedChromosomes = new HashSet<Chromosome>();

        // filter down loops by uniqueness, then size, and save the totals at each stage
        for (Chromosome chrKey : chrKeys) {
            for (String strKey : strKeys) {
                if (equivalentChromosome(strKey, chrKey)) {
                    convertedChromosomes.add(chrKey);
                    strKeys.remove(strKey);
                    break;
                }
            }
        }

        return new HashSet<Chromosome>(convertedChromosomes);
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

    /**
     * @param fileName
     * @return
     */
    public static PrintWriter openWriter(String fileName) {
        try {
            File file = new File(fileName);
            return new PrintWriter(new BufferedWriter(new FileWriter(file)), true);
        } catch (IOException e) {
            System.out.println("I/O error opening file: " + fileName);
            System.exit(0);
        }
        return null;
    }

    public static PrintWriter openWriter(File file) {
        try {
            //create a temp file
            return new PrintWriter(new BufferedWriter(new FileWriter(file)), true);
        } catch (IOException e) {
            System.out.println("I/O error opening file temp file for AutoSave. ");
            System.exit(0);
        }
        return null;
    }

    public static File openTempFile(String prefix) {
        //try{
        //create a temp file
        String pathName = tempPath + "/" + prefix + ".txt";
        //File temp = File.createTempFile(prefix, ".tmp");
        return new File(pathName);
//        } catch (IOException e) {
//            System.out.println("I/O error opening file temp file for AutoSave. ");
//            System.exit(0);
//        }
//        return null;
    }


    /**
     * Extracts matrix from hic file for a specified region.
     * By default, only the top right part of the matrix is returned if the matrix is on the diagonal.
     *
     * @return section of the matrix
     */
    public static RealMatrix extractLocalBoundedRegion(MatrixZoomData zd, int binXStart, int binXEnd,
                                                       int binYStart, int binYEnd, int numRows, int numCols,
                                                       NormalizationType normalizationType) {

        // numRows/numCols is just to ensure a set size in case bounds are approximate
        // left upper corner is reference for 0,0
        List<Block> blocks = new ArrayList<Block>();

        try {
            zd.addNormalizedBlocksToList(blocks, binXStart, binYStart, binXEnd, binYEnd, normalizationType);
        } catch (Exception e) {
            System.out.println("You do not have " + normalizationType + " normalized maps available at this resolution/region:");
            System.out.println("x1 " + binXStart + " x2 " + binXEnd + " y1 " + binYStart + " y2 " + binYEnd + " res " + zd.getBinSize());
            e.printStackTrace();
            System.exit(-6);
        }

        RealMatrix data = MatrixTools.cleanArray2DMatrix(numRows, numCols);

        if (blocks.size() > 0) {
            for (Block b : blocks) {
                for (ContactRecord rec : b.getContactRecords()) {

                    int relativeX = rec.getBinX() - binXStart;
                    int relativeY = rec.getBinY() - binYStart;

                    if (relativeX >= 0 && relativeX < numRows) {
                        if (relativeY >= 0 && relativeY < numCols) {
                            data.addToEntry(relativeX, relativeY, rec.getCounts());
                            // was used to fill LL triangle for arrowhead
                            //if (mirrorBlock && relativeY != relativeX)
                            //    data.addToEntry(relativeY, relativeX, rec.getCounts());
                        }
                    }
                }
            }
        }

        return data;
    }

    public static NormalizationType determinePreferredNormalization(Dataset ds) {
        NormalizationType[] preferredNormalization = new NormalizationType[]{NormalizationType.KR, NormalizationType.VC};
        List<NormalizationType> normalizationTypeList = ds.getNormalizationTypes();

        //System.out.println("Norms: "+normalizationTypeList);

        for (NormalizationType normalizationType : preferredNormalization) {
            if (normalizationTypeList.contains(normalizationType)) {
                System.out.println("Selected " + normalizationType + " Normalization");
                return normalizationType;
            }
            System.out.println("Did not find Normalization: " + normalizationType);
        }

        System.out.println("Could not find normalizations");
        System.exit(-5);
        return null;
    }

    public static Chromosome getChromosomeNamed(String chrName, List<Chromosome> chromosomes) {
        for (Chromosome chr : chromosomes) {
            if (equivalentChromosome(chrName, chr))
                return chr;
        }
        return null;
    }

    public static double[] extractChromosomeExpectedVector(Dataset ds, int index, HiCZoom zoom, NormalizationType normalization) {
        ExpectedValueFunction expectedValueFunction = ds.getExpectedValues(zoom, normalization);
        int n = expectedValueFunction.getLength();

        double[] expectedVector = new double[n];
        for (int i = 0; i < n; i++) {
            expectedVector[i] = expectedValueFunction.getExpectedValue(index, i);
        }
        return expectedVector;
    }


}
