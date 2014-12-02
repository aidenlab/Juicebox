/*
 * Copyright (c) 2007-2011 by The Broad Institute of MIT and Harvard.  All Rights Reserved.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
 *
 * THE SOFTWARE IS PROVIDED "AS IS." THE BROAD AND MIT MAKE NO REPRESENTATIONS OR
 * WARRANTES OF ANY KIND CONCERNING THE SOFTWARE, EXPRESS OR IMPLIED, INCLUDING,
 * WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, NONINFRINGEMENT, OR THE ABSENCE OF LATENT OR OTHER DEFECTS, WHETHER
 * OR NOT DISCOVERABLE.  IN NO EVENT SHALL THE BROAD OR MIT, OR THEIR RESPECTIVE
 * TRUSTEES, DIRECTORS, OFFICERS, EMPLOYEES, AND AFFILIATES BE LIABLE FOR ANY DAMAGES
 * OF ANY KIND, INCLUDING, WITHOUT LIMITATION, INCIDENTAL OR CONSEQUENTIAL DAMAGES,
 * ECONOMIC DAMAGES OR INJURY TO PROPERTY AND LOST PROFITS, REGARDLESS OF WHETHER
 * THE BROAD OR MIT SHALL BE ADVISED, SHALL HAVE OTHER REASON TO KNOW, OR IN FACT
 * SHALL KNOW OF THE POSSIBILITY OF THE FOREGOING.
 */

package juicebox.tools;

import htsjdk.tribble.util.LittleEndianOutputStream;
import jargs.gnu.CmdLineParser;
import juicebox.HiC;
import juicebox.data.*;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.feature.LocusScore;
import org.broad.igv.track.WindowFunction;
import org.broad.igv.util.ParsingUtils;

import java.io.*;
import java.sql.SQLException;
import java.util.*;
import java.util.regex.Pattern;

//import org.broad.igv.sam.Alignment;
//import org.broad.igv.sam.ReadMate;
//import org.broad.igv.sam.reader.AlignmentReader;
//import org.broad.igv.sam.reader.AlignmentReaderFactory;
//import htsjdk.samtools.util.CloseableIterator;

/**
 * @author Jim Robinson
 * @date 9/16/11
 */
public class HiCTools {

    private static void usage() {
        System.out.println("Usage: juicebox pairsToBin <infile> <outfile> <genomeID>");
        System.out.println("       juicebox binToPairs <infile> <outfile>");
        System.out.println("       juicebox dump <observed/oe/pearson/norm/expected/eigenvector> <NONE/VC/VC_SQRT/KR/GW_VC/GW_KR/INTER_VC/INTER_KR> <hicFile(s)> <chr1> <chr2> <BP/FRAG> <binsize> [binary outfile]");
        System.out.println("       juicebox addNorm <hicFile> [0 for no frag, 1 for no single frag]");
        System.out.println("       juicebox addGWNorm <hicFile> <min resolution>");
        System.out.println("       juicebox bigWig <bigWig path or URL> <window size in bp> [chr] [start base] [end base]");
        System.out.println("       juicebox calcKR <infile>");
        System.out.println("       juicebox pre <options> <infile> <outfile> <genomeID>");
        System.out.println("  <options>: -d only calculate intra chromosome (diagonal) [false]");
        System.out.println("           : -f <restriction site file> calculate fragment map");
        System.out.println("           : -m <int> only write cells with count above threshold m [0]");
        System.out.println("           : -q <int> filter by MAPQ score greater than or equal to q");
        System.out.println("           : -c <chromosome ID> only calculate map on specific chromosome");
        System.out.println("           : -s <statsFile> include text statistics file");
        System.out.println("           : -g <graphFile> include graph file");
        System.out.println("           : -t, --tmpdir <temporary file directory>");
        System.out.println("           : -h print help");
    }

    public static void main(String[] argv) throws IOException, CmdLineParser.UnknownOptionException, CmdLineParser.IllegalOptionValueException {

        Globals.setHeadless(true);

        CommandLineParser parser = new CommandLineParser();
        parser.parse(argv);
        String[] args = parser.getRemainingArgs();

        String tmpDir = parser.getTmpdirOption();

        if (parser.getHelpOption()) {
            usage();
            System.exit(0);
        }

        if (args.length < 1) {
            usage();
            System.exit(1);
        }

        if (args[0].equalsIgnoreCase("bigWig")) {
            if (!(args.length == 3 || args.length == 4 || args.length == 6)) {
                usage();
                System.exit(1);
            }
            String path = args[1];
            int windowSize = Integer.parseInt(args[2]);
            if (args.length == 3) {
                BigWigUtils.computeBins(path, windowSize);
            } else {
                String chr = args[3];
                if (args.length == 4) {
                    BigWigUtils.computeBins(path, chr, 0, Integer.MAX_VALUE, windowSize);
                } else {
                    int start = Integer.parseInt(args[4]) - 1;  // Convert to "zero" based coords
                    int end = Integer.parseInt(args[5]);
                    BigWigUtils.computeBins(path, chr, start, end, windowSize);

                }

            }
        } else if (args[0].equalsIgnoreCase("calcKR")) {
            if (!(args.length == 2)) {
                usage();
                System.exit(1);
            }
            NormalizationCalculations.calcKR(args[1]);

        } else if (args[0].equalsIgnoreCase("db")) {
            String[] tmp = new String[args.length - 1];
            System.arraycopy(args, 1, tmp, 0, args.length - 1);
            try {
                HiCDBUtils.main(tmp);
            } catch (SQLException e) {
                System.err.println("Sql exception: " + e.getMessage());
                e.printStackTrace();
                System.exit(1);
            }
        } else if (args[0].equalsIgnoreCase("fragmentToBed")) {
            if (args.length != 2) {
                System.out.println("Usage: juicebox fragmentToBed <fragmentFile>");
                System.exit(1);
            }
            fragmentToBed(args[1]);
        } else if (args[0].equalsIgnoreCase("bpToFrag")) {
            if (args.length != 4) {
                System.out.println("Usage: juicebox bpToFrag <fragmentFile> <inputBedFile> <outputFile>");
            }
            bpToFrag(args[1], args[2], args[3]);
        } else if (args[0].equals("pairsToBin")) {
            if (args.length != 4) {
                usage();
                System.exit(1);
            }
            String ifile = args[1];
            String ofile = args[2];
            String genomeId = args[3];
            List<Chromosome> chromosomes = loadChromosomes(genomeId);
            AsciiToBinConverter.convert(ifile, ofile, chromosomes);
        } else if (args[0].equals("binToPairs")) {
            if (args.length != 3) {
                usage();
                System.exit(1);
            }
            String ifile = args[1];
            String ofile = args[2];
            AsciiToBinConverter.convertBack(ifile, ofile);
        } else if (args[0].equals("addNorm")) {
            if (args.length < 2 || args.length > 3) {
                System.err.println("Usage: juicebox addNorm hicFile <max genome-wide resolution>");
                System.exit(1);
            }
            String file = args[1];
            if (args.length > 2) {
                int genomeWideResolution = -100;
                try {
                    genomeWideResolution = Integer.valueOf(args[2]);
                }
                catch (NumberFormatException error) {
                    System.err.println("Usage: juicebox addNorm hicFile <max genome-wide resolution>");
                    System.exit(1);
                }
                NormalizationVectorUpdater.updateHicFile(file, genomeWideResolution);
            }
            else {
                NormalizationVectorUpdater.updateHicFile(file);
            }
        } else if (args[0].equals("addGWNorm")) {
            if (args.length != 3) {
                System.err.println("Usage: juicebox addGWNorm hicFile <max genome-wide resolution>");
                System.exit(1);
            }
            String file = args[1];

            int genomeWideResolution = -100;
            try {
                genomeWideResolution = Integer.valueOf(args[2]);
            }
            catch (NumberFormatException error) {
                System.err.println("Usage: juicebox addNorm hicFile <max genome-wide resolution>");
                System.exit(1);
            }
            NormalizationVectorUpdater.addGWNorm(file, genomeWideResolution);
        } else if (args[0].equals("pre")) {
            String genomeId = "";
            try {
                genomeId = args[3];
            } catch (ArrayIndexOutOfBoundsException e) {
                System.err.println("No genome ID given");
                usage();
                System.exit(0);
            }
            List<Chromosome> chromosomes = loadChromosomes(genomeId);

            long genomeLength = 0;
            for (Chromosome c : chromosomes) {
                if (c != null)
                    genomeLength += c.getLength();
            }
            chromosomes.set(0, new Chromosome(0, "All", (int) (genomeLength / 1000)));

            String inputFile = args[1];
            String outputFile = args[2];

            Preprocessor preprocessor = new Preprocessor(new File(outputFile), genomeId, chromosomes);

            preprocessor.setIncludedChromosomes(parser.getChromosomeOption());
            preprocessor.setCountThreshold(parser.getCountThresholdOption());
            preprocessor.setMapqThreshold(parser.getMapqThresholdOption());
            preprocessor.setDiagonalsOnly(parser.getDiagonalsOption());
            preprocessor.setFragmentFile(parser.getFragmentOption());
            preprocessor.setTmpdir(tmpDir);
            preprocessor.setStatisticsFile(parser.getStatsOption());
            preprocessor.setGraphFile(parser.getGraphOption());
            preprocessor.preprocess(inputFile);
        } else if (args[0].equals("dump")) {
            //juicebox dump <observed/oe/pearson/norm/expected/eigenvector> <NONE/VC/VC_SQRT/KR> <hicFile> <chr1> <chr2> <BP/FRAG> <binsize> [outfile]")

            if (!(args[1].equals("observed") || args[1].equals("oe") ||
                    args[1].equals("pearson") || args[1].equals("norm") ||
                    args[1].equals("expected") || args[1].equals("eigenvector"))) {
                System.err.println("Matrix or vector must be one of \"observed\", \"oe\", \"pearson\", \"norm\", " +
                        "\"expected\", or \"eigenvector\".");
                usage();
                System.exit(-1);
            }

            NormalizationType norm = null;
            try {
                norm = NormalizationType.valueOf(args[2]);
            } catch (IllegalArgumentException error) {
                System.err.println("Normalization must be one of \"NONE\", \"VC\", \"VC_SQRT\", \"KR\", \"GW_KR\", \"GW_VC\", \"INTER_KR\", or \"INTER_VC\".");
                usage();
                System.exit(-1);
            }

            if (!args[3].endsWith("hic")) {
                System.err.println("Only 'hic' files are supported");
                usage();
                System.exit(-1);
            }
            List<String> files = new ArrayList<String>();
            int idx = 3;
            while (idx < args.length && args[idx].endsWith("hic")) {
                files.add(args[idx]);
                idx++;
            }
            // idx is now the next argument.  following arguments should be chr1, chr2, unit, binsize, [outfile]

            if (args.length != idx + 4 && args.length != idx + 5) {
                System.err.println("Incorrect number of arguments to \"dump\"");
                usage();
                System.exit(-1);
            }
            String chr1 = args[idx];
            String chr2 = args[idx + 1];

            Dataset dataset = null;
            if (files.size() == 1) {
                String magicString = DatasetReaderV2.getMagicString(files.get(0));

                DatasetReader reader = null;
                if (magicString.equals("HIC")) {
                    reader = new DatasetReaderV2(files.get(0));
                } else {
                    System.err.println("This version of HIC is no longer supported");
                    System.exit(-1);
                }
                dataset = reader.read();
            } else {
                DatasetReader reader = DatasetReaderFactory.getReader(files);
                if (reader == null) {
                    System.err.println("Error while reading files");
                    System.exit(-1);
                } else {
                    dataset = reader.read();
                }
            }
            List<Chromosome> chromosomeList = dataset.getChromosomes();

            Map<String, Chromosome> chromosomeMap = new HashMap<String, Chromosome>();
            for (Chromosome c : chromosomeList) {
                chromosomeMap.put(c.getName(), c);
            }

            if (!chromosomeMap.containsKey(chr1)) {
                System.err.println("Unknown chromosome: " + chr1);
                usage();
                System.exit(-1);
            }
            if (!chromosomeMap.containsKey(chr2)) {
                System.err.println("Unknown chromosome: " + chr2);
                usage();
                System.exit(-1);
            }

            HiC.Unit unit = null;

            try {
                unit = HiC.Unit.valueOf(args[idx + 2]);
            } catch (IllegalArgumentException error) {
                System.err.println("Unit must be in BP or FRAG.");
                usage();
                System.exit(1);
            }

            String binSizeSt = args[idx + 3];
            int binSize = 0;
            try {
                binSize = Integer.parseInt(binSizeSt);
            } catch (NumberFormatException e) {
                System.err.println("Integer expected for bin size.  Found: " + binSizeSt + ".");
                usage();
                System.exit(1);
            }
            HiCZoom zoom = new HiCZoom(unit, binSize);

            String type = args[1];

            //*****************************************************
            if ((type.equals("observed") || type.equals("norm")) && chr1.equals(Globals.CHR_ALL) && chr2.equals(Globals.CHR_ALL)) {
                if (zoom.getUnit() == HiC.Unit.FRAG) {
                    System.err.println("All versus All currently not supported on fragment resolution");
                    System.exit(1);
                }
                boolean includeIntra = false;
                if (args.length == idx + 5) {
                    includeIntra = true;
                }

                // Build a "whole-genome" matrix
                ArrayList<ContactRecord> recordArrayList = createWholeGenomeRecords(dataset, chromosomeList, zoom, includeIntra);

                int totalSize = 0;
                for (Chromosome c1 : chromosomeList) {
                    if (c1.getName().equals(Globals.CHR_ALL)) continue;
                    totalSize += c1.getLength() / zoom.getBinSize() + 1;
                }


                NormalizationCalculations calculations = new NormalizationCalculations(recordArrayList, totalSize);
                double[] vector = calculations.getNorm(norm);

                if (type.equals("norm")) {

                    ExpectedValueCalculation evKR = new ExpectedValueCalculation(chromosomeList, zoom.getBinSize(), null, NormalizationType.GW_KR);
                    int addY = 0;
                    // Loop through chromosomes
                    for (Chromosome chr : chromosomeList) {

                        if (chr.getName().equals(Globals.CHR_ALL)) continue;
                        final int chrIdx = chr.getIndex();
                        Matrix matrix = dataset.getMatrix(chr, chr);

                        if (matrix == null) continue;
                        MatrixZoomData zd = matrix.getZoomData(zoom);
                        Iterator<ContactRecord> iter = zd.contactRecordIterator();
                        while (iter.hasNext()) {
                            ContactRecord cr = iter.next();
                            int x = cr.getBinX();
                            int y = cr.getBinY();
                            final float counts = cr.getCounts();
                            if (vector[x + addY] > 0 && vector[y + addY] > 0 && !Double.isNaN(vector[x + addY]) && !Double.isNaN(vector[y + addY])) {
                                double value = counts / (vector[x + addY] * vector[y + addY]);
                                evKR.addDistance(chrIdx, x, y, value);
                            }
                        }

                        addY += chr.getLength() / zoom.getBinSize() + 1;
                    }
                    evKR.computeDensity();
                    double[] exp = evKR.getDensityAvg();
                    System.out.println(binSize + "\t" + vector.length + "\t" + exp.length);
                    for (double aVector : vector) {
                        System.out.println(aVector);
                    }

                    for (double aVector : exp) {
                        System.out.println(aVector);
                    }
                } else {   // type == "observed"

                    for (ContactRecord cr : recordArrayList) {
                        int x = cr.getBinX();
                        int y = cr.getBinY();
                        float value = cr.getCounts();

                        if (vector[x] != 0 && vector[y] != 0 && !Double.isNaN(vector[x]) && !Double.isNaN(vector[y])) {
                            value = (float) (value / (vector[x] * vector[y]));
                        } else {
                            value = Float.NaN;
                        }

                        System.out.println(x + "\t" + y + "\t" + value);
                    }
                }
            }

            //***********************
            else if (type.equals("oe") || type.equals("pearson") || type.equals("observed")) {
                String ofile = null;
                if (args.length == idx + 5) {
                    ofile = args[idx + 4];
                }
                dumpMatrix(dataset, chromosomeMap.get(chr1), chromosomeMap.get(chr2), norm, zoom, type, ofile);
            } else if (type.equals("norm") || type.equals("expected") || type.equals("eigenvector")) {
                PrintWriter pw = new PrintWriter(System.out);
                if (type.equals("norm")) {
                    NormalizationVector nv = dataset.getNormalizationVector(chromosomeMap.get(chr1).getIndex(), zoom, norm);
                    if (nv == null) {
                        System.err.println("Norm not available at " + chr1 + " " + binSize + " " + unit + " " + norm);
                        System.exit(-1);
                    }
                    dumpVector(pw, nv.getData(), false);

                } else if (type.equals("expected")) {
                    final ExpectedValueFunction df = dataset.getExpectedValues(zoom, norm);
                    if (df == null) {
                        System.err.println("Expected not available at " + chr1 + " " + binSize + " " + unit + " " + norm);
                        System.exit(-1);
                    }
                    int length = df.getLength();
                    if (chr1.equals("All")) { // removed cast to ExpectedValueFunctionImpl
                        dumpVector(pw, df.getExpectedValues(), false);
                    } else {
                        Chromosome c = chromosomeMap.get(chr1);
                        for (int i = 0; i < length; i++) {
                            pw.println((float) df.getExpectedValue(c.getIndex(), i));
                        }
                        pw.flush();
                        pw.close();
                    }
                } else if (type.equals("eigenvector")) {
                    dumpVector(pw, dataset.getEigenvector(chromosomeMap.get(chr1), zoom, 0, norm), true);
                }
            }
        } else {
            usage();
            System.exit(1);
        }

    }

    private static ArrayList<ContactRecord> createWholeGenomeRecords(Dataset dataset, List<Chromosome> tmp, HiCZoom zoom, boolean includeIntra) {
        ArrayList<ContactRecord> recordArrayList = new ArrayList<ContactRecord>();
        int addX = 0;
        int addY = 0;
        for (Chromosome c1 : tmp) {
            if (c1.getName().equals(Globals.CHR_ALL)) continue;
            for (Chromosome c2 : tmp) {
                if (c2.getName().equals(Globals.CHR_ALL)) continue;
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

    /**
     * Load chromosomes from given ID or file name.
     *
     * @param idOrFile Genome ID or file name where chromosome lengths written
     * @return Chromosome lengths
     * @throws IOException if chromosome length file not found
     */
    private static List<Chromosome> loadChromosomes(String idOrFile) throws IOException {

        InputStream is = null;

        try {
            // Note: to get this to work, had to edit Intellij settings
            // so that "?*.sizes" are considered sources to be copied to class path
            is = HiCTools.class.getResourceAsStream(idOrFile + ".chrom.sizes");

            if (is == null) {
                // Not an ID,  see if its a file
                File file = new File(idOrFile);
                if (file.exists()) {
                    is = new FileInputStream(file);
                } else {
                    throw new FileNotFoundException("Could not find chromosome sizes file for: " + idOrFile);
                }

            }

            List<Chromosome> chromosomes = new ArrayList<Chromosome>();
            chromosomes.add(0, null);   // Index 0 reserved for "whole genome" pseudo-chromosome

            Pattern pattern = Pattern.compile("\t");
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            String nextLine;
            long genomeLength = 0;
            int idx = 1;

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

            // Add the "pseudo-chromosome" All, representing the whole genome.  Units are in kilo-bases
            chromosomes.set(0, new Chromosome(0, "All", (int) (genomeLength / 1000)));


            return chromosomes;
        } finally {
            if (is != null) is.close();
        }

    }


    private static void bpToFrag(String fragmentFile, String inputFile, String outputDir) throws IOException {
        BufferedReader fragmentReader = null;
        Pattern pattern = Pattern.compile("\\s");
        Map<String, int[]> fragmentMap = new HashMap<String, int[]>();  // Map of chr -> site positions
        try {
            fragmentReader = new BufferedReader(new FileReader(fragmentFile));

            String nextLine;
            while ((nextLine = fragmentReader.readLine()) != null) {
                String[] tokens = pattern.split(nextLine);

                // A hack, could use IGV's genome alias definitions
                String chr = getChrAlias(tokens[0]);

                int[] sites = new int[tokens.length];
                sites[0] = 0;  // Convenient convention
                for (int i = 1; i < tokens.length; i++) {
                    sites[i] = Integer.parseInt(tokens[i]) - 1;
                }
                fragmentMap.put(chr, sites);
            }
        } finally {
            fragmentReader.close();
        }

        // inputFile contains a list of files or URLs.
        BufferedReader reader = null;
        try {
            File dir = new File(outputDir);
            if (!dir.exists() || !dir.isDirectory()) {
                System.out.println("Output directory does not exist, or is not directory");
                System.exit(1);
            }
            reader = new BufferedReader(new FileReader(inputFile));
            String nextLine;
            while ((nextLine = reader.readLine()) != null) {
                String path = nextLine.trim();
                int lastSlashIdx = path.lastIndexOf("/");
                if (lastSlashIdx < 0) lastSlashIdx = path.lastIndexOf("\\");  // Windows convention
                String fn = lastSlashIdx < 0 ? path : path.substring(lastSlashIdx);

                File outputFile = new File(dir, fn + ".sites");
                annotateWithSites(fragmentMap, path, outputFile);

            }
        } finally {
            if (reader != null) reader.close();
        }
    }


    /**
     * Find fragments that overlap the input bed file.  Its assumed the bed file is sorted by start position, otherwise
     * an exception is thrown.  If a fragment overlaps 2 or more bed featuers it is only outputted once.
     *
     * @param fragmentMap
     * @param bedFile
     * @param outputBedFile
     * @throws IOException
     */

    private static void annotateWithSites(Map<String, int[]> fragmentMap, String bedFile, File outputBedFile) throws IOException {


        BufferedReader bedReader = null;
        PrintWriter bedWriter = null;
        try {

            bedReader = ParsingUtils.openBufferedReader(bedFile);
            bedWriter = new PrintWriter(new BufferedWriter(new FileWriter(outputBedFile)));

            String nextLine;
            while ((nextLine = bedReader.readLine()) != null) {
                if (nextLine.startsWith("track") || nextLine.startsWith("browser") || nextLine.startsWith("#"))
                    continue;

                BedLikeFeature feature = new BedLikeFeature(nextLine);

                String[] tokens = Globals.whitespacePattern.split(nextLine);
                String chr = tokens[0];
                int start = Integer.parseInt(tokens[1]);
                int end = Integer.parseInt(tokens[2]);

                int[] sites = fragmentMap.get(feature.getChr());
                if (sites == null) continue;

                int firstSite = FragmentCalculation.binarySearch(sites, feature.getStart());
                int lastSite = FragmentCalculation.binarySearch(sites, feature.getEnd());

                bedWriter.print(chr + "\t" + start + "\t" + end + "\t" + firstSite + "\t" + lastSite);
                for (int i = 3; i < tokens.length; i++) {
                    bedWriter.print("\t" + tokens[i]);
                }
                bedWriter.println();


            }
        } finally {
            if (bedReader != null) bedReader.close();
            if (bedWriter != null) bedWriter.close();
        }


    }

    private static String getChrAlias(String token) {
        if (token.equals("MT")) {
            return "chrM";
        } else if (!token.startsWith("chr")) {
            return "chr" + token;
        } else {
            return token;
        }
    }

    /**
     * Convert a fragment site file to a "bed" file
     *
     * @param filename fragment site file
     * @throws IOException
     */
    private static void fragmentToBed(String filename) throws IOException {
        BufferedReader reader = null;
        PrintWriter writer = null;
        try {
            File inputFile = new File(filename);
            reader = new BufferedReader(new FileReader(inputFile));

            writer = new PrintWriter(new BufferedWriter(new FileWriter(filename + ".bed")));

            Pattern pattern = Pattern.compile("\\s");
            String nextLine;
            while ((nextLine = reader.readLine()) != null) {
                String[] tokens = pattern.split(nextLine);
                String chr = tokens[0];
                int fragNumber = 0;
                int beg = Integer.parseInt(tokens[1]) - 1;  // 1 vs 0 based coords
                for (int i = 2; i < tokens.length; i++) {
                    int end = Integer.parseInt(tokens[i]) - 1;
                    writer.println(chr + "\t" + beg + "\t" + end + "\t" + fragNumber);
                    beg = end;
                    fragNumber++;
                }

            }
        } finally {
            if (reader != null) reader.close();
        }

    }


    /**
     * Convert a BAM file containing paried-end tags to the ascii "pair" format used for HiC.
     *
     * @param inputBam
     * @param outputFile
     * @throws IOException
     */
  /*  public static void filterBam(String inputBam, String outputFile, List<Chromosome> chromosomes) throws IOException {

        CloseableIterator<Alignment> iter = null;
        AlignmentReader<Alignment> reader = null;
        PrintWriter pw = null;

        HashSet<Chromosome> allChroms = new HashSet<Chromosome>(chromosomes);

        try {
            pw = new PrintWriter(new FileWriter(outputFile));
            reader = AlignmentReaderFactory.getReader(inputBam, false);
            iter = reader.iterator();
            while (iter.hasNext()) {

                Alignment alignment = iter.next();
                ReadMate mate = alignment.getMate();

                // Filter unpaired and "normal" pairs.  Only interested in abnormals
                if (alignment.isPaired() &&
                        alignment.isMapped() &&
                        alignment.getMappingQuality() > 10 &&
                        mate != null &&
                        mate.isMapped() &&
                        allChroms.contains(alignment.getChr()) &&
                        allChroms.contains(mate.getChr()) &&
                        (!alignment.getChr().equals(mate.getChr()) || alignment.getInferredInsertSize() > 1000)) {

                    // Each pair is represented twice in the file,  keep the record with the "leftmost" coordinate
                    if (alignment.getStart() < mate.getStart()) {
                        String strand = alignment.isNegativeStrand() ? "-" : "+";
                        String mateStrand = mate.isNegativeStrand() ? "-" : "+";
                        pw.println(alignment.getReadName() + "\t" + alignment.getChr() + "\t" + alignment.getStart() +
                                "\t" + strand + "\t.\t" + mate.getChr() + "\t" + mate.getStart() + "\t" + mateStrand);
                    }
                }

            }
        } finally {
            pw.close();
            iter.close();
            reader.close();
        }
    }*/

    /**
     * Prints out a vector to the given print stream.  Mean centers if center is set
     *
     * @param pw     Stream to print to
     * @param vector Vector to print out
     * @param center Mean centers if true
     * @throws IOException
     */
    static public void dumpVector(PrintWriter pw, double[] vector, boolean center) {
        double sum = 0;
        if (center) {
            int count = 0;
            /*
            for (int idx = 0; idx < vector.length; idx++) {
                if (!Double.isNaN(vector[idx])) {
                    sum += vector[idx];
                    count++;
                }
            }
            */

            for(double element : vector){
                if (!Double.isNaN(element)) {
                    sum += element;
                    count++;
                }
            }

            sum = sum / count; // sum is now mean
        }
        // print out vector
        for (double element : vector) {
            pw.println(element - sum);
        }
        pw.flush();
        pw.close();
    }

    /**
     * Dumps the matrix.  Does more argument checking, thus this should not be called outside of this class.
     *
     * @param dataset Dataset
     * @param chr1    Chromosome 1
     * @param chr2    Chromosome 2
     * @param norm    Normalization
     * @param zoom    Zoom level
     * @param type    observed/oe/pearson
     * @param ofile   Output file string (binary output), possibly null (then prints to standard out)
     * @throws IOException
     */
    static private void dumpMatrix(Dataset dataset, Chromosome chr1, Chromosome chr2, NormalizationType norm,
                                   HiCZoom zoom, String type, String ofile) throws IOException {
        LittleEndianOutputStream les = null;
        BufferedOutputStream bos = null;

        if (ofile != null) {
            bos = new BufferedOutputStream(new FileOutputStream(ofile));
            les = new LittleEndianOutputStream(bos);
        }

        if (type.equals("oe") || type.equals("pearson")) {
            if (!chr1.equals(chr2)) {
                System.err.println("Chromosome " + chr1 + " not equal to Chromosome " + chr2);
                System.err.println("Currently only intrachromosomal O/E and Pearson's are supported.");
                usage();
                System.exit(-1);
            }
        }

        Matrix matrix = dataset.getMatrix(chr1, chr2);
        if (matrix == null) {
            System.err.println("No reads in " + chr1 + " " + chr2);
            System.exit(-1);
        }

        MatrixZoomData zd = matrix.getZoomData(zoom);
        if (zd == null) {

            System.err.println("Unknown resolution: " + zoom);
            System.err.println("This data set has the following bin sizes (in bp): ");
            for (int zoomIdx = 0; zoomIdx < dataset.getNumberZooms(HiC.Unit.BP); zoomIdx++) {
                System.err.print(dataset.getZoom(HiC.Unit.BP, zoomIdx).getBinSize() + " ");
            }
            System.err.println("\nand the following bin sizes (in frag): ");
            for (int zoomIdx = 0; zoomIdx < dataset.getNumberZooms(HiC.Unit.FRAG); zoomIdx++) {
                System.err.print(dataset.getZoom(HiC.Unit.FRAG, zoomIdx).getBinSize() + " ");
            }
            System.exit(-1);
        }

        if (type.equals("oe") || type.equals("pearson")) {
            final ExpectedValueFunction df = dataset.getExpectedValues(zd.getZoom(), norm);
            if (df == null) {
                System.err.println(type + " not available at " + chr1 + " " + zoom + " " + norm);
                System.exit(-1);
            }
            try {
                zd.dumpOE(df, type, norm, les, null);
            } finally {
                if (les != null)
                    les.close();
                if (bos != null)
                    bos.close();
            }
        } else if (type.equals("observed")) {
            double[] nv1 = null;
            double[] nv2 = null;
            if (norm != NormalizationType.NONE) {
                NormalizationVector nv = dataset.getNormalizationVector(chr1.getIndex(), zd.getZoom(), norm);
                if (nv == null) {
                    System.err.println(type + " not available at " + chr1 + " " + zoom + " " + norm);
                    System.exit(-1);
                } else {
                    nv1 = nv.getData();
                }
                if (!chr1.equals(chr2)) {
                    nv = dataset.getNormalizationVector(chr2.getIndex(), zd.getZoom(), norm);
                    if (nv == null) {
                        System.err.println(type + " not available at " + chr2 + " " + zoom + " " + norm);
                        System.exit(-1);
                    } else {
                        nv2 = nv.getData();
                    }
                } else {
                    nv2 = nv1;
                }
            }
            if (les == null) {
                zd.dump(new PrintWriter(System.out), nv1, nv2);
            } else {
                try {
                    zd.dump(les, nv1, nv2);
                } finally {
                    les.close();
                    bos.close();
                }
            }
        }
    }

    static class CommandLineParser extends CmdLineParser {
        private Option diagonalsOption = null;
        private Option chromosomeOption = null;
        private Option countThresholdOption = null;
        private Option helpOption = null;
        private Option fragmentOption = null;
        private Option tmpDirOption = null;
        private Option statsOption = null;
        private Option graphOption = null;
        private Option mapqOption = null;

        CommandLineParser() {
            diagonalsOption = addBooleanOption('d', "diagonals");
            chromosomeOption = addStringOption('c', "chromosomes");
            countThresholdOption = addIntegerOption('m', "minCountThreshold");
            fragmentOption = addStringOption('f', "restriction fragment site file");
            tmpDirOption = addStringOption('t', "tmpDir");
            helpOption = addBooleanOption('h', "help");
            statsOption = addStringOption('s', "statistics text file");
            graphOption = addStringOption('g', "graph text file");
            mapqOption = addIntegerOption('q', "mapping quality threshold");
        }

        boolean getHelpOption() {
            Object opt = getOptionValue(helpOption);
            return opt != null && (Boolean)opt;
        }

        boolean getDiagonalsOption() {
            Object opt = getOptionValue(diagonalsOption);
            return opt != null && (Boolean)opt;
        }

        Set<String> getChromosomeOption() {
            Object opt = getOptionValue(chromosomeOption);
            if (opt != null) {
                String[] tokens = opt.toString().split(",");
                return new HashSet<String>(Arrays.asList(tokens));
            } else {
                return null;
            }
        }

        String getFragmentOption() {
            Object opt = getOptionValue(fragmentOption);
            if (opt != null) {
                return opt.toString();
            } else {
                return null;
            }
        }

        String getStatsOption() {
            Object opt = getOptionValue(statsOption);
            if (opt != null) {
                return opt.toString();
            } else {
                return null;
            }
        }

        String getGraphOption() {
            Object opt = getOptionValue(graphOption);
            if (opt != null) {
                return opt.toString();
            } else {
                return null;
            }
        }

        String getTmpdirOption() {
            Object opt = getOptionValue(tmpDirOption);
            if (opt != null) {
                return opt.toString();
            } else {
                return null;
            }
        }


        int getCountThresholdOption() {
            Object opt = getOptionValue(countThresholdOption);
            return opt == null ? 0 : ((Number) opt).intValue();
        }

        int getMapqThresholdOption() {
            Object opt = getOptionValue(mapqOption);
            return opt == null ? 0 : ((Number) opt).intValue();
        }

    }


    static class BedLikeFeature implements LocusScore {

        final String chr;
        int start;
        int end;
        String name;
        final String line;

        BedLikeFeature(String line) {
            this.line = line;
            String[] tokens = Globals.whitespacePattern.split(line);
            this.chr = tokens[0];
            this.start = Integer.parseInt(tokens[1]);
            this.end = Integer.parseInt(tokens[2]);
            if (tokens.length > 3) {
                this.name = name; // TODO - is this supposed to be this.name = tokens[x]? otherwise a redundant line
            }

        }

        @Override
        public String getValueString(double position, WindowFunction windowFunction) {
            return line;
        }

        public String getChr() {
            return chr;
        }

        public int getStart() {
            return start;
        }

        public void setStart(int start) {
            this.start = start;
        }

        public int getEnd() {
            return end;
        }

        public void setEnd(int end) {
            this.end = end;
        }

        public float getScore() {
            return 0;
        }
    }
}
