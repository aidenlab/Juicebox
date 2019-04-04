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

//import juicebox.MainWindow;

import htsjdk.tribble.util.LittleEndianInputStream;
import htsjdk.tribble.util.LittleEndianOutputStream;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.ContactRecord;
import juicebox.tools.clt.CommandLineParser.Alignment;
import juicebox.windowui.NormalizationHandler;
import org.apache.commons.math.stat.StatUtils;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.tdf.BufferedByteWriter;
import org.broad.igv.util.collections.DownsampledDoubleArrayList;

import java.awt.*;
import java.io.*;
import java.util.List;
import java.util.*;
import java.util.zip.Deflater;


/**
 * @author jrobinso
 * @since Aug 16, 2010
 */
public class Preprocessor {


    private static final int VERSION = 8;
    private static final int BLOCK_SIZE = 1000;
    public static final String HIC_FILE_SCALING = "hicFileScalingFactor";
    public static final String STATISTICS = "statistics";
    public static final String GRAPHS = "graphs";
    public static final String SOFTWARE = "software";
    private static final String NVI_INDEX = "nviIndex";
    private static final String NVI_LENGTH = "nviLength";

    private final ChromosomeHandler chromosomeHandler;
    private final Map<String, Integer> chromosomeIndexes;
    private final File outputFile;
    private final Map<String, IndexEntry> matrixPositions;
    private final String genomeId;
    private final Deflater compressor;
    private LittleEndianOutputStream los;
    private long masterIndexPosition;
    private int countThreshold = 0;
    private int mapqThreshold = 0;
    private boolean diagonalsOnly = false;
    private String fragmentFileName = null;
    private String statsFileName = null;
    private String graphFileName = null;
    private String expectedVectorFile = null;
    private Set<String> randomizeFragMapFiles = null;
    private FragmentCalculation fragmentCalculation = null;
    private Set<String> includedChromosomes;
    private ArrayList<FragmentCalculation> fragmentCalculationsForRandomization = null;
    private Alignment alignmentFilter;
    private static final Random random = new Random();
    private static boolean allowPositionsRandomization = false;

    // Base-pair resolutions
    private int[] bpBinSizes = {2500000, 1000000, 500000, 250000, 100000, 50000, 25000, 10000, 5000, 1000};

    // Fragment resolutions
    private int[] fragBinSizes = {500, 200, 100, 50, 20, 5, 2, 1};

    // hic scaling factor value
    private double hicFileScalingFactor = 1;


    /**
     * The position of the field containing the masterIndex position
     */
    private long masterIndexPositionPosition;
    private long normVectorIndexPosition;
    private long normVectorLengthPosition;
    private Map<String, ExpectedValueCalculation> expectedValueCalculations;
    private File tmpDir;

    public Preprocessor(File outputFile, String genomeId, ChromosomeHandler chromosomeHandler, double hicFileScalingFactor) {
        this.genomeId = genomeId;
        this.outputFile = outputFile;
        this.matrixPositions = new LinkedHashMap<>();

        this.chromosomeHandler = chromosomeHandler;
        chromosomeIndexes = new Hashtable<>();
        for (int i = 0; i < chromosomeHandler.size(); i++) {
            chromosomeIndexes.put(chromosomeHandler.getChromosomeFromIndex(i).getName(), i);
        }

        compressor = new Deflater();
        compressor.setLevel(Deflater.DEFAULT_COMPRESSION);

        this.tmpDir = null;  // TODO -- specify this

        if (hicFileScalingFactor > 0) {
            this.hicFileScalingFactor = hicFileScalingFactor;
        }

    }

    public void setCountThreshold(int countThreshold) {
        this.countThreshold = countThreshold;
    }

    public void setMapqThreshold(int mapqThreshold) {
        this.mapqThreshold = mapqThreshold;
    }

    public void setDiagonalsOnly(boolean diagonalsOnly) {
        this.diagonalsOnly = diagonalsOnly;
    }

    public void setIncludedChromosomes(Set<String> includedChromosomes) {
        if (includedChromosomes != null && includedChromosomes.size() > 0) {
            this.includedChromosomes = new HashSet<>();
            for (String name : includedChromosomes) {
                this.includedChromosomes.add(ChromosomeHandler.cleanUpName(name));
            }
        }
    }

    public void setFragmentFile(String fragmentFileName) {
        this.fragmentFileName = fragmentFileName;
    }

    public void setExpectedVectorFile(String expectedVectorFile) {
        this.expectedVectorFile = expectedVectorFile;
    }

    public void setGraphFile(String graphFileName) {
        this.graphFileName = graphFileName;
    }

    public void setResolutions(Set<String> resolutions) {
        if (resolutions != null) {
            ArrayList<Integer> fragResolutions = new ArrayList<>();
            ArrayList<Integer> bpResolutions = new ArrayList<>();

            for (String str:resolutions) {
                boolean fragment = false;
                int index = str.indexOf("f");
                if (index != -1) {
                    str = str.substring(0, index);
                    fragment = true;
                }
                Integer myInt = null;
                try {
                    myInt = Integer.valueOf(str);
                } catch (NumberFormatException exception) {
                    System.err.println("Resolution improperly formatted.  It must be in the form of a number, such as 1000000 for 1M bp,");
                    System.err.println("or a number followed by 'f', such as 25f for 25 fragment");
                    System.exit(1);
                }
                if (fragment) fragResolutions.add(myInt);
                else          bpResolutions.add(myInt);
            }

            boolean resolutionsSet = false;
            if (fragResolutions.size() > 0) {
                resolutionsSet = true;
                Collections.sort(fragResolutions);
                Collections.reverse(fragResolutions);
                int[] frags = new int[fragResolutions.size()];
                for (int i=0; i<frags.length; i++){
                    frags[i] = fragResolutions.get(i);
                }
                fragBinSizes = frags;
            }
            else {
                fragBinSizes = new int[0];
            }
            if (bpResolutions.size() > 0) {
                resolutionsSet = true;
                Collections.sort(bpResolutions);
                Collections.reverse(bpResolutions);
                int[] bps = new int[bpResolutions.size()];
                for (int i = 0; i < bps.length; i++) {
                    bps[i] = bpResolutions.get(i);
                }
                bpBinSizes = bps;
            }
            else {
                bpBinSizes = new int[0];
            }
            if (!resolutionsSet) {
                System.err.println("No valid resolutions sent in");
                System.exit(1);
            }
        }
    }

    public void setAlignmentFilter(Alignment al) {
        this.alignmentFilter = al;
    }

    public void setRandomizeFragMaps(Set<String> fragMaps) {
        this.randomizeFragMapFiles = fragMaps;
    }

    private static int randomizePos(FragmentCalculation fragmentCalculation, String chr, int frag) {

        int low = 1;
        int high = 1;
        if (frag == 0) {
            high = fragmentCalculation.getSites(chr)[frag];
        } else if (frag >= fragmentCalculation.getNumberFragments(chr)) {
            high = fragmentCalculation.getSites(chr)[frag - 1];
            low = fragmentCalculation.getSites(chr)[frag - 2];
        } else {
            high = fragmentCalculation.getSites(chr)[frag];
            low = fragmentCalculation.getSites(chr)[frag - 1];
        }
        return random.nextInt(high - low + 1) + low;
    }

    public void setRandomizePosition(boolean allowPositionsRandomization) {
        Preprocessor.allowPositionsRandomization = allowPositionsRandomization;
    }

    private static FragmentCalculation findFragMap(List<FragmentCalculation> maps, String chr, int bp, int frag) {
        //potential maps that this strand could come from
        ArrayList<FragmentCalculation> mapsFound = new ArrayList<>();
        for (FragmentCalculation fragmentCalculation : maps) {
            int low = 1;
            int high = 1;

            if (frag > fragmentCalculation.getNumberFragments(chr)) {
                // definitely not this restriction site file for certain
                continue;
            }
            
            try {
                if (frag == 0) {
                    high = fragmentCalculation.getSites(chr)[frag];
                } else if (frag == fragmentCalculation.getNumberFragments(chr)) {
                    high = fragmentCalculation.getSites(chr)[frag - 1];
                    low = fragmentCalculation.getSites(chr)[frag - 2];
                } else {
                    high = fragmentCalculation.getSites(chr)[frag];
                    low = fragmentCalculation.getSites(chr)[frag - 1];
                }
            } catch (Exception e) {
                e.printStackTrace();
                System.out.println(String.format("fragment: %d, number of frags: %d", frag, fragmentCalculation.getNumberFragments(chr)));

            }

            // does bp fit in this range?
            if (bp >= low && bp <= high) {
                mapsFound.add(fragmentCalculation);
            }
        }
        if (mapsFound.size() == 1) {
            return mapsFound.get(0);
        }
        return null;
    }


    public void preprocess(final String inputFile) throws IOException {
        File file = new File(inputFile);

        if (!file.exists() || file.length() == 0) {
            System.err.println(inputFile + " does not exist or does not contain any reads.");
            System.exit(57);
        }

        try {
            StringBuilder stats = null;
            StringBuilder graphs = null;
            StringBuilder hicFileScaling = new StringBuilder().append(hicFileScalingFactor);
            if (fragmentFileName != null) {
                try {
                    fragmentCalculation = FragmentCalculation.readFragments(fragmentFileName);
                } catch (Exception e) {
                    System.err.println("Warning: Unable to process fragment file. Pre will continue without fragment file.");
                    fragmentCalculation = null;
                }
            } else {
                System.out.println("Not including fragment map");
            }

            if (allowPositionsRandomization) {
                if (randomizeFragMapFiles != null) {
                    fragmentCalculationsForRandomization = new ArrayList<>();
                    for (String fragmentFileName : randomizeFragMapFiles) {
                        try {
                            FragmentCalculation fragmentCalculation = FragmentCalculation.readFragments(fragmentFileName);
                            fragmentCalculationsForRandomization.add(fragmentCalculation);
                            System.out.println(String.format("added %s", fragmentFileName));
                        } catch (Exception e) {
                            System.err.println(String.format("Warning: Unable to process fragment file %s. Randomization will continue without fragment file %s.", fragmentFileName, fragmentFileName));
                        }
                    }
                } else {
                    System.out.println("Using default fragment map for randomization");
                }

            } else if (randomizeFragMapFiles != null) {
                System.err.println("Position randomizer seed not set, disregarding map options");
            }

            if (statsFileName != null) {
                FileInputStream is = null;
                try {
                    is = new FileInputStream(statsFileName);
                    BufferedReader reader = new BufferedReader(new InputStreamReader(is), HiCGlobals.bufferSize);
                    stats = new StringBuilder();
                    String nextLine;
                    while ((nextLine = reader.readLine()) != null) {
                        stats.append(nextLine).append("\n");
                    }
                } catch (IOException e) {
                    System.err.println("Error while reading stats file: " + e);
                    stats = null;
                } finally {
                    if (is != null) {
                        is.close();
                    }
                }

            }
            if (graphFileName != null) {
                FileInputStream is = null;
                try {
                    is = new FileInputStream(graphFileName);
                    BufferedReader reader = new BufferedReader(new InputStreamReader(is), HiCGlobals.bufferSize);
                    graphs = new StringBuilder();
                    String nextLine;
                    while ((nextLine = reader.readLine()) != null) {
                        graphs.append(nextLine).append("\n");
                    }
                } catch (IOException e) {
                    System.err.println("Error while reading graphs file: " + e);
                    graphs = null;
                } finally {
                    if (is != null) {
                        is.close();
                    }
                }
            }

            if (expectedVectorFile == null) {
                expectedValueCalculations = new LinkedHashMap<>();
                for (int bBinSize : bpBinSizes) {
                    ExpectedValueCalculation calc = new ExpectedValueCalculation(chromosomeHandler, bBinSize, null, NormalizationHandler.NONE);
                    String key = "BP_" + bBinSize;
                    expectedValueCalculations.put(key, calc);
                }
            }
            if (fragmentCalculation != null) {

                // Create map of chr name -> # of fragments
                Map<String, int[]> sitesMap = fragmentCalculation.getSitesMap();
                Map<String, Integer> fragmentCountMap = new HashMap<>();
                for (Map.Entry<String, int[]> entry : sitesMap.entrySet()) {
                    int fragCount = entry.getValue().length + 1;
                    String chr = entry.getKey();
                    fragmentCountMap.put(chr, fragCount);
                }

                if (expectedVectorFile == null) {
                    for (int fBinSize : fragBinSizes) {
                        ExpectedValueCalculation calc = new ExpectedValueCalculation(chromosomeHandler, fBinSize, fragmentCountMap, NormalizationHandler.NONE);
                        String key = "FRAG_" + fBinSize;
                        expectedValueCalculations.put(key, calc);
                    }
                }
            }

            try {
                los = new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(outputFile), HiCGlobals.bufferSize));
            } catch (Exception e) {
                System.err.println("Unable to write to " + outputFile);
                System.exit(70);
            }

            System.out.println("Start preprocess");

            System.out.println("Writing header");

            writeHeader(stats, graphs, hicFileScaling);

            System.out.println("Writing body");
            writeBody(inputFile);

            System.out.println();
            System.out.println("Writing footer");
            writeFooter();


        } finally {
            if (los != null)
                los.close();
        }

        updateMasterIndex();
        System.out.println("\nFinished preprocess");
    }

    private void writeHeader(StringBuilder stats, StringBuilder graphs, StringBuilder hicFileScaling) throws IOException {
        // Magic number
        byte[] magicBytes = "HIC".getBytes();
        los.write(magicBytes[0]);
        los.write(magicBytes[1]);
        los.write(magicBytes[2]);
        los.write(0);

        // VERSION
        los.writeInt(VERSION);

        // Placeholder for master index position, replaced with actual position after all contents are written
        masterIndexPositionPosition = los.getWrittenCount();
        los.writeLong(0L);


        // Genome ID
        los.writeString(genomeId);

        // Attribute dictionary
        int nAttributes = 1;
        if (stats != null) nAttributes += 1;
        if (graphs != null) nAttributes += 1;
        if (hicFileScaling != null) nAttributes += 1;
        nAttributes += 2; // NVI info

        los.writeInt(nAttributes);
        los.writeString(SOFTWARE);
        los.writeString("Juicer Tools Version " + HiCGlobals.versionNum);
        if (stats != null) {
            los.writeString(STATISTICS);
            los.writeString(stats.toString());
        }
        if (graphs != null) {
            los.writeString(GRAPHS);
            los.writeString(graphs.toString());
        }
        if (hicFileScaling != null) {
            los.writeString(HIC_FILE_SCALING);
            los.writeString(hicFileScaling.toString());
        }

        // Add NVI info
        los.writeString(NVI_INDEX);
        normVectorIndexPosition = los.getWrittenCount();
        los.writeString("0000000000000000");

        los.writeString(NVI_LENGTH);
        normVectorLengthPosition = los.getWrittenCount();
        los.writeString("0000000000000000");


        // Sequence dictionary
        int nChrs = chromosomeHandler.size();
        los.writeInt(nChrs);
        for (Chromosome chromosome : chromosomeHandler.getChromosomeArray()) {
            los.writeString(chromosome.getName());
            los.writeInt(chromosome.getLength());
        }

        //BP resolution levels
        int nBpRes = bpBinSizes.length;
        los.writeInt(nBpRes);
        for (int bpBinSize : bpBinSizes) {
            los.writeInt(bpBinSize);
        }

        //fragment resolutions
        int nFragRes = fragmentCalculation == null ? 0 : fragBinSizes.length;
        los.writeInt(nFragRes);
        for (int i = 0; i < nFragRes; i++) {
            los.writeInt(fragBinSizes[i]);
        }

        // fragment sites
        if (nFragRes > 0) {
            for (Chromosome chromosome : chromosomeHandler.getChromosomeArray()) {
                int[] sites = fragmentCalculation.getSites(chromosome.getName());
                int nSites = sites == null ? 0 : sites.length;
                los.writeInt(nSites);
                for (int i = 0; i < nSites; i++) {
                    los.writeInt(sites[i]);
                }
            }
        }
    }

    public void setPositionRandomizerSeed(long randomSeed) {
        random.setSeed(randomSeed);
    }


    /**
     * @param file List of files to read
     * @return Matrix with counts in each bin
     * @throws IOException
     */
    private MatrixPP computeWholeGenomeMatrix(String file, Alignment alignmentFilter) throws IOException {


        MatrixPP matrix;
        // NOTE: always true that c1 <= c2

        int genomeLength = chromosomeHandler.getChromosomeFromIndex(0).getLength();  // <= whole genome in KB
        int binSize = genomeLength / 500;
        if (binSize == 0) binSize = 1;
        int nBinsX = genomeLength / binSize + 1;
        int nBlockColumns = nBinsX / BLOCK_SIZE + 1;
        matrix = new MatrixPP(0, 0, binSize, nBlockColumns);

        PairIterator iter = null;

        int belowMapq = 0;
        int intraFrag = 0;
        int totalRead = 0;
        int contig = 0;
        int hicContact = 0;

        // Create an index the first time through
        try {
            iter = (file.endsWith(".bin")) ?
                    new BinPairIterator(file) :
                    new AsciiPairIterator(file, chromosomeIndexes);

            while (iter.hasNext()) {
                totalRead++;
                AlignmentPair pair = iter.next();
                if (pair.isContigPair()) {
                    contig++;
                } else {
                    int bp1 = pair.getPos1();
                    int bp2 = pair.getPos2();
                    int chr1 = pair.getChr1();
                    int chr2 = pair.getChr2();
                    int frag1 = pair.getFrag1();
                    int frag2 = pair.getFrag2();
                    int mapq1 = pair.getMapq1();
                    int mapq2 = pair.getMapq2();

                    int pos1, pos2;
                    if (diagonalsOnly && chr1 != chr2) continue;
                    if (includedChromosomes != null && chr1 != 0) {
                        String c1Name = chromosomeHandler.getChromosomeFromIndex(chr1).getName();
                        String c2Name = chromosomeHandler.getChromosomeFromIndex(chr2).getName();
                        if (!(includedChromosomes.contains(c1Name) || includedChromosomes.contains(c2Name))) {
                            continue;
                        }
                    }
                    
                    if (alignmentFilter != null && calculateAlignment(pair) != alignmentFilter) {
                        continue;
                    }


                    if (chr1 == chr2 && frag1 == frag2) {
                        intraFrag++;
                    } else if (mapq1 < mapqThreshold || mapq2 < mapqThreshold) {
                        belowMapq++;
                    } else {
                        pos1 = getGenomicPosition(chr1, bp1);
                        pos2 = getGenomicPosition(chr2, bp2);
                        matrix.incrementCount(pos1, pos2, pos1, pos2, pair.getScore());
                        hicContact++;
                    }
                }
            }
        } finally {
            if (iter != null) iter.close();
        }

        /*
            Intra-fragment Reads: 2,321 (0.19% / 0.79%)
            Below MAPQ Threshold: 44,134 (3.57% / 15.01%)
            Hi-C Contacts: 247,589 (20.02% / 84.20%)
             Ligation Motif Present: 99,245  (8.03% / 33.75%)
             3' Bias (Long Range): 73% - 27%
             Pair Type %(L-I-O-R): 25% - 25% - 25% - 25%
            Inter-chromosomal: 58,845  (4.76% / 20.01%)
            Intra-chromosomal: 188,744  (15.27% / 64.19%)
            Short Range (<20Kb): 48,394  (3.91% / 16.46%)
            Long Range (>20Kb): 140,350  (11.35% / 47.73%)

        System.err.println("contig: " + contig + " total: " + totalRead + " below mapq: " + belowMapq + " intra frag: " + intraFrag); */

        matrix.parsingComplete();
        return matrix;
    }


    private int getGenomicPosition(int chr, int pos) {
        long len = 0;
        for (int i = 1; i < chr; i++) {
            len += chromosomeHandler.getChromosomeFromIndex(i).getLength();
        }
        len += pos;

        return (int) (len / 1000);

    }

    private static Alignment calculateAlignment(AlignmentPair pair) {

        if (pair.getStrand1() == pair.getStrand2()) {
            if (pair.getStrand1()) {
                return Alignment.RR;
            } else {
                return Alignment.LL;
            }
        } else if (pair.getStrand1()) {
            if (pair.getPos1() < pair.getPos2()) {
                return Alignment.INNER;
            } else {
                return Alignment.OUTER;
            }
        } else {
            if (pair.getPos1() < pair.getPos2()) {
                return Alignment.OUTER;
            } else {
                return Alignment.INNER;
            }
        }
    }

    private void writeBody(String inputFile) throws IOException {
        MatrixPP wholeGenomeMatrix = computeWholeGenomeMatrix(inputFile, this.alignmentFilter);

        writeMatrix(wholeGenomeMatrix);

        PairIterator iter = (inputFile.endsWith(".bin")) ?
                new BinPairIterator(inputFile) :
                new AsciiPairIterator(inputFile, chromosomeIndexes);


        int currentChr1 = -1;
        int currentChr2 = -1;
        MatrixPP currentMatrix = null;
        HashSet<String> writtenMatrices = new HashSet<>();
        String currentMatrixKey = null;

        // randomization error/ambiguity stats
        int noMapFoundCount = 0;
        int mapDifferentCount = 0;

        while (iter.hasNext()) {
            AlignmentPair pair = iter.next();
            // skip pairs that mapped to contigs
            if (!pair.isContigPair()) {
                // Flip pair if needed so chr1 < chr2
                int chr1, chr2, bp1, bp2, frag1, frag2, mapq;
                if (pair.getChr1() < pair.getChr2()) {
                    bp1 = pair.getPos1();
                    bp2 = pair.getPos2();
                    frag1 = pair.getFrag1();
                    frag2 = pair.getFrag2();
                    chr1 = pair.getChr1();
                    chr2 = pair.getChr2();
                } else {
                    bp1 = pair.getPos2();
                    bp2 = pair.getPos1();
                    frag1 = pair.getFrag2();
                    frag2 = pair.getFrag1();
                    chr1 = pair.getChr2();
                    chr2 = pair.getChr1();
                }
                mapq = Math.min(pair.getMapq1(), pair.getMapq2());
                // Filters
                if (diagonalsOnly && chr1 != chr2) continue;
                if (includedChromosomes != null && chr1 != 0) {
                    String c1Name = chromosomeHandler.getChromosomeFromIndex(chr1).getName();
                    String c2Name = chromosomeHandler.getChromosomeFromIndex(chr2).getName();
                    if (!(includedChromosomes.contains(c1Name) || includedChromosomes.contains(c2Name))) {
                        continue;
                    }
                }
                if (alignmentFilter != null && calculateAlignment(pair) != alignmentFilter) {
                    continue;
                }

                // Randomize
                // Randomize
                if (fragmentCalculation != null && allowPositionsRandomization) {
                    FragmentCalculation fragMapToUse;
                    if (fragmentCalculationsForRandomization != null) {
                        FragmentCalculation fragMap1 = findFragMap(fragmentCalculationsForRandomization, chromosomeHandler.getChromosomeFromIndex(chr1).getName(), bp1, frag1);
                        FragmentCalculation fragMap2 = findFragMap(fragmentCalculationsForRandomization, chromosomeHandler.getChromosomeFromIndex(chr2).getName(), bp2, frag2);

                        if (fragMap1 == null && fragMap2 == null) {
                            noMapFoundCount += 1;
                            continue;
                        } else if (fragMap1 != null && fragMap2 != null && fragMap1 != fragMap2) {
                            mapDifferentCount += 1;
                            continue;
                        }

                        if (fragMap1 != null) {
                            fragMapToUse = fragMap1;
                        } else {
                            fragMapToUse = fragMap2;
                        }

                    } else {
                        // use default map
                        fragMapToUse = fragmentCalculation;
                    }


                    bp1 = randomizePos(fragMapToUse, chromosomeHandler.getChromosomeFromIndex(chr1).getName(), frag1);
                    bp2 = randomizePos(fragMapToUse, chromosomeHandler.getChromosomeFromIndex(chr2).getName(), frag2);
                }
                // only increment if not intraFragment and passes the mapq threshold
                if (mapq < mapqThreshold || (chr1 == chr2 && frag1 == frag2)) continue;
                if (!(currentChr1 == chr1 && currentChr2 == chr2)) {
                    // Starting a new matrix
                    if (currentMatrix != null) {
                        currentMatrix.parsingComplete();
                        writeMatrix(currentMatrix);
                        writtenMatrices.add(currentMatrixKey);
                        currentMatrix = null;
                        System.gc();
                        //System.out.println("Available memory: " + RuntimeUtils.getAvailableMemory());
                    }

                    // Start the next matrix
                    currentChr1 = chr1;
                    currentChr2 = chr2;
                    currentMatrixKey = currentChr1 + "_" + currentChr2;

                    if (writtenMatrices.contains(currentMatrixKey)) {
                        System.err.println("Error: the chromosome combination " + currentMatrixKey + " appears in multiple blocks");
                        if (outputFile != null) outputFile.deleteOnExit();
                        System.exit(58);
                    }
                    currentMatrix = new MatrixPP(currentChr1, currentChr2);
                }
                currentMatrix.incrementCount(bp1, bp2, frag1, frag2, pair.getScore());

            }
        }

        System.out.println(String.format("Randomization errors encountered: %d no map found, " +
                "%d two different maps found", noMapFoundCount, mapDifferentCount));

        if (currentMatrix != null) {
            currentMatrix.parsingComplete();
            writeMatrix(currentMatrix);
        }

        if (iter != null) iter.close();


        masterIndexPosition = los.getWrittenCount();
    }

    private void updateMasterIndex() throws IOException {
        RandomAccessFile raf = null;
        try {
            raf = new RandomAccessFile(outputFile, "rw");

            // Master index
            raf.getChannel().position(masterIndexPositionPosition);
            BufferedByteWriter buffer = new BufferedByteWriter();
            buffer.putLong(masterIndexPosition);
            raf.write(buffer.getBytes());

        } finally {
            if (raf != null) raf.close();
        }
    }


    /* todo
    private void updateNormVectorIndexInfo() throws IOException {
        RandomAccessFile raf = null;
        try {
            raf = new RandomAccessFile(outputFile, "rw");

            // NVI index
            raf.getChannel().position(normVectorIndexPosition);
            BufferedByteWriter buffer = new BufferedByteWriter();
            generateZeroPaddedString
            buffer.putNullTerminatedString(normVectorIndex);
            raf.write(buffer.getBytes());


            // NVI length
            raf.getChannel().position(normVectorLengthPosition);
            buffer = new BufferedByteWriter();
            buffer.putNullTerminatedString(normVectorLength);
            raf.write(buffer.getBytes());

        } finally {
            if (raf != null) raf.close();
        }
    }
    */


    private void writeFooter() throws IOException {

        // Index
        BufferedByteWriter buffer = new BufferedByteWriter();
        buffer.putInt(matrixPositions.size());
        for (Map.Entry<String, IndexEntry> entry : matrixPositions.entrySet()) {
            buffer.putNullTerminatedString(entry.getKey());
            buffer.putLong(entry.getValue().position);
            buffer.putInt(entry.getValue().size);
        }

        // Vectors  (Expected values,  other).
        /***  NEVA ***/
        if (expectedVectorFile == null) {
            buffer.putInt(expectedValueCalculations.size());
            for (Map.Entry<String, ExpectedValueCalculation> entry : expectedValueCalculations.entrySet()) {
                ExpectedValueCalculation ev = entry.getValue();

                ev.computeDensity();

                int binSize = ev.getGridSize();
                HiC.Unit unit = ev.isFrag ? HiC.Unit.FRAG : HiC.Unit.BP;

                buffer.putNullTerminatedString(unit.toString());
                buffer.putInt(binSize);

                // The density values
                double[] expectedValues = ev.getDensityAvg();
                buffer.putInt(expectedValues.length);
                for (double expectedValue : expectedValues) {
                    buffer.putDouble(expectedValue);
                }

                // Map of chromosome index -> normalization factor
                Map<Integer, Double> normalizationFactors = ev.getChrScaleFactors();
                buffer.putInt(normalizationFactors.size());
                for (Map.Entry<Integer, Double> normFactor : normalizationFactors.entrySet()) {
                    buffer.putInt(normFactor.getKey());
                    buffer.putDouble(normFactor.getValue());
                    //System.out.println(normFactor.getKey() + "  " + normFactor.getValue());
                }
            }
        }
        else {
            // read in expected vector file. to get # of resolutions, might have to read twice.

            int count=0;
            try (Reader reader = new FileReader(expectedVectorFile);
                 BufferedReader bufferedReader = new BufferedReader(reader)) {

                String line;
                while ((line = bufferedReader.readLine()) != null) {
                    if (line.startsWith("fixedStep"))
                        count++;
                    if (line.startsWith("variableStep")) {
                        System.err.println("Expected vector file must be in wiggle fixedStep format");
                        System.exit(19);
                    }
                }
            }
            buffer.putInt(count);
            try (Reader reader = new FileReader(expectedVectorFile);
                 BufferedReader bufferedReader = new BufferedReader(reader)) {

                String line;
                while ((line = bufferedReader.readLine()) != null) {
                    if (line.startsWith("fixedStep")) {
                        String[] words = line.split("\\s+");
                        for (String str:words){
                            if (str.contains("chrom")){
                                String[] chrs = str.split("=");
                                
                            }
                        }
                    }
                        // parse linef ixedStep  chrom=chrN
                    //start=position  step=stepInterval
                }
            }
        }

        byte[] bytes = buffer.getBytes();
        los.writeInt(bytes.length);
        los.write(bytes);
    }

    private synchronized void writeMatrix(MatrixPP matrix) throws IOException {

        long position = los.getWrittenCount();

        los.writeInt(matrix.getChr1Idx());
        los.writeInt(matrix.getChr2Idx());
        int numResolutions = 0;

        for (MatrixZoomDataPP zd : matrix.getZoomData()) {
            if (zd != null) {
                numResolutions++;
            }
        }
        los.writeInt(numResolutions);

        //fos.writeInt(matrix.getZoomData().length);
        for (MatrixZoomDataPP zd : matrix.getZoomData()) {
            if (zd != null)
                writeZoomHeader(zd);
        }

        int size = (int) (los.getWrittenCount() - position);
        matrixPositions.put(matrix.getKey(), new IndexEntry(position, size));

        for (MatrixZoomDataPP zd : matrix.getZoomData()) {
            if (zd != null) {
                List<IndexEntry> blockIndex = zd.mergeAndWriteBlocks();
                zd.updateIndexPositions(blockIndex);
            }
        }

        System.out.print(".");
    }

    private void writeZoomHeader(MatrixZoomDataPP zd) throws IOException {

        int numberOfBlocks = zd.blockNumbers.size();
        los.writeString(zd.getUnit().toString());  // Unit
        los.writeInt(zd.getZoom());     // zoom index,  lowest res is zero
        los.writeFloat((float) zd.getSum());      // sum
        los.writeFloat((float) zd.getOccupiedCellCount());
        los.writeFloat((float) zd.getPercent5());
        los.writeFloat((float) zd.getPercent95());
        los.writeInt(zd.getBinSize());
        los.writeInt(zd.getBlockBinCount());
        los.writeInt(zd.getBlockColumnCount());
        los.writeInt(numberOfBlocks);

        zd.blockIndexPosition = los.getWrittenCount();

        // Placeholder for block index
        for (int i = 0; i < numberOfBlocks; i++) {
            los.writeInt(0);
            los.writeLong(0L);
            los.writeInt(0);
        }

    }

    /**
     * Note -- compressed
     *
     * @param zd          Matrix zoom data
     * @param block       Block to write
     * @param sampledData Array to hold a sample of the data (to compute statistics)
     * @throws IOException
     */
    private void writeBlock(MatrixZoomDataPP zd, BlockPP block, DownsampledDoubleArrayList sampledData) throws IOException {

        final Map<Point, ContactCount> records = block.getContactRecordMap();//   getContactRecords();

        // System.out.println("Write contact records : records count = " + records.size());

        // Count records first
        int nRecords;
        if (countThreshold > 0) {
            nRecords = 0;
            for (ContactCount rec : records.values()) {
                if (rec.getCounts() >= countThreshold) {
                    nRecords++;
                }
            }
        } else {
            nRecords = records.size();
        }
        BufferedByteWriter buffer = new BufferedByteWriter(nRecords * 12);
        buffer.putInt(nRecords);
        zd.cellCount += nRecords;


        // Find extents of occupied cells
        int binXOffset = Integer.MAX_VALUE;
        int binYOffset = Integer.MAX_VALUE;
        int binXMax = 0;
        int binYMax = 0;
        for (Map.Entry<Point, ContactCount> entry : records.entrySet()) {
            Point point = entry.getKey();
            binXOffset = Math.min(binXOffset, point.x);
            binYOffset = Math.min(binYOffset, point.y);
            binXMax = Math.max(binXMax, point.x);
            binYMax = Math.max(binYMax, point.y);
        }


        buffer.putInt(binXOffset);
        buffer.putInt(binYOffset);


        // Sort keys in row-major order
        List<Point> keys = new ArrayList<>(records.keySet());
        Collections.sort(keys, new Comparator<Point>() {
            @Override
            public int compare(Point o1, Point o2) {
                if (o1.y != o2.y) {
                    return o1.y - o2.y;
                } else {
                    return o1.x - o2.x;
                }
            }
        });
        Point lastPoint = keys.get(keys.size() - 1);
        final short w = (short) (binXMax - binXOffset + 1);

        boolean isInteger = true;
        float maxCounts = 0;

        LinkedHashMap<Integer, List<ContactRecord>> rows = new LinkedHashMap<>();
        for (Point point : keys) {
            final ContactCount contactCount = records.get(point);
            float counts = contactCount.getCounts();
            if (counts >= countThreshold) {

                isInteger = isInteger && (Math.floor(counts) == counts);
                maxCounts = Math.max(counts, maxCounts);

                final int px = point.x - binXOffset;
                final int py = point.y - binYOffset;
                List<ContactRecord> row = rows.get(py);
                if (row == null) {
                    row = new ArrayList<>(10);
                    rows.put(py, row);
                }
                row.add(new ContactRecord(px, py, counts));
            }
        }

        // Compute size for each representation and choose smallest
        boolean useShort = isInteger && (maxCounts < Short.MAX_VALUE);
        int valueSize = useShort ? 2 : 4;

        int lorSize = 0;
        int nDensePts = (lastPoint.y - binYOffset) * w + (lastPoint.x - binXOffset) + 1;

        int denseSize = nDensePts * valueSize;
        for (List<ContactRecord> row : rows.values()) {
            lorSize += 4 + row.size() * valueSize;
        }

        buffer.put((byte) (useShort ? 0 : 1));

        if (lorSize < denseSize) {

            buffer.put((byte) 1);  // List of rows representation

            buffer.putShort((short) rows.size());  // # of rows

            for (Map.Entry<Integer, List<ContactRecord>> entry : rows.entrySet()) {

                int py = entry.getKey();
                List<ContactRecord> row = entry.getValue();
                buffer.putShort((short) py);  // Row number
                buffer.putShort((short) row.size());  // size of row

                for (ContactRecord contactRecord : row) {
                    buffer.putShort((short) (contactRecord.getBinX()));
                    final float counts = contactRecord.getCounts();

                    if (useShort) {
                        buffer.putShort((short) counts);
                    } else {
                        buffer.putFloat(counts);
                    }

                    sampledData.add(counts);
                    zd.sum += counts;
                }
            }

        } else {
            buffer.put((byte) 2);  // Dense matrix


            buffer.putInt(nDensePts);
            buffer.putShort(w);

            int lastIdx = 0;
            for (Point p : keys) {

                int idx = (p.y - binYOffset) * w + (p.x - binXOffset);
                for (int i = lastIdx; i < idx; i++) {
                    // Filler value
                    if (useShort) {
                        buffer.putShort(Short.MIN_VALUE);
                    } else {
                        buffer.putFloat(Float.NaN);
                    }
                }
                float counts = records.get(p).getCounts();
                if (useShort) {
                    buffer.putShort((short) counts);
                } else {
                    buffer.putFloat(counts);
                }
                lastIdx = idx + 1;

                sampledData.add(counts);
                zd.sum += counts;
            }
        }


        byte[] bytes = buffer.getBytes();
        byte[] compressedBytes = compress(bytes);
        los.write(compressedBytes);

    }

    public void setTmpdir(String tmpDirName) {

        if (tmpDirName != null) {
            this.tmpDir = new File(tmpDirName);

            if (!tmpDir.exists()) {
                System.err.println("Tmp directory does not exist: " + tmpDirName);
                if (outputFile != null) outputFile.deleteOnExit();
                System.exit(59);
            }
        }
    }

    public void setStatisticsFile(String statsOption) {
        statsFileName = statsOption;
    }

    private synchronized byte[] compress(byte[] data) {

        // Give the compressor the data to compress
        compressor.reset();
        compressor.setInput(data);
        compressor.finish();

        // Create an expandable byte array to hold the compressed data.
        // You cannot use an array that's the same size as the orginal because
        // there is no guarantee that the compressed data will be smaller than
        // the uncompressed data.
        ByteArrayOutputStream bos = new ByteArrayOutputStream(data.length);

        // Compress the data
        byte[] buf = new byte[1024];
        while (!compressor.finished()) {
            int count = compressor.deflate(buf);
            bos.write(buf, 0, count);
        }
        try {
            bos.close();
        } catch (IOException e) {
            System.err.println("Error clossing ByteArrayOutputStream");
            e.printStackTrace();
        }

        return bos.toByteArray();
    }

    interface BlockQueue {

        void advance() throws IOException;

        BlockPP getBlock();

    }

    public static class IndexEntry {
        public final long position;
        public final int size;
        int id;

        IndexEntry(int id, long position, int size) {
            this.id = id;
            this.position = position;
            this.size = size;
        }

        public IndexEntry(long position, int size) {
            this.position = position;
            this.size = size;
        }
    }

    static class BlockQueueFB implements BlockQueue {

        final File file;
        BlockPP block;
        long filePosition;

        BlockQueueFB(File file) {
            this.file = file;
            try {
                advance();
            } catch (IOException e) {
                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }
        }

        public void advance() throws IOException {

            if (filePosition >= file.length()) {
                block = null;
                return;
            }

            FileInputStream fis = null;

            try {
                fis = new FileInputStream(file);
                fis.getChannel().position(filePosition);


                LittleEndianInputStream lis = new LittleEndianInputStream(fis);
                int blockNumber = lis.readInt();
                int nRecords = lis.readInt();

                byte[] bytes = new byte[nRecords * 12];
                readFully(bytes, fis);

                ByteArrayInputStream bis = new ByteArrayInputStream(bytes);
                lis = new LittleEndianInputStream(bis);


                Map<Point, ContactCount> contactRecordMap = new HashMap<>(nRecords);
                for (int i = 0; i < nRecords; i++) {
                    int x = lis.readInt();
                    int y = lis.readInt();
                    float v = lis.readFloat();
                    ContactCount rec = new ContactCount(v);
                    contactRecordMap.put(new Point(x, y), rec);
                }
                block = new BlockPP(blockNumber, contactRecordMap);

                // Update file position based on # of bytes read, for next block
                filePosition = fis.getChannel().position();

            } finally {
                if (fis != null) fis.close();
            }
        }

        public BlockPP getBlock() {
            return block;
        }

        /**
         * Read enough bytes to fill the input buffer
         */
        void readFully(byte[] b, InputStream is) throws IOException {
            int len = b.length;
            if (len < 0)
                throw new IndexOutOfBoundsException();
            int n = 0;
            while (n < len) {
                int count = is.read(b, n, len - n);
                if (count < 0)
                    throw new EOFException();
                n += count;
            }
        }
    }


// class to support block merging

    static class BlockQueueMem implements BlockQueue {

        final List<BlockPP> blocks;
        int idx = 0;

        BlockQueueMem(Collection<BlockPP> blockCollection) {

            this.blocks = new ArrayList<>(blockCollection);
            Collections.sort(blocks, new Comparator<BlockPP>() {
                @Override
                public int compare(BlockPP o1, BlockPP o2) {
                    return o1.getNumber() - o2.getNumber();
                }
            });
        }

        public void advance() {
            idx++;
        }

        public BlockPP getBlock() {
            if (idx >= blocks.size()) {
                return null;
            } else {
                return blocks.get(idx);
            }
        }
    }

    /**
     * Representation of a sparse matrix block used for preprocessing.
     */
    static class BlockPP {

        private final int number;

        // Key to the map is a Point representing the x,y coordinate for the cell.
        private final Map<Point, ContactCount> contactRecordMap;


        BlockPP(int number) {
            this.number = number;
            this.contactRecordMap = new HashMap<>();
        }

        BlockPP(int number, Map<Point, ContactCount> contactRecordMap) {
            this.number = number;
            this.contactRecordMap = contactRecordMap;
        }


        int getNumber() {
            return number;
        }

        void incrementCount(int col, int row, float score) {
            Point p = new Point(col, row);
            ContactCount rec = contactRecordMap.get(p);
            if (rec == null) {
                rec = new ContactCount(score);
                contactRecordMap.put(p, rec);

            } else {
                rec.incrementCount(score);
            }
        }

        /*
         useless at present
        public void parsingComplete() {

        }
        */

        Map<Point, ContactCount> getContactRecordMap() {
            return contactRecordMap;
        }

        void merge(BlockPP other) {

            for (Map.Entry<Point, ContactCount> entry : other.getContactRecordMap().entrySet()) {

                Point point = entry.getKey();
                ContactCount otherValue = entry.getValue();

                ContactCount value = contactRecordMap.get(point);
                if (value == null) {
                    contactRecordMap.put(point, otherValue);
                } else {
                    value.incrementCount(otherValue.getCounts());
                }

            }
        }
    }

    static class ContactCount {
        float value;

        ContactCount(float value) {
            this.value = value;
        }

        void incrementCount(float increment) {
            value += increment;
        }

        float getCounts() {
            return value;
        }
    }

    /**
     * @author jrobinso
     * @since Aug 12, 2010
     */
    class MatrixPP {

        private final int chr1Idx;
        private final int chr2Idx;
        private final MatrixZoomDataPP[] zoomData;


        /**
         * Constructor for creating a matrix and initializing zoomed data at predefined resolution scales.  This
         * constructor is used when parsing alignment files.
         * c
         *
         * @param chr1Idx Chromosome 1
         * @param chr2Idx Chromosome 2
         */
        MatrixPP(int chr1Idx, int chr2Idx) {
            this.chr1Idx = chr1Idx;
            this.chr2Idx = chr2Idx;

            int nResolutions = bpBinSizes.length;
            if (fragmentCalculation != null) {
                nResolutions += fragBinSizes.length;
            }

            zoomData = new MatrixZoomDataPP[nResolutions];

            int zoom = 0; //
            for (int idx = 0; idx < bpBinSizes.length; idx++) {
                int binSize = bpBinSizes[zoom];
                Chromosome chrom1 = chromosomeHandler.getChromosomeFromIndex(chr1Idx);
                Chromosome chrom2 = chromosomeHandler.getChromosomeFromIndex(chr2Idx);

                // Size block (submatrices) to be ~500 bins wide.
                int len = Math.max(chrom1.getLength(), chrom2.getLength());
                int nBins = len / binSize + 1;   // Size of chrom in bins
                int nColumns = nBins / BLOCK_SIZE + 1;
                zoomData[idx] = new MatrixZoomDataPP(chrom1, chrom2, binSize, nColumns, zoom, false);
                zoom++;

            }

            if (fragmentCalculation != null) {
                Chromosome chrom1 = chromosomeHandler.getChromosomeFromIndex(chr1Idx);
                Chromosome chrom2 = chromosomeHandler.getChromosomeFromIndex(chr2Idx);
                int nFragBins1 = Math.max(fragmentCalculation.getNumberFragments(chrom1.getName()),
                        fragmentCalculation.getNumberFragments(chrom2.getName()));

                zoom = 0;
                for (int idx = bpBinSizes.length; idx < nResolutions; idx++) {
                    int binSize = fragBinSizes[zoom];
                    int nBins = nFragBins1 / binSize + 1;
                    int nColumns = nBins / BLOCK_SIZE + 1;
                    zoomData[idx] = new MatrixZoomDataPP(chrom1, chrom2, binSize, nColumns, zoom, true);
                    zoom++;
                }
            }
        }

        /**
         * Constructor for creating a matrix with a single zoom level at a specified bin size.  This is provided
         * primarily for constructing a whole-genome view.
         *
         * @param chr1Idx Chromosome 1
         * @param chr2Idx Chromosome 2
         * @param binSize Bin size
         */
        MatrixPP(int chr1Idx, int chr2Idx, int binSize, int blockColumnCount) {
            this.chr1Idx = chr1Idx;
            this.chr2Idx = chr2Idx;
            zoomData = new MatrixZoomDataPP[1];
            zoomData[0] = new MatrixZoomDataPP(chromosomeHandler.getChromosomeFromIndex(chr1Idx), chromosomeHandler.getChromosomeFromIndex(chr2Idx),
                    binSize, blockColumnCount, 0, false);

        }


        String getKey() {
            return "" + chr1Idx + "_" + chr2Idx;
        }


        void incrementCount(int pos1, int pos2, int frag1, int frag2, float score) throws IOException {

            for (MatrixZoomDataPP aZoomData : zoomData) {
                if (aZoomData.isFrag) {
                    aZoomData.incrementCount(frag1, frag2, score);
                } else {
                    aZoomData.incrementCount(pos1, pos2, score);
                }
            }
        }

        void parsingComplete() {
            for (MatrixZoomDataPP zd : zoomData) {
                if (zd != null) // fragment level could be null
                    zd.parsingComplete();
            }
        }

        int getChr1Idx() {
            return chr1Idx;
        }

        int getChr2Idx() {
            return chr2Idx;
        }

        MatrixZoomDataPP[] getZoomData() {
            return zoomData;
        }

    }

    /**
     * @author jrobinso
     * @since Aug 10, 2010
     */
    class MatrixZoomDataPP {

        final boolean isFrag;
        final Set<Integer> blockNumbers;  // The only reason for this is to get a count
        final List<File> tmpFiles;
        private final Chromosome chr1;  // Redundant, but convenient    BinDatasetReader
        private final Chromosome chr2;  // Redundant, but convenient
        private final int zoom;
        private final int binSize;              // bin size in bp
        private final int blockBinCount;        // block size in bins
        private final int blockColumnCount;     // number of block columns
        private final LinkedHashMap<Integer, BlockPP> blocks;
        long blockIndexPosition;
        private double sum = 0;
        private double cellCount = 0;
        private double percent5;
        private double percent95;

        /**
         * Representation of MatrixZoomData used for preprocessing
         *
         * @param chr1             index of first chromosome  (x-axis)
         * @param chr2             index of second chromosome
         * @param binSize          size of each grid bin in bp
         * @param blockColumnCount number of block columns
         * @param zoom             integer zoom (resolution) level index.  TODO Is this needed?
         */
        MatrixZoomDataPP(Chromosome chr1, Chromosome chr2, int binSize, int blockColumnCount, int zoom, boolean isFrag) {

            this.tmpFiles = new ArrayList<>();
            this.blockNumbers = new HashSet<>(1000);

            this.sum = 0;
            this.chr1 = chr1;
            this.chr2 = chr2;
            this.binSize = binSize;
            this.blockColumnCount = blockColumnCount;
            this.zoom = zoom;
            this.isFrag = isFrag;

            // Get length in proper units
            Chromosome longChr = chr1.getLength() > chr2.getLength() ? chr1 : chr2;
            int len = isFrag ? fragmentCalculation.getNumberFragments(longChr.getName()) : longChr.getLength();

            int nBinsX = len / binSize + 1;

            blockBinCount = nBinsX / blockColumnCount + 1;
            blocks = new LinkedHashMap<>(blockColumnCount * blockColumnCount);
        }

        HiC.Unit getUnit() {
            return isFrag ? HiC.Unit.FRAG : HiC.Unit.BP;
        }

        double getSum() {
            return sum;
        }

        double getOccupiedCellCount() {
            return cellCount;
        }

        double getPercent95() {
            return percent95;
        }

        double getPercent5() {
            return percent5;
        }


        int getBinSize() {
            return binSize;
        }


        Chromosome getChr1() {
            return chr1;
        }


        Chromosome getChr2() {
            return chr2;
        }

        int getZoom() {
            return zoom;
        }

        int getBlockBinCount() {
            return blockBinCount;
        }

        int getBlockColumnCount() {
            return blockColumnCount;
        }

        Map<Integer, BlockPP> getBlocks() {
            return blocks;
        }

        /**
         * Increment the count for the bin represented by the GENOMIC position (pos1, pos2)
         */
        void incrementCount(int pos1, int pos2, float score) throws IOException {

            sum += score;
            // Convert to proper units,  fragments or base-pairs

            if (pos1 < 0 || pos2 < 0) return;

            int xBin = pos1 / binSize;
            int yBin = pos2 / binSize;

            // Intra chromosome -- we'll store lower diagonal only
            if (chr1.equals(chr2)) {
                int b1 = Math.min(xBin, yBin);
                int b2 = Math.max(xBin, yBin);
                xBin = b1;
                yBin = b2;

                if (b1 != b2) {
                    sum += score;  // <= count for mirror cell.
                }

                if (expectedValueCalculations != null) {
                    String evKey = (isFrag ? "FRAG_" : "BP_") + binSize;
                    ExpectedValueCalculation ev = expectedValueCalculations.get(evKey);
                    if (ev != null) {
                        ev.addDistance(chr1.getIndex(), xBin, yBin, score);
                    }
                }
            }

            // compute block number (fist block is zero)
            int blockCol = xBin / blockBinCount;
            int blockRow = yBin / blockBinCount;
            int blockNumber = blockColumnCount * blockRow + blockCol;

            BlockPP block = blocks.get(blockNumber);
            if (block == null) {

                block = new BlockPP(blockNumber);
                blocks.put(blockNumber, block);
            }
            block.incrementCount(xBin, yBin, score);

            // If too many blocks write to tmp directory
            if (blocks.size() > 1000) {
                File tmpfile = tmpDir == null ? File.createTempFile("blocks", "bin") : File.createTempFile("blocks", "bin", tmpDir);
                //System.out.println(chr1.getName() + "-" + chr2.getName() + " Dumping blocks to " + tmpfile.getAbsolutePath());
                dumpBlocks(tmpfile);
                tmpFiles.add(tmpfile);
                tmpfile.deleteOnExit();
            }
        }


        /**
         * Dump the blocks calculated so far to a temporary file
         *
         * @param file File to write to
         * @throws IOException
         */
        private void dumpBlocks(File file) throws IOException {
            LittleEndianOutputStream los = null;
            try {
                los = new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(file), 4194304));

                List<BlockPP> blockList = new ArrayList<>(blocks.values());
                Collections.sort(blockList, new Comparator<BlockPP>() {
                    @Override
                    public int compare(BlockPP o1, BlockPP o2) {
                        return o1.getNumber() - o2.getNumber();
                    }
                });

                for (BlockPP b : blockList) {

                    // Remove from map
                    blocks.remove(b.getNumber());

                    int number = b.getNumber();
                    blockNumbers.add(number);

                    los.writeInt(number);
                    Map<Point, ContactCount> records = b.getContactRecordMap();

                    los.writeInt(records.size());
                    for (Map.Entry<Point, ContactCount> entry : records.entrySet()) {

                        Point point = entry.getKey();
                        ContactCount count = entry.getValue();

                        los.writeInt(point.x);
                        los.writeInt(point.y);
                        los.writeFloat(count.getCounts());
                    }
                }

                blocks.clear();

            } finally {
                if (los != null) los.close();

            }
        }


        // Merge and write out blocks one at a time.
        private List<IndexEntry> mergeAndWriteBlocks() throws IOException {
            DownsampledDoubleArrayList sampledData = new DownsampledDoubleArrayList(10000, 10000);

            List<BlockQueue> activeList = new ArrayList<>();

            // Initialize queues -- first whatever is left over in memory
            if (blocks.size() > 0) {
                BlockQueue bqInMem = new BlockQueueMem(blocks.values());
                activeList.add(bqInMem);
            }
            // Now from files
            for (File file : tmpFiles) {
                BlockQueue bq = new BlockQueueFB(file);
                if (bq.getBlock() != null) {
                    activeList.add(bq);
                }
            }

            List<IndexEntry> indexEntries = new ArrayList<>();

            if (activeList.size() == 0) {
                throw new RuntimeException("No reads in Hi-C contact matrices. This could be because the MAPQ filter is set too high (-q) or because all reads map to the same fragment.");
            }

            do {
                Collections.sort(activeList, new Comparator<BlockQueue>() {
                    @Override
                    public int compare(BlockQueue o1, BlockQueue o2) {
                        return o1.getBlock().getNumber() - o2.getBlock().getNumber();
                    }
                });

                BlockQueue topQueue = activeList.get(0);
                BlockPP currentBlock = topQueue.getBlock();
                topQueue.advance();
                int num = currentBlock.getNumber();


                for (int i = 1; i < activeList.size(); i++) {
                    BlockQueue blockQueue = activeList.get(i);
                    BlockPP block = blockQueue.getBlock();
                    if (block.getNumber() == num) {
                        currentBlock.merge(block);
                        blockQueue.advance();
                    }
                }

                Iterator<BlockQueue> iterator = activeList.iterator();
                while (iterator.hasNext()) {
                    if (iterator.next().getBlock() == null) {
                        iterator.remove();
                    }
                }

                // Output block
                long position = los.getWrittenCount();
                writeBlock(this, currentBlock, sampledData);
                int size = (int) (los.getWrittenCount() - position);

                indexEntries.add(new IndexEntry(num, position, size));


            } while (activeList.size() > 0);


            for (File f : tmpFiles) {
                boolean result = f.delete();
                if (!result) {
                    System.out.println("Error while deleting file");
                }
            }

            computeStats(sampledData);

            return indexEntries;
        }

        private void computeStats(DownsampledDoubleArrayList sampledData) {

            double[] data = sampledData.toArray();
            this.percent5 = StatUtils.percentile(data, 5);
            this.percent95 = StatUtils.percentile(data, 95);

        }

        void parsingComplete() {
            // Add the block numbers still in memory
            for (BlockPP block : blocks.values()) {
                blockNumbers.add(block.getNumber());
            }
        }

        void updateIndexPositions(List<IndexEntry> blockIndex) throws IOException {

            // Temporarily close output stream.  Remember position
            long losPos = los.getWrittenCount();
            los.close();

            RandomAccessFile raf = null;
            try {
                raf = new RandomAccessFile(outputFile, "rw");

                // Block indices
                long pos = blockIndexPosition;
                raf.getChannel().position(pos);

                // Write as little endian
                BufferedByteWriter buffer = new BufferedByteWriter();
                for (IndexEntry aBlockIndex : blockIndex) {
                    buffer.putInt(aBlockIndex.id);
                    buffer.putLong(aBlockIndex.position);
                    buffer.putInt(aBlockIndex.size);
                }
                raf.write(buffer.getBytes());

            } finally {

                if (raf != null) raf.close();

                // Restore
                FileOutputStream fos = new FileOutputStream(outputFile, true);
                fos.getChannel().position(losPos);
                los = new LittleEndianOutputStream(new BufferedOutputStream(fos, HiCGlobals.bufferSize));
                los.setWrittenCount(losPos);

            }
        }
    }
}
