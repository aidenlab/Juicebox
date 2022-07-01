/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2022 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.tools.utils.original;

//import juicebox.MainWindow;

import htsjdk.tribble.util.LittleEndianOutputStream;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.basics.Chromosome;
import juicebox.data.basics.ListOfDoubleArrays;
import juicebox.tools.clt.CommandLineParser.Alignment;
import juicebox.tools.utils.common.UNIXTools;
import juicebox.tools.utils.original.mnditerator.AlignmentPair;
import juicebox.tools.utils.original.mnditerator.PairIterator;
import juicebox.windowui.NormalizationHandler;
import org.broad.igv.tdf.BufferedByteWriter;
import org.broad.igv.util.Pair;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.Deflater;


/**
 * @author jrobinso
 * @since Aug 16, 2010
 */
public class Preprocessor {
    protected static final int VERSION = 9;
    protected static final int BLOCK_SIZE = 1000;
    public static final String V9_DEPTH_BASE = "v9-depth-base";
    protected int v9DepthBase = 2;
    public static final String HIC_FILE_SCALING = "hicFileScalingFactor";
    public static final String STATISTICS = "statistics";
    public static final String GRAPHS = "graphs";
    public static final String SOFTWARE = "software";
    protected static final String NVI_INDEX = "nviIndex";
    protected static final String NVI_LENGTH = "nviLength";

    protected final ChromosomeHandler chromosomeHandler;
    protected Map<String, Integer> chromosomeIndexes;
    protected final File outputFile;
    protected final Map<String, IndexEntry> matrixPositions;
    protected String genomeId;
    protected final Deflater compressor;
    protected LittleEndianOutputStream[] losArray = new LittleEndianOutputStream[1];
    protected long masterIndexPosition;
    protected int countThreshold = 0;
    protected int mapqThreshold = 0;
    protected boolean diagonalsOnly = false;
    protected String fragmentFileName = null;
    protected String statsFileName = null;
    protected String graphFileName = null;
    protected String expectedVectorFile = null;
    protected Set<String> randomizeFragMapFiles = null;
    protected FragmentCalculation fragmentCalculation = null;
    protected Set<String> includedChromosomes;
    protected ArrayList<FragmentCalculation> fragmentCalculationsForRandomization = null;
    protected Alignment alignmentFilter;
    protected static final Random random = new Random(5);
    protected static boolean allowPositionsRandomization = false;
    protected static boolean throwOutIntraFrag = false;
    public static int BLOCK_CAPACITY = 1000;
    protected double subsampleFraction = 1;
    protected Random randomSubsampleGenerator = new Random(0);
    protected static boolean fromHIC = false;
    
    // Base-pair resolutions
    protected int[] bpBinSizes = {2500000, 1000000, 500000, 250000, 100000, 50000, 25000, 10000, 5000, 1000};
    
    // Fragment resolutions
    protected int[] fragBinSizes = {500, 200, 100, 50, 20, 5, 2, 1};

    // number of resolutions
    protected int numResolutions = bpBinSizes.length + fragBinSizes.length;

    // hic scaling factor value
    protected double hicFileScalingFactor = 1;
    
    protected Long normVectorIndex = 0L, normVectorLength = 0L;
    
    /**
     * The position of the field containing the masterIndex position
     */
    protected long masterIndexPositionPosition;
    protected long normVectorIndexPosition;
    protected long normVectorLengthPosition;
    protected Map<String, ExpectedValueCalculation> expectedValueCalculations;
    protected File tmpDir;
    
    public Preprocessor(File outputFile, String genomeId, ChromosomeHandler chromosomeHandler, double hicFileScalingFactor) {
        this.genomeId = genomeId;
        this.outputFile = outputFile;
        this.matrixPositions = new LinkedHashMap<>();

        this.chromosomeHandler = chromosomeHandler;
        chromosomeIndexes = new Hashtable<>();
        for (int i = 0; i < chromosomeHandler.size(); i++) {
            chromosomeIndexes.put(chromosomeHandler.getChromosomeFromIndex(i).getName(), i);
        }

        compressor = getDefaultCompressor();

        this.tmpDir = createTempFolder(outputFile.getAbsolutePath() + "_tmp_folder");

        if (hicFileScalingFactor > 0) {
            this.hicFileScalingFactor = hicFileScalingFactor;
        }

    }

    public void setCountThreshold(int countThreshold) {
        this.countThreshold = countThreshold;
    }

    public void setV9DepthBase(int v9DepthBase) {
        if (v9DepthBase > 1 || v9DepthBase < 0) {
            this.v9DepthBase = v9DepthBase;
        }
    }

    public void setMapqThreshold(int mapqThreshold) {
        this.mapqThreshold = mapqThreshold;
    }

    public void setDiagonalsOnly(boolean diagonalsOnly) {
        this.diagonalsOnly = diagonalsOnly;
    }

    public void setIncludedChromosomes(Set<String> includedChromosomes) {
        if (includedChromosomes != null && includedChromosomes.size() > 0) {
            this.includedChromosomes = Collections.synchronizedSet(new HashSet<>());
            for (String name : includedChromosomes) {
                this.includedChromosomes.add(chromosomeHandler.cleanUpName(name));
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

    public void setGenome(String genome) {
        if (genome != null) {
            this.genomeId = genome;
        }
    }

    public void setResolutions(List<String> resolutions) {
        if (resolutions != null) {
            ArrayList<Integer> fragResolutions = new ArrayList<>();
            ArrayList<Integer> bpResolutions = new ArrayList<>();

            for (String str : resolutions) {
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

    public void setSubsampler(double subsampleFraction) {
        if (subsampleFraction == -1) {
            this.subsampleFraction = 1;
        }
        else {
            this.subsampleFraction = subsampleFraction;
        }
    }

    protected static int randomizePos(FragmentCalculation fragmentCalculation, String chr, int frag) {

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

    public void setThrowOutIntraFragOption(boolean throwOutIntraFrag) {
        Preprocessor.throwOutIntraFrag = throwOutIntraFrag;
    }

    protected static FragmentCalculation findFragMap(List<FragmentCalculation> maps, String chr, int bp, int frag) {
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

    public void setFromHIC(boolean fromHIC) {
        Preprocessor.fromHIC = fromHIC;
    }

    public void preprocess(final String inputFile, final String headerFile, final String footerFile,
                           Map<Integer, List<Chunk>> mndIndex) throws IOException {
        if (!fromHIC) {
            File file = new File(inputFile);
            if (!file.exists() || file.length() == 0) {
                System.err.println(inputFile + " does not exist or does not contain any reads.");
                System.exit(57);
            }
        }

        try {
            StringBuilder stats = null;
            StringBuilder graphs = null;
            StringBuilder hicFileScaling = new StringBuilder().append(hicFileScalingFactor);
            if (fragmentFileName != null) {
                fragmentCalculation = FragmentCalculation.readFragments(fragmentFileName, chromosomeHandler, "Pre");
            } else {
                System.out.println("Not including fragment map");
            }

            if (allowPositionsRandomization) {
                if (randomizeFragMapFiles != null) {
                    fragmentCalculationsForRandomization = new ArrayList<>();
                    for (String fragmentFileName : randomizeFragMapFiles) {
                        try {
                            FragmentCalculation fragmentCalculation = FragmentCalculation.readFragments(fragmentFileName, chromosomeHandler, "PreWithRand");
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
                expectedValueCalculations = Collections.synchronizedMap(new LinkedHashMap<>());
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

            LittleEndianOutputStream[] losFooter = new LittleEndianOutputStream[1];
            try {
                losArray[0] = new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(headerFile), HiCGlobals.bufferSize));
                if (footerFile.equalsIgnoreCase(headerFile)) {
                    losFooter = losArray;
                } else {
                    losFooter[0] = new LittleEndianOutputStream(new BufferedOutputStream(new FileOutputStream(footerFile), HiCGlobals.bufferSize));
                }
            } catch (Exception e) {
                System.err.println("Unable to write to " + outputFile);
                System.exit(70);
            }

            System.out.println("Start preprocess");

            System.out.println("Writing header");

            writeHeader(stats, graphs, hicFileScaling);

            System.out.println("Writing body");
            writeBody(inputFile, mndIndex);

            System.out.println();
            System.out.println("Writing footer");
            writeFooter(losFooter);

            if (losFooter != null && losFooter[0] != null) {
                losFooter[0].close();
            }

        } finally {
            if (losArray != null && losArray[0] != null) {
                losArray[0].close();
            }
        }

        updateMasterIndex(headerFile);
        System.out.println("\nFinished preprocess");
    }

    protected void writeHeader(StringBuilder stats, StringBuilder graphs, StringBuilder hicFileScaling) throws IOException {
        // Magic number
        byte[] magicBytes = "HIC".getBytes();
        LittleEndianOutputStream los = losArray[0];
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
    
        // Add NVI info
        //los.writeString(NVI_INDEX);
        normVectorIndexPosition = los.getWrittenCount();
        los.writeLong(0L);
    
        //los.writeString(NVI_LENGTH);
        normVectorLengthPosition = los.getWrittenCount();
        los.writeLong(0L);
    
    
        // Attribute dictionary
        int nAttributes = 1;
        if (stats != null) nAttributes += 1;
        if (graphs != null) nAttributes += 1;
        if (hicFileScaling != null) nAttributes += 1;
        if (v9DepthBase != 2) nAttributes += 1;
    
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
        if (v9DepthBase != 2) {
            los.writeString(V9_DEPTH_BASE);
            los.writeString("" + v9DepthBase);
        }


        // Sequence dictionary
        int nChrs = chromosomeHandler.size();
        los.writeInt(nChrs);
        for (Chromosome chromosome : chromosomeHandler.getChromosomeArray()) {
            los.writeString(chromosome.getName());
            los.writeLong(chromosome.getLength());
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

        numResolutions = nBpRes + nFragRes;

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

    protected MatrixPP getInitialGenomeWideMatrixPP(ChromosomeHandler chromosomeHandler) {
        long genomeLength = chromosomeHandler.getChromosomeFromIndex(0).getLength();  // <= whole genome in KB
        int binSize = (int) (genomeLength / 500); // todo
        if (binSize == 0) binSize = 1;
        int nBinsX = (int) (genomeLength / binSize + 1); // todo
        int nBlockColumns = nBinsX / BLOCK_SIZE + 1;
        return new MatrixPP(0, 0, binSize, nBlockColumns, chromosomeHandler, fragmentCalculation, countThreshold, v9DepthBase);
    }

    /**
     * @param file List of files to read
     * @return Matrix with counts in each bin
     * @throws IOException
     */
    private MatrixPP computeWholeGenomeMatrix(String file) throws IOException {

        MatrixPP matrix = getInitialGenomeWideMatrixPP(chromosomeHandler);

        PairIterator iter = null;

        //int belowMapq = 0;
        //int intraFrag = 0;
        int totalRead = 0;
        int contig = 0;
        int hicContact = 0;

        // Create an index the first time through
        try {
            iter = PairIterator.getIterator(file, chromosomeIndexes, chromosomeHandler);

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

                    int pos1, pos2;
                    if (shouldSkipContact(pair)) continue;
                    pos1 = getGenomicPosition(chr1, bp1);
                    pos2 = getGenomicPosition(chr2, bp2);
                    matrix.incrementCount(pos1, pos2, pos1, pos2, pair.getScore(), expectedValueCalculations, tmpDir);
                    hicContact++;
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

    protected boolean alignmentsAreEqual(Alignment alignment, Alignment alignmentStandard) {
        if (alignment == alignmentStandard) {
            return true;
        }
        if (alignmentStandard == Alignment.TANDEM) {
            return alignment == Alignment.LL || alignment == Alignment.RR;
        }

        return false;
    }


    protected int getGenomicPosition(int chr, int pos) {
        long len = 0;
        for (int i = 1; i < chr; i++) {
            len += chromosomeHandler.getChromosomeFromIndex(i).getLength();
        }
        len += pos;

        return (int) (len / 1000);

    }

    protected static Alignment calculateAlignment(AlignmentPair pair) {

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

    protected void writeBody(String inputFile, Map<Integer, List<Chunk>> mndIndex) throws IOException {

        MatrixPP wholeGenomeMatrix = computeWholeGenomeMatrix(inputFile);
        writeMatrix(wholeGenomeMatrix, losArray, compressor, matrixPositions, -1, false);

        PairIterator iter = PairIterator.getIterator(inputFile, chromosomeIndexes, chromosomeHandler);

        Set<String> writtenMatrices = Collections.synchronizedSet(new HashSet<>());

        int currentChr1 = -1;
        int currentChr2 = -1;
        MatrixPP currentMatrix = null;
        String currentMatrixKey = null;

        while (iter.hasNext()) {
            AlignmentPair pair = iter.next();
            // skip pairs that mapped to contigs
            if (!pair.isContigPair()) {
                if (shouldSkipContact(pair)) continue;
                // Flip pair if needed so chr1 < chr2
                int chr1, chr2, bp1, bp2, frag1, frag2;
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

                bp1 = ensureFitInChromosomeBounds(bp1, chr1);
                bp2 = ensureFitInChromosomeBounds(bp2, chr2);

                // Randomize position within fragment site
                if (allowPositionsRandomization && fragmentCalculation != null) {
                    Pair<Integer, Integer> newBPos12 = getRandomizedPositions(chr1, chr2, frag1, frag2, bp1, bp2);
                    bp1 = newBPos12.getFirst();
                    bp2 = newBPos12.getSecond();
                }
                // only increment if not intraFragment and passes the mapq threshold
                if (!(currentChr1 == chr1 && currentChr2 == chr2)) {
                    // Starting a new matrix
                    if (currentMatrix != null) {
                        currentMatrix.parsingComplete();
                        writeMatrix(currentMatrix, losArray, compressor, matrixPositions, -1, false);
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
                    currentMatrix = new MatrixPP(currentChr1, currentChr2, chromosomeHandler, bpBinSizes,
                            fragmentCalculation, fragBinSizes, countThreshold, v9DepthBase, BLOCK_CAPACITY);
                }
                currentMatrix.incrementCount(bp1, bp2, frag1, frag2, pair.getScore(), expectedValueCalculations, tmpDir);

            }
        }

        /*
        if (fragmentCalculation != null && allowPositionsRandomization) {
            System.out.println(String.format("Randomization errors encountered: %d no map found, " +
                    "%d two different maps found", noMapFoundCount, mapDifferentCount));
        }
         */

        if (currentMatrix != null) {
            currentMatrix.parsingComplete();
            writeMatrix(currentMatrix, losArray, compressor, matrixPositions, -1, false);
        }

        if (iter != null) iter.close();


        masterIndexPosition = losArray[0].getWrittenCount();
    }

    protected int ensureFitInChromosomeBounds(int bp, int chrom) {
        if (bp < 0) {
            return 0;
        }
        long maxLength = chromosomeHandler.getChromosomeFromIndex(chrom).getLength();
        if (bp > maxLength) {
            return (int) maxLength;
        }
        return bp;
    }

    protected Pair<Integer, Integer> getRandomizedPositions(int chr1, int chr2, int frag1, int frag2, int bp1, int bp2) {
        FragmentCalculation fragMapToUse;
        if (fragmentCalculationsForRandomization != null) {
            FragmentCalculation fragMap1 = findFragMap(fragmentCalculationsForRandomization, chromosomeHandler.getChromosomeFromIndex(chr1).getName(), bp1, frag1);
            FragmentCalculation fragMap2 = findFragMap(fragmentCalculationsForRandomization, chromosomeHandler.getChromosomeFromIndex(chr2).getName(), bp2, frag2);

            if (fragMap1 == null && fragMap2 == null) {
                //noMapFoundCount += 1;
                return null;
            } else if (fragMap1 != null && fragMap2 != null && fragMap1 != fragMap2) {
                //mapDifferentCount += 1;
                return null;
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

        int newBP1 = randomizePos(fragMapToUse, chromosomeHandler.getChromosomeFromIndex(chr1).getName(), frag1);
        int newBP2 = randomizePos(fragMapToUse, chromosomeHandler.getChromosomeFromIndex(chr2).getName(), frag2);

        return new Pair<>(newBP1, newBP2);
    }

    protected boolean shouldSkipContact(AlignmentPair pair) {
        int chr1 = pair.getChr1();
        int chr2 = pair.getChr2();
        if (diagonalsOnly && chr1 != chr2) return true;
        if (includedChromosomes != null && chr1 != 0) {
            String c1Name = chromosomeHandler.getChromosomeFromIndex(chr1).getName();
            String c2Name = chromosomeHandler.getChromosomeFromIndex(chr2).getName();
            if (!includedChromosomes.contains(c1Name) || !includedChromosomes.contains(c2Name)) {
                return true;
            }
        }
        if (alignmentFilter != null && !alignmentsAreEqual(calculateAlignment(pair), alignmentFilter)) {
            return true;
        }
        int mapq = Math.min(pair.getMapq1(), pair.getMapq2());
        if (mapq < mapqThreshold) return true;

        int frag1 = pair.getFrag1();
        int frag2 = pair.getFrag2();

        if (throwOutIntraFrag && chr1 == chr2 && frag1 == frag2) {return true;}

        if ( subsampleFraction < 1 && subsampleFraction > 0) {
            return randomSubsampleGenerator.nextDouble() > subsampleFraction;
        } else { return false; }
    }

    protected void updateMasterIndex(String headerFile) throws IOException {
        RandomAccessFile raf = null;
        try {
            raf = new RandomAccessFile(headerFile, "rw");

            // Master index
            raf.getChannel().position(masterIndexPositionPosition);
            BufferedByteWriter buffer = new BufferedByteWriter();
            buffer.putLong(masterIndexPosition);
            raf.write(buffer.getBytes());
            System.out.println("masterIndexPosition: " + masterIndexPosition);

        } finally {
            if (raf != null) raf.close();
        }
    }

    private void updateNormVectorIndexInfo() throws IOException {
        RandomAccessFile raf = null;
        try {
            raf = new RandomAccessFile(outputFile, "rw");
    
            // NVI index
            raf.getChannel().position(normVectorIndexPosition);
            BufferedByteWriter buffer = new BufferedByteWriter();
    
            buffer.putLong(normVectorIndex); // todo
            raf.write(buffer.getBytes());
    
            // NVI length
            raf.getChannel().position(normVectorLengthPosition);
            buffer = new BufferedByteWriter();
            buffer.putLong(normVectorLength); // todo
            raf.write(buffer.getBytes());
    
        } finally {
            if (raf != null) raf.close();
        }
    }


    protected void writeFooter(LittleEndianOutputStream[] los) throws IOException {

        // Index
        List<BufferedByteWriter> bufferList = new ArrayList<>();
        bufferList.add(new BufferedByteWriter());
        bufferList.get(bufferList.size()-1).putInt(matrixPositions.size());
        for (Map.Entry<String, IndexEntry> entry : matrixPositions.entrySet()) {
            if (Integer.MAX_VALUE - bufferList.get(bufferList.size()-1).bytesWritten() < 1000) {
                bufferList.add(new BufferedByteWriter());
            }
            bufferList.get(bufferList.size()-1).putNullTerminatedString(entry.getKey());
            bufferList.get(bufferList.size()-1).putLong(entry.getValue().position);
            bufferList.get(bufferList.size()-1).putInt(entry.getValue().size);
        }

        // Vectors  (Expected values,  other).
        /***  NEVA ***/
        if (expectedVectorFile == null) {
            if (Integer.MAX_VALUE - bufferList.get(bufferList.size()-1).bytesWritten() < 1000) {
                bufferList.add(new BufferedByteWriter());
            }
            bufferList.get(bufferList.size()-1).putInt(expectedValueCalculations.size());
            for (Map.Entry<String, ExpectedValueCalculation> entry : expectedValueCalculations.entrySet()) {
                ExpectedValueCalculation ev = entry.getValue();
    
                ev.computeDensity();
    
                int binSize = ev.getGridSize();
                HiC.Unit unit = ev.isFrag ? HiC.Unit.FRAG : HiC.Unit.BP;

                bufferList.get(bufferList.size()-1).putNullTerminatedString(unit.toString());
                bufferList.get(bufferList.size()-1).putInt(binSize);
    
                // The density values
                ListOfDoubleArrays expectedValues = ev.getDensityAvg();
                // todo @Suhas to handle buffer overflow
                bufferList.get(bufferList.size()-1).putLong(expectedValues.getLength());
                for (double[] expectedArray : expectedValues.getValues()) {
                    bufferList.add(new BufferedByteWriter());
                    for (double value : expectedArray) {
                        if (Integer.MAX_VALUE - bufferList.get(bufferList.size()-1).bytesWritten() < 1000000) {
                            bufferList.add(new BufferedByteWriter());
                        }
                        bufferList.get(bufferList.size()-1).putFloat( (float) value);
                    }
                }
    
                // Map of chromosome index -> normalization factor
                Map<Integer, Double> normalizationFactors = ev.getChrScaleFactors();
                if (Integer.MAX_VALUE - bufferList.get(bufferList.size()-1).bytesWritten() < 1000000) {
                    bufferList.add(new BufferedByteWriter());
                }
                bufferList.get(bufferList.size()-1).putInt(normalizationFactors.size());
                for (Map.Entry<Integer, Double> normFactor : normalizationFactors.entrySet()) {
                    bufferList.get(bufferList.size()-1).putInt(normFactor.getKey());
                    bufferList.get(bufferList.size()-1).putFloat(normFactor.getValue().floatValue());
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
            bufferList.get(bufferList.size()-1).putInt(count);
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
                }
            }
        }
        long nBytesV5 = 0;
        for (int i = 0; i<bufferList.size(); i++) {
            nBytesV5 += bufferList.get(i).getBytes().length;
        }
        System.out.println("nBytesV5: " + nBytesV5);

        los[0].writeLong(nBytesV5);
        for (int i = 0; i<bufferList.size(); i++) {
            los[0].write(bufferList.get(i).getBytes());
        }
    }

    protected Deflater getDefaultCompressor() {
        Deflater compressor = new Deflater();
        compressor.setLevel(Deflater.DEFAULT_COMPRESSION);
        return compressor;
    }

    protected Pair<Map<Long, List<IndexEntry>>, Long> writeMatrix(MatrixPP matrix, LittleEndianOutputStream[] losArray,
                                                                  Deflater compressor, Map<String, IndexEntry> matrixPositions, int chromosomePairIndex, boolean doMultiThreadedBehavior) throws IOException {

        if (HiCGlobals.printVerboseComments) {
            System.err.println("Used Memory for matrix");
            System.err.println(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());
        }
        LittleEndianOutputStream los = losArray[0];
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
        for ( int i = 0; i < matrix.getZoomData().length; i++) {
            MatrixZoomDataPP zd = matrix.getZoomData()[i];
            if (zd != null)
                writeZoomHeader(zd, los);
        }

        long size = los.getWrittenCount() - position;
        if (chromosomePairIndex > -1) {
            matrixPositions.put("" + chromosomePairIndex, new IndexEntry(position, (int) size));
        } else {
            matrixPositions.put(matrix.getKey(), new IndexEntry(position, (int) size));
        }

        final Map<Long, List<IndexEntry>> localBlockIndexes = new ConcurrentHashMap<>();

        for (int i = matrix.getZoomData().length-1 ; i >= 0; i--) {
            MatrixZoomDataPP zd = matrix.getZoomData()[i];
            if (zd != null) {
                List<IndexEntry> blockIndex = null;
                if (doMultiThreadedBehavior) {
                    if (losArray.length > 1) {
                        blockIndex = zd.mergeAndWriteBlocks(losArray, compressor, i, matrix.getZoomData().length);
                    } else {
                        blockIndex = zd.mergeAndWriteBlocks(losArray[0], compressor);
                    }
                    localBlockIndexes.put(zd.blockIndexPosition, blockIndex);
                } else {
                    blockIndex = zd.mergeAndWriteBlocks(losArray[0], compressor);
                    updateIndexPositions(blockIndex, losArray, true, outputFile, 0, zd.blockIndexPosition);
                }
            }
        }

        System.out.print(".");
        return new Pair<>(localBlockIndexes, position);
    }

    protected void updateIndexPositions(List<IndexEntry> blockIndex, LittleEndianOutputStream[] losArray, boolean doRestore,
                                        File outputFile, long currentPosition, long blockIndexPosition) throws IOException {

        // Temporarily close output stream.  Remember position
        long losPos = 0;
        if (doRestore) {
            losPos = losArray[0].getWrittenCount();
            losArray[0].close();
        }

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
                buffer.putLong(aBlockIndex.position + currentPosition);
                buffer.putInt(aBlockIndex.size);
            }
            raf.write(buffer.getBytes());

        } finally {
            if (raf != null) raf.close();
        }
        if (doRestore) {
            FileOutputStream fos = new FileOutputStream(outputFile, true);
            fos.getChannel().position(losPos);
            losArray[0] = new LittleEndianOutputStream(new BufferedOutputStream(fos, HiCGlobals.bufferSize));
            losArray[0].setWrittenCount(losPos);
        }
    }

    private void writeZoomHeader(MatrixZoomDataPP zd, LittleEndianOutputStream los) throws IOException {

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

    public void setTmpdir(String tmpDirName) {
        if (tmpDirName != null) {
            createTempFolder(tmpDirName);
        }
    }

    private File createTempFolder(String newPath) {
        this.tmpDir = new File(newPath);
        if (!tmpDir.exists()) {
            UNIXTools.makeDir(tmpDir);
            tmpDir.deleteOnExit();
        }
        return tmpDir;
    }

    public void setStatisticsFile(String statsOption) {
        statsFileName = statsOption;
    }
}
