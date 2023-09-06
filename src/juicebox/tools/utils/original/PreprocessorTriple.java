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
import juicebox.tools.utils.original.mnditerator.AlignmentTriple;
import juicebox.tools.utils.original.mnditerator.TripleIterator;
import juicebox.tools.utils.original.stats.ChromBpTuple;
import juicebox.windowui.NormalizationHandler;
import org.broad.igv.tdf.BufferedByteWriter;
import org.broad.igv.util.Pair;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.Deflater;


/**
 * @author jrobinso sqhang
 * @since Aug 16, 2010; Aug 24, 2023
 */
public class PreprocessorTriple {
    protected static final int VERSION = 1;
    protected static final int BLOCK_SIZE = 300; // might need to adjust according to performances;
    public static final String HIC_FILE_SCALING = "hicFileScalingFactor";
    public static final String STATISTICS = "statistics";
    public static final String GRAPHS = "graphs";
    public static final String SOFTWARE = "software";

    protected final ChromosomeHandler chromosomeHandler;
    protected Map<String, Integer> chromosomeIndexes;
    protected final File outputFile;
    protected final Map<String, IndexEntry> tensorPositions;
    protected String genomeId;
    protected final Deflater compressor;
    /*Question What's the purpose of this LittleEndianOutputStream*/
    protected LittleEndianOutputStream[] losArray = new LittleEndianOutputStream[1];
    protected long masterIndexPosition;
    protected int countThreshold = 0;
    protected int mapqThreshold = 0;
    protected boolean diagonalsOnly = false;
    protected String statsFileName = null;
    protected String graphFileName = null;
    protected String expectedVectorFile = null;
    protected Set<String> includedChromosomes;
    protected Alignment alignmentFilter;
    protected static final Random random = new Random(5);
    protected static boolean throwOutIntraFrag = false;
    public static int BLOCK_CAPACITY = 1000;
    protected double subsampleFraction = 1;
    protected Random randomSubsampleGenerator = new Random(0);
    protected static boolean fromHIC = false;

    // Base-pair resolutions
    protected int[] bpBinSizes = {2500000, 1000000, 500000, 250000, 100000, 50000, 25000, 10000, 5000, 1000};

    // number of resolutions
    protected int numResolutions = bpBinSizes.length;

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

    public PreprocessorTriple(File outputFile, String genomeId, ChromosomeHandler chromosomeHandler, double hicFileScalingFactor) {
        this.genomeId = genomeId;
        this.outputFile = outputFile;
        this.tensorPositions = new LinkedHashMap<>();

        this.chromosomeHandler = chromosomeHandler;
        chromosomeIndexes = new Hashtable<>();
        /*Question: why set chromsomeIndexes as a map?*/
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
            ArrayList<Integer> bpResolutions = new ArrayList<>();

            for (String str : resolutions) {
                int index = str.indexOf("f");
                if (index != -1) {
                    str = str.substring(0, index);
                    System.err.println("Error: Resolution should be in bp only");
                    return;
                }
                Integer myInt = null;
                try {
                    myInt = Integer.valueOf(str);
                } catch (NumberFormatException exception) {
                    System.err.println("Resolution improperly formatted.  It must be in the form of a number, such as 1000000 for 1M bp,");
                    System.err.println("or a number followed by 'f', such as 25f for 25 fragment");
                    System.exit(1);
                }
                bpResolutions.add(myInt);
            }

            boolean resolutionsSet = false;
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

    public void setSubsampler(double subsampleFraction) {
        if (subsampleFraction == -1) {
            this.subsampleFraction = 1;
        }
        else {
            this.subsampleFraction = subsampleFraction;
        }
    }

    public void setThrowOutIntraFragOption(boolean throwOutIntraFrag) {
        PreprocessorTriple.throwOutIntraFrag = throwOutIntraFrag;
    }

    public void setFromHIC(boolean fromHIC) {
        PreprocessorTriple.fromHIC = fromHIC;
    }

    public void preprocess(final String inputFile, final String headerFile, final String footerFile,
                           Map<Integer, List<Chunk>> mndIndex) throws IOException {

        /*Question: what are mndIndex? What are Chunk?*/
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

            /*Question: what are losArray and losFooter?*/
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
        byte[] magicBytes = "HICT".getBytes();
        LittleEndianOutputStream los = losArray[0];
        for (int i = 0; i < 4; i++) {
            los.write(magicBytes[i]);
        }
        /*Question: what is the purpose of writting 0 here?*/
//        los.write(0);

        // VERSION
        los.writeInt(VERSION);
    
        // Placeholder for master index position, replaced with actual position after all contents are written
        masterIndexPositionPosition = los.getWrittenCount();
        los.writeLong(0L);
    
    
        // Genome ID
        los.writeString(genomeId);

        /*Question: What are NVI_INDEX?*/
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

        numResolutions = nBpRes;
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

//        MatrixPP wholeGenomeMatrix = computeWholeGenomeMatrix(inputFile);
//        writeMatrix(wholeGenomeMatrix, losArray, compressor, matrixPositions, -1, false);

        /*Should we write similar iterator for triplets? Yes!*/
//        PairIterator iter = PairIterator.getIterator(inputFile, chromosomeIndexes, chromosomeHandler);
        TripleIterator iter = TripleIterator.getIterator(inputFile, chromosomeIndexes, chromosomeHandler);

//        Set<String> writtenMatrices = Collections.synchronizedSet(new HashSet<>());
        Set<String> writtenTensors = Collections.synchronizedSet(new HashSet<>());

        int currentChr1 = -1;
        int currentChr2 = -1;
        int currentChr3 = -1;

        TensorPP currentTensor = null;
        String currentTensorKey = null;

        while (iter.hasNext()) {
            AlignmentTriple triple = iter.next();

            // Reorder triples if needed so chr1 < chr2 < chr3 in terms of chromosome number
            int chr1, chr2, chr3, bp1, bp2, bp3;

            int[] chrs = new int[] {triple.getChr1(), triple.getChr2(), triple.getChr3()};
            int[] bps = new int[] {triple.getPos1(), triple.getPos2(), triple.getPos3()};

            List<ChromBpTuple> chrBpPairs = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                chrBpPairs.add(new ChromBpTuple(chrs[i], bps[i]));
            }

            // Sort based on chromsome number
            chrBpPairs.sort(Comparator.comparingInt(p->p.getChrom()));
            chr1 = chrBpPairs.get(0).getChrom();
            chr2 = chrBpPairs.get(1).getChrom();
            chr3 = chrBpPairs.get(2).getChrom();

            bp1 = chrBpPairs.get(0).getBp();
            bp2 = chrBpPairs.get(1).getBp();
            bp3 = chrBpPairs.get(2).getBp();

            bp1 = ensureFitInChromosomeBounds(bp1, chr1);
            bp2 = ensureFitInChromosomeBounds(bp2, chr2);
            bp3 = ensureFitInChromosomeBounds(bp3, chr3);

            /*Question: what's the meaning of intrafragment here?*/
            // only increment if not intraFragment and passes the mapq threshold
            if (!(currentChr1 == chr1 && currentChr2 == chr2 && currentChr3 == chr3)) {
                // Starting a new tensor
                if (currentTensor != null) {
                    currentTensor.parsingComplete();
                    writeTensor(currentTensor, losArray, compressor, tensorPositions, -1, false);
                    writtenTensors.add(currentTensorKey);
                    currentTensor = null;
                    System.gc();
                    //System.out.println("Available memory: " + RuntimeUtils.getAvailableMemory());
                }

                // Start the next matrix
                currentChr1 = chr1;
                currentChr2 = chr2;
                currentChr3 = chr3;
                currentTensorKey = currentChr1 + "_" + currentChr2 + "_" + currentChr3;

                if (writtenTensors.contains(currentTensorKey)) {
                    System.err.println("Error: the chromosome combination " + currentTensorKey + " appears in multiple blocks");
                    if (outputFile != null) outputFile.deleteOnExit();
                    System.exit(58);
                }
                currentTensor = new TensorPP(currentChr1, currentChr2, currentChr3, chromosomeHandler, bpBinSizes, countThreshold, BLOCK_CAPACITY);
                }
            currentTensor.incrementCount(bp1, bp2, bp3, triple.getScore(), expectedValueCalculations, tmpDir);
        }

        if (currentTensor != null) {
            currentTensor.parsingComplete();
            writeTensor(currentTensor, losArray, compressor, tensorPositions, -1, false);
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
//
//    protected boolean shouldSkipContact(AlignmentPair pair) {
//        int chr1 = pair.getChr1();
//        int chr2 = pair.getChr2();
//        if (diagonalsOnly && chr1 != chr2) return true;
//        if (includedChromosomes != null && chr1 != 0) {
//            String c1Name = chromosomeHandler.getChromosomeFromIndex(chr1).getName();
//            String c2Name = chromosomeHandler.getChromosomeFromIndex(chr2).getName();
//            if (!includedChromosomes.contains(c1Name) || !includedChromosomes.contains(c2Name)) {
//                return true;
//            }
//        }
//        if (alignmentFilter != null && !alignmentsAreEqual(calculateAlignment(pair), alignmentFilter)) {
//            return true;
//        }
//        int mapq = Math.min(pair.getMapq1(), pair.getMapq2());
//        if (mapq < mapqThreshold) return true;
//
//        int frag1 = pair.getFrag1();
//        int frag2 = pair.getFrag2();
//
//        if (throwOutIntraFrag && chr1 == chr2 && frag1 == frag2) {return true;}
//
//        if ( subsampleFraction < 1 && subsampleFraction > 0) {
//            return randomSubsampleGenerator.nextDouble() > subsampleFraction;
//        } else { return false; }
//    }

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
        bufferList.get(bufferList.size()-1).putInt(tensorPositions.size());
        for (Map.Entry<String, IndexEntry> entry : tensorPositions.entrySet()) {
            if (Integer.MAX_VALUE - bufferList.get(bufferList.size()-1).bytesWritten() < 1000) {
                bufferList.add(new BufferedByteWriter());
            }
            bufferList.get(bufferList.size()-1).putNullTerminatedString(entry.getKey());
            bufferList.get(bufferList.size()-1).putLong(entry.getValue().position);
            bufferList.get(bufferList.size()-1).putInt(entry.getValue().size);
        }

        // For now, assume the expected vectors and norm vectors are not included!
//        // Vectors  (Expected values,  other).
//        /***  NEVA ***/
//        if (expectedVectorFile == null) {
//            if (Integer.MAX_VALUE - bufferList.get(bufferList.size()-1).bytesWritten() < 1000) {
//                bufferList.add(new BufferedByteWriter());
//            }
//            bufferList.get(bufferList.size()-1).putInt(expectedValueCalculations.size());
//            for (Map.Entry<String, ExpectedValueCalculation> entry : expectedValueCalculations.entrySet()) {
//                ExpectedValueCalculation ev = entry.getValue();
//
//                ev.computeDensity();
//
//                int binSize = ev.getGridSize();
//                HiC.Unit unit = ev.isFrag ? HiC.Unit.FRAG : HiC.Unit.BP;
//
//                bufferList.get(bufferList.size()-1).putNullTerminatedString(unit.toString());
//                bufferList.get(bufferList.size()-1).putInt(binSize);
//
//                // The density values
//                ListOfDoubleArrays expectedValues = ev.getDensityAvg();
//                // todo @Suhas to handle buffer overflow
//                bufferList.get(bufferList.size()-1).putLong(expectedValues.getLength());
//                for (double[] expectedArray : expectedValues.getValues()) {
//                    bufferList.add(new BufferedByteWriter());
//                    for (double value : expectedArray) {
//                        if (Integer.MAX_VALUE - bufferList.get(bufferList.size()-1).bytesWritten() < 1000000) {
//                            bufferList.add(new BufferedByteWriter());
//                        }
//                        bufferList.get(bufferList.size()-1).putFloat( (float) value);
//                    }
//                }
//
//                // Map of chromosome index -> normalization factor
//                Map<Integer, Double> normalizationFactors = ev.getChrScaleFactors();
//                if (Integer.MAX_VALUE - bufferList.get(bufferList.size()-1).bytesWritten() < 1000000) {
//                    bufferList.add(new BufferedByteWriter());
//                }
//                bufferList.get(bufferList.size()-1).putInt(normalizationFactors.size());
//                for (Map.Entry<Integer, Double> normFactor : normalizationFactors.entrySet()) {
//                    bufferList.get(bufferList.size()-1).putInt(normFactor.getKey());
//                    bufferList.get(bufferList.size()-1).putFloat(normFactor.getValue().floatValue());
//                    //System.out.println(normFactor.getKey() + "  " + normFactor.getValue());
//                }
//            }
//        }
//        else {
//            // read in expected vector file. to get # of resolutions, might have to read twice.
//
//            int count=0;
//            try (Reader reader = new FileReader(expectedVectorFile);
//                 BufferedReader bufferedReader = new BufferedReader(reader)) {
//
//                String line;
//                while ((line = bufferedReader.readLine()) != null) {
//                    if (line.startsWith("fixedStep"))
//                        count++;
//                    if (line.startsWith("variableStep")) {
//                        System.err.println("Expected vector file must be in wiggle fixedStep format");
//                        System.exit(19);
//                    }
//                }
//            }
//            bufferList.get(bufferList.size()-1).putInt(count);
//            try (Reader reader = new FileReader(expectedVectorFile);
//                 BufferedReader bufferedReader = new BufferedReader(reader)) {
//
//                String line;
//                while ((line = bufferedReader.readLine()) != null) {
//                    if (line.startsWith("fixedStep")) {
//                        String[] words = line.split("\\s+");
//                        for (String str:words){
//                            if (str.contains("chrom")){
//                                String[] chrs = str.split("=");
//
//                            }
//                        }
//                    }
//                }
//            }
//        }
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

    protected Pair<Map<Long, List<IndexEntry>>, Long> writeTensor(TensorPP tensor, LittleEndianOutputStream[] losArray,
                                                                  Deflater compressor, Map<String, IndexEntry> matrixPositions, int chromosomeTripleIndex, boolean doMultiThreadedBehavior) throws IOException {
        if (HiCGlobals.printVerboseComments) {
            System.err.println("Used Memory for matrix");
            System.err.println(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());
        }
        LittleEndianOutputStream los = losArray[0];
        long position = los.getWrittenCount();

        los.writeInt(tensor.getChr1Idx());
        los.writeInt(tensor.getChr2Idx());
        los.writeInt(tensor.getChr3Idx());
        int numResolutions = 0;

        for (TensorZoomDataPP zd : tensor.getZoomData()) {
            if (zd != null) {
                numResolutions++;
            }
        }
        los.writeInt(numResolutions);

        //fos.writeInt(matrix.getZoomData().length);
        for ( int i = 0; i < tensor.getZoomData().length; i++) {
            TensorZoomDataPP zd = tensor.getZoomData()[i];
            if (zd != null)
                writeZoomHeader(zd, los);
        }

        long size = los.getWrittenCount() - position;
        if (chromosomeTripleIndex > -1) {
            tensorPositions.put("" + chromosomeTripleIndex, new IndexEntry(position, (int) size));
        } else {
            tensorPositions.put(tensor.getKey(), new IndexEntry(position, (int) size));
        }

        final Map<Long, List<IndexEntry>> localBlockIndexes = new ConcurrentHashMap<>();

        for (int i = tensor.getZoomData().length-1 ; i >= 0; i--) {
            TensorZoomDataPP zd = tensor.getZoomData()[i];
            if (zd != null) {
                List<IndexEntry> blockIndex = null;

                blockIndex = zd.mergeAndWriteBlocks(losArray[0], compressor);
                updateIndexPositions(blockIndex, losArray, true, outputFile, 0, zd.blockIndexPosition);

            }
        }

        System.out.print(".");
        return new Pair<>(localBlockIndexes, position);
    }

    protected void updateIndexPositions(List<IndexEntry> blockIndex, LittleEndianOutputStream[] losArray, boolean doRestore,
                                        File outputFile, long currentPosition, long blockIndexPosition) throws IOException {

        /*Question: how does it update the index positions?*/
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

    private void writeZoomHeader(TensorZoomDataPP zd, LittleEndianOutputStream los) throws IOException {

        int numberOfBlocks = zd.blockNumbers.size();
        los.writeString(zd.getUnit().toString());  // Unit
        los.writeInt(zd.getZoom());     // zoom index,  lowest res is zero
        los.writeFloat((float) zd.getSum());      // sum
        los.writeFloat((float) zd.getOccupiedCellCount());
        los.writeFloat((float) zd.getPercent5());
        los.writeFloat((float) zd.getPercent95());
        los.writeInt(zd.getBinSize());
        los.writeInt(zd.getBlockBinCountX());
        los.writeInt(zd.getBlockBinCountY());
        los.writeInt(zd.getBlockBinCountZ());
        los.writeInt(zd.getBlockXCount());
        los.writeInt(zd.getBlockYCount());
        los.writeInt(zd.getBlockZCount());
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
