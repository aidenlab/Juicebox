/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

import htsjdk.tribble.util.LittleEndianOutputStream;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.matrix.BasicMatrix;
import juicebox.track.HiCFixedGridAxis;
import juicebox.track.HiCFragmentAxis;
import juicebox.track.HiCGridAxis;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.EigenDecompositionImpl;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.RealVector;
import org.apache.log4j.Logger;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.collections.LRUCache;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;



/**
 * @author jrobinso
 * @since Aug 10, 2010
 */
public class MatrixZoomData {

    private static final Logger log = Logger.getLogger(MatrixZoomData.class);
    private final Chromosome chr1;  // Chromosome on the X axis
    private final Chromosome chr2;  // Chromosome on the Y axis
    private final HiCZoom zoom;    // Unit and bin size
    private final HiCGridAxis xGridAxis;
    private final HiCGridAxis yGridAxis;
    // Observed values are organized into sub-matrices ("blocks")
    private final int blockBinCount;   // block size in bins
    private final int blockColumnCount;     // number of block columns
    private final HashMap<NormalizationType, BasicMatrix> pearsonsMap;
    private final HashSet<NormalizationType> missingPearsonFiles;
    // Cache the last 20 blocks loaded
    private final LRUCache<String, Block> blockCache = new LRUCache<String, Block>(20);
    DatasetReader reader;
    private double averageCount = -1;
//    private static final SuperAdapter superAdapter = new SuperAdapter();
//    private static final Slideshow slideshow = superAdapter.getSlideshow();


//    float sumCounts;
//    float avgCounts;
//    float stdDev;
//    float percent95 = -1;
//    float percent80 = -1;


    /**
     * Constructor, sets the grid axes.  Called when read from file.
     *
     * @param chr1             Chromosome 1
     * @param chr2             Chromosome 2
     * @param zoom             Zoom (bin size and BP or FRAG)
     * @param blockBinCount    Number of bins divided by number of columns (around 1000)
     * @param blockColumnCount Number of bins divided by 1000 (BLOCK_SIZE)
     * @param chr1Sites        Used for looking up fragment
     * @param chr2Sites        Used for looking up fragment
     * @param reader           Pointer to file reader
     */
    public MatrixZoomData(Chromosome chr1, Chromosome chr2, HiCZoom zoom, int blockBinCount, int blockColumnCount,
                          int[] chr1Sites, int[] chr2Sites, DatasetReader reader) {

        this.reader = reader;

        this.chr1 = chr1;
        this.chr2 = chr2;
        this.zoom = zoom;
        this.blockBinCount = blockBinCount;
        this.blockColumnCount = blockColumnCount;

        int correctedBinCount = blockBinCount;
        if (reader.getVersion() < 8 && chr1.getLength() < chr2.getLength()) {
            boolean isFrag = zoom.getUnit() == HiC.Unit.FRAG;
            int len1 = isFrag ? (chr1Sites.length + 1) : chr1.getLength();
            int len2 = isFrag ? (chr2Sites.length + 1) : chr2.getLength();
            int nBinsX = Math.max(len1, len2) / zoom.getBinSize() + 1;
            correctedBinCount = nBinsX / blockColumnCount + 1;
        }

        if (zoom.getUnit() == HiC.Unit.BP) {
            this.xGridAxis = new HiCFixedGridAxis(correctedBinCount * blockColumnCount, zoom.getBinSize(), chr1Sites);
            this.yGridAxis = new HiCFixedGridAxis(correctedBinCount * blockColumnCount, zoom.getBinSize(), chr2Sites);
        } else {
            this.xGridAxis = new HiCFragmentAxis(zoom.getBinSize(), chr1Sites, chr1.getLength());
            this.yGridAxis = new HiCFragmentAxis(zoom.getBinSize(), chr2Sites, chr2.getLength());

        }

        pearsonsMap = new HashMap<NormalizationType, BasicMatrix>();
        missingPearsonFiles = new HashSet<NormalizationType>();
    }

    public Chromosome getChr1() {
        return chr1;
    }

    public Chromosome getChr2() {
        return chr2;
    }

    public HiCGridAxis getXGridAxis() {
        return xGridAxis;
    }

    public HiCGridAxis getYGridAxis() {
        return yGridAxis;
    }

    public int getBinSize() {
        return zoom.getBinSize();
    }

    public int getChr1Idx() {
        return chr1.getIndex();
    }


    public int getChr2Idx() {
        return chr2.getIndex();
    }

    public HiCZoom getZoom() {
        return zoom;
    }

    private int getBlockColumnCount() {
        return blockColumnCount;
    }

    public String getKey() {
        return chr1.getName() + "_" + chr2.getName() + "_" + zoom.getKey();
    }


    /**
     * Return the blocks of normalized, observed values overlapping the rectangular region specified.
     * The units are "bins"
     *
     * @param binY1 leftmost position in "bins"
     * @param binX2 rightmost position in "bins"
     * @param binY2 bottom position in "bins"
     * @param no    normalization type
     * @return List of overlapping blocks, normalized
     */
    public List<Block> getNormalizedBlocksOverlapping(int binX1, int binY1, int binX2, int binY2, final NormalizationType no) {

        int col1 = binX1 / blockBinCount;
        int row1 = binY1 / blockBinCount;

        int col2 = binX2 / blockBinCount;
        int row2 = binY2 / blockBinCount;

        int maxSize = (col2 - col1 + 1) * (row2 - row1 + 1);

        final List<Block> blockList = new ArrayList<Block>(maxSize);
        final List<Integer> blocksToLoad = new ArrayList<Integer>();
        for (int r = row1; r <= row2; r++) {
            for (int c = col1; c <= col2; c++) {
                int blockNumber = r * getBlockColumnCount() + c;

                String key = getKey() + "_" + blockNumber + "_" + no;
                Block b;
                if (HiCGlobals.useCache && blockCache.containsKey(key)) {
                    b = blockCache.get(key);
                    blockList.add(b);
                } else {
                    blocksToLoad.add(blockNumber);
                }
            }
        }

        final AtomicInteger errorCounter = new AtomicInteger();

        List<Thread> threads = new ArrayList<Thread>();
        for (final int blockNumber : blocksToLoad) {
            Runnable loader = new Runnable() {
                @Override
                public void run() {
                    try {
                        String key = getKey() + "_" + blockNumber + "_" + no;
                        Block b = reader.readNormalizedBlock(blockNumber, MatrixZoomData.this, no);
                        if (b == null) {
                            b = new Block(blockNumber);   // An empty block
                        }
                        if (HiCGlobals.useCache) {
                            blockCache.put(key, b);
                        }
                        blockList.add(b);
                    } catch (IOException e) {
                        errorCounter.incrementAndGet();
                    }
                }
            };

            Thread t = new Thread(loader);
            threads.add(t);
            t.start();
        }

        // Wait for all threads to complete
        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException ignore) {
            }
        }

        // untested since files got fixed - MSS
        if (errorCounter.get() > 0) {
            return null;
        }

        return blockList;
    }


    /**
     * Return the blocks of normalized, observed values overlapping the rectangular region specified.
     * The units are "bins"
     *
     * @param binY1 leftmost position in "bins"
     * @param binX2 rightmost position in "bins"
     * @param binY2 bottom position in "bins"
     * @param no    normalization type
     * @return List of overlapping blocks, normalized
     */
    public int addNormalizedBlocksToList(final List<Block> blockList, int binX1, int binY1, int binX2, int binY2, final NormalizationType no) {

        int col1 = binX1 / blockBinCount;
        int row1 = binY1 / blockBinCount;

        int col2 = binX2 / blockBinCount;
        int row2 = binY2 / blockBinCount;

        List<Integer> blocksToLoad = new ArrayList<Integer>();
        for (int r = row1; r <= row2; r++) {
            for (int c = col1; c <= col2; c++) {
                int blockNumber = r * getBlockColumnCount() + c;

                String key = getKey() + "_" + blockNumber + "_" + no;
                Block b;
                if (HiCGlobals.useCache && blockCache.containsKey(key)) {
                    b = blockCache.get(key);
                    blockList.add(b);
                } else {
                    blocksToLoad.add(blockNumber);
                }
            }
        }

        final AtomicInteger errorCounter = new AtomicInteger();

        List<Thread> threads = new ArrayList<Thread>();
        for (final int blockNumber : blocksToLoad) {
            Runnable loader = new Runnable() {
                @Override
                public void run() {
                    try {
                        String key = getKey() + "_" + blockNumber + "_" + no;
                        Block b = reader.readNormalizedBlock(blockNumber, MatrixZoomData.this, no);
                        if (b == null) {
                            b = new Block(blockNumber);   // An empty block
                        }
                        if (HiCGlobals.useCache) {
                            blockCache.put(key, b);
                        }
                        blockList.add(b);
                    } catch (IOException e) {
                        errorCounter.incrementAndGet();
                    }
                }
            };

            Thread t = new Thread(loader);
            threads.add(t);
            t.start();
        }

        // Wait for all threads to complete
        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException ignore) {
            }
        }

        // untested since files got fixed - MSS
        return errorCounter.get();
    }


    /**
     * Return the observed value at the specified location. Supports tooltip text
     * This implementation is naive, but might get away with it for tooltip.
     *
     * @param binX              X bin
     * @param binY              Y bin
     * @param normalizationType Normalization type
     */
    public float getObservedValue(int binX, int binY, NormalizationType normalizationType) {

        // Intra stores only lower diagonal
        if (chr1 == chr2) {
            if (binX > binY) {
                int tmp = binX;
                //noinspection SuspiciousNameCombination
                binX = binY;
                binY = tmp;

            }
        }

        List<Block> blocks = getNormalizedBlocksOverlapping(binX, binY, binX, binY, normalizationType);
        if (blocks == null) return 0;
        for (Block b : blocks) {
            for (ContactRecord rec : b.getContactRecords()) {
                if (rec.getBinX() == binX && rec.getBinY() == binY) {
                    return rec.getCounts();
                }
            }
        }
        // No record found for this bin
        return 0;
    }

//    /**
//     * Return a slice of the matrix at the specified Y been as a list of wiggle scores
//     *
//     * @param binY
//     */
//    public List<BasicScore> getSlice(int startBinX, int endBinX, int binY, NormalizationType normalizationType) {
//
//        // Intra stores only lower diagonal
//        if (chr1 == chr2) {
//            if (binX > binY) {
//                int tmp = binX;
//                binX = binY;
//                binY = tmp;
//
//            }
//        }
//
//        List<Block> blocks = getNormalizedBlocksOverlapping(binX, binY, binX, binY, normalizationType);
//        if (blocks == null) return 0;
//        for (Block b : blocks) {
//            for (ContactRecord rec : b.getContactRecords()) {
//                if (rec.getBinX() == binX && rec.getBinY() == binY) {
//                    return rec.getCounts();
//                }
//            }
//        }
//        // No record found for this bin
//        return 0;
//    }

    /**
     * Computes eigenvector from Pearson's.
     *
     * @param df    Expected values, needed to get Pearson's
     * @param which Which eigenvector; 0 is principal.
     * @return Eigenvector
     */
    public double[] computeEigenvector(ExpectedValueFunction df, int which) {
        BasicMatrix pearsons = getPearsons(df);
        if (pearsons == null) {
            return null;
        }

        int dim = pearsons.getRowDimension();
        double[][] data = new double[dim][dim];
        BitSet bitSet = new BitSet(dim);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                float tmp = pearsons.getEntry(i, j);
                data[i][j] = tmp;
                if (data[i][j] != 0 && !Float.isNaN(tmp)) {
                    bitSet.set(i);
                }
            }
        }

        int[] nonCentromereColumns = new int[bitSet.cardinality()];
        int count = 0;
        for (int i = 0; i < dim; i++) {
            if (bitSet.get(i)) nonCentromereColumns[count++] = i;
        }

        RealMatrix subMatrix = new Array2DRowRealMatrix(data).getSubMatrix(nonCentromereColumns, nonCentromereColumns);
        RealVector rv = (new EigenDecompositionImpl(subMatrix, 0)).getEigenvector(which);

        double[] ev = rv.toArray();

        int size = pearsons.getColumnDimension();
        double[] eigenvector = new double[size];
        int num = 0;
        for (int i = 0; i < size; i++) {
            if (num < nonCentromereColumns.length && i == nonCentromereColumns[num]) {
                eigenvector[i] = ev[num];
                num++;
            } else {
                eigenvector[i] = Double.NaN;
            }
        }
        return eigenvector;

    }

    /**
     * Returns the Pearson's matrix; read if available (currently commented out), calculate if small enough.
     *
     * @param df Expected values
     * @return Pearson's matrix or null if not able to calculate or read
     */
    public BasicMatrix getPearsons(ExpectedValueFunction df) {

        BasicMatrix pearsons = pearsonsMap.get(df.getNormalizationType());
        if (pearsons == null && !missingPearsonFiles.contains(df.getNormalizationType())) {
            /*
            // We used to put precomputed Pearson's files in the directory with the appropriate key, but don't do
            // that now.  If we ever decide to again, uncomment.
            try {
                pearsons = reader.readPearsons(chr1.getName(), chr2.getName(), zoom, df.getNormalizationType());
            } catch (IOException e) {
                log.error(e.getMessage());
            }*/
            if (pearsons != null) {
                pearsonsMap.put(df.getNormalizationType(), pearsons);
            } else {
                missingPearsonFiles.add(df.getNormalizationType());  // To keep from trying repeatedly
            }
        }
        if ((zoom.getUnit() == HiC.Unit.BP && zoom.getBinSize() >= 500000) ||
                (zoom.getUnit() == HiC.Unit.FRAG && zoom.getBinSize() >= 500)) {
            pearsons = computePearsons(df);
            pearsonsMap.put(df.getNormalizationType(), pearsons);
        }

        return pearsonsMap.get(df.getNormalizationType());
    }

    /**
     * Returns Pearson value at given bin X and Y
     *
     * @param binX X bin
     * @param binY Y bin
     * @param type Normalization type
     * @return Pearson's value at this location
     */
    public float getPearsonValue(int binX, int binY, NormalizationType type) {
        BasicMatrix pearsons = pearsonsMap.get(type);
        if (pearsons != null) {
            return pearsons.getEntry(binX, binY);
        } else {
            return 0;
        }
    }

    /**
     * Compute the Pearson's.  Read in the observed, calculate O/E from the expected value function, subtract the row
     * means, compute the Pearson's correlation on that matrix
     *
     * @param df Expected value
     * @return Pearson's correlation matrix
     */
    private BasicMatrix computePearsons(ExpectedValueFunction df) {
        if (chr1 != chr2) {
            throw new RuntimeException("Cannot compute pearsons for non-diagonal matrices");
        }

        // # of columns.  We could let the data itself define this
        int dim;
        if (zoom.getUnit() == HiC.Unit.BP) {
            dim = chr1.getLength() / zoom.getBinSize() + 1;
        } else {
            dim = ((DatasetReaderV2) reader).getFragCount(chr1) / zoom.getBinSize() + 1;
        }

        // Compute O/E column vectors
        double[][] vectors = new double[dim][];

        // Loop through all contact records
        Iterator<ContactRecord> iter = contactRecordIterator();
        while (iter.hasNext()) {

            ContactRecord record = iter.next();
            int i = record.getBinX();
            int j = record.getBinY();
            float counts = record.getCounts();
            if (Float.isNaN(counts)) continue;

            int dist = Math.abs(i - j);
            double expected = df.getExpectedValue(chr1.getIndex(), dist);
            double oeValue = counts / expected;

            double[] vi = vectors[i];
            if (vi == null) {
                vi = new double[dim]; //zeroValue) ;
                vectors[i] = vi;
            }
            vi[j] = oeValue;


            double[] vj = vectors[j];
            if (vj == null) {
                vj = new double[dim]; // zeroValue) ;
                vectors[j] = vj;
            }
            vj[i] = oeValue;

        }

        // Subtract row means
        double[] rowMeans = new double[dim];
        for (int i = 0; i < dim; i++) {
            double[] row = vectors[i];
            rowMeans[i] = row == null ? 0 : getVectorMean(row);
        }

        for (int j = 0; j < dim; j++) {
            for (int i = 0; i < dim; i++) {
                double[] column = vectors[j];
                if (column == null) continue;
                column[i] -= rowMeans[i];
            }
        }

        BasicMatrix pearsons = Pearsons.computePearsons(vectors, dim);
        pearsonsMap.put(df.getNormalizationType(), pearsons);

        return pearsons;
    }

    /**
     * Return the mean of the given vector, ignoring NaNs
     *
     * @param vector Vector to calculate the mean on
     * @return The mean of the vector, not including NaNs.
     */
    private double getVectorMean(double[] vector) {
        double sum = 0;
        int count = 0;
        for (double aVector : vector) {
            if (!Double.isNaN(aVector)) {
                sum += aVector;
                count++;
            }
        }
        return count == 0 ? 0 : sum / count;
    }

    /**
     * Utility for printing description of this matrix.
     */
    public void printDescription() {
        System.out.println("Chromosomes: " + chr1.getName() + " - " + chr2.getName());
        System.out.println("unit: " + zoom.getUnit());
        System.out.println("binSize (bp): " + zoom.getBinSize());
        System.out.println("blockBinCount (bins): " + blockBinCount);
        System.out.println("blockColumnCount (columns): " + blockColumnCount);

        System.out.println("Block size (bp): " + blockBinCount * zoom.getBinSize());
        System.out.println("");

    }

    /**
     * For a specified region, select the block numbers corresponding to it
     * @param regionIndices
     * @return
     */
    private List<Integer> getBlockNumbersForRegionFromGenomePosition(int[] regionIndices) {
        int resolution = zoom.getBinSize();
        int[] regionBinIndices = new int[4];
        for (int i = 0; i < regionBinIndices.length; i++) {
            regionBinIndices[i] = regionIndices[i] / resolution;
        }
        return getBlockNumbersForRegionFromBinPosition(regionBinIndices);
    }

    private List<Integer> getBlockNumbersForRegionFromBinPosition(int[] regionIndices) {

        int col1 = regionIndices[0] / blockBinCount;
        int col2 = (regionIndices[1] + 1) / blockBinCount;
        int row1 = regionIndices[2] / blockBinCount;
        int row2 = (regionIndices[3] + 1) / blockBinCount;

        // first check the upper triangular matrix
        Set<Integer> blocksSet = new HashSet<Integer>();
        for (int r = row1; r <= row2; r++) {
            for (int c = col1; c <= col2; c++) {
                int blockNumber = r * getBlockColumnCount() + c;
                blocksSet.add(blockNumber);
            }
        }
        // check region part that overlaps with lower left triangle
        // but only if intrachromosomal
        if (chr1.getIndex() == chr2.getIndex()) {
            for (int r = col1; r <= col2; r++) {
                for (int c = row1; c <= row2; c++) {
                    int blockNumber = r * getBlockColumnCount() + c;
                    blocksSet.add(blockNumber);
                }
            }
        }

        List<Integer> blocksToIterateOver = new ArrayList<Integer>(blocksSet);
        Collections.sort(blocksToIterateOver);
        return blocksToIterateOver;
    }


    public void dump(PrintWriter printWriter, LittleEndianOutputStream les, NormalizationType norm, MatrixType matrixType,
                     boolean useRegionIndices, int[] regionIndices, ExpectedValueFunction df) throws IOException {

        // determine which output will be used
        if (printWriter == null && les == null) {
            printWriter = new PrintWriter(System.out);
        }
        boolean usePrintWriter = printWriter != null && les == null;
        boolean isIntraChromosomal = chr1.getIndex() == chr2.getIndex();

        if (matrixType == MatrixType.PEARSON) {
            dumpPearsons(printWriter, les, df);
            return;
        }

        // Get the block index keys, and sort
        List<Integer> blocksToIterateOver;
        if (useRegionIndices) {
            blocksToIterateOver = getBlockNumbersForRegionFromGenomePosition(regionIndices);
        } else {
            blocksToIterateOver = reader.getBlockNumbers(this);
            Collections.sort(blocksToIterateOver);
        }

        for (Integer blockNumber : blocksToIterateOver) {
            Block b = reader.readNormalizedBlock(blockNumber, MatrixZoomData.this, norm);
            if (b != null) {
                for (ContactRecord rec : b.getContactRecords()) {
                    float counts = rec.getCounts();
                    int x = rec.getBinX();
                    int y = rec.getBinY();
                    int xActual = x * zoom.getBinSize();
                    int yActual = y * zoom.getBinSize();
                    float oeVal = 0f;
                    if (matrixType == MatrixType.OE) {
                        int dist = Math.abs(x - y);
                        double expected = 0;
                        try {
                            expected = df.getExpectedValue(chr1.getIndex(), dist);
                        } catch (Exception e) {
                            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
                        }
                        double observed = rec.getCounts(); // Observed is already normalized
                        oeVal = (float) (observed / expected);
                    }
                    if (!useRegionIndices || // i.e. use full matrix
                            // or check regions that overlap with upper left
                            (xActual >= regionIndices[0] && xActual <= regionIndices[1] &&
                                    yActual >= regionIndices[2] && yActual <= regionIndices[3]) ||
                            // or check regions that overlap with lower left
                            (isIntraChromosomal && yActual >= regionIndices[0] && yActual <= regionIndices[1] &&
                                    xActual >= regionIndices[2] && xActual <= regionIndices[3])) {
                        // but leave in upper right triangle coordinates
                        if (usePrintWriter) {
                            if (matrixType == MatrixType.OBSERVED) {
                                printWriter.println(xActual + "\t" + yActual + "\t" + counts);
                            } else if (matrixType == MatrixType.OE) {
                                printWriter.println(xActual + "\t" + yActual + "\t" + oeVal);
                            }
                        } else {
                            if (matrixType == MatrixType.OBSERVED) {
                                les.writeInt(x);
                                les.writeInt(y);
                                les.writeFloat(counts);
                            } else if (matrixType == MatrixType.OE) {
                                les.writeInt(x);
                                les.writeInt(y);
                                les.writeFloat(oeVal);
                            }
                        }
                    }
                }
            }
        }

        if (usePrintWriter) {
            printWriter.close();
        }
    }

    public void dump1DTrackFromCrossHairAsWig(PrintWriter printWriter, Chromosome chromosomeForPosition,
                                              int binStartPosition, boolean isIntraChromosomal, int[] regionBinIndices,
                                              NormalizationType norm, MatrixType matrixType,
                                              ExpectedValueFunction expectedValues) throws IOException {

        if (!MatrixType.isObservedOrControl(matrixType)) {
            System.out.println("This feature is only available for Observed or Control views");
            return;
        }

        int binCounter = 0;

        // Get the block index keys, and sort
        List<Integer> blocksToIterateOver = getBlockNumbersForRegionFromBinPosition(regionBinIndices);
        Collections.sort(blocksToIterateOver);

        for (Integer blockNumber : blocksToIterateOver) {
            Block b = reader.readNormalizedBlock(blockNumber, MatrixZoomData.this, norm);
            if (b != null) {
                for (ContactRecord rec : b.getContactRecords()) {
                    float counts = rec.getCounts();
                    int x = rec.getBinX();
                    int y = rec.getBinY();

                    if (    //check regions that overlap with upper left
                            (x >= regionBinIndices[0] && x <= regionBinIndices[1] &&
                                    y >= regionBinIndices[2] && y <= regionBinIndices[3]) ||
                                    // or check regions that overlap with lower left
                                    (isIntraChromosomal && x >= regionBinIndices[2] && x <= regionBinIndices[3] &&
                                            y >= regionBinIndices[0] && y <= regionBinIndices[1])) {
                        // but leave in upper right triangle coordinates

                        if (x == binStartPosition) {
                            while (binCounter < y) {
                                printWriter.println("0");
                                binCounter++;
                            }
                        } else if (y == binStartPosition) {
                            while (binCounter < x) {
                                printWriter.println("0");
                                binCounter++;
                            }
                        } else {
                            System.err.println("Something went wrong while generating 1D track");
                            System.err.println("Improper input was likely provided");
                        }

                        printWriter.println(counts);
                        binCounter++;

                    }
                }
            }
        }
    }

    private void dumpPearsons(PrintWriter pw, LittleEndianOutputStream les, ExpectedValueFunction df) throws IOException {
        BasicMatrix pearsons = getPearsons(df);
        if (pearsons != null) {
            int dim = pearsons.getRowDimension();
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    float output = pearsons.getEntry(i, j);
                    if (les != null) les.writeFloat(output);
                    else pw.print(output + " ");
                }
                if (les == null) pw.println();
            }
            pw.flush();
        } else {
            log.error("Pearson's not available at zoom " + zoom);
        }
    }

    /**
     * Returns the average count
     *
     * @return Average count
     */
    public double getAverageCount() {
        return averageCount;
    }

    /**
     * Sets the average count
     *
     * @param averageCount Average count to set
     */
    public void setAverageCount(double averageCount) {
        this.averageCount = averageCount;
    }

    /**
     * Returns iterator for contact records
     *
     * @return iterator for contact records
     */
    public Iterator<ContactRecord> contactRecordIterator() {
        return new ContactRecordIterator();
    }

    /**
     * Class for iterating over the contact records
     */
    public class ContactRecordIterator implements Iterator<ContactRecord> {

        final List<Integer> blockNumbers;
        int blockIdx;
        Iterator<ContactRecord> currentBlockIterator;

        /**
         * Initializes the iterator
         */
        public ContactRecordIterator() {
            this.blockIdx = -1;
            this.blockNumbers = reader.getBlockNumbers(MatrixZoomData.this);
        }

        /**
         * Indicates whether or not there is another block waiting; checks current block
         * iterator and creates a new one if need be
         *
         * @return true if there is another block to be read
         */
        @Override
        public boolean hasNext() {

            if (currentBlockIterator != null && currentBlockIterator.hasNext()) {
                return true;
            } else {
                blockIdx++;
                if (blockIdx < blockNumbers.size()) {
                    try {
                        int blockNumber = blockNumbers.get(blockIdx);

                        // Optionally check the cache
                        String key = getKey() + "_" + blockNumber + "_" + NormalizationType.NONE;
                        Block nextBlock;
                        if (HiCGlobals.useCache && blockCache.containsKey(key)) {
                            nextBlock = blockCache.get(key);
                        } else {
                            nextBlock = reader.readBlock(blockNumber, MatrixZoomData.this);
                        }
                        currentBlockIterator = nextBlock.getContactRecords().iterator();
                        return true;
                    } catch (IOException e) {
                        log.error("Error fetching block ", e);
                        return false;
                    }
                }
            }

            return false;
        }

        /**
         * Returns the next contact record
         *
         * @return The next contact record
         */
        @Override
        public ContactRecord next() {
            return currentBlockIterator == null ? null : currentBlockIterator.next();
        }

        /**
         * Not supported
         */
        @Override
        public void remove() {
            //Not supported
            throw new RuntimeException("remove() is not supported");
        }
    }
//    public void preloadSlides(){

//    }
}
