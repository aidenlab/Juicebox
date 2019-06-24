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


package juicebox.data;

import htsjdk.tribble.util.LittleEndianOutputStream;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.assembly.AssemblyHeatmapHandler;
import juicebox.assembly.AssemblyScaffoldHandler;
import juicebox.assembly.Scaffold;
import juicebox.gui.SuperAdapter;
import juicebox.matrix.BasicMatrix;
import juicebox.matrix.RealMatrixWrapper;
import juicebox.tools.clt.old.Pearsons;
import juicebox.track.HiCFixedGridAxis;
import juicebox.track.HiCFragmentAxis;
import juicebox.track.HiCGridAxis;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.EigenDecompositionImpl;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.RealVector;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.collections.LRUCache;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;



/**
 * @author jrobinso
 * @since Aug 10, 2010
 */
public class MatrixZoomData {

    final Chromosome chr1;  // Chromosome on the X axis
    final Chromosome chr2;  // Chromosome on the Y axis
    final HiCZoom zoom;    // Unit and bin size
    private final HiCGridAxis xGridAxis;
    private final HiCGridAxis yGridAxis;
    // Observed values are organized into sub-matrices ("blocks")
    private final int blockBinCount;   // block size in bins
    private final int blockColumnCount;     // number of block columns
    // Cache the last 20 blocks loaded
    private final LRUCache<String, Block> blockCache = new LRUCache<>(500);
    private final HashMap<NormalizationType, BasicMatrix> pearsonsMap;
    private final HashMap<NormalizationType, BasicMatrix> normSquaredMaps;
    private final HashSet<NormalizationType> missingPearsonFiles;
    DatasetReader reader;
    private double averageCount = -1;

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
        this.chr1 = chr1;
        this.chr2 = chr2;
        this.zoom = zoom;
        this.reader = reader;
        this.blockBinCount = blockBinCount;
        this.blockColumnCount = blockColumnCount;

        int correctedBinCount = blockBinCount;
        if (reader.getVersion() < 8 && chr1.getLength() < chr2.getLength()) {
            boolean isFrag = zoom.getUnit() == HiC.Unit.FRAG;
            int len1 = chr1.getLength();
            int len2 = chr2.getLength();
            if (chr1Sites != null && chr2Sites != null && isFrag) {
                len1 = chr1Sites.length + 1;
                len2 = chr2Sites.length + 1;
            }
            int nBinsX = Math.max(len1, len2) / zoom.getBinSize() + 1;
            correctedBinCount = nBinsX / blockColumnCount + 1;
        }

        if (this instanceof CustomMatrixZoomData) {
            this.xGridAxis = new HiCFixedGridAxis(chr1.getLength() / zoom.getBinSize() + 1, zoom.getBinSize(), null);
            this.yGridAxis = new HiCFixedGridAxis(chr2.getLength() / zoom.getBinSize() + 1, zoom.getBinSize(), null);
        } else if (zoom.getUnit() == HiC.Unit.BP) {
            this.xGridAxis = new HiCFixedGridAxis(correctedBinCount * blockColumnCount, zoom.getBinSize(), chr1Sites);
            this.yGridAxis = new HiCFixedGridAxis(correctedBinCount * blockColumnCount, zoom.getBinSize(), chr2Sites);
        } else {
            this.xGridAxis = new HiCFragmentAxis(zoom.getBinSize(), chr1Sites, chr1.getLength());
            this.yGridAxis = new HiCFragmentAxis(zoom.getBinSize(), chr2Sites, chr2.getLength());
        }

        pearsonsMap = new HashMap<>();
        normSquaredMaps = new HashMap<>();
        missingPearsonFiles = new HashSet<>();
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

    // i think this is how it should be? todo sxxgrc please confirm use case
    private String getKey(int chr1, int chr2) {
        return chr1 + "_" + chr2 + "_" + zoom.getKey();
    }

    public String getBlockKey(int blockNumber, NormalizationType no) {
        return getKey() + "_" + blockNumber + "_" + no;
    }

    public String getNormLessBlockKey(Block block) {
        return getKey() + "_" + block.getNumber() + "_";
    }

    private String getBlockKey(int blockNumber, NormalizationType no, int chr1, int chr2) {
        return getKey(chr1, chr2) + "_" + blockNumber + "_" + no;
    }

    public String getColorScaleKey(MatrixType displayOption, NormalizationType n1, NormalizationType n2) {
        return getKey() + displayOption + "_" + n1 + "_" + n2;
    }

    public String getTileKey(int tileRow, int tileColumn, MatrixType displayOption) {
        return getKey() + "_" + tileRow + "_" + tileColumn + "_ " + displayOption;
    }

    /**
     * Return the blocks of normalized, observed values overlapping the rectangular region specified.
     * The units are "bins"
     *
     * @param binY1 leftmost position in "bins"
     * @param binX2 rightmost position in "bins"
     * @param binY2 bottom position in "bins"
     * @param no    normalization type
     * @param isImportant used for debugging
     * @return List of overlapping blocks, normalized
     */
    public List<Block> getNormalizedBlocksOverlapping(int binX1, int binY1, int binX2, int binY2, final NormalizationType no,
                                                      boolean isImportant) {
        final List<Block> blockList = new ArrayList<>();
        Block b = new Block(1, getBlockKey(1, no));
        if (HiCGlobals.isAssemblyMatCheck) {
            return addNormalizedBlocksToList(blockList, binX1, binY1, binX2, binY2, no, 1, 1);
        } else if (SuperAdapter.assemblyModeCurrentlyActive && !HiCGlobals.isAssemblyMatCheck) {
            return addNormalizedBlocksToListAssembly(blockList, binX1, binY1, binX2, binY2, no);
        } else {
            return addNormalizedBlocksToList(blockList, binX1, binY1, binX2, binY2, no);
        }
    }

    private void populateBlocksToLoad(int r, int c, NormalizationType no, List<Block> blockList, Set<Integer> blocksToLoad) {
        int blockNumber = r * getBlockColumnCount() + c;
        String key = getBlockKey(blockNumber, no);
        Block b;
        if (HiCGlobals.useCache && blockCache.containsKey(key)) {
            b = blockCache.get(key);
            blockList.add(b);
        } else {
            blocksToLoad.add(blockNumber);
        }
    }

    /**
     * Return the blocks of normalized, observed values overlapping the rectangular region specified.
     *
     * @param binY1 leftmost position in "bins"
     * @param binX2 rightmost position in "bins"
     * @param binY2 bottom position in "bins"
     * @param no    normalization type
     * @return List of overlapping blocks, normalized
     */
    private List<Block> addNormalizedBlocksToList(final List<Block> blockList, int binX1, int binY1, int binX2, int binY2,
                                                  final NormalizationType no) {

        Set<Integer> blocksToLoad = new HashSet<>();
      
        // have to do this regardless (just in case)
        int col1 = binX1 / blockBinCount;
        int row1 = binY1 / blockBinCount;
        int col2 = binX2 / blockBinCount;
        int row2 = binY2 / blockBinCount;

        for (int r = row1; r <= row2; r++) {
            for (int c = col1; c <= col2; c++) {
                populateBlocksToLoad(r, c, no, blockList, blocksToLoad);
            }
        }

        actuallyLoadGivenBlocks(blockList, blocksToLoad, no);

        return new ArrayList<>(new HashSet<>(blockList));
    }

    private List<Block> addNormalizedBlocksToList(final List<Block> blockList, int binX1, int binY1, int binX2, int binY2,
                                                  final NormalizationType no, int chr1, int chr2) {

        Set<Integer> blocksToLoad = new HashSet<>();

        // have to do this regardless (just in case)
        int col1 = binX1 / blockBinCount;
        int row1 = binY1 / blockBinCount;
        int col2 = binX2 / blockBinCount;
        int row2 = binY2 / blockBinCount;

        for (int r = row1; r <= row2; r++) {
            for (int c = col1; c <= col2; c++) {
                populateBlocksToLoad(r, c, no, blockList, blocksToLoad);
            }
        }

        actuallyLoadGivenBlocks(blockList, blocksToLoad, no, chr1, chr2);
        System.out.println("I am block size: " + blockList.size());
        System.out.println("I am first block: " + blockList.get(0).getNumber());
        return new ArrayList<>(new HashSet<>(blockList));
    }

    private List<Block> addNormalizedBlocksToListAssembly(final List<Block> blockList, int binX1, int binY1, int binX2, int binY2,
                                                          final NormalizationType no) {

        Set<Integer> blocksToLoad = new HashSet<>();

        // get aggregate scaffold handler
        AssemblyScaffoldHandler aFragHandler = AssemblyHeatmapHandler.getSuperAdapter().getAssemblyStateTracker().getAssemblyHandler();

        final int binSize = zoom.getBinSize();
        long actualBinSize = (long) binSize;
        if (chr1.getIndex() == 0 && chr2.getIndex() == 0) {
            actualBinSize = 1000 * actualBinSize;
        }

        List<Scaffold> xAxisAggregateScaffolds = aFragHandler.getIntersectingAggregateFeatures(
                (long) (actualBinSize * binX1 * HiCGlobals.hicMapScale), (long) (actualBinSize * binX2 * HiCGlobals.hicMapScale));
        List<Scaffold> yAxisAggregateScaffolds = aFragHandler.getIntersectingAggregateFeatures(
                (long) (actualBinSize * binY1 * HiCGlobals.hicMapScale), (long) (actualBinSize * binY2 * HiCGlobals.hicMapScale));

        int x1pos, x2pos, y1pos, y2pos;

        for (Scaffold xScaffold : xAxisAggregateScaffolds) {
            for (Scaffold yScaffold : yAxisAggregateScaffolds) {

                x1pos = (int) (xScaffold.getOriginalStart() / HiCGlobals.hicMapScale);
                x2pos = (int) (xScaffold.getOriginalEnd() / HiCGlobals.hicMapScale);
                y1pos = (int) (yScaffold.getOriginalStart() / HiCGlobals.hicMapScale);
                y2pos = (int) (yScaffold.getOriginalEnd() / HiCGlobals.hicMapScale);

                // have to case long because of thumbnail, maybe fix thumbnail instead

                if (xScaffold.getCurrentStart() < actualBinSize * binX1 * HiCGlobals.hicMapScale) {
                    if (!xScaffold.getInvertedVsInitial()) {
                        x1pos = (int) ((xScaffold.getOriginalStart() + actualBinSize * binX1 * HiCGlobals.hicMapScale - xScaffold.getCurrentStart()) / HiCGlobals.hicMapScale);
                    } else {
                        x2pos = (int) ((xScaffold.getOriginalStart() - actualBinSize * binX1 * HiCGlobals.hicMapScale + xScaffold.getCurrentEnd()) / HiCGlobals.hicMapScale);
                    }
                }

                if (yScaffold.getCurrentStart() < actualBinSize * binY1 * HiCGlobals.hicMapScale) {
                    if (!yScaffold.getInvertedVsInitial()) {
                        y1pos = (int) ((yScaffold.getOriginalStart() + actualBinSize * binY1 * HiCGlobals.hicMapScale - yScaffold.getCurrentStart()) / HiCGlobals.hicMapScale);
                    } else {
                        y2pos = (int) ((yScaffold.getOriginalStart() - actualBinSize * binY1 * HiCGlobals.hicMapScale + yScaffold.getCurrentEnd()) / HiCGlobals.hicMapScale);
                    }
                }

                if (xScaffold.getCurrentEnd() > actualBinSize * binX2 * HiCGlobals.hicMapScale) {
                    if (!xScaffold.getInvertedVsInitial()) {
                        x2pos = (int) ((xScaffold.getOriginalStart() + actualBinSize * binX2 * HiCGlobals.hicMapScale - xScaffold.getCurrentStart()) / HiCGlobals.hicMapScale);
                    } else {
                        x1pos = (int) ((xScaffold.getOriginalStart() - actualBinSize * binX2 * HiCGlobals.hicMapScale + xScaffold.getCurrentEnd()) / HiCGlobals.hicMapScale);
                    }
                }

                if (yScaffold.getCurrentEnd() > actualBinSize * binY2 * HiCGlobals.hicMapScale) {
                    if (!yScaffold.getInvertedVsInitial()) {
                        y2pos = (int) ((yScaffold.getOriginalStart() + actualBinSize * binY2 * HiCGlobals.hicMapScale - yScaffold.getCurrentStart()) / HiCGlobals.hicMapScale);
                    } else {
                        y1pos = (int) ((yScaffold.getOriginalStart() - actualBinSize * binY2 * HiCGlobals.hicMapScale + yScaffold.getCurrentEnd()) / HiCGlobals.hicMapScale);
                    }
                }

                int[] genomePosition = new int[]{
                        x1pos, x2pos, y1pos, y2pos
                };

                List<Integer> tempBlockNumbers = getBlockNumbersForRegionFromGenomePosition(genomePosition);
                for (int blockNumber : tempBlockNumbers) {
                    if (!blocksToLoad.contains(blockNumber)) {
                        String key = getBlockKey(blockNumber, no);
                        Block b;
                        //temp fix for AllByAll. TODO: trace this!
                        if (HiCGlobals.useCache && blockCache.containsKey(key)) {
                            b = blockCache.get(key);
                            blockList.add(b);
                        } else {
                            blocksToLoad.add(blockNumber);
                        }
                    }
                }
            }
        }

        // Remove basic duplicates here
        // Actually load new blocks
        actuallyLoadGivenBlocks(blockList, blocksToLoad, no);

        return new ArrayList<>(new HashSet<>(blockList));
    }

//    private List<Contig2D> retrieveContigsIntersectingWithWindow(Feature2DHandler handler, Rectangle currentWindow) {
//        List<Feature2D> xAxisFeatures;
//        if (chr1.getIndex() == 0 && chr2.getIndex() == 0) {
//            xAxisFeatures = handler.getIntersectingFeatures(1, 1, currentWindow, true);
//            // helps with disappearing heatmap but doesn't fix everything
//        } else {
//            xAxisFeatures = handler.getIntersectingFeatures(chr1.getIndex(), chr2.getIndex(), currentWindow, true);
//        }
//        List<Contig2D> axisContigs = new ArrayList<>();
//        for (Feature2D feature2D : new HashSet<>(xAxisFeatures)) {
//            axisContigs.add(feature2D.toContig());
//        }
//        Collections.sort(axisContigs);
//        return AssemblyHeatmapHandler.mergeRedundantContiguousContigs(axisContigs);
//    }

    private void actuallyLoadGivenBlocks(final List<Block> blockList, Set<Integer> blocksToLoad,
                                         final NormalizationType no) {
        final AtomicInteger errorCounter = new AtomicInteger();

        ExecutorService service = Executors.newFixedThreadPool(200);

        final int binSize = getBinSize();
        final int chr1Index = chr1.getIndex();
        final int chr2Index = chr2.getIndex();

        for (final int blockNumber : blocksToLoad) {
            Runnable loader = new Runnable() {
                @Override
                public void run() {
                    try {
                        String key = getBlockKey(blockNumber, no);
                        Block b = reader.readNormalizedBlock(blockNumber, MatrixZoomData.this, no);
                        if (b == null) {
                            b = new Block(blockNumber, key);   // An empty block
                        }
                        //Run out of memory if do it here
                        if (SuperAdapter.assemblyModeCurrentlyActive) {
                            b = AssemblyHeatmapHandler.modifyBlock(b, key, binSize, chr1Index, chr2Index);
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

            service.submit(loader);
        }

        // done submitting all jobs
        service.shutdown();

        // wait for all to finish
        try {
            service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            System.err.println("Error loading mzd data " + e.getLocalizedMessage());
            if (HiCGlobals.printVerboseComments) {
                e.printStackTrace();
            }
        }

        // error printing
        if (errorCounter.get() > 0) {
            System.err.println(errorCounter.get() + " errors while reading blocks");
        }
    }

    private void actuallyLoadGivenBlocks(final List<Block> blockList, Set<Integer> blocksToLoad,
                                         final NormalizationType no, final int chr1Id, final int chr2Id) {
        final AtomicInteger errorCounter = new AtomicInteger();

        ExecutorService service = Executors.newFixedThreadPool(200);

        final int binSize = getBinSize();

        for (final int blockNumber : blocksToLoad) {
            Runnable loader = new Runnable() {
                @Override
                public void run() {
                    try {
                        String key = getBlockKey(blockNumber, no, chr1Id, chr2Id);
                        Block b = reader.readNormalizedBlock(blockNumber, MatrixZoomData.this, no);
                        if (b == null) {
                            b = new Block(blockNumber, key);   // An empty block
                        }
                        //Run out of memory if do it here
                        if (SuperAdapter.assemblyModeCurrentlyActive) {
                            b = AssemblyHeatmapHandler.modifyBlock(b, key, binSize, chr1Id, chr2Id);
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

            service.submit(loader);
        }

        // done submitting all jobs
        service.shutdown();

        // wait for all to finish
        try {
            service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            System.err.println("Error loading mzd data " + e.getLocalizedMessage());
            if (HiCGlobals.printVerboseComments) {
                e.printStackTrace();
            }
        }

        // error printing
        if (errorCounter.get() > 0) {
            System.err.println(errorCounter.get() + " errors while reading blocks");
        }
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

        List<Block> blocks = getNormalizedBlocksOverlapping(binX, binY, binX, binY, normalizationType, false);
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

    public BasicMatrix getNormSquared(NormalizationType normalizationType) {

        if (normSquaredMaps.containsKey(normalizationType) && normSquaredMaps.get(normalizationType) != null) {
            return normSquaredMaps.get(normalizationType);
        }

        // otherwise calculate
        BasicMatrix normSquared = computeNormSquared(normalizationType);
        normSquaredMaps.put(normalizationType, normSquared);
        return normSquared;
    }

    // todo only compute local region at high resolution otherwise memory gets exceeded
    private BasicMatrix computeNormSquared(NormalizationType normalizationType) {
        double[] nv1Data = reader.getNormalizationVector(getChr1Idx(), getZoom(), normalizationType).getData();
        double[] nv2Data = reader.getNormalizationVector(getChr2Idx(), getZoom(), normalizationType).getData();

        double[][] matrix = new double[nv1Data.length][nv2Data.length];
        for (int i = 0; i < nv1Data.length; i++) {
            for (int j = 0; j < nv2Data.length; j++) {
                int diff = Math.max(1, Math.abs(i - j));
                matrix[i][j] = 1 / (nv1Data[i] * nv2Data[j] * diff * diff * diff * diff);
            }
        }

        return new RealMatrixWrapper(new Array2DRowRealMatrix(matrix));
    }

    /**
     * Returns the Pearson's matrix; read if available (currently commented out), calculate if small enough.
     *
     * @param df Expected values
     * @return Pearson's matrix or null if not able to calculate or read
     */
    public BasicMatrix getPearsons(ExpectedValueFunction df) {
        boolean readPearsons = false; // check if were able to read in
        // try to get from local cache
        BasicMatrix pearsons = pearsonsMap.get(df.getNormalizationType());
        if (pearsons != null) {
            return pearsons;
        }
        else if (!missingPearsonFiles.contains(df.getNormalizationType())) {
            // try to read
            try {
                pearsons = reader.readPearsons(chr1.getName(), chr2.getName(), zoom, df.getNormalizationType());
            } catch (IOException e) {
                pearsons = null;
                System.err.println(e.getMessage());
            }
            if (pearsons != null) {
                // put it back in the map.
                pearsonsMap.put(df.getNormalizationType(), pearsons);
                readPearsons = true;
            } else {
                missingPearsonFiles.add(df.getNormalizationType());  // To keep from trying repeatedly
            }
        }
        // we weren't able to read in the Pearsons. check that the resolution is low enough to calculate
        if (!readPearsons && (zoom.getUnit() == HiC.Unit.BP && zoom.getBinSize() >= HiCGlobals.MAX_PEARSON_ZOOM) ||
                (zoom.getUnit() == HiC.Unit.FRAG && zoom.getBinSize() >= HiCGlobals.MAX_PEARSON_ZOOM/1000)) {
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
        Iterator<ContactRecord> iter = getNewContactRecordIterator();
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
    public String getDescription() {
        return chr1.getName() + " - " + chr2.getName();
    }

    public void printFullDescription() {
        System.out.println("Chromosomes: " + chr1.getName() + " - " + chr2.getName());
        System.out.println("unit: " + zoom.getUnit());
        System.out.println("binSize (bp): " + zoom.getBinSize());
        System.out.println("blockBinCount (bins): " + blockBinCount);
        System.out.println("blockColumnCount (columns): " + blockColumnCount);

        System.out.println("Block size (bp): " + blockBinCount * zoom.getBinSize());
        System.out.println();

    }

    /**
     * For a specified region, select the block numbers corresponding to it
     * @param regionIndices
     * @return
     */
    List<Integer> getBlockNumbersForRegionFromGenomePosition(int[] regionIndices) {
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
        Set<Integer> blocksSet = new HashSet<>();
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

        List<Integer> blocksToIterateOver = new ArrayList<>(blocksSet);
        Collections.sort(blocksToIterateOver);
        return blocksToIterateOver;
    }


    public void dump(PrintWriter printWriter, LittleEndianOutputStream les, NormalizationType norm, MatrixType matrixType,
                     boolean useRegionIndices, int[] regionIndices, ExpectedValueFunction df, boolean dense) throws IOException {

        // determine which output will be used
        if (printWriter == null && les == null) {
            printWriter = new PrintWriter(System.out);
        }
        boolean usePrintWriter = printWriter != null && les == null;
        boolean isIntraChromosomal = chr1.getIndex() == chr2.getIndex();

        // Get the block index keys, and sort
        List<Integer> blocksToIterateOver;
        if (useRegionIndices) {
            blocksToIterateOver = getBlockNumbersForRegionFromGenomePosition(regionIndices);
        } else {
            blocksToIterateOver = reader.getBlockNumbers(this);
            Collections.sort(blocksToIterateOver);
        }

        if (!dense) {
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
                            double expected = 0;
                            if (chr1 == chr2) {
                                if (df != null) {
                                    int dist = Math.abs(x - y);
                                    expected = df.getExpectedValue(chr1.getIndex(), dist);
                                }
                            } else {
                                expected = (averageCount > 0 ? averageCount : 1);
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
                                // TODO I suspect this is wrong - should be writing xActual - but this is for binary dumping and we never use it
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
            else {
                les.close();
            }
        }
        else {
            int maxX = 0;
            int maxY = 0;
            for (Integer blockNumber : blocksToIterateOver) {
                Block b = reader.readNormalizedBlock(blockNumber, MatrixZoomData.this, norm);
                if (b != null) {
                    for (ContactRecord rec : b.getContactRecords()) {
                        int x = rec.getBinX();
                        int y = rec.getBinY();
                        if (maxX < x) maxX = x;
                        if (maxY < y) maxY = y;
                    }
                }
            }
            if (isIntraChromosomal) {
                if (maxX < maxY) {
                    maxX = maxY;
                } else {
                    maxY = maxX;
                }
            }

            maxX++;
            maxY++;
            float[][] matrix = new float[maxX][maxY];  // auto initialized to 0

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

                            if (matrixType == MatrixType.OBSERVED) {
                                matrix[x][y] = counts;
                                if (isIntraChromosomal) {
                                    matrix[y][x] = counts;
                                }
                                // printWriter.println(xActual + "\t" + yActual + "\t" + counts);
                            } else if (matrixType == MatrixType.OE) {
                                matrix[x][y] = oeVal;
                                if (isIntraChromosomal) {
                                    matrix[y][x] = oeVal;
                                }
                                // printWriter.println(xActual + "\t" + yActual + "\t" + oeVal);
                            }
                        }
                    }
                }
            }
            if (usePrintWriter) {
                for (int i = 0; i < maxX; i++) {
                    for (int j = 0; j < maxY; j++) {
                        printWriter.print(matrix[i][j] + "\t");
                    }
                    printWriter.println();
                }
            } else {
                for (int i = 0; i < maxX; i++) {
                    for (int j = 0; j < maxY; j++) {
                        les.writeFloat(matrix[i][j]);

                    }

                }
            }

            if (usePrintWriter) {
                printWriter.close();
            }
            else {
                les.close();
            }
        }
    }

    public void dump1DTrackFromCrossHairAsWig(PrintWriter printWriter, int binStartPosition,
                                              boolean isIntraChromosomal, int[] regionBinIndices,
                                              NormalizationType norm, MatrixType matrixType) {

        if (!MatrixType.isObservedOrControl(matrixType)) {
            System.out.println("This feature is only available for Observed or Control views");
            return;
        }

        int binCounter = 0;

        // Get the block index keys, and sort
        List<Integer> blocksToIterateOver = getBlockNumbersForRegionFromBinPosition(regionBinIndices);
        Collections.sort(blocksToIterateOver);

        for (Integer blockNumber : blocksToIterateOver) {
            Block b = null;
            try {
                b = reader.readNormalizedBlock(blockNumber, MatrixZoomData.this, norm);
            } catch (Exception e) {
                System.err.println("Skipping block " + blockNumber);
            }
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
    public Iterator<ContactRecord> getNewContactRecordIterator() {
        return new ContactRecordIterator();
    }

    public List<ContactRecord> getContactRecordList() {
        List<ContactRecord> records = new ArrayList<>();
        Iterator<ContactRecord> iterator = getNewContactRecordIterator();
        while (iterator.hasNext()) {
            ContactRecord cr = iterator.next();
            records.add(cr);
        }
        return records;
    }

    public void clearCache() {
        blockCache.clear();
    }


    /**
     * Class for iterating over the contact records
     */
    class ContactRecordIterator implements Iterator<ContactRecord> {

        final List<Integer> blockNumbers;
        int blockIdx;
        Iterator<ContactRecord> currentBlockIterator;

        /**
         * Initializes the iterator
         */
        ContactRecordIterator() {
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
                        // TODO why is this always NONE, should trace to ensure hard coding doesn't cause bug?
                        String key = getBlockKey(blockNumber, NormalizationHandler.NONE);
                        Block nextBlock;
                        if (HiCGlobals.useCache && blockCache.containsKey(key)) {
                            nextBlock = blockCache.get(key);
                        } else {
                            nextBlock = reader.readNormalizedBlock(blockNumber, MatrixZoomData.this, NormalizationHandler.NONE);
                        }
                        currentBlockIterator = nextBlock.getContactRecords().iterator();
                        return true;
                    } catch (IOException e) {
                        System.err.println("Error fetching block " + e.getMessage());
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
}
