package juicebox.data;

import juicebox.HiCGlobals;
import org.apache.commons.math.linear.*;
import org.apache.log4j.Logger;
import org.broad.igv.feature.Chromosome;
import juicebox.HiC;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import juicebox.matrix.*;
import juicebox.track.HiCFixedGridAxis;
import juicebox.track.HiCFragmentAxis;
import juicebox.track.HiCGridAxis;
import org.broad.igv.ui.util.MessageUtils;
import org.broad.igv.util.collections.LRUCache;
import htsjdk.tribble.util.LittleEndianOutputStream;


//import javax.swing.*;
//import java.util.List;
import java.io.*;
import java.util.*;


/**
 * @author jrobinso
 * @date Aug 10, 2010
 */
public class MatrixZoomData {

    private static final Logger log = Logger.getLogger(MatrixZoomData.class);

    DatasetReader reader;

    private final Chromosome chr1;  // Chromosome on the X axis
    private final Chromosome chr2;  // Chromosome on the Y axis
    private final HiCZoom zoom;    // Unit and bin size

    private final HiCGridAxis xGridAxis;
    private final HiCGridAxis yGridAxis;

    // Observed values are ogranized into sub-matrices ("blocks")
    private final int blockBinCount;   // block size in bins
    private final int blockColumnCount;     // number of block columns

    private final HashMap<NormalizationType, BasicMatrix> pearsonsMap;
    private final HashSet<NormalizationType> missingPearsonFiles;
    private double averageCount = -1;


    // Cache the last 20 blocks loaded
    private final LRUCache<String, Block> blockCache = new LRUCache<String, Block>(20);


//    float sumCounts;
//    float avgCounts;
//    float stdDev;
//    float percent95 = -1;
//    float percent80 = -1;


    /**
     * @param chr1
     * @param chr2
     * @return
     * @throws IOException
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

        int[] xSites = chr1Sites;
        int[] ySites = chr2Sites;
        if (zoom.getUnit() == HiC.Unit.BP) {
            this.xGridAxis = new HiCFixedGridAxis(correctedBinCount * blockColumnCount, zoom.getBinSize(), xSites);
            this.yGridAxis = new HiCFixedGridAxis(correctedBinCount * blockColumnCount, zoom.getBinSize(), ySites);
        } else {
            this.xGridAxis = new HiCFragmentAxis(zoom.getBinSize(), xSites, chr1.getLength());
            this.yGridAxis = new HiCFragmentAxis(zoom.getBinSize(), ySites, chr2.getLength());

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

    public int getBlockColumnCount() {
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
     * @param no
     * @return
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

        final List<String> errorStrings = new ArrayList<String>();

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
                        e.printStackTrace();
                        MessageUtils.showMessage(e.getMessage());
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

        return blockList;
    }


    /**
     * Return the observed value at the specified location.   Supports tooltip text
     * This implementation is naive, but might get away with it for tooltip.
     *
     * @param binX
     * @param binY
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

   /* public boolean isPearsonAvailable(NormalizationType option) {


    }*/


    public double[] computeEigenvector(ExpectedValueFunction df, int which) {
        BasicMatrix pearsons = getPearsons(df);
        if (pearsons == null) {
            return null;
        }

        int dim = pearsons.getRowDimension();
        double[][] data = new double[dim][dim];
        BitSet bitSet = new BitSet(dim);
        for (int i=0; i<dim; i++) {
            for (int j=0; j<dim; j++) {
                float tmp = pearsons.getEntry(i,j);
                data[i][j] = tmp;
                if (data[i][j] != 0 && !Float.isNaN(tmp)) {
                    bitSet.set(i);
                }
            }
        }

        int[] nonCentromereColumns = new int[bitSet.cardinality()];
        int count=0;
        for (int i=0; i<dim; i++) {
            if (bitSet.get(i)) nonCentromereColumns[count++]=i;
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

    public BasicMatrix getPearsons(ExpectedValueFunction df) {

        BasicMatrix pearsons = pearsonsMap.get(df.getNormalizationType());
        if (pearsons == null && !missingPearsonFiles.contains(df.getNormalizationType())) {
            try {
                pearsons = reader.readPearsons(chr1.getName(), chr2.getName(), zoom, df.getNormalizationType());
            } catch (IOException e) {
                log.error(e.getMessage());
            }
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


    public float getPearsonValue(int binX, int binY, NormalizationType type) {
        BasicMatrix pearsons = pearsonsMap.get(type);
        if (pearsons != null) {
            return pearsons.getEntry(binX, binY);
        } else {
            return 0;
        }
    }

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


//
//        // Dump OE subtracted
//        try {
//            PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("OE_new_subtracted.tab")));
//            for (int i = 0; i < dim; i++) {
//                double[] row = vectors[i];
//                for (int j = 0; j < dim; j++) {
//                    double value = row == null ? 0 : row[j];
//                    pw.println(i + "\t" + j + "\t" + value);
//                }
//            }
//            pw.close();
//
//            //   ScratchPad.dumpMatrix(pearsons, "Pearsons_sparse_rowmean.tab");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        BasicMatrix pearsons = Pearsons.computePearsons(vectors, dim);

//        // Dump Pearsons
//        try {
//            PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("Pearsons_new.tab")));
//            for (int i = 0; i < dim; i++) {
//                for (int j = 0; j < dim; j++) {
//                    pw.println(i + "\t" + j + "\t" + pearsons.getEntry(i, j));
//                }
//            }
//            pw.close();
//
//            //   ScratchPad.dumpMatrix(pearsons, "Pearsons_sparse_rowmean.tab");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        pearsonsMap.put(df.getNormalizationType(), pearsons);


        return pearsons;
    }

    private boolean isZeros(double[] array) {
        for (double anArray : array)
            if (anArray != 0 && !Double.isNaN(anArray))
                return false;
        return true;
    }

    private double getVectorMean(RealVector vector) {
        double sum = 0;
        int count = 0;
        int size = vector.getDimension();
        for (int i = 0; i < size; i++) {
            if (!Double.isNaN(vector.getEntry(i))) {
                sum += vector.getEntry(i);
                count++;
            }
        }
        return sum / count;
    }

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
     * Dump observed matrix to text
     *
     * @param printWriter Text output stream
     */
    public void dump(PrintWriter printWriter, double[] nv1, double[] nv2) throws IOException {
        // Get the block index keys, and sort
        List<Integer> blockNumbers = reader.getBlockNumbers(this);
        if (blockNumbers != null) {
            Collections.sort(blockNumbers);

            for (int blockNumber : blockNumbers) {
                Block b = reader.readBlock(blockNumber, this);
                if (b != null) {
                    for (ContactRecord rec : b.getContactRecords()) {
                        float counts = rec.getCounts();
                        int x = rec.getBinX();
                        int y = rec.getBinY();
                        if (nv1 != null && nv2 != null) {
                            if (nv1[x] != 0 && nv2[y] != 0 && !Double.isNaN(nv1[x]) && !Double.isNaN(nv2[y])) {
                                counts = (float) (counts / (nv1[x] * nv2[y]));
                            } else {
                                counts = Float.NaN;
                            }
                        }
                        printWriter.println(x * zoom.getBinSize() + "\t" + y * zoom.getBinSize() + "\t" + counts);
                    }
                }
            }
        }
        printWriter.close();
    }

    /**
     * Dump observed matrix to binary.
     *
     * @param les Binary output stream
     * @throws IOException
     */
    public void dump(LittleEndianOutputStream les, double[] nv1, double[] nv2) throws IOException {

        // Get the block index keys, and sort
        List<Integer> blockNumbers = reader.getBlockNumbers(this);
        if (blockNumbers != null) {
            Collections.sort(blockNumbers);

            for (int blockNumber : blockNumbers) {
                Block b = reader.readBlock(blockNumber, this);
                if (b != null) {
                    for (ContactRecord rec : b.getContactRecords()) {
                        float counts = rec.getCounts();
                        int x = rec.getBinX();
                        int y = rec.getBinY();
                        if (nv1 != null && nv2 != null) {
                            if (nv1[x] != 0 && nv2[y] != 0 && !Double.isNaN(nv1[x]) && !Double.isNaN(nv2[y])) {
                                counts = (float) (counts / (nv1[x] * nv2[y]));
                            } else {
                                counts = Float.NaN;
                            }
                        }
                        les.writeInt(x);
                        les.writeInt(y);
                        les.writeFloat(counts);
                    }

                }
            }
        }
    }

    /**
     * Dump the O/E or Pearsons matrix to standard out in ascii format.
     *
     * @param df   Density function (expected values)
     * @param type will be "oe", "pearsons", or "expected"
     * @param les  output stream
     */
    public void dumpOE(ExpectedValueFunction df, String type, NormalizationType no, LittleEndianOutputStream les, PrintWriter pw) throws IOException {
        if (les == null && pw == null) {
            pw = new PrintWriter(System.out);
        }

        if (type.equals("oe")) {
            int nBins;

            if (zoom.getUnit() == HiC.Unit.BP) {
                nBins = chr1.getLength() / zoom.getBinSize() + 1;
            } else {
                nBins = ((DatasetReaderV2) reader).getFragCount(chr1) / zoom.getBinSize() + 1;
            }

            BasicMatrix matrix = new InMemoryMatrix(nBins);
            BitSet bitSet = new BitSet(nBins);

            List<Integer> blockNumbers = reader.getBlockNumbers(this);

            for (int blockNumber : blockNumbers) {
                Block b = null;
                try {
                    b = reader.readNormalizedBlock(blockNumber, this, df.getNormalizationType());
                    if (b != null) {
                        for (ContactRecord rec : b.getContactRecords()) {
                            int x = rec.getBinX();
                            int y = rec.getBinY();

                            int dist = Math.abs(x - y);
                            double expected = 0;
                            try {
                                expected = df.getExpectedValue(chr1.getIndex(), dist);
                            } catch (Exception e) {
                                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
                            }
                            double observed = rec.getCounts(); // Observed is already normalized
                            double normCounts = observed / expected;
                            // The apache library doesn't seem to play nice with NaNs
                            if (!Double.isNaN(normCounts)) {
                                matrix.setEntry(x, y, (float)normCounts);
                                if (x != y) {
                                    matrix.setEntry(y, x, (float)normCounts);
                                }
                                bitSet.set(x);
                                bitSet.set(y);
                            }
                        }
                    }
                } catch (IOException e) {
                    e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
                }
            }

            if (les != null) les.writeInt(nBins);

            for (int i = 0; i < nBins; i++) {
                for (int j=0; j < nBins; j++) {
                    float output;
                    if (!bitSet.get(i) && !bitSet.get(j)) {
                        output = Float.NaN;
                    }
                    else output = matrix.getEntry(i,j);
                    if (les != null) les.writeFloat(output);
                    else pw.print(output + " ");
                }
                if (les == null) pw.println();
            }
            if (les == null) {
                pw.println();
                pw.flush();
            }
        } else {
            BasicMatrix pearsons = getPearsons(df);
            if (pearsons != null) {
                int dim = pearsons.getRowDimension();
                for (int i=0; i<dim; i++) {
                    for (int j=0; j<dim; j++) {
                        float output = pearsons.getEntry(i,j);
                        if (les != null) les.writeFloat(output);
                        else pw.print(output + " ");
                    }
                    if (les == null) pw.println();
                }
                pw.flush();
            }
            else {
                log.error("Pearson's not available at zoom " + zoom);
            }
        }
    }

    public void setAverageCount(double averageCount) {
        this.averageCount = averageCount;
    }

    public double getAverageCount() {
        return averageCount;
    }

    public Iterator<ContactRecord> contactRecordIterator() {
        return new ContactRecordIterator();
    }

    public class ContactRecordIterator implements Iterator<ContactRecord> {

        int blockIdx;
        final List<Integer> blockNumbers;
        Iterator<ContactRecord> currentBlockIterator;

        public ContactRecordIterator() {
            this.blockIdx = -1;
            this.blockNumbers = reader.getBlockNumbers(MatrixZoomData.this);
        }

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

        @Override
        public ContactRecord next() {
            return currentBlockIterator == null ? null : currentBlockIterator.next();
        }

        @Override
        public void remove() {
            //Not supported
            throw new RuntimeException("remove() is not supported");
        }
    }

    private class GradientXFilter extends DefaultRealMatrixChangingVisitor {
        double previousValue = Double.MAX_VALUE;

        public double visit(int row, int column, double value) throws org.apache.commons.math.linear.MatrixVisitorException {
            double newValue;
            if (previousValue != Double.MAX_VALUE) {
                newValue = (previousValue * -1 + value) / 2;
            } else newValue = value;
            previousValue = value;
            return newValue;
        }

    }
    /*     // Actually, this isn't smart, Gaussian filter
    private class GaussianFilter extends DefaultRealMatrixChangingVisitor {
        private double[][] filter = new double[5][5];

        public GaussianFilter() {
            super();
            filter[0][0] = 0.0232; filter[0][1] = 0.0338; filter[0][2] = 0.0383; filter[0][3] = 0.0338; filter[0][4] = 0.0232;
            filter[1][0] = 0.0338; filter[1][1] = 0.0492; filter[1][2] = 0.0558; filter[1][3] = 0.0492; filter[0][0] = 0.0338;
            filter[2][0] = 0.0383; filter[2][1] = 0.0558; filter[2][2] = 0.0632; filter[2][3] = 0.0558; filter[0][0] = 0.0383;
            filter[3][0] = 0.0338; filter[3][1] = 0.0492; filter[3][2] = 0.0558; filter[3][3] = 0.0492; filter[0][0] = 0.0338;
            filter[4][0] = 0.0232; filter[4][1] = 0.0338; filter[4][2] = 0.0383; filter[4][3] = 0.0338; filter[0][0] = 0.0232;
        public double visit(int row, int column, double value) throws org.apache.commons.math.linear.MatrixVisitorException {


        }

    }    */

}
