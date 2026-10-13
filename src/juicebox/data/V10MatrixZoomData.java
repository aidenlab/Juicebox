package juicebox.data;

import juicebox.HiC;
import juicebox.data.basics.Chromosome;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import juicebox.track.HiCGridAxis;
import juicebox.track.HiCFixedGridAxis;
import java.io.IOException;
import java.math.BigInteger;
import java.util.*;

/** Desktop geometry and records backed by the common Java Straw v10 decoder. */
public class V10MatrixZoomData extends MatrixZoomData {
    final javastraw.reader.mzd.V10MatrixZoomData delegate;
    private final DatasetReaderV10 owner;
    private final int overviewStride;
    private final long overviewBins;

    public V10MatrixZoomData(Chromosome first, Chromosome second, HiCZoom zoom,
                            javastraw.reader.mzd.V10MatrixZoomData delegate, DatasetReaderV10 reader) {
        super(first, second, zoom, delegate.getZoomDescriptor().blockBinCount,
              delegate.getZoomDescriptor().blockColumnCount,
              zoom.getUnit() == HiC.Unit.FRAG ? new int[0] : null,
              zoom.getUnit() == HiC.Unit.FRAG ? new int[0] : null, reader);
        this.delegate = delegate;
        this.owner = reader;
        javastraw.reader.v10.V10AllOverview all = reader.readOverview();
        boolean overview = all != null && first.getIndex() == all.chromosomeIndex && second.getIndex() == first.getIndex();
        overviewStride = overview ? Math.toIntExact(all.storageStride) : 1;
        overviewBins = overview ? all.binCount : 0;
        setAverageCount(overview ? delegate.getZoomDescriptor().getSum() / all.binCount / all.binCount : delegate.getAverageCount());
    }
    @Override public HiCGridAxis getXGridAxis() {
        return getZoom().getUnit() == HiC.Unit.BP
                ? new HiCFixedGridAxis((overviewBins > 0 ? overviewBins : delegate.getNumberOfBinsX()), getBinSize(), null) : new FragmentAxis(owner.fragmentSites(getChr1Idx()), getChr1().getLength(), getBinSize());
    }
    @Override public HiCGridAxis getYGridAxis() {
        return getZoom().getUnit() == HiC.Unit.BP
                ? new HiCFixedGridAxis((overviewBins > 0 ? overviewBins : delegate.getNumberOfBinsY()), getBinSize(), null) : new FragmentAxis(owner.fragmentSites(getChr2Idx()), getChr2().getLength(), getBinSize());
    }
    public double getSumCount() { return delegate.getZoomDescriptor().getSum(); }
    @Override public List<Block> getNormalizedBlocksOverlapping(long x1, long y1, long x2, long y2,
            NormalizationType normalization, boolean important, boolean fillUnderDiagonal) {
        List<Block> result = new ArrayList<>();
        for (javastraw.reader.block.Block block : delegate.getNormalizedBlocksOverlapping(
                x1 * overviewStride, y1 * overviewStride,
                (x2 + 1) * overviewStride - 1, (y2 + 1) * overviewStride - 1, DatasetReaderV10.norm(normalization), fillUnderDiagonal)) {
            result.add(convert(block));
        }
        return result;
    }
    private Block convert(javastraw.reader.block.Block block) {
        List<ContactRecord> records = new ArrayList<>(block.getContactRecords().size());
        for (javastraw.reader.block.ContactRecord record : block.getContactRecords()) records.add(displayRecord(record));
        return new Block(block.getNumber(), records, block.getUniqueRegionID());
    }
    private ContactRecord displayRecord(javastraw.reader.block.ContactRecord record) {
        if (overviewStride == 1) return convert(record);
        int x = record.getBinX() / overviewStride, y = record.getBinY() / overviewStride;
        BigInteger exact = record instanceof javastraw.reader.block.V10ContactRecord ? record.getExactCount() : null;
        return exact == null ? new ContactRecord(x, y, record.getCounts()) : new ExactRecord(x, y, exact);
    }
    static ContactRecord convert(javastraw.reader.block.ContactRecord record) {
        BigInteger exact = record instanceof javastraw.reader.block.V10ContactRecord ? record.getExactCount() : null;
        return exact == null ? new ContactRecord(record.getBinX(), record.getBinY(), record.getCounts())
                : new ExactRecord(record.getBinX(), record.getBinY(), exact);
    }
    public Block readBlock(int number, NormalizationType type) throws IOException {
        javastraw.reader.block.Block block = delegate.readNormalizedBlock(number, DatasetReaderV10.norm(type));
        return block == null ? null : convert(block);
    }
    List<Integer> blockNumbers() throws IOException { return delegate.getPhysicalBlockNumbers(); }
    public Iterator<ContactRecord> directIterator() {
        final Iterator<javastraw.reader.block.ContactRecord> original = delegate.getDirectIterator();
        return new Iterator<ContactRecord>() {
            public boolean hasNext() { return original.hasNext(); }
            public ContactRecord next() { return displayRecord(original.next()); }
            public void remove() { throw new UnsupportedOperationException(); }
        };
    }
    @Override public juicebox.data.iterator.IteratorContainer getIteratorContainer() { return getFromFileIteratorContainer(); }
    @Override public juicebox.data.iterator.IteratorContainer getFromFileIteratorContainer() {
        return new juicebox.data.iterator.IteratorContainer(overviewBins > 0 ? overviewBins : delegate.getNumberOfBinsX()) {
            public Iterator<ContactRecord> getNewContactRecordIterator() { return directIterator(); }
            public juicebox.data.basics.ListOfFloatArrays sparseMultiply(juicebox.data.basics.ListOfFloatArrays vector, long length) {
                return juicebox.data.iterator.ZDIteratorContainer.matrixVectorMultiplyOnIterator(directIterator(), vector, length);
            }
            public void clear() { owner.close(); }
        };
    }
    @Override public void dump(java.io.PrintWriter writer, htsjdk.tribble.util.LittleEndianOutputStream binary,
            NormalizationType normalization, juicebox.windowui.MatrixType type, boolean regional,
            long[] region, ExpectedValueFunction expected, boolean dense) throws IOException {

        if (writer == null && binary == null) writer = new java.io.PrintWriter(System.out);
        long x1 = 0, y1 = 0, x2 = getXGridAxis().getBinCount() - 1, y2 = getYGridAxis().getBinCount() - 1;
        if (regional) { x1 = region[0] / getBinSize(); x2 = region[1] / getBinSize(); y1 = region[2] / getBinSize(); y2 = region[3] / getBinSize(); }
        Iterator<javastraw.reader.block.ContactRecord> records;
        if (regional) {
            List<javastraw.reader.block.ContactRecord> selected = new ArrayList<>();
            Set<Long> seen = new HashSet<>();
            for (javastraw.reader.block.Block block : delegate.getNormalizedBlocksOverlapping(x1 * overviewStride, y1 * overviewStride, (x2 + 1) * overviewStride - 1, (y2 + 1) * overviewStride - 1, DatasetReaderV10.norm(normalization), false)) {
                for (javastraw.reader.block.ContactRecord record : block.getContactRecords()) {
                    int x = record.getBinX() / overviewStride, y = record.getBinY() / overviewStride;
                    boolean inside = x >= x1 && x <= x2 && y >= y1 && y <= y2;
                    if (getChr1Idx() == getChr2Idx()) inside |= y >= x1 && y <= x2 && x >= y1 && x <= y2;
                    long key = ((long)y << 32) | (x & 0xffffffffL);
                    if (inside && seen.add(key)) selected.add(record);
                }
            }
            records = selected.iterator();
        } else records = delegate.getNormalizedIterator(DatasetReaderV10.norm(normalization));
        Map<Long, Float> denseValues = dense ? new HashMap<Long, Float>() : null;
        while (records.hasNext()) {
            javastraw.reader.block.ContactRecord record = records.next();
            double value = record.getCounts();
            if (type == juicebox.windowui.MatrixType.OE) {
                double denominator = getChr1Idx() == getChr2Idx()
                    ? expected == null ? Double.NaN : expected.getExpectedValue(getChr1Idx(), Math.abs(record.getBinX() - record.getBinY()))
                    : delegate.getExpectedValueForTrans();
                if (!Double.isFinite(denominator) || denominator == 0) continue;
                value /= denominator;
            } else if (type != juicebox.windowui.MatrixType.OBSERVED) throw new IOException("Unsupported v10 sparse export mode: " + type);
            long x = (long) (record.getBinX() / overviewStride) * getBinSize(), y = (long) (record.getBinY() / overviewStride) * getBinSize();
            if (dense) {
                denseValues.put(((long) (record.getBinY() / overviewStride) << 32) | ((record.getBinX() / overviewStride) & 0xffffffffL), (float)value);
                if (getChr1Idx() == getChr2Idx()) denseValues.put(((long)(record.getBinX() / overviewStride) << 32) | ((record.getBinY() / overviewStride) & 0xffffffffL), (float)value);
                continue;
            }
            if (binary != null) { binary.writeInt(Math.toIntExact(x)); binary.writeInt(Math.toIntExact(y)); binary.writeFloat((float)value); }
            else {
                String output = type == juicebox.windowui.MatrixType.OBSERVED && record instanceof javastraw.reader.block.V10ContactRecord
                    ? record.getExactCount().toString() : Double.toString(value);
                writer.println(x + "\t" + y + "\t" + output);
            }
        }
        if (dense) {
            x2 = Math.min(x2, delegate.getNumberOfBinsX() - 1);
            y2 = Math.min(y2, delegate.getNumberOfBinsY() - 1);
            for (long x = x1; x <= x2; x++) {
                for (long y = y1; y <= y2; y++) {
                    float value = denseValues.getOrDefault((y << 32) | (x & 0xffffffffL), 0f);
                    if (binary != null) binary.writeFloat(value); else writer.print(value + "\t");
                }
                if (writer != null) writer.println();
            }
        }
        if (writer != null) writer.flush();
    }

    @Override public double getExpectedValueForTrans() { return delegate.getExpectedValueForTrans(); }

    static final class FragmentAxis implements HiCGridAxis {
        private final long[] sites;
        private final long length;
        private final int binSize;
        FragmentAxis(long[] sites, long length, int binSize) { this.sites = sites; this.length = length; this.binSize = binSize; }
        public long getGenomicStart(double bin) {
            long first = Math.max(0, (long) Math.floor(bin)) * binSize;
            return first == 0 ? 0 : first > sites.length ? length : sites[(int) first - 1];
        }
        public long getGenomicEnd(double bin) {
            long next = (Math.max(0, (long) Math.floor(bin)) + 1) * binSize;
            return next > sites.length ? length : sites[(int) next - 1];
        }
        public long getGenomicMid(double bin) { long start = getGenomicStart(bin); return start + (getGenomicEnd(bin) - start) / 2; }
        public int getIGVZoom() { return (int) Math.max(0, Math.log(Math.max(1, getBinCount() / 700.0)) / Math.log(2)); }
        public long getBinCount() { return (sites.length + 1L + binSize - 1) / binSize; }
        public int getBinSize() { return binSize; }
        public int getBinNumberForFragment(int fragment) { return fragment / binSize; }
        public int getBinNumberForGenomicPosition(long position) {
            int lo = 0, hi = sites.length;
            while (lo < hi) { int mid = (lo + hi) >>> 1; if (sites[mid] <= position) lo = mid + 1; else hi = mid; }
            return lo / binSize;
        }
    }

    static final class ExactRecord extends ContactRecord {
        private BigInteger exact;
        ExactRecord(int x, int y, BigInteger count) { super(x, y, count.floatValue()); exact = count; }
        @Override public BigInteger getExactCount() { return exact; }
        @Override public void incrementCount(float score) { super.incrementCount(score); exact = null; }
    }
}
