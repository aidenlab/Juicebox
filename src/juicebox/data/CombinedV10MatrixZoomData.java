package juicebox.data;

import juicebox.track.HiCGridAxis;
import juicebox.windowui.NormalizationType;
import java.util.*;

/** Sums regional cells across maps, whose physical block layouts may differ. */
class CombinedV10MatrixZoomData extends MatrixZoomData {
    private final List<MatrixZoomData> layers;
    CombinedV10MatrixZoomData(List<MatrixZoomData> layers, DatasetReader reader) {
        super(layers.get(0).getChr1(), layers.get(0).getChr2(), layers.get(0).getZoom(), 1, 1,
              new int[0], new int[0], reader);
        this.layers = layers;
    }
    @Override public HiCGridAxis getXGridAxis() { return layers.get(0).getXGridAxis(); }
    @Override public HiCGridAxis getYGridAxis() { return layers.get(0).getYGridAxis(); }
    @Override public double getExpectedValueForTrans() {
        double mean = getAverageCount();
        return Double.isFinite(mean) && mean != 0 ? mean : Double.NaN;
    }
    @Override public double getAverageCount() {
        double sum = 0;
        for (MatrixZoomData layer : layers) if (layer.reader.isActive()) sum += layer.getAverageCount();
        return sum;
    }
    @Override public List<Block> getNormalizedBlocksOverlapping(long x1, long y1, long x2, long y2,
            NormalizationType normalization, boolean important, boolean fillUnderDiagonal) {
        Map<Long, ContactRecord> sums = new LinkedHashMap<>();
        boolean cis = getChr1Idx() == getChr2Idx();
        for (MatrixZoomData layer : layers) {
            if (!layer.reader.isActive()) continue;
            Set<Long> seen = new HashSet<>();
            for (Block block : layer.getNormalizedBlocksOverlapping(x1, y1, x2, y2, normalization, important, fillUnderDiagonal)) {
                for (ContactRecord record : block.getContactRecords()) {
                    int x = record.getBinX(), y = record.getBinY();
                    if (!((x >= x1 && x <= x2 && y >= y1 && y <= y2) ||
                          (cis && y >= x1 && y <= x2 && x >= y1 && x <= y2))) continue;
                    long key = ((long)y << 32) | (x & 0xffffffffL);
                    if (!seen.add(key)) continue;
                    ContactRecord previous = sums.get(key);
                    if (previous == null) sums.put(key, record.getExactCount() == null
                            ? new ContactRecord(x, y, record.getCounts())
                            : new V10MatrixZoomData.ExactRecord(x, y, record.getExactCount()));
                    else if (previous.getExactCount() != null && record.getExactCount() != null)
                        sums.put(key, new V10MatrixZoomData.ExactRecord(x, y,
                                previous.getExactCount().add(record.getExactCount())));
                    else previous.incrementCount(record.getCounts());
                }
            }
        }
        return Collections.singletonList(new Block(0, new ArrayList<>(sums.values()), getKey()));
    }
    @Override public juicebox.data.iterator.IteratorContainer getIteratorContainer() { return getFromFileIteratorContainer(); }
    @Override public juicebox.data.iterator.IteratorContainer getFromFileIteratorContainer() {
        return new juicebox.data.iterator.IteratorContainer(getXGridAxis().getBinCount()) {
            public Iterator<ContactRecord> getNewContactRecordIterator() {
                final List<Iterator<ContactRecord>> iterators = new ArrayList<>();
                for (MatrixZoomData layer : layers) if (layer.reader.isActive()) iterators.add(layer.getFromFileIteratorContainer().getNewContactRecordIterator());
                return new Iterator<ContactRecord>() {
                    int index;
                    public boolean hasNext() { while (index < iterators.size() && !iterators.get(index).hasNext()) index++; return index < iterators.size(); }
                    public ContactRecord next() { if (!hasNext()) throw new NoSuchElementException(); return iterators.get(index).next(); }
                };
            }
            public juicebox.data.basics.ListOfFloatArrays sparseMultiply(juicebox.data.basics.ListOfFloatArrays vector, long length) {
                return juicebox.data.iterator.ZDIteratorContainer.matrixVectorMultiplyOnIterator(getNewContactRecordIterator(), vector, length);
            }
            public void clear() { }
        };
    }
}
