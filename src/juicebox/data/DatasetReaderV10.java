package juicebox.data;

import juicebox.HiC;
import juicebox.data.basics.Chromosome;
import juicebox.data.basics.ListOfDoubleArrays;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import java.awt.event.ActionListener;
import javax.swing.JCheckBox;
import java.io.IOException;
import java.util.*;

/** Adapts Java Straw's v10 reader to the desktop data model. */
public class DatasetReaderV10 extends AbstractDatasetReader {
    private final javastraw.reader.DatasetReaderV10 delegate;
    private Dataset dataset;
    private final Map<String, Matrix> matrices = new HashMap<>();
    private boolean active = true;

    public DatasetReaderV10(String path) {
        super(path);
        delegate = new javastraw.reader.DatasetReaderV10(path, true);
    }

    @Override public int getVersion() { return 10; }
    @Override public boolean isActive() { return active; }
    @Override public void setActive(boolean value) { active = value; }
    public int getDepthBase() { return 2; }

    @Override public synchronized Dataset read() throws IOException {
        if (dataset != null) return dataset;
        javastraw.reader.Dataset original = delegate.read();
        Dataset result = new Dataset(this);
        List<Chromosome> chromosomes = new ArrayList<>();
        for (javastraw.reader.basics.Chromosome chr : delegate.getHeader().chromosomes) {
            chromosomes.add(new Chromosome(chr.getIndex(), chr.getName(), chr.getLength()));
        }
        result.setChromosomeHandler(new ChromosomeHandler(chromosomes, original.getGenomeId(), false, false, true));
        result.getChromosomeHandler().setAllOverview(delegate.getHeader().allOverview);
        result.setGenomeId(original.getGenomeId());
        result.setAttributes(new LinkedHashMap<>(original.getAttributes()));
        result.setBpZooms(descending(delegate.getHeader().binSizes(0)));
        result.setFragZooms(descending(delegate.getHeader().binSizes(1)));
        result.setFragmentCounts(original.getFragmentCounts());
        List<NormalizationType> normalizations = new ArrayList<>();
        for (javastraw.reader.type.NormalizationType norm : original.getNormalizationTypes()) {
            normalizations.add(new NormalizationType(norm.getLabel(), norm.getDescription()));
        }
        result.setNormalizationTypes(normalizations);
        Map<String, ExpectedValueFunction> expected = new HashMap<>();
        for (javastraw.reader.expected.ExpectedValueFunction function : original.getExpectedValueFunctionMap().values()) {
            ExpectedValueFunction adapter = new ExpectedAdapter(function);
            expected.put(new HiCZoom(adapter.getUnit(), adapter.getBinSize()).getKey() + "_" + adapter.getNormalizationType(), adapter);
        }
        result.setExpectedValueFunctionMap(expected);
        dataset = result;
        return result;
    }

    private static int[] descending(int[] values) {
        int[] sorted = values.clone();
        for (int i = 0; i < sorted.length / 2; i++) {
            int j = sorted.length - i - 1;
            int value = sorted[i]; sorted[i] = sorted[j]; sorted[j] = value;
        }
        return sorted;
    }

    static javastraw.reader.type.HiCZoom.HiCUnit unit(HiC.Unit unit) {
        return unit == HiC.Unit.FRAG ? javastraw.reader.type.HiCZoom.HiCUnit.FRAG : javastraw.reader.type.HiCZoom.HiCUnit.BP;
    }
    static javastraw.reader.type.NormalizationType norm(NormalizationType type) {
        return type == null ? javastraw.reader.type.NormalizationHandler.NONE :
                new javastraw.reader.type.NormalizationType(type.getLabel(), type.getDescription());
    }
    static ListOfDoubleArrays values(javastraw.reader.datastructures.ListOfDoubleArrays source) {
        if (source == null) return null;
        ListOfDoubleArrays result = new ListOfDoubleArrays(source.getLength());
        for (long i = 0; i < source.getLength(); i++) result.set(i, source.get(i));
        return result;
    }
    long[] fragmentSites(int chromosome) {
        return delegate.getHeader().fragmentSites[chromosome];
    }

    @Override public synchronized Matrix readMatrix(String key) throws IOException {
        read();
        if (matrices.containsKey(key)) return matrices.get(key);
        javastraw.reader.mzd.Matrix original = delegate.readMatrix(key, -1);
        if (original == null) {
            String[] pair = key.split("_");
            int firstIndex = Integer.parseInt(pair[0]), secondIndex = Integer.parseInt(pair[1]);
            Chromosome first = dataset.getChromosomeHandler().getChromosomeFromIndex(firstIndex);
            Chromosome second = dataset.getChromosomeHandler().getChromosomeFromIndex(secondIndex);
            List<MatrixZoomData> emptyZooms = new ArrayList<>();
            for (HiCZoom zoom : dataset.getBpZooms()) emptyZooms.add(new EmptyZoom(first, second, zoom, this));
            for (HiCZoom zoom : dataset.getFragZooms()) emptyZooms.add(new EmptyZoom(first, second, zoom, this));
            Matrix empty = new Matrix(firstIndex, secondIndex, emptyZooms);
            matrices.put(key, empty);
            return empty;
        }
        int a = original.getChr1Index(), b = original.getChr2Index();
        List<MatrixZoomData> zooms = new ArrayList<>();
        int legacyAllBin = -1;
        Chromosome first = dataset.getChromosomeHandler().getChromosomeFromIndex(a);
        if (delegate.getHeader().allOverview == null && a == b && ChromosomeHandler.isAllByAll(first)) {
            double target = first.getLength() / 500.0, distance = Double.POSITIVE_INFINITY;
            for (HiCZoom zoom : dataset.getBpZooms()) {
                javastraw.reader.mzd.V10MatrixZoomData candidate = (javastraw.reader.mzd.V10MatrixZoomData)
                    original.getZoomData(new javastraw.reader.type.HiCZoom(unit(HiC.Unit.BP), zoom.getBinSize()));
                if (candidate != null && candidate.getOccupiedCellCount() > 0 && Math.abs(zoom.getBinSize() - target) < distance) {
                    legacyAllBin = zoom.getBinSize(); distance = Math.abs(legacyAllBin - target);
                }
            }
        }
        for (HiC.Unit unit : HiC.Unit.values()) {
            List<HiCZoom> resolutions = unit == HiC.Unit.BP ? dataset.getBpZooms() : dataset.getFragZooms();
            for (HiCZoom zoom : resolutions) {
                if (legacyAllBin > 0 && (unit != HiC.Unit.BP || zoom.getBinSize() != legacyAllBin)) continue;
                javastraw.reader.v10.V10AllOverview all = delegate.getHeader().allOverview;
                if (all != null && a == all.chromosomeIndex && b == a &&
                        (unit != HiC.Unit.BP || zoom.getBinSize() != all.finestResolution)) continue;
                javastraw.reader.mzd.MatrixZoomData zd = original.getZoomData(new javastraw.reader.type.HiCZoom(unit(unit), zoom.getBinSize()));
                if (zd != null) zooms.add(new V10MatrixZoomData(
                    dataset.getChromosomeHandler().getChromosomeFromIndex(a),
                    dataset.getChromosomeHandler().getChromosomeFromIndex(b),
                    all != null && a == all.chromosomeIndex && b == a
                        ? new HiCZoom(HiC.Unit.BP, all.displayResolution) : zoom,
                    (javastraw.reader.mzd.V10MatrixZoomData) zd, this));
            }
        }
        Matrix matrix = new Matrix(a, b, zooms);
        matrices.put(key, matrix);
        return matrix;
    }

    javastraw.reader.v10.V10AllOverview readOverview() { return delegate.getHeader().allOverview; }
    private V10MatrixZoomData zoom(MatrixZoomData zd) throws IOException {
        if (zd instanceof EmptyZoom) return null;
        if (zd instanceof V10MatrixZoomData) return (V10MatrixZoomData) zd;
        Matrix matrix = readMatrix(Math.min(zd.getChr1Idx(), zd.getChr2Idx()) + "_" + Math.max(zd.getChr1Idx(), zd.getChr2Idx()));
        return matrix == null ? null : (V10MatrixZoomData) matrix.getZoomData(zd.getZoom());
    }
    @Override public Block readNormalizedBlock(int number, MatrixZoomData zd, NormalizationType normalization) throws IOException {
        V10MatrixZoomData target = zoom(zd);
        return target == null ? null : target.readBlock(number, normalization);
    }
    @Override public List<Integer> getBlockNumbers(MatrixZoomData zd) {
        try {
            V10MatrixZoomData target = zoom(zd);
            return target == null ? Collections.<Integer>emptyList() : target.blockNumbers();
        } catch (IOException e) { throw new IllegalStateException("Could not read v10 block index", e); }
    }
    public Integer getBlockSize(MatrixZoomData zd, int number) {
        try {
            V10MatrixZoomData target = zoom(zd);
            return target == null ? null : target.delegate.getBlockSize(number);
        } catch (IOException e) { throw new IllegalStateException("Could not read v10 block index", e); }
    }
    @Override public NormalizationVector readNormalizationVector(NormalizationType type, int chr, HiC.Unit unit, int size) throws IOException {
        read();
        javastraw.reader.norm.NormalizationVector vector = delegate.readNormalizationVector(norm(type), chr, unit(unit), size);
        return vector == null ? null : new NormalizationVector(type, chr, unit, size, values(vector.getData()));
    }
    @Override public NormalizationVector readNormalizationVectorPart(NormalizationType type, int chr, HiC.Unit unit, int size, int first, int last) throws IOException {
        read();
        javastraw.reader.norm.NormalizationVector vector = delegate.readNormalizationVectorPart(norm(type), chr, unit(unit), size, first, last);
        return vector == null ? null : new NormalizationVector(type, chr, unit, size, values(vector.getData()));
    }
    @Override public NormalizationVector getNormalizationVector(int chr, HiCZoom zoom, NormalizationType type) {
        try { read(); return dataset.getNormalizationVector(chr, zoom, type); }
        catch (IOException e) { throw new IllegalStateException("Could not read v10 normalization", e); }
    }
    public ListOfDoubleArrays readExpectedVectorPart(long position, long count) throws IOException {
        throw new IOException("V10 expected values use chunk indexes; use Dataset.getExpectedValues");
    }
    @Override public List<JCheckBox> getCheckBoxes(List<ActionListener> listeners) {
        JCheckBox box = new JCheckBox(path, active);
        box.addActionListener(event -> setActive(box.isSelected()));
        for (ActionListener listener : listeners) box.addActionListener(listener);
        return Collections.singletonList(box);
    }
    public void close() { delegate.clearBlockCache(); }
    public String readStats() throws IOException { read(); return delegate.getHeader().attributes.get("statistics"); }
    public juicebox.matrix.BasicMatrix readPearsons(String chromosome1, String chromosome2, HiCZoom zoom, NormalizationType type) { return null; }

    /** An omitted v10 pair is an all-zero count matrix. */
    private static class EmptyZoom extends MatrixZoomData {
        private final DatasetReaderV10 owner;
        EmptyZoom(Chromosome first, Chromosome second, HiCZoom zoom, DatasetReaderV10 owner) {
            super(first, second, zoom, 1, 1, new int[0], new int[0], owner);
            this.owner = owner;
            setAverageCount(0);
        }
        @Override public List<Block> getNormalizedBlocksOverlapping(long x1, long y1, long x2, long y2, NormalizationType norm, boolean important, boolean fill) {
            return Collections.emptyList();
        }
        @Override public double getExpectedValueForTrans() { return Double.NaN; }
        @Override public juicebox.track.HiCGridAxis getXGridAxis() { return axis(getChr1()); }
        @Override public juicebox.track.HiCGridAxis getYGridAxis() { return axis(getChr2()); }
        private juicebox.track.HiCGridAxis axis(Chromosome chromosome) {
            if (getZoom().getUnit() == HiC.Unit.FRAG) return new V10MatrixZoomData.FragmentAxis(owner.fragmentSites(chromosome.getIndex()), chromosome.getLength(), getBinSize());
            long bins = chromosome.getLength() / getBinSize() + (chromosome.getLength() % getBinSize() == 0 ? 0 : 1);
            return new juicebox.track.HiCFixedGridAxis(bins, getBinSize(), null);
        }
    }

    private static class ExpectedAdapter implements ExpectedValueFunction {
        private final javastraw.reader.expected.ExpectedValueFunction delegate;
        ExpectedAdapter(javastraw.reader.expected.ExpectedValueFunction delegate) { this.delegate = delegate; }
        @Override public double getExpectedValue(int chr, long distance) { return delegate.getExpectedValue(chr, distance); }
        @Override public long getLength() { return delegate.getLength(); }
        @Override public NormalizationType getNormalizationType() { return new NormalizationType(delegate.getNormalizationType().getLabel(), delegate.getNormalizationType().getDescription()); }
        @Override public HiC.Unit getUnit() { return delegate.getUnit() == javastraw.reader.type.HiCZoom.HiCUnit.FRAG ? HiC.Unit.FRAG : HiC.Unit.BP; }
        @Override public int getBinSize() { return delegate.getBinSize(); }
        @Override public ListOfDoubleArrays getExpectedValuesNoNormalization() { return values(delegate.getExpectedValuesNoNormalization()); }
        @Override public ListOfDoubleArrays getExpectedValuesWithNormalization(int chr) { return values(delegate.getExpectedValuesWithNormalization(chr)); }
    }
}
