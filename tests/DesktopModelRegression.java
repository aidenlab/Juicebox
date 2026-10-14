package juicebox.track;

import juicebox.Context;
import juicebox.HiC;
import juicebox.data.*;
import juicebox.data.basics.*;
import juicebox.data.anchor.GenericLocus;
import juicebox.data.censoring.RegionPair;
import juicebox.data.censoring.OneDimTrackCensoring;
import juicebox.gui.SuperAdapter;
import juicebox.mapcolorui.HeatmapPanel;
import juicebox.mapcolorui.ColorScaleHandler;
import juicebox.windowui.*;
import org.broad.igv.feature.LocusScore;
import org.broad.igv.renderer.DataRange;
import org.broad.igv.track.WindowFunction;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.lang.reflect.Method;
import org.broad.igv.util.Pair;
import java.util.*;
import java.util.List;

public class DesktopModelRegression {
    private static void check(boolean value, String message) {
        if (!value) throw new AssertionError(message);
    }

    private static class SourceZoom extends MatrixZoomData {
        boolean huge;
        final List<ContactRecord> records = Arrays.asList(new ContactRecord(0, 0, 1),
                new ContactRecord(0, 1, 3), new ContactRecord(1, 1, 2),
                new ContactRecord(2, 2, 4), new ContactRecord(3, 3, 5));
        SourceZoom(DatasetReader reader, int index) {
            super(new Chromosome(index, "chr1", 10000), new Chromosome(index, "chr1", 10000),
                    new HiCZoom(HiC.Unit.BP, 10), 100, 10, null, null, reader);
        }
        @Override public HiCGridAxis getXGridAxis() {
            return new HiCFixedGridAxis(huge ? 5000000 : 1000, 10, null);
        }
        @Override public HiCGridAxis getYGridAxis() { return getXGridAxis(); }
        @Override public List<Block> getNormalizedBlocksOverlapping(long x0, long y0, long x1, long y1,
                NormalizationType norm, boolean important, boolean fill) {
            List<ContactRecord> selected = new ArrayList<>();
            for (ContactRecord r : records) {
                if (r.getBinX() >= x0 && r.getBinX() <= x1 && r.getBinY() >= y0 && r.getBinY() <= y1) selected.add(r);
            }
            return Collections.singletonList(new Block(0, selected, "source"));
        }
    }

    private static class TestHiC extends HiC {
        final MatrixZoomData source;
        NormalizationType norm = NormalizationHandler.NONE;
        int loads;
        TestHiC(MatrixZoomData source) { super(null); this.source = source; }
        @Override public MatrixZoomData getZd() { return source; }
        @Override public MatrixZoomData getControlZd() { return source; }
        @Override public HiCZoom getZoom() { return source.getZoom(); }
        @Override public double getScaleFactor() { return 10; }
        @Override public NormalizationType getObsNormalizationType() { return norm; }
        @Override public double[] getEigenvector(int chromosome, int n, boolean control) {
            loads++;
            return new double[]{-4, 1, 2, Double.NaN, Double.POSITIVE_INFINITY};
        }
    }

    private static class Score implements LocusScore {
        int start, end;
        final float score;
        Score(int start, int end, float score) { this.start = start; this.end = end; this.score = score; }
        public String getChr() { return "chr1"; }
        public String getContig() { return "chr1"; }
        public int getStart() { return start; }
        public int getEnd() { return end; }
        public void setStart(int start) { this.start = start; }
        public void setEnd(int end) { this.end = end; }
        public float getScore() { return score; }
        public String getValueString(double p, int w, WindowFunction f) { return ""; }
    }

    private static class Adapter extends HiCDataAdapter {
        Adapter(HiC hic) { super(hic); }
        protected List<LocusScore> getLocusScores(String chr, long start, long end, int zoom, WindowFunction f) {
            return Collections.singletonList(new Score(0, 50, -2));
        }
        public String getName() { return "test"; }
        public Color getPosColor() { return Color.RED; }
        public void setColor(Color color) { }
        public Color getNegColor() { return Color.BLUE; }
        public void setNegColor(Color color) { }
        public DataRange getDataRange() { return new DataRange(-5, 5); }
        public void setDataRange(DataRange range) { }
        public Collection<WindowFunction> getAvailableWindowFunctions() { return Collections.singleton(WindowFunction.max); }
    }

    private static class CoverageDataset extends Dataset {
        final double value;
        int requested = -1;
        CoverageDataset(int index, double value) {
            super(null);
            this.value = value;
            setExpectedValueFunctionMap(Collections.emptyMap());
            List<Chromosome> chromosomes = index == 0 ? Collections.singletonList(new Chromosome(0, "chr1", 10000)) :
                    Arrays.asList(new Chromosome(0, "All", 10), new Chromosome(1, "chr1", 10000));
            setChromosomeHandler(new ChromosomeHandler(chromosomes,
                    "test", false, false, true));
        }
        @Override public NormalizationVector getNormalizationVector(int chr, HiCZoom zoom, NormalizationType norm) {
            requested = chr;
            return new NormalizationVector(norm, chr, zoom.getUnit(), zoom.getBinSize(), new ListOfDoubleArrays(10, value));
        }
        @Override public double[] getEigenvector(Chromosome chr, HiCZoom zoom, int number, NormalizationType norm) {
            check(chr.getName().equals("chr1"), "eigenvector request resolves the correct chromosome");
            requested = chr.getIndex();
            return new double[]{value};
        }
    }

    public static void main(String[] args) throws Exception {
        org.broad.igv.DirectoryManager.setIgvDirectory(java.nio.file.Files.createTempDirectory("desktop-model-").toFile());
        DatasetReader reader = DatasetReaderFactory.getReader(Collections.singletonList(args[0]));
        reader.read();
        SourceZoom source = new SourceZoom(reader, 0);
        TestHiC hic = new TestHiC(source);

        Context large = new Context(new Chromosome(1, "large", 5000000000L));
        large.setZoom(new HiCZoom(HiC.Unit.BP, 1000));
        large.setBinOrigin(3000000);
        check(large.getGenomicPositionOrigin() == 3000000000L, "genomic origins retain long coordinates");

        HiCFragmentAxis fragments = new HiCFragmentAxis(1, new int[]{10, 20}, 3000000000L);
        check(fragments.getBinNumberForGenomicPosition(15) == 1, "two-site binary search");
        check(fragments.getGenomicStart(2) == 20 && fragments.getGenomicEnd(2) == 3000000000L,
                "last fragment extends to chromosome end");
        check(fragments.getBinNumberForGenomicPosition(2500000000L) == 2, "long fragment query");
        HiCFragmentAxis noSites = new HiCFragmentAxis(1, new int[0], 100);
        check(noSites.getBinNumberForGenomicPosition(50) == 0 && noSites.getGenomicStart(0) == 0
                && noSites.getGenomicEnd(0) == 100, "chromosome without cut sites");
        HiCFragmentAxis grouped = new HiCFragmentAxis(2, new int[]{10, 20}, 100);
        check(grouped.getGenomicStart(1) == 20 && grouped.getGenomicEnd(1) == 100, "partial final fragment group");

        ContactRecord first = new ContactRecord(1, 2, 3), second = new ContactRecord(1, 2, 9);
        check(first.equals(second) && new HashSet<>(Arrays.asList(first, second)).size() == 1,
                "contact equality and hashing agree");
        check(new ContactRecord(Integer.MAX_VALUE, 0, 1).compareTo(new ContactRecord(-1, 0, 1)) > 0,
                "contact ordering avoids integer overflow");

        DynamicMatrixZoomData dynamic = new DynamicMatrixZoomData(new HiCZoom(HiC.Unit.BP, 20), source);
        check(total(dynamic, 0) == 6, "dynamic cell includes every fine bin");
        check(total(dynamic, 1) == 15 && total(dynamic, 0) == 6, "repeated dynamic windows do not mix cached cells");
        Matrix matrix = new Matrix(0, 0, Collections.singletonList(source));
        matrix.createDynamicResolutionMZD(new Pair<>(20, 10), false);
        check(matrix.getZoomData(new HiCZoom(HiC.Unit.FRAG, 20)) == null,
                "fragment request cannot select a dynamic BP matrix");
        check(matrix.getFirstPearsonZoomData(HiC.Unit.BP) == null, "short ladder without Pearson resolution");
        MatrixZoomData coarse = new MatrixZoomData(source.getChr1(), source.getChr2(),
                new HiCZoom(HiC.Unit.BP, 500000), 100, 10, null, null, reader);
        Matrix shortLadder = new Matrix(0, 0, Arrays.asList(source, coarse));
        check(shortLadder.getFirstPearsonZoomData(HiC.Unit.BP) == coarse, "Pearson selection uses available ladder");

        int[] accesses = {0};
        Block counted = new Block(0, Collections.singletonList(new ContactRecord(0, 1, 2)), "counted") {
            @Override public List<ContactRecord> getContactRecords() { accesses[0]++; return super.getContactRecords(); }
        };
        check(new ColorScaleHandler().computePercentile(Collections.singletonList(counted), 95) == 2
                && accesses[0] == 1, "color percentile fetches block records once");

        Adapter adapter = new Adapter(hic);
        HiCFixedGridAxis axis = new HiCFixedGridAxis(10, 100, null);
        Chromosome chr = new Chromosome(1, "chr1", 1000);
        HiCDataPoint[] fractional = adapter.getData(chr, 0, 1, axis, 0.5, WindowFunction.max);
        check(fractional.length == 1 && fractional[0].getValue(WindowFunction.max) == -2,
                "fractional-scale tracks and negative maxima");
        HiCDataPoint[] subdivided = adapter.getData(chr, 0, 1, axis, 2, WindowFunction.max);
        check(subdivided.length == 1 && subdivided[0].getWithInBins() == 0.5,
                "half-open score does not leak into next sub-bin");

        EigenvectorTrack eigen = new EigenvectorTrack("Eigenvector", "Eigenvector", hic, false);
        eigen.flipEigenvector(0, 10); // must work before any successful load
        BufferedImage image = new BufferedImage(100, 70, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = image.createGraphics();
        Context context = new Context(source.getChr1());
        Rectangle rect = new Rectangle(0, 10, 100, 40);
        try {
            eigen.render(g, context, rect, TrackPanel.Orientation.X, axis);
            int loads = hic.loads;
            eigen.render(g, context, rect, TrackPanel.Orientation.X, axis);
            check(hic.loads == loads, "eigenvector render uses cache");
            hic.norm = NormalizationHandler.VC;
            eigen.render(g, context, rect, TrackPanel.Orientation.X, axis);
            check(hic.loads == loads + 1, "normalization change reloads eigenvector");
        } finally { g.dispose(); }
        for (int x = 0; x < 100; x++) for (int y = 0; y < 70; y++) {
            if (y < rect.y || y >= rect.y + rect.height) check(image.getRGB(x, y) == 0, "eigenvector bars stay inside track");
        }
        HiCTrackManager tracks = new HiCTrackManager(null, hic);
        tracks.moveTrackUp(eigen);
        tracks.moveTrackDown(eigen); // missing/stale selection must be harmless

        CoverageDataset observed = new CoverageDataset(1, 1), control = new CoverageDataset(0, 100);
        hic.setDataset(observed);
        hic.setControlDataset(control);
        HiCCoverageDataSource coverage = new HiCCoverageDataSource(hic, NormalizationHandler.VC, true);
        check(coverage.getDataRange().getMaximum() == 100, "control coverage range uses control data");
        coverage.getData(observed.getChromosomeHandler().getChromosomeFromIndex(1), 0, 1, axis, 1, WindowFunction.mean);
        check(control.requested == 0, "control coverage resolves chromosome by name");
        HiC actual = new HiC(null);
        actual.setDataset(observed);
        actual.setControlDataset(control);
        actual.setChromosomeHandler(observed.getChromosomeHandler());
        check(actual.getEigenvector(0, 0, true)[0] == 100 && control.requested == 0,
                "control eigenvector uses its own chromosome table");

        HiCZoom[] requestedZoom = {null};
        long[] requestedOrigin = {0};
        HiC navigation = new HiC(new SuperAdapter() {
            @Override public boolean isResolutionLocked() { return false; }
            @Override public void executeLongRunningTask(Runnable runnable, String message) { runnable.run(); }
        }) {
            @Override public boolean unsafeActuallySetZoomAndLocation(String x, String y, HiCZoom zoom,
                    long originX, long originY, double scale, boolean reset, ZoomCallType type,
                    boolean broadcast, int locked, boolean store) {
                requestedZoom[0] = zoom; requestedOrigin[0] = originX; return true;
            }
        };
        navigation.setLocation("chr1", "chr1", HiC.Unit.FRAG, 10, 3000000000L, 0,
                1, HiC.ZoomCallType.STANDARD, "test", false);
        check(requestedOrigin[0] == 3000000000L && requestedZoom[0].getUnit() == HiC.Unit.FRAG,
                "navigation accepts initial zoom and long coordinates");
        java.lang.reflect.Field zoomField = HiC.class.getDeclaredField("currentZoom");
        zoomField.setAccessible(true);
        zoomField.set(navigation, new HiCZoom(HiC.Unit.BP, 10));
        navigation.unsafeSetLocation("chr1", "chr1", "FRAG", 10, 3000000000L, 0,
                1, HiC.ZoomCallType.STANDARD, false);
        check(requestedZoom[0].getUnit() == HiC.Unit.FRAG && requestedOrigin[0] == 3000000000L,
                "same bin size does not hide a unit change");
        GoToPanel goTo = new GoToPanel(new SuperAdapter() {
            @Override public HiC getHiC() { return hic; }
        });
        Method parseCoordinate = GoToPanel.class.getDeclaredMethod("cleanUpNumber", String.class);
        parseCoordinate.setAccessible(true);
        check((long) parseCoordinate.invoke(goTo, "3,000,000,000") == 3000000000L,
                "Goto accepts long genomic coordinates");

        Method translate = OneDimTrackCensoring.class.getDeclaredMethod("translateDataPointArray", int.class,
                HiCDataPoint[].class, GenericLocus.class, GenericLocus.class);
        translate.setAccessible(true);
        GenericLocus originalRegion = new GenericLocus("chr1", 0, 100);
        GenericLocus movedRegion = new GenericLocus("custom", 1000, 1100);
        HiCDataAdapter.DataAccumulator point = new HiCDataAdapter.DataAccumulator(0.1, 0.1, 10, 20);
        HiCDataPoint[] moved = (HiCDataPoint[]) translate.invoke(null, 100, new HiCDataPoint[]{point}, originalRegion, movedRegion);
        check(moved[0].getBinNumber() == 10.1, "custom-track translation retains fractional bins");
        check(((HiCDataPoint[]) translate.invoke(null, 100, null, originalRegion, movedRegion)).length == 0,
                "unavailable custom-track source is safe");

        RegionPair pair = RegionPair.generateRegionPair(
                new Pair<>(new GenericLocus("chr1", 3000000000L, 3000002000L), new GenericLocus("custom", 0, 2000)),
                new Pair<>(new GenericLocus("chr1", 3000000000L, 3000002000L), new GenericLocus("custom", 0, 2000)),
                control.getChromosomeHandler());
        MatrixZoomData largeSource = new MatrixZoomData(new Chromosome(0, "chr1", 3500000000L),
                new Chromosome(0, "chr1", 3500000000L), new HiCZoom(HiC.Unit.BP, 1000), 100, 10, null, null, reader);
        Method modify = CustomMatrixZoomData.class.getDeclaredMethod("modifyBlock", Block.class, String.class, MatrixZoomData.class, RegionPair.class);
        modify.setAccessible(true);
        Block transformed = (Block) modify.invoke(null, new Block(0, Arrays.asList(
                new ContactRecord(3000000, 3000000, 1), new ContactRecord(3000002, 3000002, 2)), "raw"),
                "test", largeSource, pair);
        check(transformed.getContactRecords().size() == 1 && transformed.getContactRecords().get(0).getBinX() == 0,
                "custom contacts use long coordinates and exclude the region endpoint");

        check(HiCFileTools.extractDatasetForCLT(Collections.singletonList(args[0]), false).getVersion() == 10
                && HiCFileTools.extractDatasetReaderForCLT(Collections.singletonList(args[0]), false) instanceof DatasetReaderV10,
                "shared file helpers dispatch v10 through the reader factory");

        source.huge = true;
        HeatmapPanel panel = new HeatmapPanel(new SuperAdapter() {
            @Override public HiC getHiC() { return hic; }
        });
        Image thumbnail = panel.getThumbnailImage(source, null, 64, 48, MatrixType.OEV2,
                NormalizationHandler.NONE, NormalizationHandler.NONE);
        check(thumbnail.getWidth(null) == 64 && thumbnail.getHeight(null) == 48,
                "large-grid thumbnail uses output dimensions");

        java.nio.file.Path invalid = java.nio.file.Files.createTempFile("invalid-hic-", ".hic");
        java.nio.file.Files.write(invalid, new byte[]{0, 0, 0, 0});
        try {
            DatasetReaderFactory.getReader(Arrays.asList(args[0], invalid.toString()));
            throw new AssertionError("summed dataset silently dropped invalid input");
        } catch (java.io.IOException expected) { }
        finally { java.nio.file.Files.delete(invalid); }
        System.out.println("Desktop model: coordinates, fragments, dynamic bins, tracks, control indexes, thumbnails and reader validation passed");
    }

    private static float total(DynamicMatrixZoomData zoom, int end) {
        float total = 0;
        for (Block b : zoom.getNormalizedBlocksOverlapping(0, 0, end, end, NormalizationHandler.NONE, false, false))
            for (ContactRecord r : b.getContactRecords()) total += r.getCounts();
        return total;
    }
}
