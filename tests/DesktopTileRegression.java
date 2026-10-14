package juicebox.mapcolorui;

import juicebox.HiC;
import juicebox.data.*;
import juicebox.data.basics.Chromosome;
import juicebox.gui.SuperAdapter;
import juicebox.track.HiCFixedGridAxis;
import juicebox.track.HiCGridAxis;
import juicebox.windowui.*;

import javax.swing.JPanel;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.util.Collections;
import java.util.List;

/** Headless checks for tile allocation, retry, viewport bounds and vector export. */
public class DesktopTileRegression {
    private static class TestZoomData extends MatrixZoomData {
        int queries;
        boolean available = true;

        TestZoomData(DatasetReader reader) {
            super(new Chromosome(1, "chr1", 10010), new Chromosome(2, "chr2", 7510),
                    new HiCZoom(HiC.Unit.BP, 10), 500, 3, null, null, reader);
        }

        @Override public HiCGridAxis getXGridAxis() { return new HiCFixedGridAxis(1001, 10, null); }
        @Override public HiCGridAxis getYGridAxis() { return new HiCFixedGridAxis(751, 10, null); }

        @Override
        public List<Block> getNormalizedBlocksOverlapping(long x0, long y0, long x1, long y1,
                NormalizationType norm, boolean important, boolean fill) {
            queries++;
            check(x0 >= 0 && y0 >= 0 && x1 < 1001 && y1 < 751, "query outside grid");
            return available ? Collections.emptyList() : null;
        }
    }

    private static class TestHiC extends HiC {
        boolean zoomChanged, displayChanged, normChanged;
        TestHiC() { super(null); }
        @Override public ExpectedValueFunction getExpectedValues() { return null; }
        @Override public ExpectedValueFunction getExpectedControlValues() { return null; }
        @Override public MatrixZoomData getControlZd() { return null; }
        @Override public boolean testZoomChanged() {
            boolean value = zoomChanged; zoomChanged = false; return value;
        }
        @Override public boolean testDisplayOptionChanged() {
            boolean value = displayChanged; displayChanged = false; return value;
        }
        @Override public boolean testNormalizationTypeChanged() {
            boolean value = normChanged; normChanged = false; return value;
        }
    }

    private static class TestColors extends ColorScaleHandler {
        int updates;
        @Override public org.broad.igv.renderer.ColorScale getColorScale(String key, MatrixType mode,
                boolean wholeGenome, List<Block> blocks, float max) {
            // IGV's continuous scale requires a display; use Juicebox's headless scale here.
            return new OEColorScale(MatrixType.OEV2);
        }
        @Override public void updateColorSliderFromColorScale(SuperAdapter adapter, MatrixType mode, String key) {
            updates++;
        }
    }

    private static class ExportRenderer extends HeatmapRenderer {
        int fallbackCalls;
        boolean fail;
        int maxRight, maxBottom;
        ExportRenderer(Graphics2D g, ColorScaleHandler colors) { super(g, colors); }
        @Override public void drawImage(Image image, int x0, int y0, int x1, int y1,
                int sx0, int sy0, int sx1, int sy1) {
            maxRight = Math.max(maxRight, x1);
            maxBottom = Math.max(maxBottom, y1);
            throw new UnsupportedOperationException("exercise vector fallback");
        }
        @Override public boolean render(int x, int y, int w, int h, MatrixZoomData zd,
                MatrixZoomData control, MatrixType mode, NormalizationType obs, NormalizationType ctrl,
                ExpectedValueFunction expected, ExpectedValueFunction controlExpected, boolean important) {
            fallbackCalls++;
            if (fail) throw new IllegalStateException("exercise failed vector render");
            return true;
        }
    }

    private static GeneralTileManager.ImageTile tile(HiCMapTileManager manager, TestZoomData zd,
            TestHiC hic, int row, int column) {
        return manager.getImageTile(zd, null, row, column, MatrixType.OBSERVED,
                NormalizationHandler.NONE, NormalizationHandler.NONE, hic, new JPanel());
    }

    public static void main(String[] args) throws Exception {
        org.broad.igv.DirectoryManager.setIgvDirectory(java.nio.file.Files.createTempDirectory("desktop-tiles-").toFile());
        DatasetReader reader = DatasetReaderFactory.getReader(Collections.singletonList(args[0]));
        reader.read();
        TestZoomData zd = new TestZoomData(reader);
        TestHiC hic = new TestHiC();
        TestColors colors = new TestColors();
        HiCMapTileManager manager = new HiCMapTileManager(colors);

        // JPanel is deliberately undisplayed: component.createImage() can return null here.
        GeneralTileManager.ImageTile edge = tile(manager, zd, hic, 1, 2);
        check(edge.image.getWidth(null) == 1 && edge.image.getHeight(null) == 251, "edge tile dimensions");
        check(tile(manager, zd, hic, 1, 2) == edge, "successful tile is cached");
        int queries = zd.queries;
        check(tile(manager, zd, hic, 0, 3) == null && tile(manager, zd, hic, 2, 0) == null
                && tile(manager, zd, hic, -1, 0) == null, "out-of-grid tiles skipped");
        check(zd.queries == queries, "out-of-grid tiles must not read data");
        zd.available = false;
        check(tile(manager, zd, hic, 0, 0) == null, "failed rendering returns no tile");
        zd.available = true;
        check(tile(manager, zd, hic, 0, 0) != null, "failed tile can be retried");

        BufferedImage image = new BufferedImage(600, 600, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = image.createGraphics();
        try {
            GeneralTileManager general = new GeneralTileManager(colors);
            zd.queries = 0;
            hic.zoomChanged = hic.displayChanged = hic.normChanged = true;
            general.renderHiCTiles(new HeatmapRenderer(g, colors), 0, 0, 500, 500, zd, null,
                    1, new Rectangle(400, 300, 500, 500), hic, new JPanel(), null);
            check(zd.queries == 1, "exact-boundary viewport requests only one tile");
            check(colors.updates == 1 && !hic.zoomChanged && !hic.displayChanged && !hic.normChanged,
                    "all change flags consumed in one repaint");

            g.translate(7, 11);
            g.scale(2, 3);
            AffineTransform original = g.getTransform();
            for (boolean fail : new boolean[]{false, true}) {
                ExportRenderer renderer = new ExportRenderer(g, colors);
                renderer.fail = fail;
                general.renderHiCTiles(renderer, 450, 450, 450 + 500 / 1.5, 450 + 500 / 1.5,
                        zd, null, 1.5, new Rectangle(400, 300, 500, 500), hic, new JPanel(), null);
                check(renderer.fallbackCalls == 4, "multiple export tiles exercised");
                check(original.equals(g.getTransform()), "export restores preexisting transform even on failure");
                check(renderer.maxRight <= 501 && renderer.maxBottom <= 501,
                        "clipping uses local bounds rather than component location");
            }
        } finally {
            g.dispose();
        }
        System.out.println("Desktop tiles: edge sizes, retry, visible queries, change flags and export transforms passed");
    }

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
