package juicebox.mapcolorui;
import juicebox.HiC;
import juicebox.data.*;
import juicebox.windowui.*;
import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.util.*;
public class DesktopRenderRegression {
    public static void main(String[] args) throws Exception {
        org.broad.igv.DirectoryManager.setIgvDirectory(java.nio.file.Files.createTempDirectory("desktop-render-").toFile());
        Dataset ds = DatasetReaderFactory.getReader(Collections.singletonList(args[0])).read();
        MatrixZoomData zd = ds.getMatrix(ds.getChromosomeHandler().getChromosomeFromIndex(0), ds.getChromosomeHandler().getChromosomeFromIndex(0)).getZoomData(new HiCZoom(HiC.Unit.BP, 10));
        for (MatrixType mode : new MatrixType[]{MatrixType.OEV2}) {
            BufferedImage image = new BufferedImage(8, 8, BufferedImage.TYPE_INT_ARGB);
            Graphics2D graphics = image.createGraphics();
            HeatmapRenderer renderer = new HeatmapRenderer(graphics, new ColorScaleHandler());
            if (!renderer.render(0,0,8,8,zd,null,mode,NormalizationHandler.NONE,NormalizationHandler.NONE,ds.getExpectedValues(zd.getZoom(), NormalizationHandler.NONE),null,false)) throw new AssertionError("render failed for " + mode);
            graphics.dispose();
            boolean painted = false;
            for (int x=0; x<8; x++) for (int y=0; y<8; y++) painted |= (image.getRGB(x,y) >>> 24) != 0;
            if (!painted) throw new AssertionError("no pixels painted for " + mode);
        }
        System.out.println("Desktop headless rendering: OE pixels, expected values, chromosome zero passed");
    }
}
