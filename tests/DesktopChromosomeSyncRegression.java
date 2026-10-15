package juicebox;

import juicebox.data.*;
import juicebox.data.basics.Chromosome;
import juicebox.windowui.*;
import java.lang.reflect.Field;
import java.util.*;

public class DesktopChromosomeSyncRegression {
    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
    static class Capture extends HiC {
        String xName, yName;
        int bin;
        double x, y, scale;
        Capture(Dataset dataset) { super(null); setDataset(dataset); setChromosomeHandler(dataset.getChromosomeHandler()); }
        @Override public void setLocation(String cx, String cy, Unit unit, int b, double ox, double oy,
                double s, ZoomCallType type, String message, boolean broadcast) {
            xName = cx; yName = cy; bin = b; x = ox; y = oy; scale = s;
        }
    }
    static void field(HiC hic, String name, Object value) throws Exception {
        Field field = HiC.class.getDeclaredField(name); field.setAccessible(true); field.set(hic, value);
    }
    static Dataset load(String path) throws Exception {
        return DatasetReaderFactory.getReader(Collections.singletonList(path)).read();
    }
    public static void main(String[] args) throws Exception {
        org.broad.igv.DirectoryManager.setIgvDirectory(java.nio.file.Files.createTempDirectory("desktop-sync-").toFile());
        Dataset padded = load(args[0] + "/pseudo-all.hic");
        ChromosomeHandler handler = padded.getChromosomeHandler();
        check(handler.getOverviewCoordinate(handler.getChromosomeFromName("chrB"), 10) == 11,
                "All inverse uses padded chromosome offsets");
        Capture receiver = new Capture(padded);
        String reply = new CommandExecutor(receiver).execute("setlocation All All BP 50 7 8 2 all-locus 20 chrB 10 chrA 20");
        check(reply.equals("OK") && receiver.bin == 1 && receiver.x == 11 && receiver.y == 2 && receiver.scale == 1,
                "extended All command maps named loci to the receiving All units");
        check(new CommandExecutor(receiver).execute("setlocation All All BP 1 0 0 1 all-locus NaN chrA 0 chrB 0").startsWith("Error:"),
                "non-finite overview metadata rejected");

        HiC publisher = new HiC(null); publisher.setDataset(padded); publisher.setChromosomeHandler(handler);
        Context px = new Context(handler.getChromosomeFromName("All")); px.setBinOrigin(0.25);
        Context py = new Context(handler.getChromosomeFromName("All")); py.setBinOrigin(11.5);
        field(publisher, "xContext", px); field(publisher, "yContext", py);
        field(publisher, "currentZoom", new HiCZoom(HiC.Unit.BP, 1)); field(publisher, "scaleFactor", 2.0);
        check(new CommandExecutor(receiver).execute(publisher.getLocationDescription()).equals("OK") &&
                receiver.x == 0.25 && receiver.y == 11.5, "broadcast roundtrip retains fractional All origins");

        Dataset kb = load(args[0] + "/kilobase-all.hic");
        receiver = new Capture(kb);
        new CommandExecutor(receiver).execute("setlocation All All BP 1 0 0 2 all-locus 10000 chrB 5000 chrA 10000");
        check(receiver.bin == 17 && Math.abs(receiver.x - 100.0 / 17) < 1e-12 && receiver.scale == 3.4,
                "kilobase All converts source scale and fractional origins");

        Dataset trans = load(args[0] + "/trans.hic");
        receiver = new Capture(trans);
        new CommandExecutor(receiver).execute("setlocation chrA chrA BP 15 4 6 2");
        check(receiver.bin == 20 && receiver.x == 3 && receiver.y == 4.5 && Math.abs(receiver.scale - 8.0 / 3) < 1e-12,
                "different resolution ladders preserve bp position and bp per pixel");
        new CommandExecutor(receiver).execute("setlocation All All BP 1 0 0 2 all-locus 10 chrA 20 chrB 30");
        check(receiver.xName.equals("chrA") && receiver.yName.equals("chrB") && receiver.x == 2 && receiver.y == 3,
                "a V10 file without All follows real named loci");

        new CommandExecutor(receiver).execute("setlocation All All BP 1 0 0 2 all-locus 2500000000 chrA 20 chrB 30");
        check(Math.abs(receiver.x * receiver.bin - 20) < 1e-10 && Math.abs(receiver.y * receiver.bin - 30) < 1e-10 &&
                Math.abs(receiver.scale / receiver.bin - 2.0 / 2500000000L) < 1e-20,
                "large physical All resolutions preserve coordinates without int overflow");

        Chromosome a = trans.getChromosomeHandler().getChromosomeFromName("chrA");
        Chromosome b = trans.getChromosomeHandler().getChromosomeFromName("chrB");
        MatrixZoomData nativeZD = trans.getMatrix(a, b).getZoomData(new HiCZoom(HiC.Unit.BP, 10));
        MatrixZoomData reversed = TransposedMatrixZoomData.orient(nativeZD, b.getIndex());
        check(reversed.getChr1().getName().equals("chrB") && reversed.getXGridAxis().getBinCount() == nativeZD.getYGridAxis().getBinCount(),
                "reversed trans matrix labels and grid axes agree");
        boolean found = false;
        for (Block block : reversed.getNormalizedBlocksOverlapping(2, 1, 2, 1, NormalizationHandler.NONE, false, false)) {
            for (ContactRecord record : block.getContactRecords()) if (record.getBinX() == 2 && record.getBinY() == 1) {
                check(record.getCounts() == 1 && record.getExactCount().intValueExact() == 1, "transposition retains exact counts");
                found = true;
            }
        }
        check(found, "reversed trans query returns the correct asymmetric cell");
        HiC display = new HiC(null); display.setDataset(trans); display.setControlDataset(trans);
        field(display, "xContext", new Context(b)); field(display, "yContext", new Context(a));
        field(display, "currentZoom", new HiCZoom(HiC.Unit.BP, 10));
        check(display.getZd().getChr1().getName().equals("chrB") && display.getControlZd().getChr1().getName().equals("chrB"),
                "observed and control rendering use the same named axis order");

        ChromosomeHandler first = new ChromosomeHandler(Arrays.asList(new Chromosome(0, "All", 200),
                new Chromosome(1, "chrA", 95000), new Chromosome(2, "chrB", 75000)), "test", false, false, true);
        ChromosomeHandler second = new ChromosomeHandler(Arrays.asList(new Chromosome(0, "All", 200),
                new Chromosome(1, "chrB", 75000), new Chromosome(2, "chrA", 95000)), "test", false, false, true);
        check(!first.hasSameOverviewLayout(second), "reordered legacy All layouts cannot be overlaid");
        check(second.getOverviewCoordinate(second.getChromosomeFromName("chrA"), 5000) == 80,
                "legacy All coordinate follows the target chromosome order");
        System.out.println("Desktop chromosome sync: name mapping, All units, missing All, resolution ladders and trans axes passed");
    }
}
