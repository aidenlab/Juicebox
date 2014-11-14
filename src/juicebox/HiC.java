package juicebox;

import org.apache.log4j.Logger;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.ui.color.ColorUtilities;
import org.broad.igv.util.ParsingUtils;
import org.broad.igv.util.ResourceLocator;
import juicebox.data.*;
import juicebox.track.*;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.*;
import java.util.List;

/**
 * This is the "model" class for the HiC viewer.
 *
 * @author Jim Robinson
 * @since 4/8/12
 */
public class HiC {

    private static Logger log = Logger.getLogger(HiC.class);

    private double scaleFactor;

    public enum Unit {BP, FRAG}

    private MainWindow mainWindow;
    private MainWindow.MatrixType displayOption;
    private NormalizationType normalizationType;
    private java.util.List<Chromosome> chromosomes;

    private Dataset dataset;
    private Dataset controlDataset;
    private HiCZoom zoom;

    private Context xContext;
    private Context yContext;

    private Map<String, Feature2DList> loopLists;
    private boolean showLoops;

    private EigenvectorTrack eigenvectorTrack;

    private HiCTrackManager trackManager;
    private ResourceTree resourceTree;
    private LoadEncodeAction encodeAction;

    private Point cursorPoint;
    private Point selectedBin;

    private boolean linkedMode;

    private boolean m_zoomChanged;


    public HiC(MainWindow mainWindow) {
        this.mainWindow = mainWindow;
        this.trackManager = new HiCTrackManager(mainWindow, this);
        this.loopLists = new HashMap<String, Feature2DList>();
    }

    public void reset() {
        dataset = null;
        xContext = null;
        yContext = null;
        eigenvectorTrack = null;
        resourceTree = null;
        encodeAction = null;
        trackManager.clearTracks();
        loopLists.clear();
        showLoops = true;
    }

    public double getScaleFactor() {
        return scaleFactor;
    }

    public void setScaleFactor(double scaleFactor) {
        this.scaleFactor = Math.min(50, scaleFactor);
    }

    public void setControlDataset(Dataset controlDataset) {
        this.controlDataset = controlDataset;
    }

    public void loadEigenvectorTrack() {
        if (eigenvectorTrack == null) {
            eigenvectorTrack = new EigenvectorTrack("Eigenvector", "Eigenvector", this);
        }
        trackManager.add(eigenvectorTrack);
    }


    public ResourceTree getResourceTree() {
        return resourceTree;
    }

    public void setResourceTree(ResourceTree rTree) {
        resourceTree = rTree;
    }

    public LoadEncodeAction getEncodeAction() { return encodeAction; }

    public void setEncodeAction(LoadEncodeAction eAction) {encodeAction = eAction;}

    public List<Feature2D> getVisibleLoopList(int chrIdx1, int chrIdx2) {
        if (!showLoops) return null;
        List<Feature2D> visibleLoopList = new ArrayList<Feature2D>();
        for (Feature2DList list : loopLists.values()) {
            if (list.isVisible()) {
                List<Feature2D> currList = list.get(chrIdx1, chrIdx2);
                if (currList != null) {
                    for (Feature2D feature2D : currList) {
                        visibleLoopList.add(feature2D);
                    }
                }
            }
        }
        return visibleLoopList;
    }

    public void setShowLoops(boolean showLoops1) {
        showLoops = showLoops1;

    }

    public void setLoopsInvisible(String path) {
        loopLists.get(path).setVisible(false);
    }

    public boolean isLinkedMode() {
        return linkedMode;
    }

    public void setLinkedMode(boolean linkedMode) {
        this.linkedMode = linkedMode;
    }

    public java.util.List<HiCTrack> getLoadedTracks() {
        return trackManager == null ? new ArrayList<HiCTrack>() : trackManager.getLoadedTracks();
    }

    public void loadHostedTracks(List<ResourceLocator> locators) {
        trackManager.load(locators);
    }

    public void loadTrack(String path) {
        trackManager.loadTrack(path);
    }


    public void loadCoverageTrack(NormalizationType no) {
        trackManager.loadCoverageTrack(no);
    }

    public void removeTrack(HiCTrack track) {
        if (resourceTree != null) resourceTree.remove(track.getLocator());
        if (encodeAction != null) encodeAction.remove(track.getLocator());
        trackManager.removeTrack(track);
    }

    public void removeTrack(ResourceLocator locator) {
        trackManager.removeTrack(locator);
    }

    public Dataset getDataset() {
        return dataset;
    }

    public void setDataset(Dataset dataset) {
        this.dataset = dataset;
    }

    public void setSelectedChromosomes(Chromosome chrX, Chromosome chrY) {
        this.xContext = new Context(chrX);
        this.yContext = new Context(chrY);

        if (eigenvectorTrack != null) {
            eigenvectorTrack.forceRefresh();
        }

    }

    public HiCZoom getZoom() {
        return zoom;
    }

    public MatrixZoomData getZd() {
        Matrix matrix = getMatrix();
        if (matrix == null || zoom == null) {
            return null;
        } else {
            return matrix.getZoomData(zoom);
        }
    }

    public MatrixZoomData getControlZd() {
        Matrix matrix = getControlMatrix();
        if (matrix == null || zoom == null) {
            return null;
        } else {
            return matrix.getZoomData(zoom);
        }
    }

    public Matrix getControlMatrix() {
        if (controlDataset == null || xContext == null || zoom == null) return null;

        return controlDataset.getMatrix(xContext.getChromosome(), yContext.getChromosome());
    }

    public Context getXContext() {
        return xContext;
    }

    public Context getYContext() {
        return yContext;
    }

    public void resetContexts() {
        this.xContext = null;
        this.yContext = null;
    }

    public void setCursorPoint(Point point) {
        this.cursorPoint = point;

    }

    public Point getCursorPoint() {
        return cursorPoint;
    }

    public Matrix getMatrix() {
        return dataset == null || xContext == null ? null : getDataset().getMatrix(xContext.getChromosome(), yContext.getChromosome());

    }

    public void setSelectedBin(Point point) {
        if (point.equals(this.selectedBin)) {
            this.selectedBin = null;
        } else {
            this.selectedBin = point;
        }
    }

    public MainWindow.MatrixType getDisplayOption() {
        return displayOption;
    }

    public boolean isControlLoaded() {return controlDataset != null;}

    public boolean isWholeGenome() {
        return xContext != null && xContext.getChromosome().getName().equals("All");
    }

    public void setChromosomes(List<Chromosome> chromosomes) {
        this.chromosomes = chromosomes;
    }

    public java.util.List<Chromosome> getChromosomes() {
        return chromosomes;
    }

    public static boolean isPrivateHic(String string) {
        return string.contains("igvdata/hic/files");
    }

    /**
     * Change zoom level and recenter.  Triggered by the resolutionSlider, or by a double-click in the
     * heatmap panel.
     */
    public boolean setZoom(HiCZoom newZoom, final double centerGenomeX, final double centerGenomeY) {

        if(dataset == null) return false;

        final Chromosome chrX = xContext.getChromosome();
        final Chromosome chrY = yContext.getChromosome();

        // Verify that all datasets have this zoom level

        Matrix matrix = dataset.getMatrix(chrX, chrY);
        if (matrix == null) return false;

        MatrixZoomData newZD;
        if (chrX.getName().equals("All")) {
            newZD = matrix.getFirstZoomData(Unit.BP);
        } else {
            newZD = matrix.getZoomData(newZoom);
        }
        if (newZD == null) {
            JOptionPane.showMessageDialog(mainWindow, "Sorry, this zoom is not available", "Zoom unavailable", JOptionPane.WARNING_MESSAGE);
            return false;
        }

        // Assumption is all datasets share the same grid axis
        HiCGridAxis xGridAxis = newZD.getXGridAxis();
        HiCGridAxis yGridAxis = newZD.getYGridAxis();

        zoom = newZoom;

        xContext.setZoom(zoom);
        yContext.setZoom(zoom);

        int xBinCount = xGridAxis.getBinCount();
        int yBinCount = yGridAxis.getBinCount();
        int maxBinCount = Math.max(xBinCount, yBinCount);

        double scalefactor = Math.max(1.0, (double) mainWindow.getHeatmapPanel().getMinimumDimension() / maxBinCount);

        setScaleFactor(scalefactor);

        //Point binPosition = zd.getBinPosition(genomePositionX, genomePositionY);
        int binX = xGridAxis.getBinNumberForGenomicPosition((int) centerGenomeX);
        int binY = yGridAxis.getBinNumberForGenomicPosition((int) centerGenomeY);

        center(binX, binY);

        //Notify Heatmap panel render that the zoom has been changed. In that case,
        //Render should update zoom slider (only once) with previous map range values
        setZoomChanged();

        if (linkedMode) {
            broadcastState();
        }

        return true;
    }

    private void setZoomChanged()
    {
        m_zoomChanged = true;
    }

    //Check zoom change value and reset.
    synchronized boolean testZoomChanged()
    {
        if (m_zoomChanged)
        {
            m_zoomChanged = false;
            return true;
        }
        return false;
    }

    // Called from alt-drag
    public void zoomTo(final int xBP0, final int yBP0, double targetBinSize) {


        if (dataset == null) return;  // No data in view


        final Chromosome chr1 = xContext.getChromosome();
        final Chromosome chr2 = yContext.getChromosome();
        final Matrix matrix = dataset.getMatrix(chr1, chr2);


        HiC.Unit unit = zoom.getUnit();

        // Find the new resolution,
        HiCZoom newZoom = zoom;
        if (!mainWindow.isResolutionLocked()) {
            List<HiCZoom> zoomList = unit == HiC.Unit.BP ? dataset.getBpZooms() : dataset.getFragZooms();
            zoomList.get(zoomList.size() - 1);   // Highest zoom level by defaul
            for (int i = zoomList.size() - 1; i >= 0; i--) {
                if (zoomList.get(i).getBinSize() > targetBinSize) {
                    newZoom = zoomList.get(i);
                    break;
                }
            }
        }

        final MatrixZoomData newZD = matrix.getZoomData(newZoom);

        int binX0 = newZD.getXGridAxis().getBinNumberForGenomicPosition(xBP0);
        int binY0 = newZD.getYGridAxis().getBinNumberForGenomicPosition(yBP0);

        final double scaleFactor = newZD.getBinSize() / targetBinSize;

        zoom = newZD.getZoom();


        mainWindow.updateZoom(zoom);

        setScaleFactor(scaleFactor);
        setScaleFactor(scaleFactor);

        xContext.setBinOrigin(binX0);
        yContext.setBinOrigin(binY0);


        if (linkedMode) {
            broadcastState();
        }

        mainWindow.refresh();


    }

    public void centerFragment(int fragmentX, int fragmentY) {
        if (zoom != null) {

            MatrixZoomData zd = getMatrix().getZoomData(zoom);
            HiCGridAxis xAxis = zd.getXGridAxis();
            HiCGridAxis yAxis = zd.getYGridAxis();

            int binX = xAxis.getBinNumberForFragment(fragmentX);
            int binY = yAxis.getBinNumberForFragment(fragmentY);
            center(binX, binY);

        }
    }

    public void centerBP(int bpX, int bpY) {
        if (zoom != null) {
            MatrixZoomData zd = getMatrix().getZoomData(zoom);
            HiCGridAxis xAxis = zd.getXGridAxis();
            HiCGridAxis yAxis = zd.getYGridAxis();

            int binX = xAxis.getBinNumberForGenomicPosition(bpX);
            int binY = yAxis.getBinNumberForGenomicPosition(bpY);
            center(binX, binY);

        }
    }

    /**
     * Center the bins in view at the current resolution.
     *
     * @param binX center X
     * @param binY center Y
     */
    public void center(double binX, double binY) {

        double w = mainWindow.getHeatmapPanel().getWidth() / getScaleFactor();  // view width in bins
        int newOriginX = (int) (binX - w / 2);

        double h = mainWindow.getHeatmapPanel().getHeight() / getScaleFactor();  // view hieght in bins
        int newOriginY = (int) (binY - h / 2);
        moveTo(newOriginX, newOriginY);
    }

    /**
     * Move by the specified delta (in bins)
     *
     * @param dxBins -- delta x in bins
     * @param dyBins -- delta y in bins
     */
    public void moveBy(double dxBins, double dyBins) {
        final double newX = xContext.getBinOrigin() + dxBins;
        final double newY = yContext.getBinOrigin() + dyBins;
        moveTo(newX, newY);
    }

    /**
     * Move to the specified origin (in bins)
     *
     * @param newBinX new location X
     * @param newBinY new location Y
     */
    private void moveTo(double newBinX, double newBinY) {

        MatrixZoomData zd = getZd();

        final double wBins = (mainWindow.getHeatmapPanel().getWidth() / getScaleFactor());
        double maxX = zd.getXGridAxis().getBinCount() - wBins;

        final double hBins = (mainWindow.getHeatmapPanel().getHeight() / getScaleFactor());
        double maxY = zd.getYGridAxis().getBinCount() - hBins;

        double x = Math.max(0, Math.min(maxX, newBinX));
        double y = Math.max(0, Math.min(maxY, newBinY));

        xContext.setBinOrigin(x);
        yContext.setBinOrigin(y);

//        String locus1 = "chr" + (xContext.getChromosome().getName()) + ":" + x + "-" + (int) (x + bpWidthX);
//        String locus2 = "chr" + (yContext.getChromosome().getName()) + ":" + x + "-" + (int) (y + bpWidthY);
//        IGVUtils.sendToIGV(locus1, locus2);

        mainWindow.repaint();

        if (linkedMode) {
            broadcastState();
        }
    }


    public void setDisplayOption(MainWindow.MatrixType newDisplay) {

        if (this.displayOption != newDisplay) {
            this.displayOption = newDisplay;
        }
    }

    public void setNormalizationType(NormalizationType option) {
        this.normalizationType = option;
    }

    public NormalizationType getNormalizationType() {
        return normalizationType;
    }

    public double[] getEigenvector(final int chrIdx, final int n) {

        if (dataset == null) return null;

        Chromosome chr = chromosomes.get(chrIdx);
        return dataset.getEigenvector(chr, zoom, n, normalizationType);

    }

    public ExpectedValueFunction getExpectedValues() {
        if (dataset == null) return null;
        return dataset.getExpectedValues(zoom, normalizationType);
    }

    public NormalizationVector getNormalizationVector(int chrIdx) {
        if (dataset == null) return null;
        return dataset.getNormalizationVector(chrIdx, zoom, normalizationType);
    }

    // Note - this is an inefficient method, used to support tooltip text only.
    public float getNormalizedObservedValue(int binX, int binY) {

        return getZd().getObservedValue(binX, binY, normalizationType);

    }


    public void loadLoopList(String path) throws IOException {

        if (loopLists.get(path) != null) {
            loopLists.get(path).setVisible(true);
            return;
        }

        int attCol = 7;
        BufferedReader br = null;

        Feature2DList newList = new Feature2DList();

        try {
            mainWindow.showGlassPane();
            br = ParsingUtils.openBufferedReader(path);
            String nextLine;

            // header
            nextLine = br.readLine();
            String[] headers = Globals.tabPattern.split(nextLine);

            int errorCount = 0;
            while ((nextLine = br.readLine()) != null) {
                String[] tokens = Globals.tabPattern.split(nextLine);
                if (tokens.length > headers.length) {
                    throw new IOException("Improperly formatted file");
                }
                if (tokens.length < 6) {
                    continue;
                }
                String chr1Name = tokens[0];
                int start1 = Integer.parseInt(tokens[1]);
                int end1 = Integer.parseInt(tokens[2]);

                String chr2Name = tokens[3];
                int start2 = Integer.parseInt(tokens[4]);
                int end2 = Integer.parseInt(tokens[5]);

                Color c = tokens.length > 6 ? ColorUtilities.stringToColor(tokens[6].trim()) : Color.black;

                Map<String, String> attrs = new LinkedHashMap<String, String>();
                for (int i = attCol; i < tokens.length; i++) {

                    attrs.put(headers[i], tokens[i]);
                }

                Chromosome chr1 = this.getChromosomeNamed(chr1Name);
                Chromosome chr2 = this.getChromosomeNamed(chr2Name);
                if (chr1 == null || chr2 == null) {
                    if (errorCount < 100) {
                        log.error("Skipping line: " + nextLine);
                    } else if (errorCount == 100) {
                        log.error("Maximum error count exceeded.  Further errors will not be logged");
                    }

                    errorCount++;
                    continue;
                }

                // Convention is chr1 is lowest "index". Swap if necessary
                Feature2D feature = chr1.getIndex() <= chr2.getIndex() ?
                        new Feature2D(chr1Name, start1, end1, chr2Name, start2, end2, c, attrs) :
                        new Feature2D(chr2Name, start2, end2, chr1Name, start1, end1, c, attrs);

                newList.add(chr1.getIndex(), chr2.getIndex(), feature);

            }
            loopLists.put(path, newList);
            mainWindow.hideGlassPane();
        } finally {
            if (br != null) br.close();
            mainWindow.hideGlassPane();
        }

    }

    private Chromosome getChromosomeNamed(String token) {
        for (Chromosome chr : chromosomes) {
            if (token.equals(chr.getName())) return chr;
        }
        return null;
    }


    /**
     * Change zoom level and recenter.  Triggered by the resolutionSlider, or by a double-click in the
     * heatmap panel.
     */
    public void setState(String chrXName, String chrYName, String unitName, int binSize, double xOrigin, double yOrigin, double scalefactor) {

        if (!chrXName.equals(xContext.getChromosome().getName()) || !chrYName.equals(yContext.getChromosome().getName())) {

            Chromosome chrX = getChromosomeNamed(chrXName);
            Chromosome chrY = getChromosomeNamed(chrYName);
            this.xContext = new Context(chrX);
            this.yContext = new Context(chrY);
            mainWindow.setSelectedChromosomesNoRefresh(chrX, chrY);
            if (eigenvectorTrack != null) {
                eigenvectorTrack.forceRefresh();
            }
        }

        HiCZoom newZoom = new HiCZoom(Unit.valueOf(unitName), binSize);
        if (!newZoom.equals(zoom)) {
            zoom = newZoom;
            xContext.setZoom(zoom);
            yContext.setZoom(zoom);
            mainWindow.updateZoom(newZoom);
        }

        setScaleFactor(scalefactor);
        setScaleFactor(scalefactor);
        xContext.setBinOrigin(xOrigin);
        yContext.setBinOrigin(yOrigin);

        mainWindow.refresh();

    }

    public void broadcastState() {
        String command = "setstate " +
                xContext.getChromosome().getName() + " " +
                yContext.getChromosome().getName() + " " +
                zoom.getUnit().toString() + " " +
                zoom.getBinSize() + " " +
                xContext.getBinOrigin() + " " +
                yContext.getBinOrigin() + " " +
                getScaleFactor();

        CommandBroadcaster.broadcast(command);
    }
    public void saveState() {
        String command = "setstate " +
                xContext.getChromosome().getName() + " " +
                yContext.getChromosome().getName() + " " +
                zoom.getUnit().toString() + " " +
                zoom.getBinSize() + " " +
                xContext.getBinOrigin() + " " +
                yContext.getBinOrigin() + " " +
                getScaleFactor();

        CommandBroadcaster.broadcast(command);
    }
}
