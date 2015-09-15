/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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


package juicebox;

import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;
import juicebox.data.*;
import juicebox.gui.SuperAdapter;
import juicebox.mapcolorui.Feature2DHandler;
import juicebox.track.*;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;
import org.apache.log4j.Logger;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * This is the "model" class for the HiC viewer.
 *
 * @author Jim Robinson
 * @since 4/8/12
 */
public class HiC {

    private static final Logger log = Logger.getLogger(HiC.class);
    private static final Splitter MY_SPLITTER = Splitter.on(CharMatcher.BREAKING_WHITESPACE).trimResults().omitEmptyStrings();

    //private final MainWindow mainWindow;
    private final Feature2DHandler feature2DHandler;
    private final HiCTrackManager trackManager;
    private final HashMap<String, Integer> binSizeDictionary = new HashMap<String, Integer>();
    private final SuperAdapter superAdapter;
    private double scaleFactor;
    private String xPosition;
    private String yPosition;
    private MatrixType displayOption;
    private NormalizationType normalizationType;
    private List<Chromosome> chromosomes;
    private Dataset dataset;
    private Dataset controlDataset;
    private HiCZoom zoom;
    //private MatrixZoomData matrixForReloadState;
    private Context xContext;
    private Context yContext;
    private EigenvectorTrack eigenvectorTrack;
    private ResourceTree resourceTree;
    private LoadEncodeAction encodeAction;
    private Point cursorPoint;
    private Point selectedBin;
    private boolean linkedMode;
    private boolean m_zoomChanged;
    private boolean m_displayOptionChanged;
    private boolean m_normalizationTypeChanged;

    public HiC(SuperAdapter superAdapter) {
        this.superAdapter = superAdapter;
        trackManager = new HiCTrackManager(superAdapter, this);
        feature2DHandler = new Feature2DHandler();
        m_zoomChanged = false;
        m_displayOptionChanged = false;
        m_normalizationTypeChanged = false;
        initBinSizeDictionary();
    }


    public void reset() {
        dataset = null;
        xContext = null;
        yContext = null;
        eigenvectorTrack = null;
        resourceTree = null;
        encodeAction = null;
        clearFeatures();
    }

    // TODO zgire - why iterate through trackstoremove if you end up calling clearFeatures() ?
    public void clearTracksForReloadState() {
        ArrayList<HiCTrack> tracksToRemove = new ArrayList<HiCTrack>(trackManager.getLoadedTracks());
        for (HiCTrack trackToRemove : tracksToRemove) {
            if (trackToRemove.getName().equals("eigenvector")) {
                eigenvectorTrack = null;
            } else {
                trackManager.removeTrack(trackToRemove);
            }
        }
        clearFeatures();
        superAdapter.updateTrackPanel();
    }

    private void clearFeatures() {
        trackManager.clearTracks();
        feature2DHandler.clearLists();
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

    public void refreshEigenvectorTrackIfExists() {
        if (eigenvectorTrack != null) {
            eigenvectorTrack.forceRefresh();
        }
    }

    public ResourceTree getResourceTree() {
        return resourceTree;
    }

    public void setResourceTree(ResourceTree rTree) {
        resourceTree = rTree;
    }

    public LoadEncodeAction getEncodeAction() {
        return encodeAction;
    }

    public void setEncodeAction(LoadEncodeAction eAction) {
        encodeAction = eAction;
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
        trackManager.safeTrackLoad(locators);
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
        if (resourceTree != null) resourceTree.remove(locator);
        if (encodeAction != null) encodeAction.remove(locator);
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
        // TODO - every function which calls this needs to check for null values
        // maybe throw an Exception to force this check
        if (matrix == null) {
            System.err.println("Matrix is null");
            return null;
        } else if (zoom == null) {
            System.err.println("Zoom is null");
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

    public Point getCursorPoint() {
        return cursorPoint;
    }

    public void setCursorPoint(Point point) {
        this.cursorPoint = point;

    }

    public String getXPosition() {
        return xPosition;
    }

    public void setXPosition(String txt) {
        this.xPosition = txt;
    }

    public String getYPosition() {
        return yPosition;
    }

    public void setYPosition(String txt) {
        this.yPosition = txt;
    }

    public Matrix getMatrix() {
        if (dataset == null) {
            System.err.println("Dataset is null");
            return null;
        } else if (xContext == null) {
            System.err.println("xContext is null");
            return null;
        } else if (yContext == null) {
            System.err.println("yContext is null");
            return null;
        }
        return dataset.getMatrix(xContext.getChromosome(), yContext.getChromosome());

    }

    public void setSelectedBin(Point point) {
        if (point.equals(this.selectedBin)) {
            this.selectedBin = null;
        } else {
            this.selectedBin = point;
        }
    }

    public MatrixType getDisplayOption() {
        return displayOption;
    }

    public void setDisplayOption(MatrixType newDisplay) {
        if (this.displayOption != newDisplay) {
            this.displayOption = newDisplay;
            setDisplayOptionChanged();
        }
    }

    public boolean isControlLoaded() {
        return controlDataset != null;
    }

    public boolean isWholeGenome() {
        return xContext != null && xContext.getChromosome().getName().equals("All");
    }

    public java.util.List<Chromosome> getChromosomes() {
        return chromosomes;
    }

    public void setChromosomes(List<Chromosome> chromosomes) {
        this.chromosomes = chromosomes;
    }

    /**
     * Change zoom level and recenter.  Triggered by the resolutionSlider, or by a double-click in the
     * heatmap panel.
     */
    public boolean setZoom(HiCZoom newZoom, final double centerGenomeX, final double centerGenomeY) {

        if (dataset == null) return false;

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
            superAdapter.launchGenericMessageDialog("Sorry, this zoom is not available", "Zoom unavailable",
                    JOptionPane.WARNING_MESSAGE);
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

        double scalefactor = Math.max(1.0, (double) superAdapter.getHeatmapPanel().getMinimumDimension() / maxBinCount);

        setScaleFactor(scalefactor);

        //Point binPosition = zd.getBinPosition(genomePositionX, genomePositionY);
        int binX = xGridAxis.getBinNumberForGenomicPosition((int) centerGenomeX);
        int binY = yGridAxis.getBinNumberForGenomicPosition((int) centerGenomeY);

        center(binX, binY);

        //Notify Heatmap panel render that the zoom has been changed. In that case,
        //Render should update zoom slider (only once) with previous map range values
        setZoomChanged();

        if (linkedMode) {
            broadcastLocation();
        }

        return true;
    }

    private void setZoomChanged() {
        m_zoomChanged = true;
    }

    //Check zoom change value and reset.
    public synchronized boolean testZoomChanged() {
        if (m_zoomChanged) {
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
        if (!superAdapter.isResolutionLocked()) {
            List<HiCZoom> zoomList = unit == HiC.Unit.BP ? dataset.getBpZooms() : dataset.getFragZooms();
            zoomList.get(zoomList.size() - 1);   // Highest zoom level by default
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


        superAdapter.updateZoom(zoom);

        setScaleFactor(scaleFactor);

        xContext.setBinOrigin(binX0);
        yContext.setBinOrigin(binY0);


        if (linkedMode) {
            broadcastLocation();
        }

//        try {
//            mainWindow.refresh();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
    }

    public void centerFragment(int fragmentX, int fragmentY) {
        if (zoom != null) {

            MatrixZoomData zd = getMatrix().getZoomData(zoom);
            HiCGridAxis xAxis = zd.getXGridAxis();
            HiCGridAxis yAxis = zd.getYGridAxis();
            int binX;
            int binY;
            try {
                binX = xAxis.getBinNumberForFragment(fragmentX);
                //noinspection SuspiciousNameCombination
                binY = yAxis.getBinNumberForFragment(fragmentY);
                center(binX, binY);
            } catch (RuntimeException error) {
                superAdapter.launchGenericMessageDialog(error.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            }

        }
    }

    public void centerBP(int bpX, int bpY) {
        if (zoom != null && getMatrix() != null) {
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

        double w = superAdapter.getHeatmapPanel().getWidth() / getScaleFactor();  // view width in bins
        int newOriginX = (int) (binX - w / 2);

        double h = superAdapter.getHeatmapPanel().getHeight() / getScaleFactor();  // view hieght in bins
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

        final double wBins = (superAdapter.getHeatmapPanel().getWidth() / getScaleFactor());
        double maxX = zd.getXGridAxis().getBinCount() - wBins;

        final double hBins = (superAdapter.getHeatmapPanel().getHeight() / getScaleFactor());
        double maxY = zd.getYGridAxis().getBinCount() - hBins;

        double x = Math.max(0, Math.min(maxX, newBinX));
        double y = Math.max(0, Math.min(maxY, newBinY));

        xContext.setBinOrigin(x);
        yContext.setBinOrigin(y);

//        String locus1 = "chr" + (xContext.getChromosome().getName()) + ":" + x + "-" + (int) (x + bpWidthX);
//        String locus2 = "chr" + (yContext.getChromosome().getName()) + ":" + x + "-" + (int) (y + bpWidthY);
//        IGVUtils.sendToIGV(locus1, locus2);

        superAdapter.repaint();

        if (linkedMode) {
            broadcastLocation();
        }
    }

    private void setDisplayOptionChanged() {
        m_displayOptionChanged = true;
    }

    //Check zoom change value and reset.
    public synchronized boolean testDisplayOptionChanged() {
        if (m_displayOptionChanged) {
            m_displayOptionChanged = false;
            return true;
        }
        return false;
    }

    private void setNormalizationTypeChanged() {
        m_normalizationTypeChanged = true;
    }

    //Check zoom change value and reset.
    public synchronized boolean testNormalizationTypeChanged() {
        if (m_normalizationTypeChanged) {
            m_normalizationTypeChanged = false;
            return true;
        }
        return false;
    }

    public NormalizationType getNormalizationType() {
        return normalizationType;
    }

    public void setNormalizationType(NormalizationType option) {
        if (this.normalizationType != option) {
            this.normalizationType = option;
            setNormalizationTypeChanged();
        }
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



    /**
     * Change zoom level and recenter.  Triggered by the resolutionSlider, or by a double-click in the
     * heatmap panel.
     */
    //reloading the previous location
    public void setLocation(String chrXName, String chrYName, String unitName, int binSize, double xOrigin, double yOrigin, double scalefactor) {

        if (!chrXName.equals(xContext.getChromosome().getName()) || !chrYName.equals(yContext.getChromosome().getName())) {

            Chromosome chrX = HiCFileTools.getChromosomeNamed(chrXName, chromosomes);
            Chromosome chrY = HiCFileTools.getChromosomeNamed(chrYName, chromosomes);

            if (chrX == null || chrY == null) {
                //Chromosomes do not appear to exist in current map.
                log.info("Chromosome(s) not found.");
                log.info("Most probably origin is a different species saved location or sync/link between two different species maps.");
                return;
            }

            this.xContext = new Context(chrX);
            this.yContext = new Context(chrY);
            superAdapter.setSelectedChromosomesNoRefresh(chrX, chrY);
            if (eigenvectorTrack != null) {
                eigenvectorTrack.forceRefresh();
            }
        }

        // if (!newZoom.equals(zoom) || (xContext.getZoom() == null) || (yContext.getZoom() == null))
        HiCZoom newZoom = new HiCZoom(Unit.valueOf(unitName), binSize);
        zoom = newZoom;
        xContext.setZoom(zoom);
        yContext.setZoom(zoom);
        setZoom(newZoom, xOrigin, yOrigin);
        superAdapter.updateZoom(newZoom);

        setScaleFactor(scalefactor);
        xContext.setBinOrigin(xOrigin);
        yContext.setBinOrigin(yOrigin);

        try {
            superAdapter.refresh();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void broadcastLocation() {
        String command = getLocationDescription();
        CommandBroadcaster.broadcast(command);
    }

    public String getLocationDescription() {
        String xChr = xContext.getChromosome().getName();
        String yChr = yContext.getChromosome().getName();

        if (!(xChr.toLowerCase().contains("chr"))) xChr = "chr" + xChr;
        if (!(yChr.toLowerCase().contains("chr"))) yChr = "chr" + yChr;

        return "setstate " + xChr + " " + yChr + " " + zoom.getUnit().toString() + " " + zoom.getBinSize() + " " +
                xContext.getBinOrigin() + " " + yContext.getBinOrigin() + " " + getScaleFactor();
    }

    public String getDefaultLocationDescription() {

        String xChr = xContext.getChromosome().getName();
        String yChr = yContext.getChromosome().getName();

        if (!(xChr.toLowerCase().contains("chr"))) xChr = "chr" + xChr;
        if (!(yChr.toLowerCase().contains("chr"))) yChr = "chr" + yChr;

        return xChr + "@" + (long) (xContext.getBinOrigin() * zoom.getBinSize()) + "_" +
                yChr + "@" + (long) (yContext.getBinOrigin() * zoom.getBinSize());
    }


    public void restoreLocation(String cmd) {
        CommandExecutor cmdExe = new CommandExecutor(this);
        cmdExe.execute(cmd);
        if (linkedMode) {
            broadcastLocation();
        }
    }

    public int validateBinSize(String key) {
        if (binSizeDictionary.containsKey(key)) {
            return Integer.valueOf(String.valueOf(binSizeDictionary.get(key)));
        } else {
            return Integer.MIN_VALUE;
        }
    }

    private void initBinSizeDictionary() {
        // TODO remove magic strings or move this to hicglobals?
        //BP Bin size:
        binSizeDictionary.put("2.5M", 2500000);
        binSizeDictionary.put("1M", 1000000);
        binSizeDictionary.put("500K", 500000);
        binSizeDictionary.put("250K", 250000);
        binSizeDictionary.put("100K", 100000);
        binSizeDictionary.put("50K", 50000);
        binSizeDictionary.put("25K", 25000);
        binSizeDictionary.put("10K", 10000);
        binSizeDictionary.put("5K", 5000);
        binSizeDictionary.put("1K", 1000);
        binSizeDictionary.put("2.5m", 2500000);
        binSizeDictionary.put("1m", 1000000);
        binSizeDictionary.put("500k", 500000);
        binSizeDictionary.put("250k", 250000);
        binSizeDictionary.put("100k", 100000);
        binSizeDictionary.put("50k", 50000);
        binSizeDictionary.put("25k", 25000);
        binSizeDictionary.put("10k", 10000);
        binSizeDictionary.put("5k", 5000);
        binSizeDictionary.put("1k", 1000);
        binSizeDictionary.put("2500000", 2500000);
        binSizeDictionary.put("1000000", 1000000);
        binSizeDictionary.put("500000", 500000);
        binSizeDictionary.put("250000", 250000);
        binSizeDictionary.put("100000", 100000);
        binSizeDictionary.put("50000", 50000);
        binSizeDictionary.put("25000", 25000);
        binSizeDictionary.put("10000", 10000);
        binSizeDictionary.put("5000", 5000);
        binSizeDictionary.put("1000", 1000);

        //FRAG Bin size:
        binSizeDictionary.put("500f", 500);
        binSizeDictionary.put("200f", 200);
        binSizeDictionary.put("100f", 100);
        binSizeDictionary.put("50f", 50);
        binSizeDictionary.put("20f", 20);
        binSizeDictionary.put("5f", 5);
        binSizeDictionary.put("2f", 2);
        binSizeDictionary.put("1f", 1);
    }

    public void setShowLoops(boolean showLoops) {
        feature2DHandler.setShowLoops(showLoops);
    }

    public void setLoopsInvisible(String path) {
        feature2DHandler.setLoopsInvisible(path);
    }

    public void loadLoopList(String path) {
        feature2DHandler.loadLoopList(path, chromosomes);
    }

    public List<Feature2DList> getAllVisibleLoopLists() {
        return feature2DHandler.getAllVisibleLoopLists();
    }

    public List<Feature2D> getVisibleFeatures(int chrIdx1, int chrIdx2) {
        return feature2DHandler.getVisibleFeatures(chrIdx1, chrIdx2);
    }

    public List<Feature2D> findNearbyFeatures(MatrixZoomData zd, int chrIdx1, int chrIdx2, int x, int y, int n) {

        double binOriginX = getXContext().getBinOrigin();
        double binOriginY = getYContext().getBinOrigin();
        double scale = getScaleFactor();

        return feature2DHandler.findNearbyFeatures(zd, chrIdx1, chrIdx2, x, y, n, binOriginX, binOriginY, scale);
    }

    public List<Pair<Rectangle, Feature2D>> findNearbyFeaturePairs(MatrixZoomData zd, int chrIdx1, int chrIdx2, int x,
                                                                   int y, int n) {
        double binOriginX = getXContext().getBinOrigin();
        double binOriginY = getYContext().getBinOrigin();
        double scale = getScaleFactor();

        return feature2DHandler.findNearbyFeaturePairs(zd, chrIdx1, chrIdx2, x, y, n, binOriginX, binOriginY, scale);
    }

    public void setSparseFeaturePlotting(boolean status) {
        feature2DHandler.setSparseFeaturePlotting(status);
    }

    public void enlarge2DFeaturePlotting(boolean status) {
        feature2DHandler.enlarge2DFeaturePlotting(status);
    }

    public void toggleFeatureOpacity(boolean status) {
        feature2DHandler.toggleFeatureOpacity(status);
    }

    public void removeLoadedAnnotation(String path) {

        feature2DHandler.removeFeaturePath(path);
    }

    public enum Unit {BP, FRAG}
}

