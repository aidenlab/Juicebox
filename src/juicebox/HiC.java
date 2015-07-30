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
import juicebox.encode.EncodeFileBrowser;
import juicebox.mapcolorui.HeatmapRenderer;
import juicebox.state.ReadXMLForReload;
import juicebox.state.ReloadPreviousState;
import juicebox.state.XMLForReloadState;
import juicebox.tools.utils.common.HiCFileTools;
import juicebox.track.*;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;
import org.apache.log4j.Logger;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This is the "model" class for the HiC viewer.
 *
 * @author Jim Robinson
 * @since 4/8/12
 */
public class HiC {

    private static final Logger log = Logger.getLogger(HiC.class);
    private static final Splitter MY_SPLITTER = Splitter.on(CharMatcher.BREAKING_WHITESPACE).trimResults().omitEmptyStrings();
    private static String mapName;
    private static String stateID;
    private static String mapPath;
    final File currentStates = new File(HiCGlobals.stateFileName);
    private final MainWindow mainWindow;
    private final Map<String, Feature2DList> loopLists;
    private final HiCTrackManager trackManager;
    private final File JuiceboxStatesXML = new File("JuiceboxStatesXML.txt");
    private final HashMap<String, Integer> binSizeDictionary = new HashMap<String, Integer>();
    File currentStatesToXML = new File(HiCGlobals.xmlFileName);
    private double scaleFactor;
    private String xPosition;
    private String yPosition;
    private MatrixType displayOption;
    private NormalizationType normalizationType;
    private java.util.List<Chromosome> chromosomes;
    private Dataset dataset;
    private Dataset controlDataset;
    private HiCZoom zoom;
    private MatrixZoomData matrixForReloadState;
    private Context xContext;
    private Context yContext;
    private boolean showLoops;
    private HeatmapRenderer heatmapRenderer;
    private List<HiCTrack> trackLabels;
    private EncodeFileBrowser encodeFileBrowser;
    private TrackConfigDialog configDialog;
    private HiCTrack hiCTrack;
    private EigenvectorTrack eigenvectorTrack;
    private ResourceTree resourceTree;
    private LoadEncodeAction encodeAction;
    private Point cursorPoint;
    private Point selectedBin;
    private boolean linkedMode;
    private boolean m_zoomChanged;
    private boolean m_displayOptionChanged;
    private boolean m_normalizationTypeChanged;

    public HiC(MainWindow mainWindow) {
        this.mainWindow = mainWindow;
        this.trackManager = new HiCTrackManager(mainWindow, this);
        this.loopLists = new HashMap<String, Feature2DList>();
        this.m_zoomChanged = false;
        this.m_displayOptionChanged = false;
        this.m_normalizationTypeChanged = false;
        initBinSizeDictionary();
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

    public void clearTracksForReloadState(){

        ArrayList<HiCTrack> tracksToRemove = new ArrayList<HiCTrack>(trackManager.getLoadedTracks());
        for(HiCTrack trackToRemove : tracksToRemove){
            if(trackToRemove.getName().equals("eigenvector")){
                eigenvectorTrack = null;
            }else {
                trackManager.removeTrack(trackToRemove);
            }
        }
        trackManager.clearTracks();
        loopLists.clear();
        mainWindow.updateTrackPanel();
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

    public LoadEncodeAction getEncodeAction() {
        return encodeAction;
    }

    public void setEncodeAction(LoadEncodeAction eAction) {
        encodeAction = eAction;
    }

    public List<Feature2DList> getAllVisibleLoopLists() {
        if (!showLoops) return null;
        List<Feature2DList> visibleLoopList = new ArrayList<Feature2DList>();
        for (Feature2DList list : loopLists.values()) {
            if (list.isVisible()) {
                visibleLoopList.add(list);
            }
        }
        return visibleLoopList;
    }

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

    private MatrixZoomData setZoomDataForReloadState(HiCZoom newZoom, Chromosome crX, Chromosome crY) {

        Matrix matrix = dataset.getMatrix(crX,crY);
        matrixForReloadState = matrix.getZoomData(newZoom);
        return matrix.getZoomData(newZoom);
    }

    public MatrixZoomData getZoomDataForReloadState(){
        return matrixForReloadState;
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
        return dataset == null || xContext == null ? null : getDataset().getMatrix(xContext.getChromosome(), yContext.getChromosome());

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
        if (!mainWindow.isResolutionLocked()) {
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


        mainWindow.updateZoom(zoom);

        setScaleFactor(scaleFactor);

        xContext.setBinOrigin(binX0);
        yContext.setBinOrigin(binY0);


        if (linkedMode) {
            broadcastState();
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
                JOptionPane.showMessageDialog(mainWindow, error.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            }

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

    public void loadLoopList(String path) {

        if (loopLists.get(path) != null) {
            loopLists.get(path).setVisible(true);
            return;
        }

        Feature2DList newList = Feature2DParser.parseLoopFile(path, chromosomes, false, 0, 0, 0, true, null);
        loopLists.put(path, newList);
    }

    /**
     * Change zoom level and recenter.  Triggered by the resolutionSlider, or by a double-click in the
     * heatmap panel.
     */
    //reloading the previous location
    public void setState(String chrXName, String chrYName, String unitName, int binSize, double xOrigin, double yOrigin, double scalefactor) {

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
            mainWindow.setSelectedChromosomesNoRefresh(chrX, chrY);
            if (eigenvectorTrack != null) {
                eigenvectorTrack.forceRefresh();
            }
        }

        HiCZoom newZoom = new HiCZoom(Unit.valueOf(unitName), binSize);
        if (!newZoom.equals(zoom) || (xContext.getZoom() == null) || (yContext.getZoom() == null)) {
            zoom = newZoom;
            xContext.setZoom(zoom);
            yContext.setZoom(zoom);
            mainWindow.updateZoom(newZoom);
        }

        setScaleFactor(scalefactor);
        xContext.setBinOrigin(xOrigin);
        yContext.setBinOrigin(yOrigin);

        try {
            mainWindow.refresh();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private void storeMapName() {
        mapName=mainWindow.getRecentMapMenu().getRecentMapName();
    }
    public void storeStateID(){
        stateID=mainWindow.getPrevousStateMenu().getRecentMapName();
    }
    public String getMapPath(String currentMapPath){
        mapPath = currentMapPath;
        return mapPath;
    }
    public String currentMapName(){
        return mapPath;
    }

    private void resetMap(String[] temp) {
        boolean control = isControlLoaded();
        List<String> files = new ArrayList<String>();
        files.add(temp[1]);
        System.out.println(mainWindow.getTitle());
        if(!mainWindow.getTitle().contains(temp[0])){
            mainWindow.safeLoadForReloadState(files, control, temp[0]);
        }
    }
    //reloading the previous state
    // TODO--Use XML File instead
    public void safeSetReloadState(final String mapURL , final String chrXName, final String chrYName, final String unitName, final int binSize,
                                   final double xOrigin, final double yOrigin, final double scalefactor,
                                   final MatrixType displaySelection, final NormalizationType normSelection, final double minColor,
                                   final double lowColor, final double upColor, final double maxColor, final ArrayList<String> trackNames){
        Runnable runnable = new Runnable() {
            public void run() {
                try {
                    unsafeSetReloadState(mapURL , chrXName,  chrYName, unitName,binSize, xOrigin, yOrigin, scalefactor,
                            displaySelection, normSelection, minColor, lowColor, upColor, maxColor,trackNames);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        };
        mainWindow.executeLongRunningTask(runnable, "Mouse Click Set Chr");
    }

    private void unsafeSetReloadState(String mapURL, String chrXName, String chrYName, String unitName, int binSize,
                                      double xOrigin, double yOrigin, double scalefactor,
                                      MatrixType displaySelection, NormalizationType normSelection, double minColor, double lowColor,
                                      double upColor, double maxColor, ArrayList<String> trackNames) {

        boolean control = isControlLoaded();
        String delimeter = "@@";
        String[] temp = mapURL.split(delimeter);
        resetMap(temp);

        //if (!chrXName.equals(xContext.getChromosome().getName()) || !chrYName.equals(yContext.getChromosome().getName())) {

        Chromosome chrX = HiCFileTools.getChromosomeNamed(chrXName, chromosomes);
        Chromosome chrY = HiCFileTools.getChromosomeNamed(chrYName, chromosomes);

        if (chrX == null || chrY == null) {
            //Chromosomes do not appear to exist in current map.
            log.info("Chromosome(s) not found.");
            log.info("Most probably origin is a different species saved location or sync/link between two different species maps.");
            return;
        }
        setSelectedChromosomes(chrX,chrY);
        mainWindow.setSelectedChromosomesNoRefresh(chrX, chrY);
        if (eigenvectorTrack != null) {
            eigenvectorTrack.forceRefresh();
        }
        //}

        HiCZoom newZoom = new HiCZoom(Unit.valueOf(unitName), binSize);
        if (!newZoom.equals(zoom) || (xContext.getZoom() == null) || (yContext.getZoom() == null)) {
            setZoomDataForReloadState(newZoom,chrX,chrY);
            zoom = newZoom;
            xContext.setZoom(newZoom);
            yContext.setZoom(newZoom);
            mainWindow.updateZoom(newZoom);
        }

        setScaleFactor(scalefactor);
        xContext.setBinOrigin(xOrigin);
        yContext.setBinOrigin(yOrigin);
        mainWindow.setDisplayBox(displaySelection.ordinal());
        mainWindow.setNormalizationBox(normSelection.ordinal());
        mainWindow.updateColorSlider(minColor, lowColor, upColor, maxColor);

        LoadEncodeAction loadEncodeAction = new LoadEncodeAction("Check Encode boxes", mainWindow, this);
        LoadAction loadAction = new LoadAction("Check track boxes", mainWindow, this);

        if (!trackNames.isEmpty()) {
            //System.out.println("trackNames: " + trackNames); for debugging
            for (String currentTrackName : trackNames) {
                String[] tempTrackName = currentTrackName.split("\\*\\*\\*");
                if (tempTrackName[0].equals("Eigenvector")) {
                    loadEigenvectorTrack();
                } else if (tempTrackName[0].toLowerCase().contains("coverage") || tempTrackName[0].toLowerCase().contains("balanced")
                        || tempTrackName[0].equals("Loaded")) {
                    loadCoverageTrack(NormalizationType.enumValueFromString(tempTrackName[0]));
                } else if (tempTrackName[0].contains("peaks") || tempTrackName[0].contains("blocks") || tempTrackName[0].contains("superloop")) {
                    resourceTree.checkTrackBoxesForReloadState(tempTrackName[0]);
                    loadLoopList(tempTrackName[0]);
                } else if (currentTrackName.contains("goldenPath")||currentTrackName.toLowerCase().contains("ensembl")) {
                    loadTrack(tempTrackName[0]);
                    loadEncodeAction.checkEncodeBoxes(tempTrackName[1]);
                } else {
                    loadTrack(tempTrackName[0]);
                }
                //renaming
                for(HiCTrack loadedTrack: getLoadedTracks()){
                    if(tempTrackName[0].contains(loadedTrack.getName())){
                        loadedTrack.setName(tempTrackName[1]);
                    }
                    loadAction.checkBoxesForReload(tempTrackName[1]);
                }
            }

        }
        mainWindow.updateTrackPanel();
    }


    public void unsafeSetReloadStateFromXML(String[] initialInfo, int binSize, double[] doubleInfo,
                                            MatrixType displaySelection, NormalizationType normSelection,
                                            String[] tracks){
        String mapName = initialInfo[0];
        String mapURL = initialInfo[1];
        String chrXName = initialInfo[2];
        String chrYName = initialInfo[3];
        String unitName = initialInfo[4];
        double xOrigin = doubleInfo[0];
        double yOrigin = doubleInfo[1];
        double scalefactor = doubleInfo[2];
        double minColor = doubleInfo[3];
        double lowColor = doubleInfo[4];
        double upColor = doubleInfo[5];
        double maxColor = doubleInfo[6];

        boolean control = isControlLoaded();
        String[] temp = new String[2];
        temp[0] = mapName;
        temp[1] = mapURL;
        resetMap(temp);

        //if (!chrXName.equals(xContext.getChromosome().getName()) || !chrYName.equals(yContext.getChromosome().getName())) {

        Chromosome chrX = HiCFileTools.getChromosomeNamed(chrXName, chromosomes);
        Chromosome chrY = HiCFileTools.getChromosomeNamed(chrYName, chromosomes);

        if (chrX == null || chrY == null) {
            //Chromosomes do not appear to exist in current map.
            log.info("Chromosome(s) not found.");
            log.info("Most probably origin is a different species saved location or sync/link between two different species maps.");
            JOptionPane.showMessageDialog(mainWindow, "Error:\n" + "Chromosome(s) not found. Please check chromosome" , "Error",
                    JOptionPane.ERROR_MESSAGE);
            return;
        }
        setSelectedChromosomes(chrX,chrY);
        mainWindow.setSelectedChromosomesNoRefresh(chrX, chrY);
        if (eigenvectorTrack != null) {
            eigenvectorTrack.forceRefresh();
        }
        //}

        HiCZoom newZoom = new HiCZoom(Unit.valueOf(unitName), binSize);
        if (!newZoom.equals(zoom) || (xContext.getZoom() == null) || (yContext.getZoom() == null)) {
            setZoomDataForReloadState(newZoom,chrX,chrY);
            zoom = newZoom;
            xContext.setZoom(newZoom);
            yContext.setZoom(newZoom);
            mainWindow.updateZoom(newZoom);
        } else{
            JOptionPane.showMessageDialog(mainWindow, "Error:\n" + "Please check zoom data" , "Error",
                    JOptionPane.ERROR_MESSAGE);
        }

        setScaleFactor(scalefactor);
        xContext.setBinOrigin(xOrigin);
        yContext.setBinOrigin(yOrigin);
        mainWindow.setDisplayBox(displaySelection.ordinal());
        mainWindow.setNormalizationBox(normSelection.ordinal());
        mainWindow.updateColorSlider(minColor, lowColor, upColor, maxColor);

        LoadEncodeAction loadEncodeAction = new LoadEncodeAction("Check Encode boxes", mainWindow, this);
        LoadAction loadAction = new LoadAction("Check track boxes", mainWindow, this);

        String[] trackURLs = tracks[0].split("\\,");
        String[] trackNames = tracks[1].split("\\,");

        try {
            if (tracks.length > 0 && !tracks[1].contains("none")) {
                //System.out.println("trackNames: " + trackNames); for debugging
                for(int i=0; i<trackURLs.length; i++) {
                        String currentTrack = trackURLs[i].trim();
                        if (!currentTrack.isEmpty()) {
                            if (currentTrack.equals("Eigenvector")) {
                                loadEigenvectorTrack();
                            } else if (currentTrack.toLowerCase().contains("coverage") || currentTrack.toLowerCase().contains("balanced")
                                    || currentTrack.equals("Loaded")) {
                                loadCoverageTrack(NormalizationType.enumValueFromString(currentTrack));
                            } else if (currentTrack.contains("peaks") || currentTrack.contains("blocks") || currentTrack.contains("superloop")) {
                                resourceTree.checkTrackBoxesForReloadState(currentTrack.trim());
                                loadLoopList(currentTrack);
                            } else if (currentTrack.contains("goldenPath") || currentTrack.toLowerCase().contains("ensembl")) {
                                loadTrack(currentTrack);
                                loadEncodeAction.checkEncodeBoxes(trackNames[i].trim());
                            } else {
                                loadTrack(currentTrack);
                                loadAction.checkBoxesForReload(trackNames[i].trim());
                            }
                            //renaming
                    }
                }
                for(HiCTrack loadedTrack: getLoadedTracks()){
                    for(int i = 0; i<trackNames.length; i++){
                        if(trackURLs[i].contains(loadedTrack.getName())){
                            loadedTrack.setName(trackNames[i].trim());
                        }
                    }
                }

            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        mainWindow.updateTrackPanel();

    }

    public void broadcastState() {
        String xChr = xContext.getChromosome().getName();
        String yChr = yContext.getChromosome().getName();

        if (!(xChr.toLowerCase().contains("chr"))) {
            xChr = "chr" + xChr;
        }
        if (!(yChr.toLowerCase().contains("chr"))) {
            yChr = "chr" + yChr;
        }

        String command = "setstate " +
                xChr + " " +
                yChr + " " +
                zoom.getUnit().toString() + " " +
                zoom.getBinSize() + " " +
                xContext.getBinOrigin() + " " +
                yContext.getBinOrigin() + " " +
                getScaleFactor();

        CommandBroadcaster.broadcast(command);
    }

    public String saveState() {
        String xChr = xContext.getChromosome().getName();
        String yChr = yContext.getChromosome().getName();

        if (!(xChr.toLowerCase().contains("chr"))) {
            xChr = "chr" + xChr;
        }
        if (!(yChr.toLowerCase().contains("chr"))) {
            yChr = "chr" + yChr;
        }

        return "setstate " +
                xChr + " " +
                yChr + " " +
                zoom.getUnit().toString() + " " +
                zoom.getBinSize() + " " +
                xContext.getBinOrigin() + " " +
                yContext.getBinOrigin() + " " +
                getScaleFactor();
        // CommandBroadcaster.broadcast(command);
    }
    // Creating XML file
    private void createXMLForReload(File tempState) {
        XMLForReloadState xml = new XMLForReloadState();
        xml.begin();
    }

    public void readXML(String mapPath){
        ReadXMLForReload readFile = new ReadXMLForReload(this);
        readFile.readXML(HiCGlobals.xmlFileName, mapPath);
    }

    public void writeState() {
        try {
            BufferedWriter buffWriter = new BufferedWriter(new FileWriter(currentStates,true));
            String xChr = xContext.getChromosome().getName();
            String yChr = yContext.getChromosome().getName();
            String colorVals = mainWindow.getColorRangeValues();
            List<HiCTrack> currentTracks = getLoadedTracks();
            String currentTrack = "";
            storeMapName();
            buffWriter.newLine();

            //tracks true & loops true
            if(currentTracks!=null && !currentTracks.isEmpty() && getAllVisibleLoopLists()!=null && !getAllVisibleLoopLists().isEmpty()) {

                for(HiCTrack track: currentTracks) {
                    //System.out.println("trackLocator: "+track.getLocator()); for debugging
                    //System.out.println("track name: " + track.getName());
                    currentTrack+="$$"+track.getLocator()+"***"+track.getName();
                }

                buffWriter.write(stateID + "--currentState:$$" + mapName + "$$" + xChr + "$$" + yChr + "$$" + zoom.getUnit().toString() + "$$" +
                        zoom.getBinSize() + "$$" + xContext.getBinOrigin() + "$$" + yContext.getBinOrigin() + "$$" +
                        getScaleFactor() + "$$" + displayOption.name() + "$$" + getNormalizationType().name()
                        + "$$" + colorVals + currentTrack + "$$" + dataset.getPeaks().toString() + "$$" + dataset.getBlocks().toString() + "$$" + dataset.getSuperLoops().toString());
            }//tracks true & loops false
            else if(currentTracks!=null && !currentTracks.isEmpty()) {

                for(HiCTrack track: currentTracks) {
                    //System.out.println("trackLocator: "+track.getLocator()); for debugging
                    //System.out.println("track name: "+track.getName());
                    currentTrack+="$$"+track.getLocator()+"***"+track.getName();
                }

                buffWriter.write(stateID+"--currentState:$$"+ mapName + "$$" + xChr + "$$" + yChr + "$$" + zoom.getUnit().toString() + "$$" +
                        zoom.getBinSize() + "$$" + xContext.getBinOrigin() + "$$" + yContext.getBinOrigin() + "$$" +
                        getScaleFactor() + "$$" + displayOption.name() + "$$" + getNormalizationType().name()
                        + "$$" + colorVals + currentTrack);
                //loops true & tracks false
            } else if(getAllVisibleLoopLists()!=null && !getAllVisibleLoopLists().isEmpty()){

                //System.out.println(dataset.getPeaks().toString());
                //System.out.println(dataset.getBlocks().toString()); for debugging
                buffWriter.write(stateID+"--currentState:$$"+ mapName + "$$" + xChr + "$$" + yChr + "$$" + zoom.getUnit().toString() + "$$" +
                        zoom.getBinSize() + "$$" + xContext.getBinOrigin() + "$$" + yContext.getBinOrigin() + "$$" +
                        getScaleFactor() + "$$" + displayOption.name() + "$$" + getNormalizationType().name()
                        + "$$" + colorVals + "$$" + dataset.getPeaks().toString() + "$$" + dataset.getBlocks().toString() + "$$" + dataset.getSuperLoops().toString());

            }
            else{ //false & false
                buffWriter.write(stateID+"--currentState:$$"+ mapName + "$$" + xChr + "$$" + yChr + "$$" + zoom.getUnit().toString() + "$$" +
                        zoom.getBinSize() + "$$" + xContext.getBinOrigin() + "$$" + yContext.getBinOrigin() + "$$" +
                        getScaleFactor() + "$$" + displayOption.name() + "$$" + getNormalizationType().name()
                        + "$$" + colorVals);
            }

            //("currentState,xChr,yChr,resolution,zoom level,xbin,ybin,scale factor,display selection,
            // normalization type,color range values, tracks")
            buffWriter.close();
            System.out.println("stuff saved"); //check
            createXMLForReload(currentStates);

        }catch (IOException e){
            e.printStackTrace();
        }
    }

    public void writeStateForXML() {
        try {
            BufferedWriter buffWriter = new BufferedWriter(new FileWriter(JuiceboxStatesXML,true));
            String xChr = xContext.getChromosome().getName();
            String yChr = yContext.getChromosome().getName();
            String colorVals = mainWindow.getColorRangeValues();
            List<HiCTrack> currentTracks = getLoadedTracks();
            String currentTrack = "";
            String currentTrackName = "";
            storeMapName();
            buffWriter.newLine();

            //tracks true & loops true
            if(currentTracks!=null && !currentTracks.isEmpty() && getAllVisibleLoopLists()!=null && !getAllVisibleLoopLists().isEmpty()) {

                for(HiCTrack track: currentTracks) {
                    //System.out.println("trackLocator: "+track.getLocator()); for debugging
                    System.out.println("track name: " + track.getName());
                    currentTrack+=track.getLocator()+", ";
                    currentTrackName+=track.getName()+", ";
                }

                buffWriter.write(stateID + "--currentState:$$" + mapName + "$$" + xChr + "$$" + yChr + "$$" + zoom.getUnit().toString() + "$$" +
                        zoom.getBinSize() + "$$" + xContext.getBinOrigin() + "$$" + yContext.getBinOrigin() + "$$" +
                        getScaleFactor() + "$$" + displayOption.name() + "$$" + getNormalizationType().name()
                        + "$$" + colorVals +  "$$" + currentTrack + "$$" + currentTrackName + "$$" + dataset.getPeaks().toString() + "$$" + dataset.getBlocks().toString() + "$$" + dataset.getSuperLoops().toString());
            }//tracks true & loops false
            else if(currentTracks!=null && !currentTracks.isEmpty()) {

                for(HiCTrack track: currentTracks) {
                    //System.out.println("trackLocator: "+track.getLocator()); for debugging
                    System.out.println("track name: "+track.getName());
                    currentTrack+=track.getLocator()+", ";
                    currentTrackName+=track.getName()+", ";
                }

                buffWriter.write(stateID+"--currentState:$$"+ mapName + "$$" + xChr + "$$" + yChr + "$$" + zoom.getUnit().toString() + "$$" +
                        zoom.getBinSize() + "$$" + xContext.getBinOrigin() + "$$" + yContext.getBinOrigin() + "$$" +
                        getScaleFactor() + "$$" + displayOption.name() + "$$" + getNormalizationType().name()
                        + "$$" + colorVals + "$$" + currentTrack + "$$" + currentTrackName);
                //loops true & tracks false
            } else if(getAllVisibleLoopLists()!=null && !getAllVisibleLoopLists().isEmpty()){

                //System.out.println(dataset.getPeaks().toString());
                //System.out.println(dataset.getBlocks().toString()); for debugging
                buffWriter.write(stateID+"--currentState:$$"+ mapName + "$$" + xChr + "$$" + yChr + "$$" + zoom.getUnit().toString() + "$$" +
                        zoom.getBinSize() + "$$" + xContext.getBinOrigin() + "$$" + yContext.getBinOrigin() + "$$" +
                        getScaleFactor() + "$$" + displayOption.name() + "$$" + getNormalizationType().name()
                        + "$$" + colorVals + "$$" + dataset.getPeaks().toString() + "$$" + dataset.getBlocks().toString() + "$$" + dataset.getSuperLoops().toString());

            }
            else{ //false & false
                buffWriter.write(stateID+"--currentState:$$"+ mapName + "$$" + xChr + "$$" + yChr + "$$" + zoom.getUnit().toString() + "$$" +
                        zoom.getBinSize() + "$$" + xContext.getBinOrigin() + "$$" + yContext.getBinOrigin() + "$$" +
                        getScaleFactor() + "$$" + displayOption.name() + "$$" + getNormalizationType().name()
                        + "$$" + colorVals);
            }

            //("currentState,xChr,yChr,resolution,zoom level,xbin,ybin,scale factor,display selection,
            // normalization type,color range values, tracks")
            buffWriter.close();
            System.out.println("stuff saved"); //check
            createXMLForReload(JuiceboxStatesXML);

        }catch (IOException e){
            e.printStackTrace();
        }
    }

    public String getDefaultLocationDescription() {

        String xChr = xContext.getChromosome().getName();
        String yChr = yContext.getChromosome().getName();

        if (!(xChr.toLowerCase().contains("chr"))) {
            xChr = "chr" + xChr;
        }
        if (!(yChr.toLowerCase().contains("chr"))) {
            yChr = "chr" + yChr;
        }

        return xChr + "@" +
                (long) (xContext.getBinOrigin() * zoom.getBinSize()) + "_" +
                yChr + "@" +
                (long) (yContext.getBinOrigin() * zoom.getBinSize());
        // CommandBroadcaster.broadcast(command);
    }

    public void reloadPreviousState(File tempState){
        ReloadPreviousState rld = new ReloadPreviousState(this);
        rld.reload(tempState);
    }

    public void reloadPreviousStateFromXML(String[] infoToReload){
        ReloadPreviousState rld = new ReloadPreviousState(this);
        rld.reloadXML(infoToReload);
    }
    public void restoreState(String cmd) {
        CommandExecutor cmdExe = new CommandExecutor(this);
        cmdExe.execute(cmd);
        if (linkedMode) {
            broadcastState();
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

    public enum Unit {BP, FRAG}
}

