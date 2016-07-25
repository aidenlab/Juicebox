/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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
import oracle.net.jdbc.nl.UninitializedObjectException;
import org.apache.log4j.Logger;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.ui.util.MessageUtils;
import org.broad.igv.util.Pair;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
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
    private final String eigString = "Eigenvector";
    private final String ctrlEigString = "Ctrl_Eigenvector";
    private double scaleFactor;
    private String xPosition;
    private String yPosition;
    private MatrixType displayOption;
    private NormalizationType normalizationType;
    private List<Chromosome> chromosomes;
    private Dataset dataset;
    private Dataset controlDataset;
    private HiCZoom currentZoom;
    //private MatrixZoomData matrixForReloadState;
    private Context xContext;
    private Context yContext;
    private EigenvectorTrack eigenvectorTrack, controlEigenvectorTrack;
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

    /**
     * @param string
     * @return string with replacements of 000000 with M and 000 with K
     */
    private static String cleanUpNumbersInName(String string) {
        string = (new StringBuilder(string)).reverse().toString();
        string = string.replaceAll("000000", "M").replaceAll("000", "K");
        return (new StringBuilder(string)).reverse().toString();
    }

    public static HiC.Unit valueOfUnit(String unit) {
        if (unit.equalsIgnoreCase(Unit.BP.toString())) {
            return Unit.BP;
        } else if (unit.equalsIgnoreCase(Unit.FRAG.toString())) {
            return Unit.FRAG;
        }
        return null;
    }

    public void reset() {
        dataset = null;
        resetContexts();
        chromosomes = null;
        eigenvectorTrack = null;
        controlEigenvectorTrack = null;
        resourceTree = null;
        encodeAction = null;
        normalizationType = NormalizationType.NONE;
        clearFeatures();
    }

    // TODO zgire - why iterate through tracksToRemove if you end up calling clearFeatures() at the end?
    public void clearTracksForReloadState() {
        ArrayList<HiCTrack> tracksToRemove = new ArrayList<HiCTrack>(trackManager.getLoadedTracks());
        for (HiCTrack trackToRemove : tracksToRemove) {
            if (trackToRemove.getName().equals(eigString)) {
                eigenvectorTrack = null;
            } else if (trackToRemove.getName().equals(ctrlEigString)) {
                controlEigenvectorTrack = null;
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
        this.scaleFactor = Math.max(Math.min(50, scaleFactor), 1e-10);
    }

    public void setControlDataset(Dataset controlDataset) {
        this.controlDataset = controlDataset;
    }

    public void loadEigenvectorTrack() {
        if (eigenvectorTrack == null) {
            eigenvectorTrack = new EigenvectorTrack(eigString, eigString, this, false);
        }
        if (controlEigenvectorTrack == null && isControlLoaded()) {
            controlEigenvectorTrack = new EigenvectorTrack(ctrlEigString, ctrlEigString, this, true);
        }
        trackManager.add(eigenvectorTrack);
        if (controlEigenvectorTrack != null && isControlLoaded()) {
            trackManager.add(controlEigenvectorTrack);
        }
    }

    private void refreshEigenvectorTrackIfExists() {
        if (eigenvectorTrack != null) {
            eigenvectorTrack.forceRefresh();
        }
        if (controlEigenvectorTrack != null) {
            controlEigenvectorTrack.forceRefresh();
        }
    }

    public ResourceTree getResourceTree() {
        return resourceTree;
    }

    public void setResourceTree(ResourceTree rTree) {
        resourceTree = rTree;
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

    public void unsafeLoadHostedTracks(List<ResourceLocator> locators) {
        trackManager.unsafeTrackLoad(locators);
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

    public void moveTrack(HiCTrack track, boolean thisShouldBeMovedUp) {
        if (thisShouldBeMovedUp) {
            //move the track up
            trackManager.moveTrackUp(track);
        } else {
            //move the track down
            trackManager.moveTrackDown(track);
        }
    }

    public List<HiCTrack> getLoadedTrackList() {
        return trackManager.getLoadedTracks();

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
        refreshEigenvectorTrackIfExists();
    }

    public HiCZoom getZoom() {
        return currentZoom;
    }

    public MatrixZoomData getZd() throws UninitializedObjectException {
        Matrix matrix = getMatrix();
        if (matrix == null) {
            throw new UninitializedObjectException("Uninitialized matrix");
        } else if (currentZoom == null) {
            throw new UninitializedObjectException("Uninitialized zoom");
        } else {
            return matrix.getZoomData(currentZoom);
        }
    }

    public MatrixZoomData getControlZd() {
        Matrix matrix = getControlMatrix();
        if (matrix == null || currentZoom == null) {
            return null;
        } else {
            return matrix.getZoomData(currentZoom);
        }
    }

    public Matrix getControlMatrix() {
        if (controlDataset == null || xContext == null || currentZoom == null) return null;
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

    public int[] getCurrentRegionWindowGenomicPositions() {

        // address int overflow or exceeding bound issues
        int xEndEdge = xContext.getGenomicPositionOrigin() +
                (int) ((double) getZoom().getBinSize() * superAdapter.getHeatmapPanel().getWidth() / getScaleFactor());
        if (xEndEdge < 0 || xEndEdge > xContext.getChromosome().getLength()) {
            xEndEdge = xContext.getChromosome().getLength();
        }

        int yEndEdge = yContext.getGenomicPositionOrigin() +
                (int) ((double) getZoom().getBinSize() * superAdapter.getHeatmapPanel().getHeight() / getScaleFactor());
        if (yEndEdge < 0 || yEndEdge > yContext.getChromosome().getLength()) {
            yEndEdge = yContext.getChromosome().getLength();
        }

        return new int[]{xContext.getGenomicPositionOrigin(), xEndEdge, yContext.getGenomicPositionOrigin(), yEndEdge};
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
            //System.err.println("Dataset is null");
            return null;
        } else if (xContext == null) {
            //System.err.println("xContext is null");
            return null;
        } else if (yContext == null) {
            //System.err.println("yContext is null");
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
        return xContext != null && HiCFileTools.isAllChromosome(xContext.getChromosome());
    }

    public java.util.List<Chromosome> getChromosomes() {
        return chromosomes;
    }

    public void setChromosomes(List<Chromosome> chromosomes) {
        this.chromosomes = chromosomes;
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

    public void centerFragment(int fragmentX, int fragmentY) {
        if (currentZoom != null) {

            MatrixZoomData zd = getMatrix().getZoomData(currentZoom);
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
        if (currentZoom != null && getMatrix() != null) {
            MatrixZoomData zd = getMatrix().getZoomData(currentZoom);
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

        double h = superAdapter.getHeatmapPanel().getHeight() / getScaleFactor();  // view height in bins
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
        try {
            MatrixZoomData zd = getZd();

            final double wBins = (superAdapter.getHeatmapPanel().getWidth() / getScaleFactor());
            double maxX = zd.getXGridAxis().getBinCount() - wBins;

            final double hBins = (superAdapter.getHeatmapPanel().getHeight() / getScaleFactor());
            double maxY = zd.getYGridAxis().getBinCount() - hBins;

            double x = Math.max(0, Math.min(maxX, newBinX));
            double y = Math.max(0, Math.min(maxY, newBinY));

            xContext.setBinOrigin(x);
            yContext.setBinOrigin(y);

            superAdapter.repaint();
        } catch (Exception e) {
            e.printStackTrace();
        }

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

    public double[] getEigenvector(final int chrIdx, final int n, boolean isControl) {

        if (isControl) {
            if (controlDataset == null) return null;

            Chromosome chr = chromosomes.get(chrIdx);
            return controlDataset.getEigenvector(chr, currentZoom, n, normalizationType);
        } else {
            if (dataset == null) return null;

            Chromosome chr = chromosomes.get(chrIdx);
            return dataset.getEigenvector(chr, currentZoom, n, normalizationType);
        }
    }

    public ExpectedValueFunction getExpectedValues() {
        if (dataset == null) return null;
        return dataset.getExpectedValues(currentZoom, normalizationType);
    }

    public NormalizationVector getNormalizationVector(int chrIdx) {
        if (dataset == null) return null;
        return dataset.getNormalizationVector(chrIdx, currentZoom, normalizationType);
    }

    // Note - this is an inefficient method, used to support tooltip text only.
    public float getNormalizedObservedValue(int binX, int binY) {
        float val = Float.NaN;
        try {
            val = getZd().getObservedValue(binX, binY, normalizationType);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return val;
    }

    /**
     * Called from alt-drag zoom
     *
     * @param xBP0
     * @param yBP0
     * @param targetBinSize
     */
    public void zoomToDrawnBox(final int xBP0, final int yBP0, final double targetBinSize) {

        HiCZoom newZoom = currentZoom;
        if (!isResolutionLocked()) {
            List<HiCZoom> zoomList = currentZoom.getUnit() == HiC.Unit.BP ? dataset.getBpZooms() : dataset.getFragZooms();
            for (int i = zoomList.size() - 1; i >= 0; i--) {
                if (zoomList.get(i).getBinSize() >= targetBinSize) {
                    newZoom = zoomList.get(i);
                    break;
                }
            }

            // this addresses draw box to zoom when down from low res pearsons
            // it can't zoom all the way in, but can zoom in a little more up to 500K
            if (isInPearsonsMode() && newZoom.getBinSize() < HiCGlobals.MAX_PEARSON_ZOOM) {
                for (int i = zoomList.size() - 1; i >= 0; i--) {
                    if (zoomList.get(i).getBinSize() >= HiCGlobals.MAX_PEARSON_ZOOM) {
                        newZoom = zoomList.get(i);
                        break;
                    }
                }
            }
        }

        safeActuallySetZoomAndLocation(newZoom, xBP0, yBP0, newZoom.getBinSize() / targetBinSize, false,
                ZoomCallType.DRAG, "DragZoom", true);
    }

    /**
     * Triggered by syncs, goto, and load state.
     */
    //reloading the previous location
    public void setLocation(String chrXName, String chrYName, HiC.Unit unit, int binSize, double xOrigin,
                            double yOrigin, double scaleFactor, ZoomCallType zoomCallType, String message,
                            boolean allowLocationBroadcast) {

        HiCZoom newZoom = currentZoom;
        if (currentZoom.getBinSize() != binSize) {
            newZoom = new HiCZoom(unit, binSize);
        }
        safeActuallySetZoomAndLocation(chrXName, chrYName, newZoom, (int) xOrigin, (int) yOrigin, scaleFactor,
                true, zoomCallType, message, allowLocationBroadcast);
    }

    public void unsafeSetLocation(String chrXName, String chrYName, String unitName, int binSize, double xOrigin,
                                  double yOrigin, double scaleFactor, ZoomCallType zoomCallType, boolean allowLocationBroadcast) {

        HiCZoom newZoom = currentZoom;
        if (currentZoom.getBinSize() != binSize) {
            newZoom = new HiCZoom(HiC.valueOfUnit(unitName), binSize);
        }
        unsafeActuallySetZoomAndLocation(chrXName, chrYName, newZoom, (int) xOrigin, (int) yOrigin, scaleFactor,
                true, zoomCallType, allowLocationBroadcast);
    }

    private boolean safeActuallySetZoomAndLocation(HiCZoom newZoom, int genomeX, int genomeY, double scaleFactor,
                                                   boolean resetZoom, ZoomCallType zoomCallType, String message,
                                                   boolean allowLocationBroadcast) {
        return safeActuallySetZoomAndLocation("", "", newZoom, genomeX, genomeY, scaleFactor, resetZoom, zoomCallType,
                message, allowLocationBroadcast);
    }

    /*  TODO Undo Zoom implementation mss2 _UZI
     private boolean canUndoZoomChange = false;
     private boolean canRedoZoomChange = false;
     private ZoomState previousZoomState, tempZoomState;

     public boolean isCanUndoZoomChangeAvailable(){
         return canUndoZoomChange;
     }

     public boolean isCanRedoZoomChangeAvailable(){
         return canRedoZoomChange;
     }

     public void undoZoomChange(){
         if(canUndoZoomChange){
             System.err.println(previousZoomState);
             System.err.println(previousZoomState.loadZoomState());
             System.err.println(previousZoomState+"\n\n");
             // override when undoing zoom
             canUndoZoomChange = false;
             canRedoZoomChange = true;
         }

     }

     public void redoZoomChange(){
         if(canRedoZoomChange){
             System.err.println(previousZoomState);
             System.err.println(previousZoomState.loadZoomState());
             System.err.println(previousZoomState+"\n\n");
             // override when redoing zoom
             canRedoZoomChange = false;
             canUndoZoomChange = true;
         }
     }

        private class ZoomState {
         String chr1Name, chr2Name;
         HiCZoom zoom;
         int genomeX, genomeY;
         double scaleFactor;
         boolean resetZoom;
         ZoomCallType zoomCallType;

         ZoomState(String chr1Name, String chr2Name,
                   HiCZoom zoom, int genomeX, int genomeY, double scaleFactor,
                   boolean resetZoom, ZoomCallType zoomCallType){
             this.chr1Name = chr1Name;
             this.chr2Name = chr2Name;
             this.zoom = zoom;
             this.genomeX = genomeX;
             this.genomeY = genomeY;
             this.scaleFactor = scaleFactor;
             this.resetZoom = resetZoom;
             this.zoomCallType = zoomCallType;
         }

         boolean loadZoomState(){
             return actuallySetZoomAndLocation(chr1Name, chr2Name, zoom, genomeX, genomeY, scaleFactor, resetZoom, zoomCallType);
         }

         @Override
         public String toString(){
             return ""+chr1Name+" "+chr2Name+" "+zoom;
         }
      }
     */

    private boolean safeActuallySetZoomAndLocation(final String chrXName, final String chrYName,
                                                   final HiCZoom newZoom, final int genomeX, final int genomeY,
                                                   final double scaleFactor, final boolean resetZoom,
                                                   final ZoomCallType zoomCallType, String message,
                                                   final boolean allowLocationBroadcast) {
        final boolean[] returnVal = new boolean[1];
        superAdapter.executeLongRunningTask(new Runnable() {
            @Override
            public void run() {
                returnVal[0] = unsafeActuallySetZoomAndLocation(chrXName, chrYName, newZoom, genomeX, genomeY, scaleFactor,
                        resetZoom, zoomCallType, allowLocationBroadcast);
            }
        }, message);
        return returnVal[0];
    }

    /**
     * *************************************************************
     * Official Method for setting the zoom and location for heatmap
     * DO NOT IMPLEMENT A NEW FUNCTION
     * Make the necessary customizations, then call this function
     * *************************************************************
     *
     * @param newZoom
     * @param genomeX
     * @param genomeY
     * @param scaleFactor (pass -1 if scaleFactor should be calculated)
     * @return
     */
    public boolean unsafeActuallySetZoomAndLocation(String chrXName, String chrYName,
                                                    HiCZoom newZoom, int genomeX, int genomeY, double scaleFactor,
                                                    boolean resetZoom, ZoomCallType zoomCallType,
                                                    boolean allowLocationBroadcast) {


        if (dataset == null) return false;  // No data in view

        //Check this zoom operation is possible, if not, fail it here:
//        if (superAdapter.testNewZoom(newZoom))
//        {
//            return false;
//        }

        //String chr1OriginalName = xContext.getChromosome().getName();
        //String chr2OriginalName = yContext.getChromosome().getName();
        if (chrXName.length() > 0 && chrYName.length() > 0) {
            setChromosomesFromBroadcast(chrXName, chrYName);
            //We might end with All->All view, make sure normalization state is updates accordingly...
            superAdapter.getMainViewPanel().setNormalizationDisplayState(superAdapter.getHiC());
        }

        if (newZoom == null) {
            System.err.println("Invalid zoom " + newZoom);
        }

        Chromosome chr1 = xContext.getChromosome();
        Chromosome chr2 = yContext.getChromosome();
        final Matrix matrix = dataset.getMatrix(chr1, chr2);

        if (matrix == null) {
            superAdapter.launchGenericMessageDialog("Sorry, this region is not available", "Matrix unavailable",
                    JOptionPane.WARNING_MESSAGE);
            return false;
        }

        MatrixZoomData newZD = matrix.getZoomData(newZoom);
        if (HiCFileTools.isAllChromosome(chr1)) {
            newZD = matrix.getFirstZoomData(Unit.BP);
        }

        if (newZD == null) {
            superAdapter.launchGenericMessageDialog("Sorry, this zoom is not available", "Zoom unavailable",
                    JOptionPane.WARNING_MESSAGE);
            return false;
        }

        /* TODO Undo Zoom implementation mss2 _UZI
        if(currentZoom != null) {
            tempZoomState = new ZoomState(chr1OriginalName, chr2OriginalName, currentZoom.clone(), (int) xContext.getBinOrigin(),
                    (int) yContext.getBinOrigin(), getScaleFactor(), resetZoom, ZoomCallType.GOTO);
        }
        */

        currentZoom = newZoom;
        xContext.setZoom(currentZoom);
        yContext.setZoom(currentZoom);

        if (scaleFactor > 0) {
            setScaleFactor(scaleFactor);
        } else {
            int maxBinCount = Math.max(newZD.getXGridAxis().getBinCount(), newZD.getYGridAxis().getBinCount());
            double defaultScaleFactor = Math.max(1.0, (double) superAdapter.getHeatmapPanel().getMinimumDimension() / maxBinCount);
            setScaleFactor(defaultScaleFactor);
        }

        int binX = newZD.getXGridAxis().getBinNumberForGenomicPosition(genomeX);
        int binY = newZD.getYGridAxis().getBinNumberForGenomicPosition(genomeY);
        switch (zoomCallType) {
            case INITIAL:
            case STANDARD:
                center(binX, binY);
                break;
            case DRAG:
                xContext.setBinOrigin(binX);
                yContext.setBinOrigin(binY);
                break;
            case DIRECT:
                xContext.setBinOrigin(genomeX);
                yContext.setBinOrigin(genomeY);
                break;
        }

        // Notify HeatmapPanel render that zoom has changed. Render should update zoom slider once with previous range values

        setZoomChanged();
        if (resetZoom) {
            superAdapter.updateAndResetZoom(newZoom);
        } else {
            superAdapter.updateZoom(newZoom);
        }
        superAdapter.refresh();

        if (linkedMode && allowLocationBroadcast) {
            broadcastLocation();
        }
        /*
         TODO Undo Zoom implementation mss2 _UZI
         if(zoomCallType == ZoomCallType.INITIAL || tempZoomState == null || chrXName.equals(Globals.CHR_ALL) || chrYName.equals(Globals.CHR_ALL)
                 || tempZoomState.chr1Name.equals(Globals.CHR_ALL) || tempZoomState.chr2Name.equals(Globals.CHR_ALL)){
             canRedoZoomChange = false;
             canUndoZoomChange = false;
         }
         else {
             // defauts for a normal zoom operation
             canRedoZoomChange = false;
             canUndoZoomChange = true;
             previousZoomState = tempZoomState;
         }
         */

        return true;
    }

    private void setChromosomesFromBroadcast(String chrXName, String chrYName) {
        if (!chrXName.equals(xContext.getChromosome().getName()) || !chrYName.equals(yContext.getChromosome().getName())) {
            Chromosome chrX = HiCFileTools.getChromosomeNamed(chrXName, chromosomes);
            Chromosome chrY = HiCFileTools.getChromosomeNamed(chrYName, chromosomes);

            if (chrX == null || chrY == null) {
                //log.info("Most probably origin is a different species saved location or sync/link between two different species maps.");
                return;
            }

            this.xContext = new Context(chrX);
            this.yContext = new Context(chrY);
            superAdapter.setSelectedChromosomesNoRefresh(chrX, chrY);
            refreshEigenvectorTrackIfExists();
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

        return "setlocation " + xChr + " " + yChr + " " + currentZoom.getUnit().toString() + " " + currentZoom.getBinSize() + " " +
                xContext.getBinOrigin() + " " + yContext.getBinOrigin() + " " + getScaleFactor();
    }

    public String getDefaultLocationDescription() {

        String xChr = xContext.getChromosome().getName();
        String yChr = yContext.getChromosome().getName();

        if (!(xChr.toLowerCase().contains("chr"))) xChr = "chr" + xChr;
        if (!(yChr.toLowerCase().contains("chr"))) yChr = "chr" + yChr;

        return xChr + "@" + (long) (xContext.getBinOrigin() * currentZoom.getBinSize()) + "_" +
                yChr + "@" + (long) (yContext.getBinOrigin() * currentZoom.getBinSize());
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

    public void generateTrackFromLocation(int mousePos, boolean isHorizontal) {

        if (!MatrixType.isObservedOrControl(displayOption)) {
            MessageUtils.showMessage("This feature is only available for Observed or Control views");
            return;
        }

        // extract the starting position
        int binStartPosition = (int) (getXContext().getBinOrigin() + mousePos / getScaleFactor());
        if (isHorizontal) binStartPosition = (int) (getYContext().getBinOrigin() + mousePos / getScaleFactor());

        // Initialize default file name
        String filename = displayOption == MatrixType.OBSERVED ? "obs" : "ctrl";
        filename += isHorizontal ? "_horz" : "_vert";
        filename += "_bin" + binStartPosition + "_res" + currentZoom.getBinSize();
        filename = cleanUpNumbersInName(filename);

        // allow user to customize or change the name
        filename = MessageUtils.showInputDialog("Enter a name for the resulting .wig file", filename);
        if (filename == null || filename.equalsIgnoreCase("null"))
            return;

        File outputWigFile = new File(DirectoryManager.getHiCDirectory(), filename + ".wig");
        MessageUtils.showMessage("Data will be saved to " + outputWigFile.getAbsolutePath());

        Chromosome chromosomeForPosition = getXContext().getChromosome();
        if (isHorizontal) chromosomeForPosition = getYContext().getChromosome();

        safeSave1DTrackToWigFile(chromosomeForPosition, outputWigFile, binStartPosition);
    }

    private void safeSave1DTrackToWigFile(final Chromosome chromosomeForPosition, final File outputWigFile,
                                          final int binStartPosition) {
        superAdapter.getMainWindow().executeLongRunningTask(new Runnable() {
            @Override
            public void run() {
                try {
                    PrintWriter printWriter = new PrintWriter(outputWigFile);
                    unsafeSave1DTrackToWigFile(chromosomeForPosition, printWriter, binStartPosition);
                    printWriter.close();
                    if (outputWigFile.exists() && outputWigFile.length() > 0) {
                        // TODO this still doesn't add to the resource tree / load annotation dialog box
                        //superAdapter.getTrackLoadAction();
                        //getResourceTree().add1DCustomTrack(outputWigFile);
                        HiC.this.loadTrack(outputWigFile.getAbsolutePath());
                        LoadAction loadAction = superAdapter.getTrackLoadAction();
                        loadAction.checkBoxesForReload(outputWigFile.getName());
                    }
                } catch (Exception e) {
                    System.err.println("Unable to generate and save 1D HiC track");
                }
            }
        }, "Saving_1D_track_as_wig");

    }

    private void unsafeSave1DTrackToWigFile(Chromosome chromosomeForPosition, PrintWriter printWriter,
                                            int binStartPosition) throws IOException {
        int resolution = getZoom().getBinSize();
        for (Chromosome chromosome : chromosomes) {
            if (chromosome.getName().equals(Globals.CHR_ALL)) continue;
            Matrix matrix = null;
            if (displayOption == MatrixType.OBSERVED) {
                matrix = dataset.getMatrix(chromosomeForPosition, chromosome);
            } else if (displayOption == MatrixType.CONTROL) {
                matrix = controlDataset.getMatrix(chromosomeForPosition, chromosome);
            }

            if (matrix == null) continue;

            MatrixZoomData zd = matrix.getZoomData(currentZoom);
            printWriter.println("fixedStep chrom=chr" + chromosome.getName().replace("chr", "")
                    + " start=1 step=" + resolution + " span=" + resolution);

            int[] regionIndices;
            if (chromosomeForPosition.getIndex() < chromosome.getIndex()) {
                regionIndices = new int[]{binStartPosition, binStartPosition, 0, chromosome.getLength()};
            } else {
                regionIndices = new int[]{0, chromosome.getLength(), binStartPosition, binStartPosition};
            }

            zd.dump1DTrackFromCrossHairAsWig(printWriter, chromosomeForPosition, binStartPosition,
                    chromosomeForPosition.getIndex() == chromosome.getIndex(), regionIndices,
                    normalizationType, displayOption, getExpectedValues());
        }
    }

    public boolean isInPearsonsMode() {
        return getDisplayOption() == MatrixType.PEARSON;
    }

    public boolean isPearsonEdgeCaseEncountered(HiCZoom zoom) {
        return isInPearsonsMode() && zoom.getBinSize() < HiCGlobals.MAX_PEARSON_ZOOM;
    }

    public boolean isResolutionLocked() {
        return superAdapter.isResolutionLocked() ||
                // pearson can't zoom in
                (isInPearsonsMode() && currentZoom.getBinSize() == HiCGlobals.MAX_PEARSON_ZOOM);
    }

    public enum ZoomCallType {STANDARD, DRAG, DIRECT, INITIAL}

    public enum Unit {BP, FRAG}
}

