/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */


package juicebox;

import juicebox.data.*;
import juicebox.data.anchor.GenericLocus;
import juicebox.data.anchor.MotifAnchor;
import juicebox.data.basics.Chromosome;
import juicebox.gui.SuperAdapter;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.track.*;
import juicebox.track.feature.Feature2D;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationHandler;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.ui.util.MessageUtils;
import org.broad.igv.util.Pair;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * This is the "model" class for the HiC viewer.
 */
public class HiC {

    private final HiCTrackManager trackManager;
    private final SuperAdapter superAdapter;
    private final String eigString = "Eigenvector";
    private final String ctrlEigString = "Ctrl_Eigenvector";
    private final ZoomActionTracker zoomActionTracker = new ZoomActionTracker();
    private final List<Feature2D> highlightedFeatures = new ArrayList<>();
    private double scaleFactor;
    private String xPosition;
    private String yPosition;
    private MatrixType displayOption = MatrixType.OBSERVED;
    private NormalizationType obsNormalizationType = NormalizationHandler.NONE;
    private NormalizationType ctrlNormalizationType = NormalizationHandler.NONE;
    private ChromosomeHandler chromosomeHandler;
    private Dataset dataset;
    private Dataset controlDataset;
    private HiCZoom currentZoom;
    //private MatrixZoomData matrixForReloadState;
    private Context xContext;
    private Context yContext;
    private EigenvectorTrack eigenvectorTrack, controlEigenvectorTrack;
    private ResourceTree resourceTree;
    private LoadEncodeAction encodeAction;
    private Point cursorPoint, diagonalCursorPoint, gwCursorPoint;
    private Point selectedBin;
    private boolean linkedMode;
    private boolean m_zoomChanged;
    private boolean m_displayOptionChanged;
    private boolean m_normalizationTypeChanged;
    private boolean showFeatureHighlight;

    public HiC(SuperAdapter superAdapter) {
        this.superAdapter = superAdapter;
        trackManager = new HiCTrackManager(superAdapter, this);
        //feature2DHandler = new Feature2DHandler();
        m_zoomChanged = false;
        m_displayOptionChanged = false;
        m_normalizationTypeChanged = false;
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

    public Unit getDefaultUnit() {
        return dataset.getBpZooms().size() > 0 ? Unit.BP : Unit.FRAG;
    }

    public void reset() {
        dataset = null;
        controlDataset = null;
        displayOption = MatrixType.OBSERVED;
        currentZoom = null;
        resetContexts();
        chromosomeHandler = null;
        eigenvectorTrack = null;
        controlEigenvectorTrack = null;
        resourceTree = null;
        encodeAction = null;
        obsNormalizationType = NormalizationHandler.NONE;
        ctrlNormalizationType = NormalizationHandler.NONE;
        zoomActionTracker.clear();
        clearFeatures();
    }

    //Todo why iterate through tracksToRemove if you end up calling clearFeatures() at the end?
    public void clearTracksForReloadState() {
        ArrayList<HiCTrack> tracksToRemove = new ArrayList<>(trackManager.getLoadedTracks());
        for (HiCTrack trackToRemove : tracksToRemove) {
            String name = trackToRemove.getName();
            if (name.equalsIgnoreCase(eigString)) {
                eigenvectorTrack = null;
            } else if (name.equalsIgnoreCase(ctrlEigString)) {
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
        // feature2DHandler.clearLists();
    }

    /**
     * Use setCursorPoint() to set cursorPoint to the last known mouse click position before calling this method
     */
    public void undoZoomAction() {
        zoomActionTracker.undoZoom();
        ZoomAction currentZoomAction = zoomActionTracker.getCurrentZoomAction();
        unsafeActuallySetZoomAndLocation(currentZoomAction.getChromosomeX(), currentZoomAction.getChromosomeY(),
                currentZoomAction.getHiCZoom(), currentZoomAction.getGenomeX(), currentZoomAction.getGenomeY(),
                currentZoomAction.getScaleFactor(), currentZoomAction.getResetZoom(), currentZoomAction.getZoomCallType(),
                true, currentZoomAction.getResolutionLocked(), false);
    }

    /**
     * Use setCursorPoint() to set cursorPoint to the last known mouse click position before calling this method
     */
    public void redoZoomAction() {
        zoomActionTracker.redoZoom();
        ZoomAction currentZoomAction = zoomActionTracker.getCurrentZoomAction();
        unsafeActuallySetZoomAndLocation(currentZoomAction.getChromosomeX(), currentZoomAction.getChromosomeY(),
                currentZoomAction.getHiCZoom(), currentZoomAction.getGenomeX(), currentZoomAction.getGenomeY(),
                currentZoomAction.getScaleFactor(), currentZoomAction.getResetZoom(), currentZoomAction.getZoomCallType(),
                true, currentZoomAction.getResolutionLocked(), false);
    }

    public double getScaleFactor() {
        return scaleFactor;
    }

    private void setScaleFactor(double scaleFactor) {
        this.scaleFactor = Math.max(Math.min(50, scaleFactor), 1e-10);
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

    public void refreshEigenvectorTrackIfExists() {
        if (eigenvectorTrack != null) {
            eigenvectorTrack.forceRefreshCache();
        }
        if (controlEigenvectorTrack != null) {
            controlEigenvectorTrack.forceRefreshCache();
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

    public List<HiCTrack> getLoadedTracks() {
        return trackManager == null ? new ArrayList<>() : trackManager.getLoadedTracks();
    }

    public void unsafeLoadHostedTracks(List<ResourceLocator> locators) {
        trackManager.unsafeTrackLoad(locators);
    }

    public void unsafeLoadTrack(String path) {
        trackManager.unsafeLoadTrackDirectPath(path);
    }

    public void loadCoverageTrack(String label) {
        NormalizationType no = dataset.getNormalizationHandler().getNormTypeFromString(label);
        trackManager.loadCoverageTrack(no, false);
        if (isControlLoaded()) {
            trackManager.loadCoverageTrack(no, true);
        }
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

    public Dataset getDataset() {
        return dataset;
    }

    public void setDataset(Dataset dataset) {
        this.dataset = dataset;
    }

    public Dataset getControlDataset() {
        return controlDataset;
    }

    public void setControlDataset(Dataset controlDataset) {
        this.controlDataset = controlDataset;
    }

    public void setSelectedChromosomes(Chromosome chrX, Chromosome chrY) {
        this.xContext = new Context(chrX);
        this.yContext = new Context(chrY);
        refreshEigenvectorTrackIfExists();
    }

    public HiCZoom getZoom() {
        return currentZoom;
    }

    public MatrixZoomData getZd() throws NullPointerException {
        Matrix matrix = getMatrix();
        if (matrix == null) {
            throw new NullPointerException("Uninitialized matrix");
        } else if (currentZoom == null) {
            throw new NullPointerException("Uninitialized zoom");
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

    public Point getDiagonalCursorPoint() {
        return diagonalCursorPoint;
    }

    public void setDiagonalCursorPoint(Point point) {
        this.diagonalCursorPoint = point;
    }

    public Point getGWCursorPoint() {
        return gwCursorPoint;
    }

    public void setGWCursorPoint(Point point) {
        gwCursorPoint = point;
    }

    public long[] getCurrentRegionWindowGenomicPositions() {
    
        // address int overflow or exceeding bound issues
        long xEndEdge = xContext.getGenomicPositionOrigin() +
                (int) ((double) getZoom().getBinSize() * superAdapter.getHeatmapPanel().getWidth() / getScaleFactor());
        if (xEndEdge < 0 || xEndEdge > xContext.getChromosome().getLength()) {
            xEndEdge = xContext.getChromosome().getLength();
        }
    
        long yEndEdge = yContext.getGenomicPositionOrigin() +
                (int) ((double) getZoom().getBinSize() * superAdapter.getHeatmapPanel().getHeight() / getScaleFactor());
        if (yEndEdge < 0 || yEndEdge > yContext.getChromosome().getLength()) {
            yEndEdge = yContext.getChromosome().getLength();
        }
    
        return new long[]{xContext.getGenomicPositionOrigin(), xEndEdge, yContext.getGenomicPositionOrigin(), yEndEdge};
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
            if (HiCGlobals.printVerboseComments) {
                System.err.println("Dataset is null");
            }
            return null;
        } else if (xContext == null) {
            if (HiCGlobals.printVerboseComments) {
                System.err.println("xContext is null");
            }
            return null;
        } else if (yContext == null) {
            if (HiCGlobals.printVerboseComments) {
                System.err.println("yContext is null");
            }
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
        return xContext != null && ChromosomeHandler.isAllByAll(xContext.getChromosome());
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

    public NormalizationType getObsNormalizationType() {
        return obsNormalizationType;
    }

    public void setObsNormalizationType(String label) {
        NormalizationType option = dataset.getNormalizationHandler().getNormTypeFromString(label);
        if (this.obsNormalizationType != option) {
            this.obsNormalizationType = option;
            setNormalizationTypeChanged();
        }
    }

    public NormalizationType getControlNormalizationType() {
        return ctrlNormalizationType;
    }

    public void setControlNormalizationType(String label) {
        NormalizationType option = dataset.getNormalizationHandler().getNormTypeFromString(label);
        if (this.ctrlNormalizationType != option) {
            this.ctrlNormalizationType = option;
            setNormalizationTypeChanged();
        }
    }

    public double[] getEigenvector(final int chrIdx, final int n, boolean isControl) {

        if (isControl) {
            if (controlDataset == null) return null;

            Chromosome chr = chromosomeHandler.getChromosomeFromIndex(chrIdx);
            return controlDataset.getEigenvector(chr, currentZoom, n, ctrlNormalizationType);
        } else {
            if (dataset == null) return null;

            Chromosome chr = chromosomeHandler.getChromosomeFromIndex(chrIdx);
            return dataset.getEigenvector(chr, currentZoom, n, obsNormalizationType);
        }
    }

    public ExpectedValueFunction getExpectedValues() {
        if (dataset == null) return null;
        return dataset.getExpectedValues(currentZoom, obsNormalizationType);
    }

    public ExpectedValueFunction getExpectedControlValues() {
        if (controlDataset == null) return null;
        return controlDataset.getExpectedValues(currentZoom, ctrlNormalizationType);
    }

    public NormalizationVector getNormalizationVector(int chrIdx) {
        if (dataset == null) return null;
        return dataset.getNormalizationVector(chrIdx, currentZoom, obsNormalizationType);
    }

    public NormalizationVector getControlNormalizationVector(int chrIdx) {
        if (controlDataset == null) return null;
        return controlDataset.getNormalizationVector(chrIdx, currentZoom, ctrlNormalizationType);
    }

    // Note - this is an inefficient method, used to support tooltip text only.
    public float getNormalizedObservedValue(int binX, int binY) {
        float val = Float.NaN;
        try {
            val = getZd().getObservedValue(binX, binY, obsNormalizationType);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return val;
    }

    public float getNormalizedControlValue(int binX, int binY) {
        float val = Float.NaN;
        try {
            val = getControlZd().getObservedValue(binX, binY, ctrlNormalizationType);
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
    public void zoomToDrawnBox(final long xBP0, final long yBP0, final double targetBinSize) {
        
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
                true, zoomCallType, allowLocationBroadcast, isResolutionLocked() ? 1 : 0, true);
    }
    
    private boolean safeActuallySetZoomAndLocation(HiCZoom newZoom, long genomeX, long genomeY, double scaleFactor,
                                                   boolean resetZoom, ZoomCallType zoomCallType, String message,
                                                   boolean allowLocationBroadcast) {
        return safeActuallySetZoomAndLocation(xContext.getChromosome().toString(), yContext.getChromosome().toString(),
                newZoom, genomeX, genomeY, scaleFactor, resetZoom, zoomCallType, message, allowLocationBroadcast);
    }

    private boolean safeActuallySetZoomAndLocation(final String chrXName, final String chrYName,
                                                   final HiCZoom newZoom, final long genomeX, final long genomeY,
                                                   final double scaleFactor, final boolean resetZoom,
                                                   final ZoomCallType zoomCallType, String message,
                                                   final boolean allowLocationBroadcast) {
        final boolean[] returnVal = new boolean[1];
        superAdapter.executeLongRunningTask(new Runnable() {
            @Override
            public void run() {
                returnVal[0] = unsafeActuallySetZoomAndLocation(chrXName, chrYName, newZoom, genomeX, genomeY, scaleFactor,
                        resetZoom, zoomCallType, allowLocationBroadcast, isResolutionLocked() ? 1 : 0, true);
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
     * @param scaleFactor      (pass -1 if scaleFactor should be calculated)
     * @param resolutionLocked (pass -1 if status of lock button should not be saved)
     * @param storeZoomAction  (pass false if function is being used to undo/redo zoom, true otherwise)
     * @return
     */
    public boolean unsafeActuallySetZoomAndLocation(String chrXName, String chrYName,
                                                    HiCZoom newZoom, long genomeX, long genomeY, double scaleFactor,
                                                    boolean resetZoom, ZoomCallType zoomCallType,
                                                    boolean allowLocationBroadcast, int resolutionLocked, boolean storeZoomAction) {

        if (dataset == null) return false;  // No data in view

        boolean chromosomesChanged = !(xContext.getChromosome().equals(chromosomeHandler.getChromosomeFromName(chrXName)) &&
                yContext.getChromosome().equals(chromosomeHandler.getChromosomeFromName(chrYName)));

        if (chrXName.length() > 0 && chrYName.length() > 0) {
            setChromosomesFromBroadcast(chrXName, chrYName);
            //We might end with All->All view, make sure normalization state is updated accordingly...
            superAdapter.getMainViewPanel().setNormalizationDisplayState(superAdapter.getHiC());
        }

        if (newZoom == null) {
            System.err.println("Invalid zoom " + newZoom);
        }

        Chromosome chrX = chromosomeHandler.getChromosomeFromName(chrXName);
        Chromosome chrY = chromosomeHandler.getChromosomeFromName(chrYName);
        final Matrix matrix = dataset.getMatrix(chrX, chrY);

        if (matrix == null) {
            superAdapter.launchGenericMessageDialog("Sorry, this region is not available", "Matrix unavailable",
                    JOptionPane.WARNING_MESSAGE);
            return false;
        }

        MatrixZoomData newZD = matrix.getZoomData(newZoom);

        if (ChromosomeHandler.isAllByAll(chrX)) {
            newZD = matrix.getFirstZoomData(Unit.BP);
        }

        if (newZD == null) {
            superAdapter.launchGenericMessageDialog("Sorry, this zoom is not available", "Zoom unavailable",
                    JOptionPane.WARNING_MESSAGE);
            return false;
        }

        Matrix preZoomMatrix = getMatrix();
        double preZoomScaleFactor = getScaleFactor();
        Context preZoomXContext = xContext;
        Context preZoomYContext = yContext;
        HiCZoom preZoomHiCZoom = currentZoom;

        if (resolutionLocked >= 0) {
            adjustLockButton(resolutionLocked != 0);
        }

        currentZoom = newZoom;
        xContext.setZoom(currentZoom);
        yContext.setZoom(currentZoom);

        if (scaleFactor > 0) {
            setScaleFactor(scaleFactor);
        } else {
            long maxBinCount = Math.max(newZD.getXGridAxis().getBinCount(), newZD.getYGridAxis().getBinCount());
            double defaultScaleFactor = Math.max(1.0, (double) superAdapter.getHeatmapPanel().getMinimumDimension() / maxBinCount);
            setScaleFactor(defaultScaleFactor);
        }

        int binX = newZD.getXGridAxis().getBinNumberForGenomicPosition(genomeX);
        int binY = newZD.getYGridAxis().getBinNumberForGenomicPosition(genomeY);

        if (zoomCallType == ZoomCallType.INITIAL || zoomCallType == ZoomCallType.STANDARD) {
            if (storeZoomAction || chromosomesChanged) {
                center(binX, binY);
            } else if (preZoomHiCZoom != null && getCursorPoint() != null) {
                Point standardUnzoomCoordinates = computeStandardUnzoomCoordinates(preZoomMatrix, preZoomXContext,
                        preZoomYContext, newZD, preZoomHiCZoom, preZoomScaleFactor);
                center(standardUnzoomCoordinates.getX(), standardUnzoomCoordinates.getY());
            }
        } else if (zoomCallType == ZoomCallType.DRAG) {
            xContext.setBinOrigin(binX);
            yContext.setBinOrigin(binY);
        } else if (zoomCallType == ZoomCallType.DIRECT) {
            xContext.setBinOrigin(genomeX);
            yContext.setBinOrigin(genomeY);
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

        if (storeZoomAction) {
            ZoomAction newZoomAction = new ZoomAction(chrXName, chrYName, newZoom, genomeX, genomeY, scaleFactor,
                    resetZoom, zoomCallType, resolutionLocked);
            if (zoomActionTracker.getCurrentZoomAction() == null) {
                this.zoomActionTracker.addZoomState(newZoomAction);
            } else if (!zoomActionTracker.getCurrentZoomAction().equals(newZoomAction)) {
                this.zoomActionTracker.addZoomState(newZoomAction);
            }
        }

        return true;
    }

    private Point computeStandardUnzoomCoordinates(Matrix preZoomMatrix, Context preZoomXContext, Context preZoomYContext,
                                                   MatrixZoomData newZD, HiCZoom preZoomHiCZoom, double preZoomScaleFactor) {
    
        double xMousePos = cursorPoint.getX();
        double yMousePos = cursorPoint.getY();
    
        double preZoomCenterBinX = preZoomXContext.getBinOrigin() + xMousePos / preZoomScaleFactor;
        double preZoomCenterBinY = preZoomYContext.getBinOrigin() + yMousePos / preZoomScaleFactor;
    
        long preZoomBinCountX = preZoomMatrix.getZoomData(preZoomHiCZoom).getXGridAxis().getBinCount();
        long preZoomBinCountY = preZoomMatrix.getZoomData(preZoomHiCZoom).getYGridAxis().getBinCount();
    
        long postZoomBinCountX = newZD.getXGridAxis().getBinCount();
        long postZoomBinCountY = newZD.getYGridAxis().getBinCount();
    
        return new Point((int) (preZoomCenterBinX / preZoomBinCountX * postZoomBinCountX),
                (int) (preZoomCenterBinY / preZoomBinCountY * postZoomBinCountY));
    }

    private void adjustLockButton(boolean savedResolutionLocked) {
        if (isResolutionLocked() != savedResolutionLocked) {
            superAdapter.getMainViewPanel().getResolutionSlider().setResolutionLocked(savedResolutionLocked);
        }
    }

    private void setChromosomesFromBroadcast(String chrXName, String chrYName) {
        if (!chrXName.equals(xContext.getChromosome().getName()) || !chrYName.equals(yContext.getChromosome().getName())) {
            Chromosome chrX = chromosomeHandler.getChromosomeFromName(chrXName);
            Chromosome chrY = chromosomeHandler.getChromosomeFromName(chrYName);

            if (chrX == null || chrY == null) {
                //System.out.println("Most probably origin is a different species saved location or sync/link between two different species maps.");
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

//        if (!xChr.toLowerCase().equals("assembly") && !(xChr.toLowerCase().contains("chr"))) xChr = "chr" + xChr;
//        if (!yChr.toLowerCase().equals("assembly") && !(yChr.toLowerCase().contains("chr"))) yChr = "chr" + yChr;


        return "setlocation " + xChr + " " + yChr + " " + currentZoom.getUnit().toString() + " " + currentZoom.getBinSize() + " " +
                xContext.getBinOrigin() + " " + yContext.getBinOrigin() + " " + getScaleFactor();
    }

    public String getDefaultLocationDescription() {

        String xChr = xContext.getChromosome().getName();
        String yChr = yContext.getChromosome().getName();

//        if (!xChr.toLowerCase().equals("assembly") && !(xChr.toLowerCase().contains("chr"))) xChr = "chr" + xChr;
//        if (!yChr.toLowerCase().equals("assembly") && !(yChr.toLowerCase().contains("chr"))) yChr = "chr" + yChr;


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

    public void loadLoopList(String path) {
        superAdapter.getActiveLayerHandler().loadLoopList(path, chromosomeHandler);
    }

    public void generateTrackFromLocation(int mousePos, boolean isHorizontal) {
    
        if (!MatrixType.isObservedOrControl(displayOption)) {
            SuperAdapter.showMessageDialog("This feature is only available for Observed or Control views");
            return;
        }
    
        // extract the starting position
        long binStartPosition = (long) (getXContext().getBinOrigin() + mousePos / getScaleFactor());
        if (isHorizontal) binStartPosition = (long) (getYContext().getBinOrigin() + mousePos / getScaleFactor());
    
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
        SuperAdapter.showMessageDialog("Data will be saved to " + outputWigFile.getAbsolutePath());

        Chromosome chromosomeForPosition = getXContext().getChromosome();
        if (isHorizontal) chromosomeForPosition = getYContext().getChromosome();

        safeSave1DTrackToWigFile(chromosomeForPosition, outputWigFile, binStartPosition);
    }

    /*
    public List<Feature2DList> getAllVisibleLoops() {
        return feature2DHandler.getAllVisibleLoops();
    }

    /*
    public List<Feature2D> getVisibleFeatures(int chrIdx1, int chrIdx2) {
        return feature2DHandler.getVisibleFeatures(chrIdx1, chrIdx2);
    }

    public List<Feature2D> findNearbyFeatures(MatrixZoomData zd, int chrIdx1, int chrIdx2, int x, int y, int n) {

        double binOriginX = getXContext().getBinOrigin();
        double binOriginY = getYContext().getBinOrigin();
        double scale = getScaleFactor();

        return feature2DHandler.getNearbyFeatures(zd, chrIdx1, chrIdx2, x, y, n, binOriginX, binOriginY, scale);
    }


    public List<Pair<Rectangle, Feature2D>> findNearbyFeaturePairs(MatrixZoomData zd, int chrIdx1, int chrIdx2, int x,
                                                                   int y, int n) {
        double binOriginX = getXContext().getBinOrigin();
        double binOriginY = getYContext().getBinOrigin();
        double scale = getScaleFactor();

        return feature2DHandler.getNearbyFeaturePairs(zd, chrIdx1, chrIdx2, x, y, n, binOriginX, binOriginY, scale);
    }
    */

    /*
    public void removeLoadedAnnotation(String path) {

        feature2DHandler.removeFeaturePath(path);
    }
    */
    
    private void safeSave1DTrackToWigFile(final Chromosome chromosomeForPosition, final File outputWigFile,
                                          final long binStartPosition) {
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
                        HiC.this.unsafeLoadTrack(outputWigFile.getAbsolutePath());
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
                                            long binStartPosition) {
        // todo could crash with custom chromosomes - so make sure this doesn't get called on those chromosomes
        int resolution = getZoom().getBinSize();
        for (Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {
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
            
            long[] regionIndices;
            if (chromosomeForPosition.getIndex() < chromosome.getIndex()) {
                regionIndices = new long[]{binStartPosition, binStartPosition, 0, chromosome.getLength()};
            } else {
                regionIndices = new long[]{0, chromosome.getLength(), binStartPosition, binStartPosition};
            }

            zd.dump1DTrackFromCrossHairAsWig(printWriter, binStartPosition,
                    chromosomeForPosition.getIndex() == chromosome.getIndex(), regionIndices,
                    obsNormalizationType, displayOption);
        }
    }

    public void generateRainbowBed() {

        // Initialize default file name
        String filename = "temp.rainbow";

        File outputBedFile = new File(DirectoryManager.getHiCDirectory(), filename + ".bed");
//        SuperAdapter.showMessageDialog("Data will be saved to " + outputWigFile.getAbsolutePath());

        Chromosome chromosome = getXContext().getChromosome();

        safeGenerateRainbowBed(chromosome, outputBedFile);
    }

    private void safeGenerateRainbowBed(final Chromosome chromosome, final File outputBedFile) {
        superAdapter.getMainWindow().executeLongRunningTask(new Runnable() {
            @Override
            public void run() {
                try {
                    PrintWriter printWriter = new PrintWriter(outputBedFile);
                    unsafeGenerateRainbowBed(chromosome, printWriter);
                    printWriter.close();
                    if (outputBedFile.exists() && outputBedFile.length() > 0) {
                        // TODO this still doesn't add to the resource tree / load annotation dialog box
                        //superAdapter.getTrackLoadAction();
                        //getResourceTree().add1DCustomTrack(outputWigFile);
                        HiC.this.unsafeLoadTrack(outputBedFile.getAbsolutePath());
                        LoadAction loadAction = superAdapter.getTrackLoadAction();
                        loadAction.checkBoxesForReload(outputBedFile.getName());
                    }
                } catch (Exception e) {
                    System.err.println("Unable to generate rainbow track");
                }
            }
        }, "Saving rainbow bed file.");

    }

    private void unsafeGenerateRainbowBed(Chromosome chromosome, PrintWriter printWriter) {

        printWriter.println("track name=\"Rainbow track\" description=\"Rainbow track\" visibility=2 itemRgb=\"On\"");
        int resolution = getZoom().getBinSize();
        long size = chromosome.getLength() / resolution + 1;
        for (int i = 0; i < size; i++) {
            printWriter.println(chromosome.getName() + "\t" + i * resolution + "\t" + ((i + 1) * resolution) + "\t-\t0\t+\t" + i * resolution + "\t" + ((i + 1) * resolution) + "\t" + getRgb(i, size));
        }
    }

    private String getRgb(int i, long size) {
        int red = (int) Math.floor(127 * Math.sin(Math.PI / size * 2 * i + 0 * Math.PI * 2 / 3)) + 128;
        int blue = (int) Math.floor(127 * Math.sin(Math.PI / size * 2 * i + 1 * Math.PI * 2 / 3)) + 128;
        int green = (int) Math.floor(127 * Math.sin(Math.PI / size * 2 * i + 2 * Math.PI * 2 / 3)) + 128;
        return red + "," + green + "," + blue;
    }

    public boolean isInPearsonsMode() {
        return MatrixType.isPearsonType(displayOption);
    }

    public boolean isPearsonEdgeCaseEncountered(HiCZoom zoom) {
        return isInPearsonsMode() && zoom.getBinSize() < HiCGlobals.MAX_PEARSON_ZOOM;
    }

    public boolean isResolutionLocked() {
        return superAdapter.isResolutionLocked() ||
                // pearson can't zoom in
                // even though it should never be less, I think we should try to catch it
                // (i.e. <= rather than ==)?
                (isInPearsonsMode() && currentZoom.getBinSize() <= HiCGlobals.MAX_PEARSON_ZOOM);
    }

    public boolean isPearsonsNotAvailableForFile(boolean isControl) {
        try {
            if (isControl) {
                MatrixZoomData cZd = getControlZd();
                return cZd.getPearsons(controlDataset.getExpectedValues(cZd.getZoom(), ctrlNormalizationType)) == null;
            } else {
                MatrixZoomData zd = getZd();
                return zd.getPearsons(dataset.getExpectedValues(zd.getZoom(), obsNormalizationType)) == null;
            }
        } catch (Exception e) {
            return true;
        }
    }

    public boolean isPearsonsNotAvailableAtSpecificZoom(boolean isControl, HiCZoom zoom) {
        try {
            if (isControl) {
                return getControlZd().getPearsons(controlDataset.getExpectedValues(zoom, ctrlNormalizationType)) == null;
            } else {
                return getZd().getPearsons(dataset.getExpectedValues(zoom, obsNormalizationType)) == null;
            }
        } catch (Exception e) {
            return true;
        }
    }

    public Color getColorForRuler() {
        if (MatrixType.isPearsonType(displayOption)) {
            return Color.WHITE;
        } else {
            if (HiCGlobals.isDarkulaModeEnabled) {
                return HiCGlobals.DARKULA_RULER_LINE_COLOR;
            } else {
                return HiCGlobals.RULER_LINE_COLOR;
            }
        }
    }

    public boolean isVSTypeDisplay() {
        return MatrixType.isVSTypeDisplay(displayOption);
    }

    public boolean isInControlPearsonsMode() {
        return MatrixType.isControlPearsonType(displayOption);
    }

    public String getColorScaleKey() {
        try {
            return getZd().getColorScaleKey(displayOption, obsNormalizationType, ctrlNormalizationType);
        } catch (Exception ignored) {
        }
        return null;
    }

    public List<Feature2D> getHighlightedFeatures() {
        if (showFeatureHighlight) {
            return highlightedFeatures;
        }
        return new ArrayList<>();
    }

    public void setHighlightedFeatures(List<Feature2D> highlightedFeatures) {
        this.highlightedFeatures.clear();
        this.highlightedFeatures.addAll(highlightedFeatures);
    }

    public void setShowFeatureHighlight(boolean showFeatureHighlight) {
        this.showFeatureHighlight = showFeatureHighlight;
    }

    public ChromosomeHandler getChromosomeHandler() {
        return chromosomeHandler;
    }

    public void setChromosomeHandler(ChromosomeHandler chromosomeHandler) {
        this.chromosomeHandler = chromosomeHandler;
        dataset.setChromosomeHandler(chromosomeHandler);
        if (controlDataset != null) controlDataset.setChromosomeHandler(chromosomeHandler);
    }

    public ZoomActionTracker getZoomActionTracker() {
        return this.zoomActionTracker;
    }

    public void clearAllMatrixZoomDataCache() {
        clearAllCacheForDataset(dataset);
        if (isControlLoaded()) {
            clearAllCacheForDataset(controlDataset);
        }
    }

    private void clearAllCacheForDataset(Dataset ds) {
        ds.clearCache(false);
    }

    public List<Pair<GenericLocus, GenericLocus>> getRTreeHandlerIntersectingFeatures(String name, int g1, int g2) {
        try {
            return ((CustomMatrixZoomData) getZd()).getRTreeHandlerIntersectingFeatures(name, g1, g2);
        } catch (Exception ignored) {
            return new ArrayList<>();
        }
    }

    public String[] getNormalizationOptions(boolean isControl) {
        if (isControl) {
            if (controlDataset == null || controlDataset.getVersion() < HiCGlobals.minVersion) {
                return new String[]{NormalizationHandler.NONE.getDescription()};
            } else {
                ArrayList<String> tmp = new ArrayList<>();
                tmp.add(NormalizationHandler.NONE.getDescription());
                for (NormalizationType t : controlDataset.getNormalizationTypes()) {
                    tmp.add(t.getDescription());
                }
                return tmp.toArray(new String[tmp.size()]);
            }
        } else {
            if (dataset.getVersion() < HiCGlobals.minVersion) {
                return new String[]{NormalizationHandler.NONE.getDescription()};
            } else {
                ArrayList<String> tmp = new ArrayList<>();
                tmp.add(NormalizationHandler.NONE.getDescription());
                for (NormalizationType t : dataset.getNormalizationTypes()) {
                    tmp.add(t.getDescription());
                }
                return tmp.toArray(new String[tmp.size()]);
            }
        }
    }

    public void exportDataCenteredAboutRegion(int xBinPos, int yBinPos) throws IOException {

        int radius = 20;

        // Initialize default file name
        String filename = displayOption == MatrixType.OBSERVED ? "obs" : "ctrl";
        filename += "_bin_" + xBinPos + "_" + yBinPos + "_res_" + currentZoom.getBinSize();
        filename = cleanUpNumbersInName(filename);

        // allow user to customize or change the name
        filename = MessageUtils.showInputDialog("Enter a name for the resulting .txt file", filename);
        if (filename == null || filename.equalsIgnoreCase("null"))
            return;

        String radiusSize = MessageUtils.showInputDialog("What radius of pixels around the selected point would you like", "" + radius);
        try {
            radius = Integer.parseInt(radiusSize);
        } catch (Exception ignored) {
            radius = 20;
        }

        File outputMatrixFile = new File(DirectoryManager.getHiCDirectory(), filename + ".txt");
        SuperAdapter.showMessageDialog("Data will be saved to " + outputMatrixFile.getAbsolutePath());

        // extract the starting position
        int xbinStartPosition = (int) (getXContext().getBinOrigin() + xBinPos / getScaleFactor());
        int ybinStartPosition = (int) (getYContext().getBinOrigin() + yBinPos / getScaleFactor());

        int binXStart = xbinStartPosition - radius;
        int binXEnd = xbinStartPosition + radius;
        int binYStart = ybinStartPosition - radius;
        int binYEnd = ybinStartPosition + radius;
        int matrixWidth = 2 * radius + 1;

        MatrixZoomData requestedZd = getZd();
        NormalizationType requestedNormType = getObsNormalizationType();
        if (MatrixType.isSimpleControlType(displayOption)) {
            requestedZd = getControlZd();
            requestedNormType = getControlNormalizationType();
        }

        RealMatrix realMatrix = HiCFileTools.extractLocalBoundedRegion(requestedZd, binXStart, binXEnd, binYStart, binYEnd,
                matrixWidth, matrixWidth, requestedNormType, true);

        MatrixTools.saveMatrixTextV2(outputMatrixFile.getAbsolutePath(), realMatrix);
    }

    public void createNewDynamicResolutions(int newResolution) {
        dataset.addDynamicResolution(newResolution);
        if (controlDataset != null) {
            controlDataset.addDynamicResolution(newResolution);
        }
    }

    // use REVERSE for only undoing and redoing zoom actions
    public enum ZoomCallType {
        STANDARD, DRAG, DIRECT, INITIAL, REVERSE
    }

    public enum Unit {BP, FRAG}
}

