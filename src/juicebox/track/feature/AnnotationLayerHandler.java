/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

package juicebox.track.feature;

import juicebox.HiC;
import juicebox.data.ChromosomeHandler;
import juicebox.data.MatrixZoomData;
import juicebox.gui.SuperAdapter;
import juicebox.mapcolorui.Feature2DHandler;
import juicebox.mapcolorui.FeatureRenderer;
import juicebox.track.HiCGridAxis;
import juicebox.windowui.SaveAnnotationsDialog;
import org.broad.igv.ui.color.ColorChooserPanel;
import org.broad.igv.util.Pair;

import javax.swing.*;
import java.awt.*;
import java.util.*;
import java.util.List;

/**
 * Created by Marie on 6/4/15.
 * Modified by muhammadsaadshamim
 */
public class AnnotationLayerHandler {

    private static boolean importAnnotationsEnabled = false;
    // displacement in terms of gene pos
    private final int peakDisplacement = 3;
    private Rectangle selectionRegion;
    private Feature2D.FeatureType featureType;
    private Feature2D lastResizeLoop = null;
    private int lastChr1Idx = -1;
    private int lastChr2Idx = -1;
    private Pair<Integer, Integer> lastStarts = null;
    private Pair<Integer, Integer> lastEnds = null;
    private AnnotationLayer annotationLayer;
    private String layerName;
    private FeatureRenderer.PlottingOption plottingStyle = FeatureRenderer.PlottingOption.EVERYTHING;
    private boolean canExport = false, canUndo = false;
    private JButton exportButton, undoButton, importAnnotationsButton, deleteLayerButton;
    private JToggleButton activeLayerButton;
    private Color defaultColor = Color.BLUE;
    private JButton plottingStyleButton;
    private ColorChooserPanel colorChooserPanel;
    private JTextField nameTextField;

    public AnnotationLayerHandler() {
        featureType = Feature2D.FeatureType.NONE;
        this.annotationLayer = new AnnotationLayer();
        resetSelection();
        layerName = "Layer " + annotationLayer.getId();
    }

    private void resetSelection() {
        selectionRegion = null;
        //selectionPoint = null;
        featureType = Feature2D.FeatureType.NONE;
    }

    public boolean isEnabled() {
        return featureType != Feature2D.FeatureType.NONE;
    }

    public boolean isPeak() {
        return featureType == Feature2D.FeatureType.PEAK;
    }

    public void doGeneric() {
        featureType = Feature2D.FeatureType.GENERIC;
    }

    public void doPeak() {
        featureType = Feature2D.FeatureType.PEAK;
    }

    private void doDomain() {
        featureType = Feature2D.FeatureType.DOMAIN;
    }

    public void setStationaryStart(int start1, int start2) {
        lastStarts = new Pair<>(start1, start2);
        lastEnds = null;
    }

    public void setStationaryEnd(int end1, int end2) {
        lastEnds = new Pair<>(end1, end2);
        lastStarts = null;
    }

    // Update selection region from new rectangle
    public void updateSelectionRegion(Rectangle newRegion) {
        doDomain();
        selectionRegion = newRegion;
    }

//    // Update selection region from new coordinates
//    public Rectangle updateSelectionRegion(int x, int y, int deltaX, int deltaY) {
//
//        int x2, y2;
//        doDomain();
//        Rectangle lastRegion, damageRect;
//
//        lastRegion = selectionRegion;
//
//        if (deltaX == 0 || deltaY == 0) {
//            return null;
//        }
//
//        x2 = deltaX > 0 ? x : x + deltaX;
//        y2 = deltaY > 0 ? y : y + deltaY;
//        selectionRegion = new Rectangle(x2, y2, Math.abs(deltaX), Math.abs(deltaY));
//
//        damageRect = lastRegion == null ? selectionRegion : selectionRegion.union(lastRegion);
//        damageRect.x--;
//        damageRect.y--;
//        damageRect.width += 2;
//        damageRect.height += 2;
//        return damageRect;
//    }

    public void setLastItem(int idx1, int idx2, Feature2D lastLoop) {
        lastChr1Idx = idx1;
        lastChr2Idx = idx2;
        lastResizeLoop = lastLoop;
    }

    private void clearLastItem() {
        lastChr1Idx = -1;
        lastChr2Idx = -1;
        lastResizeLoop = null;
    }

    /*
    public void updateSelectionPoint(int x, int y) {
        selectionPoint = new Point(x, y);
    }
    */

    // Adds to lower lefthand side, for consistency.
    public Feature2D addFeature(HiC hic) {
        if (selectionRegion == null) return null;

        int start1, start2, end1, end2;
        Feature2D newFeature;
        setExportAbility(true);
        setUndoAbility(true);
        clearLastItem();
        String chr1 = hic.getXContext().getChromosome().getName();
        String chr2 = hic.getYContext().getChromosome().getName();
        int chr1Idx = hic.getXContext().getChromosome().getIndex();
        int chr2Idx = hic.getYContext().getChromosome().getIndex();
        HashMap<String, String> attributes = new HashMap<>(); //here
        int rightBound = hic.getXContext().getChromosome().getLength();
        int bottomBound = hic.getYContext().getChromosome().getLength();
        int leftBound = 0;
        int x = selectionRegion.x;
        int y = selectionRegion.y;
        int width = selectionRegion.width;
        int height = selectionRegion.height;

        start1 = geneXPos(hic, x, 0);
        end1 = geneXPos(hic, x + width, 0);
        start2 = geneYPos(hic, y, 0);
        end2 = geneYPos(hic, y + height, 0);

        // Snap if close to diagonal
        if (chr1Idx == chr2Idx && (pointsShouldSnapToDiagonal(hic, x, y, width, height)
                || regionsOverlapSignificantly(start1, end1, start2, end2, .6))) {

            if (start1 < start2) {
                // snap to the right i.e. use y values
                start1 = start2;
                end1 = end2;
            } else {
                // snap down i.e. use x values
                start2 = start1;
                end2 = end1;
            }

            /*
            TODO meh - I don't think this is doing what we think it is?
            // Snap to min of horizontal stretch and vertical stretch
            if (width <= y) {
                start2 = start1;
                end2 = end1;
            } else {
                start2 = geneYPos(hic, y, 0);
                end2 = geneYPos(hic, y + height, 0);
                start1 = start2;
                end1 = end2;
            }
            */

            // Otherwise record as drawn
        }

        // Make sure bounds aren't unreasonable (out of HiC map)
//                int rightBound = hic.getChromosomes().get(0).getLength();
//                int bottomBound = hic.getChromosomes().get(1).getLength();
        start1 = Math.min(Math.max(start1, leftBound), rightBound);
        start2 = Math.min(Math.max(start2, leftBound), bottomBound);
        end1 = Math.max(Math.min(end1, rightBound), leftBound);
        end2 = Math.max(Math.min(end2, bottomBound), leftBound);

        // Check for anchored corners
        if (lastStarts != null) {
            if (lastStarts.getFirst() < end1 && lastStarts.getSecond() < end2) {
                start1 = lastStarts.getFirst();
                start2 = lastStarts.getSecond();
            }
        } else if (lastEnds != null) {
            if (start1 < lastEnds.getFirst() && start2 < lastEnds.getSecond()) {
                end1 = lastEnds.getFirst();
                end2 = lastEnds.getSecond();
            }
        }


        // Add new feature
        newFeature = new Feature2D(Feature2D.FeatureType.DOMAIN, chr1, start1, end1, chr2, start2, end2,
                defaultColor, attributes); // could be here need to find a way to get list of
        annotationLayer.add(chr1Idx, chr2Idx, newFeature);
        lastStarts = null;
        lastEnds = null;
        return newFeature;
    }

    private boolean regionsOverlapSignificantly(int start1, int end1, int start2, int end2, double tolerance) {

        // must cross diagonal for overlap
        if ((start1 < end2 && end1 > start2) || (start1 > end2 && end1 < start2)) {
            double areaFeatureScaled = (end1 - start1) / 100.0 * (end2 - start2) / 100.0;
            double areaOverlapScaled = Math.pow((Math.min(end1, end2) - Math.max(start1, start2)) / 100.0, 2);

            return areaOverlapScaled / areaFeatureScaled > tolerance;
        }

        return false;
    }

    private boolean pointsShouldSnapToDiagonal(HiC hic, int x, int y, int width, int height) {
        return nearDiagonal(hic, x, y) && nearDiagonal(hic, x + width, y + height)
                && !allPointsAreOnSameSideOfDiagonal(hic, x, y, width, height);
    }

    /**
     * assumes that points are on same chromosome
     *
     * @param hic
     * @param x
     * @param y
     * @param width
     * @param height
     * @return
     */
    private boolean allPointsAreOnSameSideOfDiagonal(HiC hic, int x, int y, int width, int height) {

        int x1 = getXBin(hic, x);
        int y1 = getYBin(hic, y);
        int x2 = getXBin(hic, x + width);
        int y2 = getYBin(hic, y + height);

        // if this has been called, we can assume that x1,y1 and x2,y2 are near the diagonal
        // now we need to check if x1,y2 and x2,y1 are on the same side of the diagonal or not
        // if they are on same side of diagonal, don't snap to grid

        // i.e. x<y for all points or x >y for all points
        // if x1,y2 and x2,y1 are on the same side, then x1,y1 and x2,y2 will be on that side as well (rectangle)

        return (x1 < y2 && x2 < y1) || (x1 > y2 && x2 > y1);
    }

    /*
    public void addVisibleLoops(HiC hic) {
        try {
            hic.getZd();
        } catch (Exception e) {
            return;
        }

        if (hic.getXContext() == null || hic.getYContext() == null)
            return;

        java.util.List<Feature2DList> loops = hic.getAllVisibleLoopLists();
        if (loops == null) return;
        if (customAnnotation == null) {
            System.err.println("Error! Custom annotations should not be null!");
            return;
        }

        // Add each loop list to the custom annotation list
        if (loops.size() > 0) {
            setExportAbility(true);
            for (Feature2DList list : loops) {
                customAnnotation.addVisibleToCustom(list);
            }
        }
    }
    */

    public void undo(JButton undoButton) {
        annotationLayer.undo();
        if (lastResizeLoop != null) {
            annotationLayer.add(lastChr1Idx, lastChr2Idx, lastResizeLoop);
            resetSelection();
        }
        undoButton.setEnabled(false);
    }

    private boolean nearDiagonal(HiC hic, int x, int y) {
        int start1 = getXBin(hic, x);
        int start2 = getYBin(hic, y);

        int threshold = 15;
        return Math.abs(start1 - start2) < threshold;
    }

    /*
    TODO - consider scaling threshold with resolution?
    private int getThreshold(HiC hic) {
        try{
            return (int)Math.ceil(threshold*Math.log(hic.getZd().getBinSize()));
        }
        catch (Exception e){
            return threshold;
        }
    }
    */

    //helper for getannotatemenu
    private int geneXPos(HiC hic, int x, int displacement) {
        try {
            final MatrixZoomData zd = hic.getZd();
            if (zd == null) return -1;
            HiCGridAxis xGridAxis = zd.getXGridAxis();
            int binX = getXBin(hic, x) + displacement;
            return xGridAxis.getGenomicStart(binX);
        } catch (Exception e) {
            return -1;
        }
    }

    //helper for getannotatemenu
    private int geneYPos(HiC hic, int y, int displacement) {
        try {
            final MatrixZoomData zd = hic.getZd();
            if (zd == null) return -1;
            HiCGridAxis yGridAxis = zd.getYGridAxis();
            int binY = getYBin(hic, y) + displacement;
            return yGridAxis.getGenomicStart(binY);
        } catch (Exception e) {
            return -1;
        }
    }

    private int getXBin(HiC hic, int x) {
        return (int) (hic.getXContext().getBinOrigin() + x / hic.getScaleFactor());
    }

    private int getYBin(HiC hic, int y) {
        return (int) (hic.getYContext().getBinOrigin() + y / hic.getScaleFactor());
    }

    public boolean getLayerVisibility() {
        return annotationLayer.getLayerVisibility();
    }

    public void setLayerVisibility(boolean showCustom) {
        annotationLayer.setLayerVisibility(showCustom);
    }

    public void clearAnnotations() {
        annotationLayer.clearAnnotations();
    }

    public void deleteTempFile() {
        annotationLayer.deleteTempFile();
    }

    public AnnotationLayer getAnnotationLayer() {
        return annotationLayer;
    }

    public void setAnnotationLayer(AnnotationLayer annotationLayer) {
        this.annotationLayer = annotationLayer;
    }

    public List<Feature2D> getNearbyFeatures(MatrixZoomData zd, int chr1Idx, int chr2Idx, int centerX, int centerY,
                                             int numberOfLoopsToFind, double binOriginX,
                                             double binOriginY, double scaleFactor) {
        return annotationLayer.getNearbyFeatures(zd, chr1Idx, chr2Idx, centerX, centerY, numberOfLoopsToFind,
                binOriginX, binOriginY, scaleFactor);
    }

    public List<Feature2D> getContainedFeatures(HiC hic) {
        if (selectionRegion == null) return null;

        int start1, end1;
        int x = selectionRegion.x;
        int width = selectionRegion.width;
        int chr1Idx = hic.getXContext().getChromosome().getIndex();
        int chr2Idx = hic.getYContext().getChromosome().getIndex();

        // Get starting chrx and ending chrx
        start1 = geneXPos(hic, x, 0);
        end1 = geneXPos(hic, x + width, 0);

        net.sf.jsi.Rectangle selectionWindow = new net.sf.jsi.Rectangle(start1, start1, end1, end1);

        return annotationLayer.getContainedFeatures(chr1Idx, chr2Idx, selectionWindow);
    }

    /*
     * Gets the contained featuers within the selction region, including the features that the
     * selection starts and ends in
     */
    public List<Feature2D> getSelectedFeatures(HiC hic, int lastX, int lastY) {
        List<Feature2D> selectedFeatures = new ArrayList<Feature2D>();
        int chr1Idx = hic.getXContext().getChromosome().getIndex();
        int chr2Idx = hic.getYContext().getChromosome().getIndex();
        // Multiple regions selected
        if (selectionRegion != null) {
            int start1, end1;
            int x = selectionRegion.x;
            int width = selectionRegion.width;

            // Get starting chrx and ending chrx and window
            start1 = geneXPos(hic, x, 0);
            end1 = geneXPos(hic, x + width, 0);
            net.sf.jsi.Rectangle selectionWindow = new net.sf.jsi.Rectangle(start1, start1, end1, end1);

            // Get inner selection of loops
            selectedFeatures.addAll(annotationLayer.getContainedFeatures(chr1Idx, chr2Idx, selectionWindow));

            try {
                // Find closest loop to starting selection
                annotationLayer.getFeatureHandler().setSparsePlottingEnabled(true);
                selectedFeatures.addAll(getNearbyFeatures(hic.getZd(), chr1Idx, chr2Idx,
                        x, x, 1, hic.getXContext().getBinOrigin(),
                        hic.getYContext().getBinOrigin(), hic.getScaleFactor()));
                // Find closest loop to ending selection
                selectedFeatures.addAll(getNearbyFeatures(hic.getZd(), chr1Idx, chr2Idx,
                        x + width, x + width, 1, hic.getXContext().getBinOrigin(),
                        hic.getYContext().getBinOrigin(), hic.getScaleFactor()));
                annotationLayer.getFeatureHandler().setSparsePlottingEnabled(false);
            } catch (Exception e) {
                annotationLayer.getFeatureHandler().setSparsePlottingEnabled(false);
                selectionRegion = null;
                return selectedFeatures;
            }
            selectionRegion = null;
            return new ArrayList<>(new HashSet<>(selectedFeatures));

        // Single region selected
        } else {
            try {
                // Find closest loop to starting selection
                annotationLayer.getFeatureHandler().setSparsePlottingEnabled(true);
                selectedFeatures.addAll(getNearbyFeatures(hic.getZd(), chr1Idx, chr2Idx,
                        lastX, lastY, 1, hic.getXContext().getBinOrigin(),
                        hic.getYContext().getBinOrigin(), hic.getScaleFactor()));
                annotationLayer.getFeatureHandler().setSparsePlottingEnabled(false);
            } catch (Exception e) {
                System.out.println("error:" + e);
                annotationLayer.getFeatureHandler().setSparsePlottingEnabled(false);
                return selectedFeatures;
            }
            return selectedFeatures;
        }
    }

    public void removeFromList(MatrixZoomData zd, int chr1Idx, int chr2Idx, int centerX, int centerY, int numberOfLoopsToFind,
                               double binOriginX, double binOriginY, double scaleFactor, Feature2D feature) {
        annotationLayer.removeFromList(zd, chr1Idx, chr2Idx, centerX, centerY, numberOfLoopsToFind,
                binOriginX, binOriginY, scaleFactor, feature);
    }

    public boolean hasLoop(MatrixZoomData zd, int chr1Idx, int chr2Idx, int centerX, int centerY, int numberOfLoopsToFind,
                           double binOriginX, double binOriginY, double scaleFactor, Feature2D feature) {
        return annotationLayer.hasLoop(zd, chr1Idx, chr2Idx, centerX, centerY, numberOfLoopsToFind,
                binOriginX, binOriginY, scaleFactor, feature);
    }

    public String getLayerName() {
        return layerName;
    }

    public void setLayerName(String layerName) {
        this.layerName = layerName;
    }

    public void setLayerNameAndField(String layerName) {
        this.layerName = layerName;
        if (nameTextField != null) nameTextField.setText(layerName);
    }



    public Feature2DHandler getFeatureHandler() {
        return annotationLayer.getFeatureHandler();
    }

    public boolean getIsTransparent() {
        return getFeatureHandler().getIsTransparent();
    }

    public void setIsTransparent(boolean isTransparent) {
        getFeatureHandler().setIsTransparent(isTransparent);
    }

    public boolean getIsEnlarged() {
        return getFeatureHandler().getIsEnlarged();
    }

    public void setIsEnlarged(boolean isEnlarged) {
        getFeatureHandler().setIsEnlarged(isEnlarged);
    }

    public FeatureRenderer.PlottingOption getPlottingStyle() {
        return plottingStyle;
    }

    public void setPlottingStyle(FeatureRenderer.PlottingOption plottingStyle) {
        this.plottingStyle = plottingStyle;
    }

    public void exportAnnotations() {
        new SaveAnnotationsDialog(getAnnotationLayer(), getLayerName());
    }

    public void setImportAnnotationButton(JButton importAnnotationsButton) {
        this.importAnnotationsButton = importAnnotationsButton;
    }

    private boolean getImportAnnotationsEnabled() {
        return importAnnotationsEnabled;
    }

    public void setImportAnnotationsEnabled(boolean status) {
        importAnnotationsEnabled = status;
        if (importAnnotationsButton != null) {
            importAnnotationsButton.setEnabled(importAnnotationsEnabled);
        }
    }

    public void loadLoopList(String path, ChromosomeHandler chromosomeHandler) {
        Feature2DHandler.resultContainer result = getFeatureHandler().loadLoopList(path, chromosomeHandler);
        if (result.n > 0) {
            setExportAbility(true);
            if (result.color != null) {
                setDefaultColor(result.color);
            }
            if (result.attributes != null) {
                annotationLayer.setAttributeKeys(result.attributes);
            }
        }
    }

    public List<Feature2DList> getAllVisibleLoopLists() {
        return getFeatureHandler().getAllVisibleLoopLists();
    }


    public void setExportAbility(boolean allowed) {
        canExport = allowed;
        if (exportButton != null) {
            exportButton.setEnabled(true);
        }
    }

    public void setExportButton(JButton exportButton) {
        this.exportButton = exportButton;
    }

    public boolean getExportCapability() {
        return canExport;
    }

    public void setUndoAbility(boolean allowed) {
        canUndo = allowed;
        if (undoButton != null) {
            undoButton.setEnabled(true);
        }
    }

    public void setUndoButton(JButton undoButton) {
        this.undoButton = undoButton;
    }

    public boolean getUndoCapability() {
        return canUndo;
    }

    public boolean isActiveLayer(SuperAdapter superAdapter) {
        return annotationLayer.getId() == superAdapter.getActiveLayer().getAnnotationLayer().getId();
    }

    public void setActiveLayerButtonStatus(boolean status) {
        if (activeLayerButton != null) {
            activeLayerButton.setSelected(status);
            activeLayerButton.revalidate();
        }
    }

    public void setActiveLayerButton(JToggleButton activeLayerButton) {
        this.activeLayerButton = activeLayerButton;
    }

    public int getNumberOfFeatures() {
        return annotationLayer.getNumberOfFeatures();
    }

    public void setColorOfAllAnnotations(Color color) {
        defaultColor = color;
        annotationLayer.setColorOfAllAnnotations(color);
    }

    public Color getDefaultColor() {
        return defaultColor;
    }

    private void setDefaultColor(Color defaultColor) {
        this.defaultColor = defaultColor;
        if (colorChooserPanel != null) colorChooserPanel.setSelectedColor(defaultColor);
    }

    public void setDeleteLayerButtonStatus(boolean status) {
        if (deleteLayerButton != null) {
            deleteLayerButton.setEnabled(status);
            deleteLayerButton.revalidate();
        }
    }

    public void setDeleteLayerButton(JButton deleteLayerButton) {
        this.deleteLayerButton = deleteLayerButton;
    }

    public boolean getIsSparse() {
        return annotationLayer.getIsSparse();
    }

    public void setIsSparse(boolean isSparse) {
        annotationLayer.setIsSparse(isSparse);
    }

    public void duplicateDetailsFrom(AnnotationLayerHandler handlerOriginal) {
        featureType = handlerOriginal.featureType;
        System.out.println("...p...");

        setLayerNameAndField("Copy of " + handlerOriginal.getLayerName());
        setLayerVisibility(handlerOriginal.getLayerVisibility());
        setDefaultColor(handlerOriginal.getDefaultColor());
        setIsTransparent(handlerOriginal.getIsTransparent());
        setIsEnlarged(handlerOriginal.getIsEnlarged());
        setPlottingStyle(handlerOriginal.getPlottingStyle());

        Collection<Feature2DList> origLists = handlerOriginal.getAnnotationLayer().getAllFeatureLists();
        Collection<Feature2DList> dupLists = new ArrayList<>();
        for (Feature2DList list : origLists) {
            dupLists.add(list.deepCopy());
        }

        annotationLayer.createMergedLoopLists(dupLists);
        setImportAnnotationsEnabled(handlerOriginal.getImportAnnotationsEnabled());
        setExportAbility(handlerOriginal.getExportCapability());
        setIsSparse(handlerOriginal.getIsSparse());
    }

    public void mergeDetailsFrom(Collection<AnnotationLayerHandler> originalHandlers) {

        StringBuilder cleanedTitle = new StringBuilder();
        for (AnnotationLayerHandler originalHandler : originalHandlers) {
            featureType = originalHandler.featureType;

            cleanedTitle.append("-").append(originalHandler.getLayerName().toLowerCase().replaceAll("layer", "").replaceAll("\\s", ""));

            setLayerVisibility(originalHandler.getLayerVisibility());
            setColorOfAllAnnotations(originalHandler.getDefaultColor());

            annotationLayer.createMergedLoopLists(originalHandler.getAnnotationLayer().getAllFeatureLists());
            importAnnotationsEnabled |= originalHandler.getImportAnnotationsEnabled();

            canExport |= originalHandler.getExportCapability();
        }

        setExportAbility(canExport);
        setLayerName("Merger" + cleanedTitle);
    }

    public void togglePlottingStyle() {
        try {
            plottingStyleButton.doClick();
        } catch (Exception e) {
            setPlottingStyle(FeatureRenderer.getNextState(getPlottingStyle()));
        }
    }

    public void setPlottingStyleButton(JButton plottingStyleButton) {
        this.plottingStyleButton = plottingStyleButton;
    }

    public void setColorChooserPanel(ColorChooserPanel colorChooserPanel) {
        this.colorChooserPanel = colorChooserPanel;
    }

    public void setNameTextField(JTextField nameTextField) {
        this.nameTextField = nameTextField;
    }
}
