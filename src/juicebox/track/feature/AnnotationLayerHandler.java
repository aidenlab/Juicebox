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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.track.feature;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.MatrixZoomData;
import juicebox.gui.SuperAdapter;
import juicebox.mapcolorui.Feature2DHandler;
import juicebox.mapcolorui.FeatureRenderer;
import juicebox.track.HiCGridAxis;
import juicebox.windowui.layers.MiniAnnotationsLayerPanel;
import juicebox.windowui.layers.PlottingStyleButton;
import juicebox.windowui.layers.SaveAnnotationsDialog;
import org.broad.igv.ui.color.ColorChooserPanel;
import org.broad.igv.util.Pair;

import javax.swing.*;
import java.awt.*;
import java.util.List;
import java.util.*;

/**
 * Created by Marie on 6/4/15.
 * Modified by muhammadsaadshamim
 */
public class AnnotationLayerHandler {
	
	private static boolean importAnnotationsEnabled = false;
	private Rectangle selectionRegion;
	private Feature2D.FeatureType featureType = Feature2D.FeatureType.NONE;
	private Feature2D lastResizeLoop = null;
	private int lastChr1Idx = -1;
	private int lastChr2Idx = -1;
	private Pair<Long, Long> lastStarts = null;
	private Pair<Long, Long> lastEnds = null;
	private AnnotationLayer annotationLayer;
	private String layerName;
	private FeatureRenderer.PlottingOption plottingStyle = FeatureRenderer.PlottingOption.EVERYTHING;
	private FeatureRenderer.LineStyle lineStyle = FeatureRenderer.LineStyle.SOLID;
	private boolean canExport = false, canUndo = false;
	private JButton exportButton, undoButton, importAnnotationsButton, deleteLayerButton, censorButton;
	private final List<JToggleButton> activeLayerButtons = new ArrayList<>();
	private Color defaultColor = Color.BLUE;
	private final List<PlottingStyleButton> plottingStyleButtons = new ArrayList<>();
	private final List<ColorChooserPanel> colorChooserPanels = new ArrayList<>();
    private JTextField nameTextField;
    private JLabel miniNameLabel;

    public AnnotationLayerHandler() {
        annotationLayer = new AnnotationLayer();
        resetSelection();
        layerName = "Layer " + annotationLayer.getId();
    }

    public AnnotationLayerHandler(Feature2DList feature2DList) {
        annotationLayer = new AnnotationLayer(feature2DList);
        resetSelection();
        layerName = "Layer " + annotationLayer.getId();
    }

    private void resetSelection() {
        selectionRegion = null;
        //selectionPoint = null;
        featureType = Feature2D.FeatureType.NONE;
    }

    public void doPeak() {
        featureType = Feature2D.FeatureType.PEAK;
    }

    private void doDomain() {
        featureType = Feature2D.FeatureType.DOMAIN;
    }
	
	public void setStationaryStart(long start1, long start2) {
		lastStarts = new Pair<>(start1, start2);
		lastEnds = null;
	}
	
	public void setStationaryEnd(long end1, long end2) {
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

    public Feature2D generateFeature(HiC hic) {
        if (selectionRegion == null) return null;
	
		long start1, start2, end1, end2;
		Feature2D newFeature;
		setExportAbility(true);
		setUndoAbility(true);
		clearLastItem();
		String chr1 = hic.getXContext().getChromosome().getName();
		String chr2 = hic.getYContext().getChromosome().getName();
		int chr1Idx = hic.getXContext().getChromosome().getIndex();
		int chr2Idx = hic.getYContext().getChromosome().getIndex();
		HashMap<String, String> attributes = new HashMap<>();
		long rightBound = hic.getXContext().getChromosome().getLength();
		long bottomBound = hic.getYContext().getChromosome().getLength();
		int leftBound = 0;
		int x = selectionRegion.x;
		int y = selectionRegion.y;
		int width = selectionRegion.width;
		int height = selectionRegion.height;
	
		start1 = geneXPos(hic, x, 0);
		end1 = geneXPos(hic, x + width, 0);
		start2 = geneYPos(hic, y, 0);
		end2 = geneYPos(hic, y + height, 0);

//        System.out.println(start1 + "\t" + end1);


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
        if (HiCGlobals.splitModeEnabled) {
        }
        newFeature = new Feature2D(Feature2D.FeatureType.DOMAIN, chr1, start1, end1, chr2, start2, end2,
                defaultColor, attributes);
        lastStarts = null;
        lastEnds = null;
        return newFeature;
    }

    public Feature2D addFeature(HiC hic) {
        // Add new feature
        Feature2D newFeature = generateFeature(hic);
        annotationLayer.add(hic.getXContext().getChromosome().getIndex(), hic.getYContext().getChromosome().getIndex(), newFeature);
        return newFeature;
    }

    private Feature2D generateTempSelectedGroup(List<Feature2D> selectedFeatures, HiC hiC) {
		Collections.sort(selectedFeatures);
	
		Feature2D firstSelectedContig = selectedFeatures.get(0);
		Feature2D lastSelectedContig = selectedFeatures.get(selectedFeatures.size() - 1);
	
		String chrX = hiC.getXContext().getChromosome().getName();
		String chrY = hiC.getYContext().getChromosome().getName();
	
		long startX = firstSelectedContig.getStart1();
		long startY = firstSelectedContig.getStart2();
		long endX = lastSelectedContig.getEnd1();
		long endY = lastSelectedContig.getEnd2();
	
		HashMap<String, String> attributes = new HashMap<>();
	
		return new Feature2D(Feature2D.FeatureType.SELECTED_GROUP, chrX, startX, endX, chrY, startY, endY, getDefaultColor(), attributes);
	}

    public Feature2D addTempSelectedGroup(List<Feature2D> selectedFeatures, HiC hiC) {
        Feature2D tempSelectedGroup = generateTempSelectedGroup(selectedFeatures, hiC);
        annotationLayer.add(hiC.getXContext().getChromosome().getIndex(), hiC.getYContext().getChromosome().getIndex(), tempSelectedGroup);
        return tempSelectedGroup;
    }

    private List<Feature2D> getTempSelectedGroups(int chr1Idx, int chr2Idx) {
        List<Feature2D> tempSelectedGroups = new ArrayList<>();
        List<Feature2D> allVisibleLoops = this.getAllVisibleLoops().getFeatureList(Feature2DList.getKey(chr1Idx, chr2Idx));
        if (allVisibleLoops != null) {
            for (Feature2D feature2D : this.getAllVisibleLoops().getFeatureList(Feature2DList.getKey(chr1Idx, chr2Idx))) {
                if (feature2D.getFeatureType() == Feature2D.FeatureType.SELECTED_GROUP) {
                    tempSelectedGroups.add(feature2D);
                }
            }
        }
        return tempSelectedGroups;
    }

    public void filterTempSelectedGroup(int chr1Idx, int chr2Idx) {
        List<Feature2D> tempSelectedGroups = getTempSelectedGroups(chr1Idx, chr2Idx);
        for (Feature2D feature2D : tempSelectedGroups) {
            this.getAllVisibleLoops().checkAndRemoveFeature(chr1Idx, chr2Idx, feature2D);
        }
    }
	
	private boolean regionsOverlapSignificantly(long start1, long end1, long start2, long end2, double tolerance) {
		
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

        java.util.List<Feature2DList> loops = hic.getAllVisibleLoops();
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
	private long geneXPos(HiC hic, int x, int displacement) {
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
	private long geneYPos(HiC hic, int y, int displacement) {
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

    public AnnotationLayer.LayerType getAnnotationLayerType() {
        return annotationLayer.getLayerType();
    }

    public void clearAnnotations() {
        annotationLayer.clearAnnotations();
    }

    public AnnotationLayer getAnnotationLayer() {
        return annotationLayer;
    }

    private void setAnnotationLayer(AnnotationLayer annotationLayer) {
        this.annotationLayer = annotationLayer;
    }

    public List<Feature2D> getNearbyFeatures(MatrixZoomData zd, int chr1Idx, int chr2Idx, int centerX, int centerY,
                                             int numberOfLoopsToFind, double binOriginX,
                                             double binOriginY, double scaleFactor) {
        return annotationLayer.getNearbyFeatures(zd, chr1Idx, chr2Idx, centerX, centerY, numberOfLoopsToFind,
                binOriginX, binOriginY, scaleFactor);
    }

    private List<Feature2D> getIntersectingFeatures(int chr1Idx, int chr2Idx, net.sf.jsi.Rectangle selectionWindow) {
        return annotationLayer.getIntersectingFeatures(chr1Idx, chr2Idx, selectionWindow);
    }

    /*
     * Gets the contained features within the selection region, including the features that the
     * selection starts and ends in
     */
    public List<Feature2D> getSelectedFeatures(HiC hic, int lastX, int lastY) {
        List<Feature2D> selectedFeatures = new ArrayList<>();
        int chr1Idx = hic.getXContext().getChromosome().getIndex();
        int chr2Idx = hic.getYContext().getChromosome().getIndex();

        boolean previousStatus = annotationLayer.getFeatureHandler().getIsSparsePlottingEnabled();

        // Multiple regions selected
        if (selectionRegion != null) {
			long startX, endX;
			long startY, endY;
	
			int x = selectionRegion.x;
			int width = selectionRegion.width;
	
			int y = selectionRegion.y;
			int height = selectionRegion.height;
	
			// Get starting chrX and ending chrX and window
			startX = geneXPos(hic, x, 0);
			endX = geneXPos(hic, x + width, 0);

            // Get starting chrY and ending chrY and window
            startY = geneYPos(hic, y, 0);
            endY = geneYPos(hic, y + height, 0);

            net.sf.jsi.Rectangle selectionWindow = new net.sf.jsi.Rectangle(startX, startY, endX, endY);

            try {
                //annotationLayer.getFeatureHandler().setSparsePlottingEnabled(true);

                // Get features that are both contained by and touching (nearest single neighbor)
                // the selection rectangle
                List<Feature2D> intersectingFeatures = getIntersectingFeatures(chr1Idx, chr2Idx, selectionWindow);
                selectedFeatures.addAll(intersectingFeatures);

                annotationLayer.getFeatureHandler().setSparsePlottingEnabled(previousStatus);
            } catch (Exception e) {
                //annotationLayer.getFeatureHandler().setSparsePlottingEnabled(false);
                selectionRegion = null;
                return selectedFeatures;
            }
            selectionRegion = null;
            return new ArrayList<>(new HashSet<>(selectedFeatures));

            // Single region selected
        } else {
            try {
                // Find feature that contains selection point
                annotationLayer.getFeatureHandler().setSparsePlottingEnabled(true);
                selectedFeatures.addAll(selectSingleRegion(chr1Idx, chr2Idx, lastX, lastY, hic.getZd(), hic));
                annotationLayer.getFeatureHandler().setSparsePlottingEnabled(previousStatus);
            } catch (Exception e) {
                System.out.println("error:" + e);
                annotationLayer.getFeatureHandler().setSparsePlottingEnabled(false);
                return selectedFeatures;
            }
            return selectedFeatures;
        }
    }

    private List<Feature2D> selectSingleRegion(int chr1Idx, int chr2Idx, int unscaledX, int unscaledY, MatrixZoomData zd, HiC hic) {
		List<Feature2D> selectedFeatures = new ArrayList<>();
	
		final HiCGridAxis xAxis = zd.getXGridAxis();
		final HiCGridAxis yAxis = zd.getYGridAxis();
		final double binOriginX = hic.getXContext().getBinOrigin();
		final double binOriginY = hic.getYContext().getBinOrigin();
		final double scale = hic.getScaleFactor();
	
		long x = (long) (((unscaledX / scale) + binOriginX) * xAxis.getBinSize());
		long y = (long) (((unscaledY / scale) + binOriginY) * yAxis.getBinSize());
	
		Feature2DList features = this.getAnnotationLayer().getFeatureHandler().getAllVisibleLoops();
		List<Feature2D> contigs = features.get(chr1Idx, chr2Idx);
		for (Feature2D feature2D : contigs) {
			if (feature2D.containsPoint(x, y)) {
				selectedFeatures.add(feature2D);
			}
		}
		return selectedFeatures;
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

    public void setLayerNameAndOtherField(String layerName) {
        this.layerName = layerName;
        if (miniNameLabel != null) {
            miniNameLabel.setText(MiniAnnotationsLayerPanel.shortenedName(layerName));
            miniNameLabel.setToolTipText(layerName);
        }
    }

    public void setLayerNameAndField(String layerName) {
        setLayerNameAndOtherField(layerName);
        if (nameTextField != null) {
            nameTextField.setText(layerName);
        }
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
        for (PlottingStyleButton button : plottingStyleButtons) {
            button.setCurrentState(plottingStyle);
        }
    }

    public void exportAnnotations() {
        new SaveAnnotationsDialog(getAnnotationLayer(), getLayerName());
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
        Feature2DHandler.resultContainer result = getFeatureHandler().setLoopList(path, chromosomeHandler);
        if (result.n > 0) {
            setExportAbility(true);
            if (result.color != null) {
                setDefaultColor(result.color);
            }
        }
    }

    public Feature2DList getAllVisibleLoops() {
        return getFeatureHandler().getAllVisibleLoops();
    }


    public void setExportAbility(boolean allowed) {
        canExport = allowed;
        if (exportButton != null) {
            exportButton.setEnabled(true);
        }
        if (censorButton != null) {
            censorButton.setEnabled(true);
        }
    }

    public void setExportButton(JButton exportButton) {
        this.exportButton = exportButton;
    }

    public void setCensorButton(JButton censorButton) {
        this.censorButton = censorButton;
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
        return annotationLayer.getId() == superAdapter.getActiveLayerHandler().getAnnotationLayer().getId();
    }

    public void setActiveLayerButtonStatus(boolean status) {
        for (JToggleButton button : activeLayerButtons) {
            button.setSelected(status);
            button.revalidate();
        }
    }

    public void setActiveLayerButton(JToggleButton activeLayerButton) {
        activeLayerButtons.add(activeLayerButton);
    }

    public int getNumberOfFeatures() {
        return annotationLayer.getNumberOfFeatures();
    }

    public void setColorOfAllAnnotations(Color color) {
        setDefaultColor(color);
        annotationLayer.setColorOfAllAnnotations(color);
    }

    public Color getDefaultColor() {
        return defaultColor;
    }

    private void setDefaultColor(Color defaultColor) {
        this.defaultColor = defaultColor;
        if (colorChooserPanels.size() > 0) {
            for (ColorChooserPanel colorChooserPanel : colorChooserPanels)
                colorChooserPanel.setSelectedColor(defaultColor);
        }
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

    public FeatureRenderer.LineStyle getLineStyle() {
        return this.lineStyle;
    }

    public void setLineStyle(FeatureRenderer.LineStyle lineStyle) {
        this.lineStyle = lineStyle;
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
        Feature2DList origLists = handlerOriginal.getAnnotationLayer().getFeatureList();

        annotationLayer.createMergedLoopLists(origLists.deepCopy());
        setImportAnnotationsEnabled(handlerOriginal.getImportAnnotationsEnabled());
        setExportAbility(handlerOriginal.getExportCapability());
        setIsSparse(handlerOriginal.getIsSparse());
    }

    public void mergeDetailsFrom(Collection<AnnotationLayerHandler> originalHandlers) {

        StringBuilder cleanedTitle = new StringBuilder();
        List<Feature2DList> allLists = new ArrayList<>();
        for (AnnotationLayerHandler originalHandler : originalHandlers) {
            featureType = originalHandler.featureType;
            allLists.add(originalHandler.getAnnotationLayer().getFeatureList());
            cleanedTitle.append("-").append(originalHandler.getLayerName().toLowerCase().replaceAll("layer", "").replaceAll("\\s", ""));
            setLayerVisibility(originalHandler.getLayerVisibility());
            setColorOfAllAnnotations(originalHandler.getDefaultColor());
            importAnnotationsEnabled |= originalHandler.getImportAnnotationsEnabled();
            canExport |= originalHandler.getExportCapability();
        }
        annotationLayer.createMergedLoopLists(allLists);

        setExportAbility(canExport);
        setLayerNameAndField("Merger" + cleanedTitle);
    }

    public void togglePlottingStyle() {
        try {
            plottingStyleButtons.get(0).doClick();
        } catch (Exception e) {
            setPlottingStyle(FeatureRenderer.getNextState(getPlottingStyle()));
        }
    }

    public void setPlottingStyleButton(PlottingStyleButton plottingStyleButton) {
        plottingStyleButtons.add(plottingStyleButton);
    }

    public void setColorChooserPanel(ColorChooserPanel colorChooserPanel) {
        colorChooserPanels.add(colorChooserPanel);
    }

    public void setNameTextField(JTextField nameTextField) {
        this.nameTextField = nameTextField;
    }

    public void setMiniNameLabelField(JLabel miniNameLabel) {
        this.miniNameLabel = miniNameLabel;
    }

    public void setProperties(AnnotationLayer scaffoldLayer, String name, Color color) {
        setAnnotationLayer(scaffoldLayer);
        setLayerNameAndField(name);
        setColorOfAllAnnotations(color);
    }

    public List<Feature2DGuiContainer> convertToFeaturePairs(AnnotationLayerHandler handler,
                                                             List<Feature2D> loops, MatrixZoomData zd, double binOriginX, double binOriginY, double scaleFactor) {
        return annotationLayer.getFeatureHandler().convertFeaturesToFeaturePairs(handler, loops, zd, binOriginX, binOriginY, scaleFactor);
    }
}
