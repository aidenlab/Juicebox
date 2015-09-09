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

package juicebox.track.feature;

import com.sun.org.apache.xalan.internal.utils.FeatureManager;
import juicebox.HiC;
import juicebox.data.ExpectedValueFunction;
import juicebox.data.MatrixZoomData;
import juicebox.gui.MainMenuBar;
import juicebox.track.HiCGridAxis;

import java.awt.*;
import java.io.PrintWriter;
import java.util.HashMap;

/**
 * Created by Marie on 6/4/15.
 */
public class CustomAnnotationHandler {

    // displacement in terms of gene pos
    private final int peakDisplacement = 3;
    // threshold in terms of pixel pos
    private final int threshold = 10;
    String id;
    private PrintWriter outputFile;
    private Rectangle selectionRegion;
    private Point selectionPoint;
    private FeatureType featureType;
    private boolean hasPoint, hasRegion;

    public CustomAnnotationHandler() {
        featureType = FeatureType.NONE;
        resetSelection();
    }

    private void resetSelection() {
        hasPoint = false;
        hasRegion = false;
        selectionRegion = null;
        selectionPoint = null;
        featureType = FeatureType.NONE;
    }

    public boolean isEnabled() {
        return featureType != FeatureType.NONE;
    }

    public boolean isPeak() {
        return featureType == FeatureType.PEAK;
    }

    public void doGeneric() {
        featureType = FeatureType.GENERIC;
    }

    public void doPeak() {
        featureType = FeatureType.PEAK;
    }

    private void doDomain() {
        featureType = FeatureType.DOMAIN;
    }

    // Update selection region from new rectangle
    public void updateSelectionRegion(Rectangle newRegion) {
        hasPoint = false;
        hasRegion = true;
        doDomain();
        selectionRegion = newRegion;
    }

    // Update selection region from new coordinates
    public Rectangle updateSelectionRegion(int x, int y, int deltaX, int deltaY) {

        int x2, y2;
        hasPoint = false;
        hasRegion = true;
        doDomain();
        Rectangle lastRegion, damageRect;

        lastRegion = selectionRegion;

        if (deltaX == 0 || deltaY == 0) {
            return null;
        }

        x2 = deltaX > 0 ? x : x + deltaX;
        y2 = deltaY > 0 ? y : y + deltaY;
        selectionRegion = new Rectangle(x2, y2, Math.abs(deltaX), Math.abs(deltaY));

        damageRect = lastRegion == null ? selectionRegion : selectionRegion.union(lastRegion);
        damageRect.x--;
        damageRect.y--;
        damageRect.width += 2;
        damageRect.height += 2;
        return damageRect;
    }

    public void updateSelectionPoint(int x, int y) {
        selectionPoint = new Point(x, y);
        hasPoint = true;
    }

    public void addFeature(HiC hic, CustomAnnotation customAnnotations) {

        int start1, start2, end1, end2;
        Feature2D newFeature;
        MainMenuBar.exportAnnotationsMI.setEnabled(true);
        MainMenuBar.undoMenuItem.setEnabled(true);
        String chr1 = hic.getXContext().getChromosome().getName();
        String chr2 = hic.getYContext().getChromosome().getName();
        int chr1Idx = hic.getXContext().getChromosome().getIndex();
        int chr2Idx = hic.getYContext().getChromosome().getIndex();
        HashMap<String, String> attributes = new HashMap<String, String>();

        switch (featureType) {
            case GENERIC:
                start1 = geneXPos(hic, selectionRegion.x, 0);
                start2 = geneYPos(hic, selectionRegion.y, 0);
                end1 = geneXPos(hic, selectionRegion.x + selectionRegion.width, 0);
                end2 = geneYPos(hic, selectionRegion.y + selectionRegion.height, 0);
                newFeature = new Feature2D(Feature2D.generic, chr1, start1, end1, chr2, start2, end2,
                        java.awt.Color.orange, attributes);
                customAnnotations.add(chr1Idx, chr2Idx, newFeature);
                break;
            case PEAK:
                start1 = geneXPos(hic, selectionPoint.x, -1 * peakDisplacement);
                end1 = geneXPos(hic, selectionPoint.x, peakDisplacement);

                if (chr1Idx == chr2Idx && nearDiagonal(hic, selectionPoint.x, selectionPoint.y)) {
                    start2 = start1;
                    end2 = end1;
                } else {
                    //Displacement inside before geneYPos scales to resolution
                    start2 = geneYPos(hic, selectionPoint.y, -1 * peakDisplacement);
                    end2 = geneYPos(hic, selectionPoint.y, peakDisplacement);
                }

                //UNCOMMENT to take out annotation data
                boolean exportData = true;
                if (exportData) {
                    int tempBinX0 = getXBin(hic, selectionPoint.x);
                    int tempBinY0 = getYBin(hic, selectionPoint.y);
                    int tempBinX, tempBinY;
                    final MatrixZoomData zd = hic.getZd();

                    float totObserved = 0;
                    float totExpected = 0;
                    int count = 0;
                    float observedValue;

                    for (int i = -1 * peakDisplacement; i <= peakDisplacement; i++) {
                        tempBinX = tempBinX0 + i;
                        for (int j = -1 * peakDisplacement; j <= peakDisplacement; j++) {
                            tempBinY = tempBinY0 + j;
                            observedValue = hic.getNormalizedObservedValue(tempBinX, tempBinY);

                            double ev = zd.getAverageCount();
                            ExpectedValueFunction df = hic.getExpectedValues();
                            if (df != null) {
                                int distance = Math.abs(tempBinX - tempBinY);
                                ev = df.getExpectedValue(chr1Idx, distance);

                            }
                            totObserved += observedValue;
                            totExpected += ev;
                            count++;
                        }
                    }
                }

                newFeature = new Feature2D(Feature2D.peak, chr1, start1, end1, chr2, start2, end2,
                        Color.DARK_GRAY, attributes);
                customAnnotations.add(chr1Idx, chr2Idx, newFeature);
                break;
            case DOMAIN:
                start1 = geneXPos(hic, selectionRegion.x, 0);
                end1 = geneXPos(hic, selectionRegion.x + selectionRegion.width, 0);

                // Snap if close to diagonal
                if (chr1Idx == chr2Idx && nearDiagonal(hic, selectionRegion.x, selectionRegion.y)) {
                    // Snap to min of horizontal stretch and vertical stretch
                    if (selectionRegion.width <= selectionRegion.y) {
                        start2 = start1;
                        end2 = end1;
                    } else {
                        start2 = geneYPos(hic, selectionRegion.y, 0);
                        end2 = geneYPos(hic, selectionRegion.y + selectionRegion.height, 0);
                        start1 = start2;
                        end1 = end2;
                    }
                    // Otherwise record as drawn
                } else {
                    start2 = geneYPos(hic, selectionRegion.y, 0);
                    end2 = geneYPos(hic, selectionRegion.y + selectionRegion.height, 0);
                }

                newFeature = new Feature2D(Feature2D.domain, chr1, start1, end1, chr2, start2, end2,
                        Color.GREEN, attributes);
                customAnnotations.add(chr1Idx, chr2Idx, newFeature);
                break;
            default:
                resetSelection();
        }
    }

    public CustomAnnotation addVisibleLoops(HiC hic, CustomAnnotation customAnnotations) {
        final MatrixZoomData zd = hic.getZd();
        if (zd == null || hic.getXContext() == null) return customAnnotations;

        java.util.List<Feature2DList> loops = hic.getAllVisibleLoopLists();
        if (loops == null) return customAnnotations;
        if (customAnnotations == null)
            return null;

        for (Feature2DList list : loops) {
            customAnnotations.addVisibleToCustom(list);
        }
        return customAnnotations;
    }

    public void undo(CustomAnnotation customAnnotations) {
        customAnnotations.undo();
        MainMenuBar.undoMenuItem.setEnabled(false);
    }

    private boolean nearDiagonal(HiC hic, int x, int y) {
        int start1 = getXBin(hic, x);
        int start2 = getYBin(hic, y);

        return Math.abs(start1 - start2) < threshold;
    }

    //helper for getannotatemenu
    private int geneXPos(HiC hic, int x, int displacement) {
        final MatrixZoomData zd = hic.getZd();
        if (zd == null) return -1;
        HiCGridAxis xGridAxis = zd.getXGridAxis();
        int binX = getXBin(hic, x) + displacement;
        return xGridAxis.getGenomicStart(binX) + 1;
    }

    //helper for getannotatemenu
    private int geneYPos(HiC hic, int y, int displacement) {
        final MatrixZoomData zd = hic.getZd();
        if (zd == null) return -1;
        HiCGridAxis yGridAxis = zd.getYGridAxis();
        int binY = getYBin(hic, y) + displacement;
        return yGridAxis.getGenomicStart(binY) + 1;
    }

    private int getXBin(HiC hic, int x) {
        return (int) (hic.getXContext().getBinOrigin() + x / hic.getScaleFactor());
    }

    private int getYBin(HiC hic, int y) {
        return (int) (hic.getYContext().getBinOrigin() + y / hic.getScaleFactor());
    }

    // TODO merge with Feature2D as public enum type
    enum FeatureType {NONE, PEAK, DOMAIN, GENERIC}

}
