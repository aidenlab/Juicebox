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

import juicebox.HiC;
import juicebox.data.MatrixZoomData;
import juicebox.gui.MainMenuBar;
import juicebox.track.HiCGridAxis;

import java.awt.*;
import java.util.HashMap;
import org.broad.igv.util.Pair;

/**
 * Created by Marie on 6/4/15.
 */
public class CustomAnnotationHandler {

    // displacement in terms of gene pos
    private final int peakDisplacement = 3;
    // threshold in terms of pixel pos
    private final int threshold = 15;
    private Rectangle selectionRegion;
    private Point selectionPoint;
    private Feature2D.FeatureType featureType;
    private Feature2D lastResizeLoop = null;
    private int lastChr1Idx = -1;
    private int lastChr2Idx = -1;
    private Pair<Integer, Integer> lastStarts = null;
    private Pair<Integer, Integer> lastEnds = null;

    public CustomAnnotationHandler() {
        featureType = Feature2D.FeatureType.NONE;
        resetSelection();
    }

    private void resetSelection() {
        selectionRegion = null;
        selectionPoint = null;
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

    public void setStationaryStart(int start1, int start2){
        lastStarts = new Pair<Integer, Integer>(start1, start2);
        lastEnds = null;
    }

    public void setStationaryEnd(int end1, int end2){
        lastEnds = new Pair<Integer, Integer>(end1, end2);
        lastStarts = null;
    }

    // Update selection region from new rectangle
    public void updateSelectionRegion(Rectangle newRegion) {
        doDomain();
        selectionRegion = newRegion;
    }

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

    public void updateSelectionPoint(int x, int y) {
        selectionPoint = new Point(x, y);
    }

    // Adds to lower lefthand side, for consistency.
    public void addFeature(HiC hic, CustomAnnotation customAnnotations) {

        int start1, start2, end1, end2;
        Feature2D newFeature;
        MainMenuBar.exportAnnotationsMI.setEnabled(true);
        MainMenuBar.undoMenuItem.setEnabled(true);
        clearLastItem();
        String chr1 = hic.getXContext().getChromosome().getName();
        String chr2 = hic.getYContext().getChromosome().getName();
        int chr1Idx = hic.getXContext().getChromosome().getIndex();
        int chr2Idx = hic.getYContext().getChromosome().getIndex();
        HashMap<String, String> attributes = new HashMap<String, String>();
        int rightBound = hic.getXContext().getChromosome().getLength();
        int bottomBound = hic.getYContext().getChromosome().getLength();
        int leftBound = 0;
        int x = selectionRegion.x;
        int y = selectionRegion.y;
        int width = selectionRegion.width;
        int height = selectionRegion.height;

        start1 = geneXPos(hic, x, 0);
        end1 = geneXPos(hic, x + width, 0);

        // Snap if close to diagonal
        if (chr1Idx == chr2Idx && nearDiagonal(hic, x, y) && nearDiagonal(hic, x + width, y + height)) {
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
            // Otherwise record as drawn
        } else {
            start2 = geneYPos(hic, y, 0);
            end2 = geneYPos(hic, y + height, 0);
        }

        // Make sure bounds aren't unreasonable (out of HiC map)
//                int rightBound = hic.getChromosomes().get(0).getLength();
//                int bottomBound = hic.getChromosomes().get(1).getLength();
        start1 = Math.min(Math.max(start1, leftBound), rightBound);
        start2 = Math.min(Math.max(start2, leftBound), bottomBound);
        end1 = Math.max(Math.min(end1, rightBound), leftBound);
        end2 = Math.max(Math.min(end2, bottomBound), leftBound);

        // Check for anchored corners
        if (lastStarts != null){
            if (lastStarts.getFirst() < end1 && lastStarts.getSecond() < end2) {
                start1 = lastStarts.getFirst();
                start2 = lastStarts.getSecond();
            }
        } else if (lastEnds != null){
            if (start1 < lastEnds.getFirst() && start2 < lastEnds.getSecond()) {
                end1 = lastEnds.getFirst();
                end2 = lastEnds.getSecond();
            }
        }

        // Add new feature
        newFeature = new Feature2D(Feature2D.FeatureType.DOMAIN, chr1, start1, end1, chr2, start2, end2,
                Color.GREEN, attributes);
        customAnnotations.add(chr1Idx, chr2Idx, newFeature);
        lastStarts = null;
        lastEnds = null;
    }

    public CustomAnnotation addVisibleLoops(HiC hic, CustomAnnotation customAnnotations) {
        try {
            hic.getZd();
        } catch (Exception e) {
            return customAnnotations;
        }

        if (hic.getXContext() == null || hic.getYContext() == null)
            return customAnnotations;

        java.util.List<Feature2DList> loops = hic.getAllVisibleLoopLists();
        if (loops == null) return customAnnotations;
        if (customAnnotations == null) {
            System.out.println("Error! Custom annotations should not be null!");
            return null;
        }

        // Add each loop list to the custom annotation list
        for (Feature2DList list : loops) {
            customAnnotations.addVisibleToCustom(list);
        }
        return customAnnotations;
    }

    public void undo(CustomAnnotation customAnnotations) {
        customAnnotations.undo();
        if (lastResizeLoop != null) {
            customAnnotations.add(lastChr1Idx, lastChr2Idx, lastResizeLoop);
            resetSelection();
        }
        MainMenuBar.undoMenuItem.setEnabled(false);
    }

    private boolean nearDiagonal(HiC hic, int x, int y) {
        int start1 = getXBin(hic, x);
        int start2 = getYBin(hic, y);

        return Math.abs(start1 - start2) < threshold;
    }

    //helper for getannotatemenu
    private int geneXPos(HiC hic, int x, int displacement) {
        try {
            final MatrixZoomData zd = hic.getZd();
            if (zd == null) return -1;
            HiCGridAxis xGridAxis = zd.getXGridAxis();
            int binX = getXBin(hic, x) + displacement;
            return xGridAxis.getGenomicStart(binX) + 1;
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
            return yGridAxis.getGenomicStart(binY) + 1;
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
}
