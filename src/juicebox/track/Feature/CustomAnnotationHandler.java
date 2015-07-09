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
import juicebox.MainWindow;
import juicebox.data.MatrixZoomData;
import juicebox.track.HiCGridAxis;

import java.awt.*;
import java.io.PrintWriter;
import java.util.*;

/**
 * Created by Marie on 6/4/15.
 */
public class CustomAnnotationHandler {

    // displacement in terms of gene pos
    private int peakDisplacement = 3;
    private PrintWriter outputFile;
    // threshold in terms of pixel pos
    private int threshold = 10;
    private Rectangle selectionRegion;
    private Point selectionPoint;
    private HiC hic;
    FeatureType featureType;
    private boolean hasPoint, hasRegion;
    MainWindow mainWindow;
    String id;

    enum FeatureType {NONE, PEAK, DOMAIN, GENERIC}

    public CustomAnnotationHandler(MainWindow mainWIndow, HiC hic){
        this.mainWindow = mainWindow;
        this.hic = hic;
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
        if (featureType != FeatureType.NONE){
            return true;
        }
        return false;
    }

    public boolean isPeak() {
        if (featureType == FeatureType.PEAK){
            return true;
        }
        return false;
    }

    public void doGeneric(){
        featureType = FeatureType.GENERIC;
    }

    public void doPeak(){
        featureType = FeatureType.PEAK;
    }

    public void doDomain(){
        featureType = FeatureType.DOMAIN;
    }

    // Update selection region from new rectangle
    public void updateSelectionRegion(Rectangle newRegion){
        hasPoint = false;
        hasRegion = true;
        doDomain();
        selectionRegion = newRegion;
    }

    // Update selection region from new coordinates
    public Rectangle updateSelectionRegion(int x, int y, int deltaX, int deltaY){

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

    public void updateSelectionPoint(int x, int y){
        selectionPoint = new Point(x, y);
        hasPoint = true;
    }

    public void addFeature(CustomAnnotation customAnnotations){

        int start1, start2, end1, end2;
        Feature2D newFeature;
        MainWindow.exportAnnotationsMI.setEnabled(true);
        MainWindow.undoMenuItem.setEnabled(true);
        String chr1 = hic.getXContext().getChromosome().getName();
        String chr2 = hic.getYContext().getChromosome().getName();
        int chr1Idx = hic.getXContext().getChromosome().getIndex();
        int chr2Idx = hic.getYContext().getChromosome().getIndex();

        switch (featureType) {
            case GENERIC:
                start1 = geneXPos(selectionRegion.x, 0);
                start2 = geneYPos(selectionRegion.y, 0);
                end1 = geneXPos(selectionRegion.x + selectionRegion.width, 0);
                end2 = geneYPos(selectionRegion.y + selectionRegion.height, 0);
                newFeature = new Feature2D(Feature2D.generic, chr1, start1, end1, chr2, start2, end2,
                        java.awt.Color.orange, new HashMap<String,String>());
                customAnnotations.add(chr1Idx, chr2Idx, newFeature);
                break;
            case PEAK:
                start1 = geneXPos(selectionPoint.x, -1 * peakDisplacement);
                end1 = geneXPos(selectionPoint.x,  peakDisplacement);

                if (chr1Idx == chr2Idx && nearDiagonal(selectionPoint.x, selectionPoint.y)){
                    start2 = start1;
                    end2 = end1;
                } else {
                    //Displacement inside before geneYPos scales to resolution
                    start2 = geneYPos(selectionPoint.y, -1 * peakDisplacement);
                    end2 = geneYPos(selectionPoint.y, peakDisplacement);
                }

                newFeature = new Feature2D(Feature2D.peak, chr1, start1, end1, chr2, start2, end2,
                        Color.DARK_GRAY, new HashMap<String,String>());
                customAnnotations.add(chr1Idx, chr2Idx, newFeature);
                break;
            case DOMAIN:
                start1 = geneXPos(selectionRegion.x, 0);
                end1 = geneXPos(selectionRegion.x + selectionRegion.width, 0);

                if (chr1Idx == chr2Idx && nearDiagonal(selectionRegion.x, selectionRegion.y)){
                    start2 = start1;
                    end2 = end1;
                } else {
                    start2 = geneYPos(selectionRegion.y, 0);
                    end2 = geneYPos(selectionRegion.y + selectionRegion.height, 0);
                }

                newFeature = new Feature2D(Feature2D.domain, chr1, start1, end1, chr2, start2, end2,
                        Color.GREEN, new HashMap<String,String>());
                customAnnotations.add(chr1Idx, chr2Idx, newFeature);
                break;
            default:
                resetSelection();
        }
    }

    public CustomAnnotation addVisibleLoops(CustomAnnotation customAnnotations){
        final MatrixZoomData zd = hic.getZd();
        if (zd == null || hic.getXContext() == null) return customAnnotations;

        java.util.List<Feature2DList> loops = hic.getAllVisibleLoopLists();
        if (loops == null) return customAnnotations;
        if (customAnnotations == null)
            return null;

        for(Feature2DList list : loops){
            customAnnotations.addVisibleToCustom(list);
        }
        return customAnnotations;
    }

    public void undo(CustomAnnotation customAnnotations){
        customAnnotations.undo();
        MainWindow.undoMenuItem.setEnabled(false);
    }

    private boolean nearDiagonal(int x, int y){
        int start1 = getXBin(x);
        int start2 = getYBin(y);

        if (Math.abs(start1 - start2) < threshold){
            return true;
        }
        return false;
    }

    private int getXBin(int x){
        return (int) (hic.getXContext().getBinOrigin() + x / hic.getScaleFactor());
    }

    private int getYBin(int y){
        return (int) (hic.getYContext().getBinOrigin() + y / hic.getScaleFactor());
    }

    //helper for getannotatemenu
    private int geneXPos(int x, int displacement){
        final MatrixZoomData zd = hic.getZd();
        if (zd == null) return -1;
        HiCGridAxis xGridAxis = zd.getXGridAxis();
        int binX = getXBin(x) + displacement;
        return xGridAxis.getGenomicStart(binX) + 1;
    }

    //helper for getannotatemenu
    private int geneYPos(int y, int displacement){
        final MatrixZoomData zd = hic.getZd();
        if (zd == null) return -1;
        HiCGridAxis yGridAxis = zd.getYGridAxis();
        int binY = getYBin(y) + displacement;
        return yGridAxis.getGenomicStart(binY) + 1;
    }


}
