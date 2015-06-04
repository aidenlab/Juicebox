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

package juicebox.track.Feature;

import juicebox.HiC;
import juicebox.data.MatrixZoomData;
import juicebox.mapcolorui.HeatmapPanel;
import juicebox.track.HiCGridAxis;

import java.awt.*;
import java.util.HashMap;

/**
 * Created by Marie on 6/4/15.
 */
public class CustomAnnotationHandler {

    private int displacement;
    private Rectangle selectionRegion;
    private Point selectionPoint;
    private HiC hic;
    FeatureType featureType;
    private boolean hasPoint, hasRegion;

    enum FeatureType {NONE, PEAK, DOMAIN, GENERIC}

    public CustomAnnotationHandler(HiC hic){
        this.hic = hic;
        featureType = FeatureType.NONE;
        reset();
    }

    private void reset() {
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

    public Rectangle updateSelectionRegion(int x, int y, int deltaX, int deltaY){

        Rectangle lastRegion, damageRect;

        lastRegion = selectionRegion;

        if (deltaX == 0 || deltaY == 0) {
            return null;
        }

        // Constrain aspect ratio of zoom rectangle to that of panel
//                    double aspectRatio2 = (double) getWidth() / getHeight();
//                    if (deltaX * aspectRatio2 > deltaY) {
//                        deltaY = (int) (deltaX / aspectRatio2);
//                    } else {
//                        deltaX = (int) (deltaY * aspectRatio2);
//                    }

        x = deltaX > 0 ? x : x + deltaX;
        y = deltaY > 0 ? y : y + deltaY;
        selectionRegion = new Rectangle(x, y, Math.abs(deltaX), Math.abs(deltaY));

        damageRect = lastRegion == null ? selectionRegion : selectionRegion.union(lastRegion);
        damageRect.x--;
        damageRect.y--;
        damageRect.width += 2;
        damageRect.height += 2;
        hasRegion = true;
        return damageRect;
    }

    public void updateSelectionPoint(int x, int y){
        selectionPoint = new Point(x, y);
        hasPoint = true;
    }

    public void addFeature(CustomAnnotation customAnnotations){
        int start1, start2, end1, end2, chr1Idx, chr2Idx;
        String chr1, chr2;
        Feature2D newFeature;

        switch (featureType) {
            case GENERIC:
                start1 = geneXPos(selectionRegion.x);
                start2 = geneYPos(selectionRegion.y);
                end1 = geneXPos(selectionRegion.x + selectionRegion.height);
                end2 = geneYPos(selectionRegion.y + selectionRegion.width);
                chr1Idx = hic.getXContext().getChromosome().getIndex();
                chr2Idx = hic.getYContext().getChromosome().getIndex();
                chr1 = hic.getXContext().getChromosome().getName();
                chr2 = hic.getYContext().getChromosome().getName();
                newFeature = new Feature2D(Feature2D.generic, chr1, start1, end1, chr2, start2, end2,
                        java.awt.Color.orange, new HashMap<String,String>());
                customAnnotations.add(chr1Idx, chr2Idx, newFeature);
                break;
            case PEAK:
                start1 = geneXPos(selectionPoint.x - displacement);
                start2 = geneYPos(selectionPoint.y - displacement);
                end1 = geneXPos(selectionPoint.x + displacement);
                end2 = geneYPos(selectionPoint.y - displacement);
                chr1Idx = hic.getXContext().getChromosome().getIndex();
                chr2Idx = hic.getYContext().getChromosome().getIndex();
                chr1 = hic.getXContext().getChromosome().getName();
                chr2 = hic.getYContext().getChromosome().getName();
                newFeature = new Feature2D(Feature2D.peak, chr1, start1, end1, chr2, start2, end2,
                        Color.DARK_GRAY, new HashMap<String,String>());
                customAnnotations.add(chr1Idx, chr2Idx, newFeature);
                break;
            case DOMAIN:
                start1 = geneXPos(selectionRegion.x);
                start2 = start1;
                end1 = geneXPos(selectionRegion.x + selectionRegion.height);
                end2 = end1;
                chr1Idx = hic.getXContext().getChromosome().getIndex();
                chr2Idx = hic.getYContext().getChromosome().getIndex();
                chr1 = hic.getXContext().getChromosome().getName();
                chr2 = hic.getYContext().getChromosome().getName();
                newFeature = new Feature2D(Feature2D.domain, chr1, start1, end1, chr2, start2, end2,
                        Color.GREEN, new HashMap<String,String>());
                customAnnotations.add(chr1Idx, chr2Idx, newFeature);
                break;
        }
    }

    // Add a generic feature annotation
    public void addGeneric(CustomAnnotation customAnnotations) {
        final int start1 = geneXPos(selectionRegion.x);
        final int start2 = geneYPos(selectionRegion.y);
        final int end1 = geneXPos(selectionRegion.x + selectionRegion.height);
        final int end2 = geneYPos(selectionRegion.y + selectionRegion.width);
        final int chr1Idx = hic.getXContext().getChromosome().getIndex();
        final int chr2Idx = hic.getYContext().getChromosome().getIndex();
        final String chr1 = hic.getXContext().getChromosome().getName();
        final String chr2 = hic.getYContext().getChromosome().getName();
        Feature2D newFeature = new Feature2D(Feature2D.generic, chr1, start1, end1, chr2, start2, end2,
                java.awt.Color.orange, new HashMap<String,String>());
        customAnnotations.add(chr1Idx, chr2Idx, newFeature);
    }

    // Add a peak annotation
    public void addPeak(CustomAnnotation customAnnotations){
        final int start1 = geneXPos(selectionPoint.x - displacement);
        final int start2 = geneYPos(selectionPoint.y - displacement);
        final int end1 = geneXPos(selectionPoint.x + displacement);
        final int end2 = geneYPos(selectionPoint.y - displacement);
        final int chr1Idx = hic.getXContext().getChromosome().getIndex();
        final int chr2Idx = hic.getYContext().getChromosome().getIndex();
        final String chr1 = hic.getXContext().getChromosome().getName();
        final String chr2 = hic.getYContext().getChromosome().getName();
        Feature2D newFeature = new Feature2D(Feature2D.peak, chr1, start1, end1, chr2, start2, end2,
                Color.DARK_GRAY, new HashMap<String,String>());
        customAnnotations.add(chr1Idx, chr2Idx, newFeature);
    }

    // Add a domain annotation
    public void addDomain(CustomAnnotation customAnnotations){
        final int start1 = geneXPos(selectionRegion.x);
        final int start2 = start1;
        final int end1 = geneXPos(selectionRegion.x + selectionRegion.height);
        final int end2 = end1;
        final int chr1Idx = hic.getXContext().getChromosome().getIndex();
        final int chr2Idx = hic.getYContext().getChromosome().getIndex();
        final String chr1 = hic.getXContext().getChromosome().getName();
        final String chr2 = hic.getYContext().getChromosome().getName();
        Feature2D newFeature = new Feature2D(Feature2D.domain, chr1, start1, end1, chr2, start2, end2,
                Color.GREEN, new HashMap<String,String>());
        customAnnotations.add(chr1Idx, chr2Idx, newFeature);
    }

    //helper for getannotatemenu
    private int geneXPos(int x){
        final MatrixZoomData zd = hic.getZd();
        if (zd == null) return -1;
        HiCGridAxis xGridAxis = zd.getXGridAxis();
        int binX = (int) (hic.getXContext().getBinOrigin() + x / hic.getScaleFactor());
        return xGridAxis.getGenomicStart(binX) + 1;
    }

    //helper for getannotatemenu
    private int geneYPos(int y){
        final MatrixZoomData zd = hic.getZd();
        if (zd == null) return -1;
        HiCGridAxis yGridAxis = zd.getYGridAxis();
        int binY = (int) (hic.getYContext().getBinOrigin() + y / hic.getScaleFactor());
        return yGridAxis.getGenomicStart(binY) + 1;
    }


}
