/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

import juicebox.DirectoryManager;
import juicebox.HiCGlobals;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.mapcolorui.Feature2DHandler;

import java.awt.*;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Marie on 6/3/15.
 * Modified by muhammadsaadshamim
 */
public class AnnotationLayer {

    private static int i;
    private final int id;
    private CustomAnnotationRTree2DHandler customAnnotationRTreeHandler;
    private boolean unsavedEdits;
    private boolean nothingSavedYet;
    private Feature2D lastItem;
    private int lastChr1Idx;
    private int lastChr2Idx;
    private PrintWriter tempWriter;
    private File tempFile;
    private ArrayList<String> attributeKeys;
    private LayerType layerType;

    public AnnotationLayer() {
        id = i++;
        nothingSavedYet = true;
        reset();
        layerType = LayerType.DEFAULT;
    }
    public AnnotationLayer(Feature2DList inputList) {
        this();
        this.customAnnotationRTreeHandler = new CustomAnnotationRTree2DHandler(inputList);
    }

    public AnnotationLayer(Feature2DHandler feature2DHandler, LayerType layerType) {
        this(feature2DHandler.getFeatureList());
        setLayerType(layerType);
    }

    // Clear all annotations
    private void reset() {
        nothingSavedYet = true;
        lastChr1Idx = -1;
        lastChr2Idx = -1;
        lastItem = null;
        unsavedEdits = false;
        customAnnotationRTreeHandler = new CustomAnnotationRTree2DHandler(new Feature2DList());
        attributeKeys = new ArrayList<>(); // TODO make sure attributes get copied over
    }

    public void clearAnnotations() {
        reset();
        deleteTempFile();
    }

    public LayerType getLayerType() {
        return layerType;
    }

    public void setLayerType(LayerType layerType) {
        this.layerType = layerType;
    }

    //add annotation to feature2D list
    public void add(int chr1Idx, int chr2Idx, Feature2D feature) {
        if (feature == null) {
            return;
        }
        // Add attributes to feature
        List<String> featureKeys = feature.getAttributeKeys();
        for (String customKey : attributeKeys) {
            if (!featureKeys.contains(customKey)) {
                feature.addStringAttribute(customKey, "null");
                //System.out.println("Added" + customKey);
            }
        }
        getAndAddAttributes(featureKeys);

        customAnnotationRTreeHandler.add(chr1Idx, chr2Idx, feature);

        lastChr1Idx = chr1Idx;
        lastChr2Idx = chr2Idx;
        lastItem = feature;

        // Autosave the information
        unsavedEdits = true;
        if (nothingSavedYet) {
            makeTempFile();
            nothingSavedYet = false;
        }
        updateAutoSave();
    }

    // Requires that lastItem is not null
    private void makeTempFile() {
        tempFile = getAutosaveFile();
        tempWriter = HiCFileTools.openWriter(tempFile);

        Feature2D singleFeature = customAnnotationRTreeHandler.extractSingleFeature();
        if (singleFeature == null) {
            tempWriter.println(Feature2D.getDefaultOutputFileHeader());
        } else {
            tempWriter.println(singleFeature.getOutputFileHeader());
        }
    }

    private void deleteTempFile() {
        if (tempWriter != null) {
            tempWriter.close();
        }
        if (tempFile == null) {
            tempFile = getAutosaveFile();
        }
        if (tempFile.exists()) {
            tempFile.delete();
        }
        this.unsavedEdits = false;
    }

    public boolean getLayerVisibility() {
        return customAnnotationRTreeHandler.getLayerVisibility();
    }

    // Set show loops
    public void setLayerVisibility(boolean newStatus) {
        customAnnotationRTreeHandler.setLayerVisibility(newStatus);
    }

    // Creates unique identifier for Feature2D based on start and end positions.
    private String getIdentifier(Feature2D feature) {
        return "" + feature.getStart1() + feature.getEnd1() + feature.getStart2() + feature.getEnd2();
    }

    // Undo last move
    public void undo() {
        removeRecentFromList(lastChr1Idx, lastChr2Idx, lastItem);
    }

    /**
     * Export feature list to given file path
     */
    private void updateAutoSave() {
        if (unsavedEdits && lastItem != null) {
            customAnnotationRTreeHandler.autoSaveNew(tempWriter, lastItem);
        }
    }

    /**
     * Export feature list to given file path, including header
     */
    private void reSaveAll() {
        deleteTempFile();
        makeTempFile();
        customAnnotationRTreeHandler.autoSaveAll(tempWriter);
    }

    public boolean hasLoop(MatrixZoomData zd, int chrIdx1, int chrIdx2, int x, int y, int n,
                           double binOriginX, double binOriginY, double scale, Feature2D feature) {
        if (chrIdx1 > 0 && chrIdx2 > 0) {
            List<Feature2D> featureList = getNearbyFeatures(zd, chrIdx1, chrIdx2, x, y, n, binOriginX, binOriginY, scale);
            return featureList.contains(feature) || featureList.contains(feature.reflectionAcrossDiagonal());
        }
        return false;
    }

    private void removeRecentFromList(int idx1, int idx2, Feature2D feature) {

        if (idx1 > 0 && idx2 > 0) {
            List<Feature2D> lastList;
            String mirrorIdentity = "" + feature.getStart2() + feature.getEnd2() + feature.getStart1() + feature.getEnd1();

            lastList = customAnnotationRTreeHandler.get(idx1, idx2);
            unsavedEdits = customAnnotationRTreeHandler.checkAndRemoveFeature(idx1, idx2, feature);

            if (!unsavedEdits) {
                Feature2D removeFeature = null;
                for (Feature2D aFeature : lastList) {
                    if (getIdentifier(aFeature).compareTo(mirrorIdentity) == 0) {
                        removeFeature = aFeature;
                        unsavedEdits = true;
                    }
                }
                customAnnotationRTreeHandler.checkAndRemoveFeature(idx1, idx2, removeFeature);
            }
        }
        reSaveAll();
    }

    public void removeFromList(MatrixZoomData zd, int chrIdx1, int chrIdx2, int x, int y, int n,
                               double binOriginX, double binOriginY, double scale, Feature2D feature) {

        if (chrIdx1 > 0 && chrIdx2 > 0) {
            List<Feature2D> lastList;
            String mirrorIdentity = "" + feature.getStart2() + feature.getEnd2() + feature.getStart1() + feature.getEnd1();

            lastList = getNearbyFeatures(zd, chrIdx1, chrIdx2, x, y, n, binOriginX, binOriginY, scale);
            unsavedEdits = customAnnotationRTreeHandler.checkAndRemoveFeature(chrIdx1, chrIdx2, feature);

            if (!unsavedEdits) {
                Feature2D removeFeature = null;
                for (Feature2D aFeature : lastList) {
                    if (getIdentifier(aFeature).compareTo(mirrorIdentity) == 0) {
                        removeFeature = aFeature;
                        unsavedEdits = true;
                    }
                }
                customAnnotationRTreeHandler.checkAndRemoveFeature(chrIdx1, chrIdx2, removeFeature);
            }
        }
        reSaveAll();
    }

    // Export annotations
    public boolean exportAnnotations(String outputFilePath) {
        boolean somethingExported = customAnnotationRTreeHandler.exportFeatureList(new File(outputFilePath), false, Feature2DList.ListFormat.NA);
        if (somethingExported) {
            deleteTempFile();
        }
        return somethingExported;
    }

    public boolean exportOverlap(Feature2DList otherAnnotations, String outputFilePath) {
        if (customAnnotationRTreeHandler.getOverlap(otherAnnotations).exportFeatureList(
                new File(outputFilePath), false, Feature2DList.ListFormat.NA)) {
            unsavedEdits = false;
        }
        return false;
    }

    private void getAndAddAttributes(List<String> featureKeys) {
        // Add feature's unique attributes to all others
        for (String key : featureKeys) {
            if (!attributeKeys.contains(key)) {
                attributeKeys.add(key);
                customAnnotationRTreeHandler.addAttributeFieldToAll(key, "null");
            }
        }
    }

    public void addAllAttributeValues(String key, String newValue) {
        attributeKeys.add(key);
        customAnnotationRTreeHandler.addAttributeFieldToAll(key, newValue);
    }

    public List<Feature2D> getNearbyFeatures(MatrixZoomData zd, int chrIdx1, int chrIdx2, int x, int y, int n,
                                             double binOriginX, double binOriginY, double scale) {
        return customAnnotationRTreeHandler.getNearbyFeatures(zd, chrIdx1, chrIdx2, x, y, n,
                binOriginX, binOriginY, scale);
    }

    public List<Feature2D> getIntersectingFeatures(int chrIdx1, int chrIdx2, net.sf.jsi.Rectangle selectionWindow) {
        return customAnnotationRTreeHandler.getIntersectingFeatures(chrIdx1, chrIdx2, selectionWindow, false);
    }

    public Feature2DHandler getFeatureHandler() {
        return customAnnotationRTreeHandler;
    }

    public int getId() {
        return id;
    }

    public void resetCounter() {
        i = 0;
    }

    public int getNumberOfFeatures() {
        return customAnnotationRTreeHandler.getNumberOfFeatures();
    }

    public void setColorOfAllAnnotations(Color color) {
        customAnnotationRTreeHandler.setColorOfAllAnnotations(color);
    }

    public boolean getIsSparse() {
        return customAnnotationRTreeHandler.getIsSparsePlottingEnabled();
    }

    public void setIsSparse(boolean isSparse) {
        customAnnotationRTreeHandler.setSparsePlottingEnabled(isSparse);
    }

    public void createMergedLoopLists(List<Feature2DList> lists) {
        customAnnotationRTreeHandler.createNewMergedLoopLists(lists);
    }

    public void createMergedLoopLists(Feature2DList list) {
        List<Feature2DList> lists = new ArrayList<>();
        lists.add(list);
        customAnnotationRTreeHandler.createNewMergedLoopLists(lists);
    }

    public Feature2DList getFeatureList() {
        return customAnnotationRTreeHandler.getFeatureList();
    }

    private String getAutosaveFilename() {
        return HiCGlobals.BACKUP_FILE_STEM + id + ".bedpe";
    }

    private File getAutosaveFile() {
        return new File(DirectoryManager.getHiCDirectory(), getAutosaveFilename());
    }

    public enum LayerType {DEFAULT, EDIT, SCAFFOLD, SUPERSCAFFOLD}

}
