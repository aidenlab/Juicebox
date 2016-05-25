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

package juicebox.track.feature;

import juicebox.DirectoryManager;
import juicebox.data.HiCFileTools;
import juicebox.gui.MainMenuBar;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Marie on 6/3/15.
 */
public class CustomAnnotation {

    private final String id;
    private Feature2DList customAnnotationList;
    private boolean isVisible;
    private boolean unsavedEdits;
    private boolean firstSave;
    private Feature2D lastItem;
    private int lastChr1Idx;
    private int lastChr2Idx;
    private PrintWriter tempWriter;
    private File tempFile;
    private ArrayList<String> attributeKeys;

    public CustomAnnotation(String id) {
        this.id = id;
        isVisible = true;
        firstSave = true;
        reset();
    }

    public CustomAnnotation(Feature2DList inputList, String id) {
        this.id = id;
        isVisible = true;
        firstSave = true;
        reset();
        this.customAnnotationList = inputList;
    }

    // Clear all annotations
    private void reset() {
        firstSave = true;
        lastChr1Idx = -1;
        lastChr2Idx = -1;
        lastItem = null;
        unsavedEdits = false;
        customAnnotationList = new Feature2DList();
        attributeKeys = new ArrayList<String>();
    }

    public void clearAnnotations() {
        reset();
        deleteTempFile();
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

        customAnnotationList.add(chr1Idx, chr2Idx, feature);

        lastChr1Idx = chr1Idx;
        lastChr2Idx = chr2Idx;
        lastItem = feature;

        // Autosave the information
        unsavedEdits = true;
        if (firstSave) {
            makeTempFile();
            firstSave = false;
        }
        updateAutoSave();
    }

    // Requires that lastItem is not null
    private void makeTempFile() {
        String prefix = "unsaved-hiC-annotations" + id;
        tempFile = new File(DirectoryManager.getHiCDirectory(), prefix + ".txt");
        tempWriter = HiCFileTools.openWriter(tempFile);

        Feature2D singleFeature = customAnnotationList.extractSingleFeature();
        if (singleFeature == null) {
            tempWriter.println(Feature2D.getDefaultOutputFileHeader());
        } else {
            tempWriter.println(singleFeature.getOutputFileHeader());
        }
    }

    public void deleteTempFile() {
        if (tempWriter != null) {
            tempWriter.close();
        }
        if (tempFile == null) {
            String prefix = "unsaved-hiC-annotations" + id;
            tempFile = new File(DirectoryManager.getHiCDirectory(), prefix + ".txt");
        }
        if (tempFile.exists()) {
            tempFile.delete();
        }
    }

    // Set show loops
    public void setShowCustom(boolean newStatus) {
        isVisible = newStatus;
    }

    // Get visible loop list (note: only one)
    public List<Feature2D> getVisibleLoopList(int chrIdx1, int chrIdx2) {
        if (this.isVisible && customAnnotationList.isVisible()) {
            return customAnnotationList.get(chrIdx1, chrIdx2);
        }
        // Empty to prevent null pointer exception
        return new ArrayList<Feature2D>();
    }

    // Creates unique identifier for Feature2D based on start and end positions.
    private String getIdentifier(Feature2D feature) {
        return "" + feature.getStart1() + feature.getEnd1() + feature.getStart2() + feature.getEnd2();
    }

    // Undo last move
    public void undo() {
        removeFromList(lastChr1Idx, lastChr2Idx, lastItem);
    }

    /**
     * Export feature list to given file path
     */
    private void updateAutoSave() {
        if (unsavedEdits && lastItem != null) {
            customAnnotationList.autoSaveNew(tempWriter, lastItem);
        }
    }

    /**
     * Export feature list to given file path, including header
     */
    private void reSaveAll() {
        deleteTempFile();
        makeTempFile();
        customAnnotationList.autoSaveAll(tempWriter);
    }

    public boolean hasLoop(int idx1, int idx2, Feature2D feature) {
        if (idx1 > 0 && idx2 > 0) {
            List<Feature2D> featureList = customAnnotationList.get(idx1, idx2);
            if (featureList.contains(feature) || featureList.contains(feature.reflectionAcrossDiagonal())) {
                return true;
            }
        }
        return false;
    }

    public void removeFromList(int idx1, int idx2, Feature2D feature) {

        if (idx1 > 0 && idx2 > 0) {
            List<Feature2D> lastList;
            String mirrorIdentity = "" + feature.getStart2() + feature.getEnd2() + feature.getStart1() + feature.getEnd1();

            lastList = customAnnotationList.get(idx1, idx2);
            unsavedEdits = lastList.remove(feature);

            if (!unsavedEdits) {
                Feature2D removeFeature = null;
                for (Feature2D aFeature : lastList) {
                    if (getIdentifier(aFeature).compareTo(mirrorIdentity) == 0) {
                        removeFeature = aFeature;
                        unsavedEdits = true;
                    }
                }
                lastList.remove(removeFeature);
                customAnnotationList.checkAndRemoveEmptyList(idx1, idx2);
            }
        }
        reSaveAll();
    }

    // Export annotations
    public int exportAnnotations(String outputFilePath) {
        int ok;
        ok = customAnnotationList.exportFeatureList(new File(outputFilePath), false, Feature2DList.ListFormat.NA);
        if (ok < 0)
            return ok;
        unsavedEdits = false;
        return ok;
    }

    // Note assumes that all attributes are already correctly formatted. Ok to assume
    // because loaded list must have consistent formatting.
    public void addVisibleToCustom(Feature2DList newAnnotations) {
        MainMenuBar.exportAnnotationsMI.setEnabled(true);
        Feature2D featureZero = newAnnotations.extractSingleFeature();
        // Add attributes to feature
        List<String> featureKeys = featureZero.getAttributeKeys();
        for (String customKey : attributeKeys) {
            if (!featureKeys.contains(customKey)) {
                newAnnotations.addAttributeFieldToAll(customKey, "null");
            }
        }
        getAndAddAttributes(featureKeys);
        //customAnnotationList.addUnique(newAnnotations);
        customAnnotationList.add(newAnnotations);

        // Autosave the information
        unsavedEdits = true;
        if (firstSave) {
            makeTempFile();
            firstSave = false;
        }
        updateAutoSave();
    }

    public int exportOverlap(Feature2DList otherAnnotations, String outputFilePath) {
        int ok;
        ok = customAnnotationList.getOverlap(otherAnnotations).exportFeatureList(
                new File(outputFilePath), false, Feature2DList.ListFormat.NA);
        if (ok < 0)
            return ok;
        unsavedEdits = false;
        return ok;
    }

    private void getAndAddAttributes(List<String> featureKeys) {
        // Add feature's unique attributes to all others
        for (String key : featureKeys) {
            if (!attributeKeys.contains(key)) {
                attributeKeys.add(key);
                customAnnotationList.addAttributeFieldToAll(key, "null");
            }
        }
    }

    public void addAllAttributeValues(String key, String newValue) {
        attributeKeys.add(key);
        customAnnotationList.addAttributeFieldToAll(key, newValue);
    }

    public boolean hasUnsavedEdits() {
        return unsavedEdits;
    }

    // Remove annotation from feature2dlist
    //remove annotation by key
}
