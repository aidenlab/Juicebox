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

import juicebox.mapcolorui.Feature2DHandler;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 11/2/16.
 */
class CustomAnnotationRTree2DHandler extends Feature2DHandler {

    public CustomAnnotationRTree2DHandler(Feature2DList inputList) {
        clearLists();
        setLoopList(inputList);
    }

    public void add(int chr1Idx, int chr2Idx, Feature2D feature) {
        loopList.add(chr1Idx, chr2Idx, feature);
        // TODO can be optimized further i.e. no need to remake entire rtree, just add necessary nodes
        remakeRTree();
    }

    public int getNumberOfFeatures() {
        return loopList.getNumTotalFeatures();
    }

    public void addAttributeFieldToAll(String key, String aNull) {
        loopList.addDefaultAttribute(key, aNull);
    }

    /**
     * Export feature list to given file path
     *
     * @param outputFile
     */
    public void autoSaveNew(PrintWriter outputFile, Feature2D feature) {
        loopList.autoSaveNew(outputFile, feature);
    }

    /**
     * Export feature list to given file path
     *
     * @param outputFile
     */
    public void autoSaveAll(PrintWriter outputFile) {
        loopList.autoSaveAll(outputFile);
    }

    public boolean checkAndRemoveFeature(int idx1, int idx2, Feature2D feature2D) {
        boolean somethingWasDeleted = loopList.checkAndRemoveFeature(idx1, idx2, feature2D);
        if (somethingWasDeleted)
            remakeRTree();
        // TODO can be optimized further i.e. no need to remake entire rtree, just delete necessary nodes

        return somethingWasDeleted;
    }

    public Feature2DList getOverlap(Feature2DList inputList) {
        Feature2DList overlapFeature2DList = new Feature2DList();
        overlapFeature2DList.add(loopList.getOverlap(inputList));
        return overlapFeature2DList;
    }

    public Feature2D extractSingleFeature() {
        //noinspection LoopStatementThatDoesntLoop
        return loopList.extractSingleFeature();
    }

    public List<Feature2D> get(int chrIdx1, int chrIdx2) {

        return new ArrayList<>(loopList.get(chrIdx1, chrIdx2));
    }

    public boolean exportFeatureList(File file, boolean b, Feature2DList.ListFormat na) {
        Feature2DList mergedList = new Feature2DList();
        mergedList.add(loopList);
        return mergedList.exportFeatureList(file, b, na);
    }
}
