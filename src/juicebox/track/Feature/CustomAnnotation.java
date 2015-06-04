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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by Marie on 6/3/15.
 */
public class CustomAnnotation {

    Feature2DList customAnnotationList;
    boolean isVisible, unsavedEdits;
    //String lastItem;
    Feature2D lastItem;
    int lastChr1Idx, lastChr2Idx;

    public CustomAnnotation () {
        isVisible = true;
        unsavedEdits = false;
        lastItem = null;

        customAnnotationList = new Feature2DList();
    }

    //add annotation to feature2D list
    public void add(int chr1Idx, int chr2Idx, Feature2D feature) {
        if (feature == null){
            System.out.println("feature is null. why?");
        }
        customAnnotationList.add(chr1Idx, chr2Idx, feature);

        System.out.println("Added new feature! ");
        for (Feature2D featureTest : customAnnotationList.get(chr1Idx, chr2Idx)){
            System.out.println(featureTest.toString());
        }

        lastChr1Idx = chr1Idx;
        lastChr2Idx = chr2Idx;
        lastItem = feature;
        //lastItem = getIdentifier(feature);
        unsavedEdits = true;
    }

    //set show loops
    public void setShowCustom (boolean newStatus){
        isVisible = newStatus;
    }

    //get visible loop list (note: only one)
    public List<Feature2D> getVisibleLoopList(int chrIdx1, int chrIdx2) {
        if (this.isVisible && customAnnotationList.isVisible()) {

            return customAnnotationList.get(chrIdx1, chrIdx2);
        }
        //return null;
        return new ArrayList<Feature2D>();
    }

    // Creates unique identifier based on start and end positions.
    private String getIdentifier(Feature2D feature){
        return "" + feature.getStart1() + feature.getEnd1() + feature.getStart2() + feature.getEnd2();
    }

    // Undo last move
    public void undo() {
        removeFromList(lastChr1Idx, lastChr2Idx, lastItem);
//        List<Feature2D> lastList;
//        lastList = customAnnotationList.get(lastChr1Idx, lastChr1Idx);
//        for (Feature2D feature : lastList){
//            if (getIdentifier(feature) == lastItem){
//                //TODO: might not work, if object pointers incorrect
//                lastList.remove(feature);
//            }
//        }
//        unsavedEdits = true;
    }

    private void removeFromList(int idx1, int idx2, Feature2D feature){
        List<Feature2D> lastList;
        String featureIdentifier = getIdentifier(feature);
        lastList = customAnnotationList.get(idx1, idx2);
        for (Feature2D aFeature : lastList){
            if (featureIdentifier.compareTo(getIdentifier(aFeature)) == 0){
                //TODO: might not work, if object pointers incorrect
                lastList.remove(feature);
            }
        }
        unsavedEdits = true;
    }

    // Export annotations
    public void exportAnnotations (String outputFilePath){
        unsavedEdits = false;
        customAnnotationList.exportFeatureList(outputFilePath);
    }

    // Clear all annotations
    public void clearAnnotations(){
        lastItem = null;
        unsavedEdits = false;
        customAnnotationList = new Feature2DList();
        //TODO: add something that pops up if unsavedEdits is true
    }

    public boolean hasUnsavedEdits(){
        return unsavedEdits;
    }

    // Remove annotation from feature2dlist
    //remove annotation by key
}
