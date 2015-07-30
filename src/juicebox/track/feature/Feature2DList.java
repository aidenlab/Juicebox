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

import juicebox.tools.utils.common.HiCFileTools;
import juicebox.tools.utils.juicer.hiccups.HiCCUPSUtils;
import org.broad.igv.feature.Chromosome;

import java.awt.*;
import java.io.PrintWriter;
import java.util.*;
import java.util.List;

/**
 * List of two-dimensional features.  Hashtable for each chromosome for quick viewing.
 * Visibility depends on user selection.
 *
 * @author Neva Durand
 * @modified Muhammad Shamim
 */
public class Feature2DList {

    /**
     * List of 2D features stored by chromosome
     */
    private Map<String, List<Feature2D>> featureList;

    /**
     * Visibility as set by user
     */
    private boolean isVisible;

    /**
     * Initialized hashtable
     */
    public Feature2DList() {
        featureList = new HashMap<String, List<Feature2D>>();
        isVisible = true;
    }

    /**
     * Helper method to get the key, lowest ordinal chromosome first
     *
     * @param chr1Idx First chromosome index
     * @param chr2Idx Second chromosome index
     * @return key
     */
    private static String getKey(int chr1Idx, int chr2Idx) {

        int c1;
        int c2;
        if (chr1Idx < chr2Idx) {
            c1 = chr1Idx;
            c2 = chr2Idx;
        } else {
            c1 = chr2Idx;
            c2 = chr1Idx;
        }

        return "" + c1 + "_" + c2;
    }

    /**
     * Helper method to get the key given chromosomes
     *
     * @param chr1 First chromosome
     * @param chr2 Second chromosome
     * @return key
     */
    public static String getKey(Chromosome chr1, Chromosome chr2) {
        return getKey(chr1.getIndex(), chr2.getIndex());
    }

    /**
     * Returns list of features on this chromosome pair
     *
     * @param chr1Idx First chromosome index
     * @param chr2Idx Second chromosome index
     * @return List of 2D features at that point
     */
    public List<Feature2D> get(int chr1Idx, int chr2Idx) {
        List<Feature2D> returnVal = featureList.get(getKey(chr1Idx, chr2Idx));
        if (returnVal == null){
            return new ArrayList<Feature2D>();
        } else {
            return returnVal;
        }
        //return featureList.get(getKey(chr1Idx, chr2Idx));
    }

    /**
     * Adds feature to appropriate chromosome pair list; key stored so that first chromosome always less than second
     *
     * @param chr1Idx First chromosome index
     * @param chr2Idx Second chromosome index
     * @param feature feature to add
     */
    public void add(int chr1Idx, int chr2Idx, Feature2D feature) {

        String key = getKey(chr1Idx, chr2Idx);
        addByKey(key,feature);

    }

    private void addByKey(String key, Feature2D feature) {

        List<Feature2D> loops = featureList.get(key);
        if (loops == null) {
            loops = new ArrayList<Feature2D>();
            loops.add(feature);
            featureList.put(key, loops);
        } else {
            loops.add(feature);
        }

    }

    /**
     * Returns visibility of list
     *
     * @return If list is visible
     */
    public boolean isVisible() {
        return isVisible;
    }

    /**
     * Set visibility of list
     *
     * @param flag Visibility
     */
    public void setVisible(boolean flag) {
        isVisible = flag;
    }

    /**
     * Export feature list to given file path
     *
     * @param outputFilePath
     */
    public int exportFeatureList(String outputFilePath, final boolean useOldHiccupsOutput) {
        if (featureList != null && featureList.size() > 0) {

            final PrintWriter outputFile = HiCFileTools.openWriter(outputFilePath);

            Feature2D featureZero = extractSingleFeature();
            outputFile.println(featureZero.getOutputFileHeader());

            processLists(new FeatureFunction() {
                @Override
                public void process(String chr, List<Feature2D> feature2DList) {
                    for (Feature2D feature : feature2DList) {
                        if (useOldHiccupsOutput)
                            outputFile.println(HiCCUPSUtils.oldOutput(feature));
                        else
                            outputFile.println(feature);
                    }
                }
            });

            outputFile.close();

            return 0;
        }
        return -1;
    }

    /**
     * Export feature list to given file path
     *
     * @param outputFile
     */
    public int autoSaveNew(PrintWriter outputFile, Feature2D feature) {

        if (featureList != null && featureList.size() > 0) {
            outputFile.println(feature);
            return 0;
        }
        return -1;
    }

    /**
     * Export feature list to given file path
     *
     * @param outputFile
     */
    public int autoSaveAll(PrintWriter outputFile) {
        if (featureList != null && featureList.size() > 0) {

            for (String key : featureList.keySet()) {
                for (Feature2D feature : featureList.get(key)) {
                    outputFile.println(feature);
                }
            }

            return 0;
        }
        return -1;
    }

    /**
     * Get first feature found
     * @return feature
     */
    public Feature2D extractSingleFeature() {
        return featureList.get(featureList.keySet().iterator().next()).iterator().next();
    }

    /*
     * Set color for the features
     * @param color
     */
    public void setColor(final Color color) {
        processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D feature : feature2DList) {
                    feature.setColor(color);
                }
            }
        });
    }

    /**
     * Calculate FDR values for all peaks
     * @param fdrLogBL
     * @param fdrLogDonut
     * @param fdrLogH
     * @param fdrLogV
     */
    public void calculateFDR(final float[][] fdrLogBL, final float[][] fdrLogDonut,
                             final float[][] fdrLogH, final float[][] fdrLogV) {
        processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D feature : feature2DList) {
                    HiCCUPSUtils.calculateFDR(feature, fdrLogBL, fdrLogDonut, fdrLogH, fdrLogV);
                }
            }
        });
    }

    /**
     * Adds features to appropriate chromosome pair list;
     * key stored so that first chromosome always less than second
     *
     * @param inputList
     * @return
     */
    public void add(Feature2DList inputList) {

        Set<String> inputKeySet = inputList.getKeySet();

        for(String inputKey : inputKeySet){
            List<Feature2D> inputFeatures = inputList.getFeatureList(inputKey);

            List<Feature2D> features = featureList.get(inputKey);
            if (features == null) {
                features = new ArrayList<Feature2D>();
                features.addAll(inputFeatures);
                featureList.put(inputKey, features);
            } else {
                features.addAll(inputFeatures);
            }
        }
    }

    /**
     * Adds features to appropriate chromosome pair list if same
     * or similar point not already in list;
     * key stored so that first chromosome always less than second
     *
     * @param inputList
     * @return
     */
    public void addUnique(Feature2DList inputList) {

        Set<String> inputKeySet = inputList.getKeySet();

        for(String inputKey : inputKeySet){
            List<Feature2D> inputFeatures = inputList.getFeatureList(inputKey);

            List<Feature2D> features = featureList.get(inputKey);
            if (features == null) {
                features = new ArrayList<Feature2D>();
                features.addAll(inputFeatures);
                featureList.put(inputKey, features);
            } else {
                //features.addAll(inputFeatures);
                addAllUnique(inputFeatures, features);
            }
        }
    }

    public Feature2DList getOverlap(Feature2DList inputList) {
        Feature2DList output = new Feature2DList();
        Set<String> inputKeySet = inputList.getKeySet();
        for(String inputKey : inputKeySet){
            List<Feature2D> inputFeatures = inputList.getFeatureList(inputKey);
            // there are features in both lists
            List<Feature2D> myFeatures = featureList.get(inputKey);
            if (myFeatures != null) {
                for (Feature2D myFeature : myFeatures){
                    if (doesOverlap(myFeature, inputFeatures)) {
                        output.addByKey(inputKey, myFeature);
                    }
                }
            }
        }
        return output;
    }

    // Compares a feature against all other featuers in list
    private boolean doesOverlap(Feature2D feature, List<Feature2D> existingFeatures){
        boolean repeat = false;
        for (Feature2D existingFeature : existingFeatures){
            if (existingFeature.overlapsWith(feature)){
                repeat = true;
            }
        }
        return repeat;
    }

    // Iterate through new features and see if there is any overlap
    // TODO: implement this more efficiently
    private void addAllUnique(List<Feature2D> inputFeatures, List<Feature2D> existingFeatures){
        for (Feature2D inputFeature : inputFeatures){
            // Compare input with existing points
            if (!doesOverlap(inputFeature, existingFeatures)) {
                existingFeatures.add(inputFeature);
            }

        }

    }

    public void addAttributeFieldToAll(final String newAttributeName, final String newAttributeValue) {
        processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D feature : feature2DList) {
                    if (feature.getAttribute(newAttributeName) == null)
                        feature.addFeature(newAttributeName, newAttributeValue);
                }
            }
        });
    }


    /**
     * Get all keys (chromosome pairs) for hashmap
     * @return keySet
     */
    private Set<String> getKeySet() {
        return featureList.keySet();
    }

    /**
     * Get feature list corresponding to key (chromosome pair)
     * @param key
     * @return
     */
    private List<Feature2D> getFeatureList(String key) {
        return featureList.get(key);
    }

    /**
     * pass interface implementing a filter for features
     *
     * @param filter
     */
    public void filterLists(FeatureFilter filter) {
        for (String chr : featureList.keySet()) {
            featureList.put(chr, filter.filter(chr, featureList.get(chr)));
        }
    }

    /**
     * pass interface implementing a process for all features
     * TODO - alter above functions to use this
     *
     * @param function
     */
    public void processLists(FeatureFunction function) {
        for (String chr : featureList.keySet()) {
            function.process(chr, featureList.get(chr));
        }
    }


}
