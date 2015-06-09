/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2014 Broad Institute, Aiden Lab
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

import juicebox.tools.utils.Common.HiCFileTools;
import juicebox.tools.utils.Juicer.HiCCUPS.HiCCUPSUtils;
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
    private final Map<String, List<Feature2D>> featureList;

    /*
     * Metrics resulting from APA filtering
     */
    private Map<String, Integer[]> filterMetrics;

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
     * @param feature Feature to add
     */
    public void add(int chr1Idx, int chr2Idx, Feature2D feature) {

        String key = getKey(chr1Idx, chr2Idx);
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
     * Helper method to get the key, lowest ordinal chromosome first
     *
     * @param chr1Idx First chromosome index
     * @param chr2Idx Second chromosome index
     * @return key
     */
    private String getKey(int chr1Idx, int chr2Idx) {

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
    private String getKey(Chromosome chr1, Chromosome chr2) {
        return getKey(chr1.getIndex(), chr2.getIndex());
    }

    /**
     * Remove duplicates and filters by size
     * Also save internal metrics for these measures
     *
     * @param minPeakDist
     * @param maxPeakDist
     * @param resolution
     */
    public void apaFiltering(double minPeakDist, double maxPeakDist, int resolution) {

        filterMetrics = new HashMap<String, Integer[]>();
        Set<String> keys = featureList.keySet();

        for (String key : keys) {
            List<Feature2D> features = featureList.remove(key);
            List<Feature2D> uniqueFeatures = filterFeaturesByUniqueness(features);
            List<Feature2D> filteredUniqueFeatures = filterFeaturesBySize(uniqueFeatures,
                    minPeakDist, maxPeakDist, resolution);


            featureList.put(key, filteredUniqueFeatures);
            filterMetrics.put(key,
                    new Integer[]{filteredUniqueFeatures.size(), uniqueFeatures.size(), features.size()});
        }
    }

    /**
     * remove duplicates by using a hashset intermediate
     *
     * @param features
     * @return
     */
    private static ArrayList<Feature2D> filterFeaturesByUniqueness(List<Feature2D> features) {
        return new ArrayList<Feature2D>(new HashSet<Feature2D>(features));
    }

    /**
     * Size filtering of loops
     *
     * @param features
     * @param minPeakDist
     * @param maxPeakDist
     * @return
     */
    private static ArrayList<Feature2D> filterFeaturesBySize(List<Feature2D> features,
                                                             double minPeakDist, double maxPeakDist, int resolution) {

        ArrayList<Feature2D> sizeFilteredFeatures = new ArrayList<Feature2D>();

        for (Feature2D feature : features) {
            int xMidPt = feature.getMidPt1();
            int yMidPt = feature.getMidPt2();
            int dist = Math.abs(xMidPt - yMidPt);

            if (dist >= minPeakDist * resolution)
                if (dist <= maxPeakDist * resolution)
                    sizeFilteredFeatures.add(feature);
        }
        return new ArrayList<Feature2D>(sizeFilteredFeatures);
    }

    /**
     * [NumUniqueFiltered, NumUnique, NumTotal]
     *
     * @param chr
     * @return
     */
    public Integer[] getFilterMetrics(Chromosome chr) {
        return filterMetrics.get(getKey(chr, chr));
    }

    /**
     * Export feature list to given file path
     *
     * @param outputFilePath
     */
    public int exportFeatureList(String outputFilePath) {
        if (featureList != null && featureList.size() > 0) {

            PrintWriter outputFile = HiCFileTools.openWriter(outputFilePath);

            Feature2D featureZero = extractSingleFeature();
            outputFile.println(featureZero.getOutputFileHeader());

            for (String key : featureList.keySet()) {
                for (Feature2D feature : featureList.get(key)) {
                    outputFile.println(feature);
                }
            }
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
        for (String key : featureList.keySet()) {
            for (Feature2D feature : featureList.get(key)) {
                return feature;
            }
        }
        return null;
    }

    /*
     * Set color for the features
     * @param color
     */
    public void setColor(Color color){
        for (String key : featureList.keySet()) {
            for (Feature2D feature : featureList.get(key)) {
                feature.setColor(color);
            }
        }
    }

    /**
     * Calculate FDR values for all peaks
     * @param fdrLogBL
     * @param fdrLogDonut
     * @param fdrLogH
     * @param fdrLogV
     */
    public void calculateFDR(float[][] fdrLogBL, float[][] fdrLogDonut, float[][] fdrLogH, float[][] fdrLogV) {
        for (String key : featureList.keySet()) {
            for (Feature2D feature : featureList.get(key)) {
                HiCCUPSUtils.calculateFDR(feature, fdrLogBL, fdrLogDonut, fdrLogH, fdrLogV);
            }
        }
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
    private List<Feature2D> getFeatureList(String key){
        return featureList.get(key);
    }

}
