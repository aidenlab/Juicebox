/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.track.feature;

import juicebox.HiCGlobals;
import juicebox.data.HiCFileTools;
import juicebox.data.basics.Chromosome;

import java.awt.*;
import java.io.File;
import java.io.PrintWriter;
import java.util.List;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * List of two-dimensional features.  Hashtable for each chromosome for quick viewing.
 * Visibility depends on user selection.
 *
 * @author Neva Durand, Muhammad Shamim, Marie Hoeger
 *         <p/>
 *         TODO cleanup code and eliminate this class
 *         It should become GenomeWideList<Feature2D>
 *         Helper functions should be relocated to Feature2DTools
 */
public class Feature2DList {

    /**
     * List of 2D features stored by chromosome
     */
    private final Map<String, List<Feature2D>> featureList = new HashMap<>();

    private Map<String, String> defaultAttributes = new HashMap<>();

    public void parallelizedProcessLists(FeatureFunction featureFunction) {
        List<String> keys = new ArrayList<>(featureList.keySet());
        Collections.sort(keys);
        ExecutorService executor = Executors.newFixedThreadPool(keys.size());
        for (String key : keys) {
            Runnable worker = new Runnable() {
                @Override
                public void run() {
                    featureFunction.process(key, featureList.get(key));
                }
            };
            executor.execute(worker);
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    /**
     * pass interface implementing a filter for features
     *
     * @param filter
     */
    public synchronized void filterLists(FeatureFilter filter) {
        List<String> keys = new ArrayList<>(featureList.keySet());
        Collections.sort(keys);
        for (String key : keys) {
            featureList.put(key, filter.filter(key, featureList.get(key)));
        }
    }

    public Feature2DList() {
    }

    public Feature2DList(Feature2DList list) {
        add(list);
        defaultAttributes.putAll(list.defaultAttributes);
    }

    /**
     * Helper method to get the key, lowest ordinal chromosome first
     *
     * @param chr1Idx First chromosome index
     * @param chr2Idx Second chromosome index
     * @return key
     */
    public static String getKey(int chr1Idx, int chr2Idx) {

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
     * values from list A that are common to list B within tolerance
     *
     * @param listA
     * @param listB
     * @return
     */
    public static Feature2DList getIntersection(final Feature2DList listA, Feature2DList listB) {

        Feature2DList commonFeatures = new Feature2DList(listB);
        commonFeatures.filterLists(new FeatureFilter() {
            @Override
            public List<Feature2D> filter(String chr, List<Feature2D> feature2DList) {
                List<Feature2D> commonVals = new ArrayList<>();
                if (listA.containsKey(chr)) {
                    List<Feature2D> listAFeatures = listA.getFeatureList(chr);
                    for (Feature2D feature : listAFeatures) {
                        if (feature2DList.contains(feature)) {
                            commonVals.add(feature);
                        }
                    }
                }
                return commonVals;
            }
        });


        commonFeatures.removeDuplicates();
        return commonFeatures;
    }

    // Iterate through new features and see if there is any overlap
    // TODO: implement this more efficiently, maybe rtree
    private synchronized static void addAllUnique(List<Feature2D> inputFeatures, List<Feature2D> existingFeatures) {
        for (Feature2D inputFeature : inputFeatures) {
            // Compare input with existing points
            if (!Feature2DTools.doesOverlap(inputFeature, existingFeatures)) {
                existingFeatures.add(inputFeature);
            }
        }
    }

    /**
     * Returns list of features on this chromosome pair
     *
     * @param chr1Idx First chromosome index
     * @param chr2Idx Second chromosome index
     * @return List of 2D features at that point
     */
    public List<Feature2D> get(int chr1Idx, int chr2Idx) {
        String key = getKey(chr1Idx, chr2Idx);
        if (!featureList.containsKey(key)) {
            List<Feature2D> features = new ArrayList<>();
            featureList.put(key, features);
        }
        return featureList.get(key);
    }

    /**
     * Returns list of features on this chromosome pair
     * Warning, this should be used carefully, assumes proper key nomenclature is used
     * should only be used when comparing equivalent lists
     *
     * @return List of 2D features for given key
     */
    public List<Feature2D> get(String key) {
        if (!featureList.containsKey(key)) {
            List<Feature2D> features = new ArrayList<>();
            featureList.put(key, features);
        }
        return featureList.get(key);
    }

    /**
     * Adds feature to appropriate chromosome pair list; key stored so that first chromosome always less than second
     *
     * @param chr1Idx First chromosome index
     * @param chr2Idx Second chromosome index
     * @param feature feature to add
     */
    public synchronized void add(int chr1Idx, int chr2Idx, Feature2D feature) {

        String key = getKey(chr1Idx, chr2Idx);
        addByKey(key, feature);

    }

    /**
     * Adds feature to appropriate chromosome pair list; key stored so that first chromosome always less than second
     *
     * @param key     chromosomal pair key
     * @param feature to add
     */
    public synchronized void addByKey(String key, Feature2D feature) {

        feature = updateAttributeForFeature(feature);

        if (featureList.containsKey(key)) {
            featureList.get(key).add(feature);//could be an issue
        } else {
            List<Feature2D> loops = new ArrayList<>();
            loops.add(feature);
            featureList.put(key, loops);
        }
    }

    /**
     * Adds features to appropriate chromosome pair list; key stored so that first chromosome always less than second
     *
     * @param key      chromosomal pair key
     * @param features to add
     */
    public synchronized void addByKey(String key, List<Feature2D> features) {
        features = updateAttributes(features);
        if (featureList.containsKey(key)) {
            featureList.get(key).addAll(features);
        } else {
            List<Feature2D> loops = new ArrayList<>(features);
            featureList.put(key, loops);
        }
    }

    private synchronized Feature2D updateAttributeForFeature(Feature2D feature) {
        if (defaultAttributes != null) {
            if (feature.getAttributeKeys() == null) {
                for (String attribute : defaultAttributes.keySet()) {
                    feature.addStringAttribute(attribute, defaultAttributes.get(attribute));
                    System.out.println("Added:1 " + attribute); //TODO find out why this doesn't work
                }
            } else {
                List<String> featureKeys = feature.getAttributeKeys();

                for (String customKey : defaultAttributes.keySet()) {
                    if (!featureKeys.contains(customKey)) {
                        feature.addStringAttribute(customKey, defaultAttributes.get(customKey));
                    }
                }
            }
        }
        return feature;
    }

    private synchronized List<Feature2D> updateAttributes(List<Feature2D> features) {
        processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D feature : feature2DList) {
                    updateAttributeForFeature(feature);
                }
            }
        });
        return features;
    }

    private synchronized void putFeature(String key, List<Feature2D> loops) {
        featureList.put(key, loops);
    }

    /**
     * Export feature list to given file path
     *
     * @param outputFile
     * @return true if successful
     */
    public boolean exportFeatureList(File outputFile, boolean formattedOutput, ListFormat listFormat) {
        if (featureList != null && featureList.size() > 0) {
            final PrintWriter outputFilePrintWriter = HiCFileTools.openWriter(outputFile);
            if (exportFeatureList(outputFilePrintWriter, formattedOutput, listFormat)) {
                System.out.println(getNumTotalFeatures() + " features written to file: " + outputFile.getAbsolutePath());
            } else {
                System.err.println("Error features not written to file: " + outputFile.getAbsolutePath());
            }
            return true;
        }
        return false;
    }

    /**
     * Export feature list to given file path
     *
     * @param outputFilePrintWriter
     */
    private boolean exportFeatureList(final PrintWriter outputFilePrintWriter, final boolean formattedOutput,
                                      final ListFormat listFormat) {
        if (featureList != null && featureList.size() > 0) {

            Feature2D featureZero = extractSingleFeature();
            if (featureZero != null) {
                if (formattedOutput) {
                    StringBuilder header = new StringBuilder(Feature2D.genericHeader);
                    final ArrayList<String> outputKeys = new ArrayList<>();
                    if (listFormat == ListFormat.ENRICHED) {
                        outputKeys.addAll(Arrays.asList("observed", "expectedBL", "expectedDonut", "expectedH",
                                "expectedV", "binBL", "binDonut", "binH", "binV", "fdrBL", "fdrDonut", "fdrH", "fdrV"));
                    } else if (listFormat == ListFormat.FINAL) {
                        outputKeys.addAll(Arrays.asList("observed", "expectedBL", "expectedDonut", "expectedH",
                                "expectedV", "fdrBL", "fdrDonut", "fdrH", "fdrV", "numCollapsed", "centroid1", "centroid2", "radius"));
                    } else if (listFormat == ListFormat.LOCALIZED) {
                        if (featureZero.attributes.containsKey("motif_start_1") && featureZero.attributes.containsKey("expectedBL")) {
                            outputKeys.addAll(Arrays.asList("observed", "expectedBL", "expectedDonut", "expectedH",
                                    "expectedV", "fdrBL", "fdrDonut", "fdrH", "fdrV", "numCollapsed", "centroid1", "centroid2", "radius",
                                    "motif_start_1", "motif_end_1", "sequence_1", "orientation_1", "uniqueness_1",
                                    "motif_start_2", "motif_end_2", "sequence_2", "orientation_2", "uniqueness_2",
                                    "highRes_start_1", "highRes_end_1", "highRes_start_2", "highRes_end_2",
                                    "upstream_start_1", "upstream_end_1", "downstream_start_2", "downstream_end_2",
                                    "localX", "localY", "localObserved", "localPval", "localZscore", "localPeakID"));
                        } else if (featureZero.attributes.containsKey("motif_start_1")) {
                            outputKeys.addAll(Arrays.asList("motif_start_1", "motif_end_1", "sequence_1", "orientation_1", "uniqueness_1",
                                    "motif_start_2", "motif_end_2", "sequence_2", "orientation_2", "uniqueness_2",
                                    "highRes_start_1", "highRes_end_1", "highRes_start_2", "highRes_end_2",
                                    "upstream_start_1", "upstream_end_1", "downstream_start_2", "downstream_end_2",
                                    "localX", "localY", "localObserved", "localPval", "localZscore", "localPeakID"));
                        } else if (featureZero.attributes.containsKey("expectedBL")){
                            outputKeys.addAll(Arrays.asList("observed", "expectedBL", "expectedDonut", "expectedH",
                                    "expectedV", "fdrBL", "fdrDonut", "fdrH", "fdrV", "numCollapsed", "centroid1", "centroid2", "radius",
                                    "highRes_start_1", "highRes_end_1", "highRes_start_2", "highRes_end_2",
                                    "upstream_start_1", "upstream_end_1", "downstream_start_2", "downstream_end_2",
                                    "localX", "localY", "localObserved", "localPval", "localZscore", "localPeakID"));
                        } else {
                            outputKeys.addAll(Arrays.asList("highRes_start_1", "highRes_end_1", "highRes_start_2", "highRes_end_2",
                                    "upstream_start_1", "upstream_end_1", "downstream_start_2", "downstream_end_2",
                                    "localX", "localY", "localObserved", "localPval", "localZscore", "localPeakID"));
                        }
                    }else if (listFormat == ListFormat.ARROWHEAD) {
                        outputKeys.addAll(Arrays.asList("score", "uVarScore", "lVarScore", "upSign", "loSign"));
                    }
                    for (String key : outputKeys) {
                        header.append("\t").append(key);
                    }
                    outputFilePrintWriter.println(header);
                    outputFilePrintWriter.println("# juicer_tools version " + HiCGlobals.versionNum);
                    processLists(new FeatureFunction() {
                        @Override
                        public void process(String chr, List<Feature2D> feature2DList) {
                            for (Feature2D feature : feature2DList) {
                                StringBuilder output = new StringBuilder(feature.simpleStringWithColor());
                                for (String key : outputKeys) {
                                    output.append("\t").append(feature.attributes.get(key));
                                }
                                outputFilePrintWriter.println(output);
                            }
                        }
                    });
                } else {
                    outputFilePrintWriter.println(featureZero.getOutputFileHeader());
                    processLists(new FeatureFunction() {
                        @Override
                        public void process(String chr, List<Feature2D> feature2DList) {
                            Collections.sort(feature2DList);
                            for (Feature2D feature : feature2DList) {
                                outputFilePrintWriter.println(feature);
                            }
                        }
                    });
                }
            }
            outputFilePrintWriter.close();

            return true;
        }
        return false;
    }

    /**
     * Export feature list to given file path
     *
     * @param outputFile
     */
    public void autoSaveNew(PrintWriter outputFile, Feature2D feature) {
        if (featureList != null && featureList.size() > 0) {
            outputFile.println(feature);
        }
    }

    /**
     * Export feature list to given file path
     *
     * @param outputFile
     */
    public void autoSaveAll(PrintWriter outputFile) {
        if (featureList != null && featureList.size() > 0) {
            for (String key : featureList.keySet()) {
                for (Feature2D feature : featureList.get(key)) {
                    outputFile.println(feature);
                }
            }
        }
    }

    /**
     * Get first feature found
     *
     * @return feature
     */
    public Feature2D extractSingleFeature() {
        for (List<Feature2D> features : featureList.values()) {
            //noinspection LoopStatementThatDoesntLoop
            for (Feature2D feature : features) {
                return feature;
            }
        }
        return null;
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
     * Adds features to appropriate chromosome pair list;
     * key stored so that first chromosome always less than second
     *
     * @param inputList
     * @return
     */
    public synchronized void add(Feature2DList inputList) {

        Set<String> inputKeySet = inputList.getKeySet();

        for (String inputKey : inputKeySet) {
            List<Feature2D> inputFeatures = inputList.getFeatureList(inputKey);

            if (featureList.containsKey(inputKey)) {
                featureList.get(inputKey).addAll(inputFeatures);
            } else {
                List<Feature2D> features = new ArrayList<>(inputFeatures);
                putFeature(inputKey, features);
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
    public synchronized void addUnique(Feature2DList inputList) {

        Set<String> inputKeySet = inputList.getKeySet();

        for (String inputKey : inputKeySet) {
            List<Feature2D> inputFeatures = inputList.getFeatureList(inputKey);

            if (featureList.containsKey(inputKey)) {
                //features.addAll(inputFeatures);
                addAllUnique(inputFeatures, featureList.get(inputKey));
            } else {
                List<Feature2D> features = new ArrayList<>(inputFeatures);
                // featureList.put(inputKey, features);
                putFeature(inputKey, features);
            }
        }
    }

    public Feature2DList getOverlap(Feature2DList inputList) {
        Feature2DList output = new Feature2DList();
        Set<String> inputKeySet = inputList.getKeySet();
        for (String inputKey : inputKeySet) {
            List<Feature2D> inputFeatures = inputList.getFeatureList(inputKey);
            // there are features in both lists

            if (featureList.containsKey(inputKey)) {
                for (Feature2D myFeature : featureList.get(inputKey)) {
                    if (Feature2DTools.doesOverlap(myFeature, inputFeatures)) {
                        output.addByKey(inputKey, myFeature);
                    }
                }
            }
        }
        return output;
    }

    public synchronized void setDefaultAttributes(Map<String, String> defaultAttributes) {
        this.defaultAttributes = defaultAttributes;
    }


    public synchronized void addDefaultAttribute(String attribute, String value) {
        defaultAttributes.put(attribute, value);
        addAttributeFieldToAll(attribute, value);
    }


    public synchronized void addAttributeFieldToAll(final String newAttributeName, final String newAttributeValue) {
        processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D feature : feature2DList) {
                    if (feature.doesNotContainAttributeKey(newAttributeName))
                        feature.addStringAttribute(newAttributeName, newAttributeValue);
                }
            }
        });
    }

    /**
     * Simple removal of exact duplicates (memory address)
     * TODO more detailed filtering by size/position/etc? NOTE that this is used by HiCCUPS
     */
    public void removeDuplicates() {
        filterLists(new FeatureFilter() {
            @Override
            public List<Feature2D> filter(String chr, List<Feature2D> feature2DList) {
                return new ArrayList<>(new HashSet<>(feature2DList));
            }
        });
    }

    /**
     * Get all keys (chromosome pairs) for hashmap
     *
     * @return keySet
     */
    private Set<String> getKeySet() {
        return featureList.keySet();
    }

    /**
     * Get feature list corresponding to key (chromosome pair)
     *
     * @param key
     * @return
     */
    List<Feature2D> getFeatureList(String key) {
        return featureList.get(key);
    }

    public enum ListFormat {ENRICHED, FINAL, LOCALIZED, ARROWHEAD, NA}

    /**
     * pass interface implementing a process for all features
     *
     * @param function
     */
    public synchronized void processLists(FeatureFunction function) {
        List<String> keys = new ArrayList<>(featureList.keySet());
        Collections.sort(keys);
        for (String key : keys) {
            function.process(key, featureList.get(key));
        }
    }

    /**
     * @return true if features available for this region (key = "chr1_chr2")
     */
    public boolean containsKey(String key) {
        return featureList.containsKey(key);
    }

    public int getIndex(Chromosome chrX, Chromosome chrY, Feature2D feature2D) {
        final String key = Feature2DList.getKey(chrX, chrY);
        //this.convertFeaturesToContigs(key);
        List<Feature2D> contigs = this.get(key);

        for (int i = 0; i < contigs.size(); i++) {
            Feature2D currentContig = contigs.get(i);
            if (currentContig.equals(feature2D)) {
                return i;
            }
        }
        return -1;  // return -1 if input feature2D object is not present for the specified chromosomes
    }

    public int getNumTotalFeatures() {
        int total = 0;
        for (List<Feature2D> chrList : featureList.values()) {
            total += chrList.size();
        }
        return total;
    }

    public synchronized boolean checkAndRemoveFeature(int idx1, int idx2, Feature2D feature) {
        boolean somethingWasDeleted = false;
        String key = getKey(idx1, idx2);

        if (featureList.containsKey(key)) {
            try {
                somethingWasDeleted = featureList.get(key).remove(feature);
            } catch (Exception e) {
                System.err.println("Error encountered removing feature");
            }
            if (featureList.get(key).size() == 0)
                featureList.remove(key);
        }
        return somethingWasDeleted;
    }

    public Feature2D searchForFeature(final int c1, final int start1, final int end1,
                                      final int c2, final int start2, final int end2) {
        final Feature2D[] feature = new Feature2D[1];
        processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D f : feature2DList) {
                    if (f.getChr1().contains("" + c1) && f.getChr2().contains("" + c2) && f.start1 == start1 &&
                            f.start2 == start2 && f.end1 == end1 && f.end2 == end2) {
                        feature[0] = f;
                        break;
                    }
                }
            }
        });
        return feature[0];
    }

    public void clearAllAttributes() {
        processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D feature : feature2DList) {
                    feature.clearAttributes();
                }
            }
        });
    }

    public Feature2DList deepCopy() {
        Feature2DList clone = new Feature2DList();
        for (String key : featureList.keySet()) {
            List<Feature2D> cloneList = new ArrayList<>();
            for (Feature2D f : featureList.get(key)) {
                cloneList.add(f.deepCopy());
            }
            clone.featureList.put(key, cloneList);
        }
        return clone;
    }

    public String printChromosomeRegionKeys() {
        StringBuilder features = new StringBuilder();
        for (List<Feature2D> feature2DS : featureList.values()) {
            for (Feature2D feature2D : feature2DS) {
                features.append(feature2D.getLocationKey()).append(", ");
            }
        }
        return features.toString();
    }
}
