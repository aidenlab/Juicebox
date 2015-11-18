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

package juicebox.data.feature;

import juicebox.data.HiCFileTools;
import org.broad.igv.feature.Chromosome;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 11/17/15.
 */
public class GenomeWideList<T extends Feature> {

    /**
     * Genome-wide list of features, where each string is a key for an
     * inter or intra-chromosomal region
     */
    private Map<String, List<T>> featureLists = new HashMap<String, List<T>>();

    /** Constructors**/

    /**
     * Private constructor only used for cloning
     */
    public GenomeWideList() {
    }

    /**
     * @param chromosomes for genome
     */
    public GenomeWideList(List<Chromosome> chromosomes) {
        for (Chromosome chr : chromosomes) {
            featureLists.put(chr.getName(), new ArrayList<T>());
        }
    }

    /**
     * @param genomeID
     */
    public GenomeWideList(String genomeID) {
        this(HiCFileTools.loadChromosomes(genomeID));
    }

    /**
     * @param chromosomes for genome
     * @param features     to be added to list
     */
    public GenomeWideList(List<Chromosome> chromosomes, List<T> features) {
        this(chromosomes);
        addAll(features);
    }

    /**
     * @param genomeID
     * @param features  to be added to list
     */
    public GenomeWideList(String genomeID, List<T> features) {
        this(HiCFileTools.loadChromosomes(genomeID), features);
    }

    /**
     * Basic methods/functions
     */

    /**
     * @param key
     * @return
     */
    public boolean containsKey(String key) {
        return featureLists.containsKey(key);
    }

    /**
     * @param key
     * @param features
     */
    public void setFeatures(String key, List<T> features) {
        featureLists.put(key, features);
    }

    /**
     * @param key
     * @return features for corresponding region
     */
    public List<T> getFeatures(String key) {
        return featureLists.get(key);
    }

    /**
     * @return number of features in full list
     */
    public int size() {
        int val = 0;
        for (List<T> features : featureLists.values()) {
            val += features.size();
        }
        return val;
    }

    /**
     * @param features to be added to this list
     */
    private void addAll(List<T> features) {
        for (T feature : features) {
            featureLists.get(feature.getKey()).add(feature);
        }
    }

    /**
     * pass interface implementing a filter for anchors
     *
     * @param filter
     */
    public void filterLists(FeatureFilter<T> filter) {
        for (String chr : featureLists.keySet()) {
            featureLists.put(chr, filter.filter(chr, featureLists.get(chr)));
        }
    }

    /**
     * pass interface implementing a process for all anchors
     *
     * @param function
     */
    public void processLists(FeatureFunction<T> function) {
        for (String key : featureLists.keySet()) {
            function.process(key, featureLists.get(key));
        }
    }

    /** methods to create copies **/

    /**
     * @return deep copy of the anchor list
     */
    public GenomeWideList<T> deepClone() {
        GenomeWideList<T> clone = new GenomeWideList<T>();
        for (String key : featureLists.keySet()) {
            clone.featureLists.put(key, cloneFeatureList(featureLists.get(key)));
        }
        return clone;
    }

    /**
     * @param features
     * @return deep copy of the list of features
     */
    private List<T> cloneFeatureList(List<T> features) {
        List<T> clonedFeatures = new ArrayList<T>();
        for (T feature : features) {
            clonedFeatures.add(feature.<T>deepClone());
        }
        return clonedFeatures;
    }
}
