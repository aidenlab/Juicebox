/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
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

import juicebox.data.ChromosomeHandler;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by muhammadsaadshamim on 11/17/15.
 */
public class GenomeWideList<T extends Feature> {

    /**
     * Genome-wide list of features, where each string is a key for an
     * inter or intra-chromosomal region
     */
    private final Map<String, List<T>> featureLists = new HashMap<>();

    /** Constructors**/

    /**
     * Private constructor only used for cloning
     * todo delete / make private @mss
     */
    public GenomeWideList() {
    }

    /**
     * @param handler
     */
    public GenomeWideList(ChromosomeHandler handler) {
        for (Chromosome c : handler.getChromosomeArray()) {
            featureLists.put("" + c.getIndex(), new ArrayList<T>());
        }
    }

    /**
     * @param handler
     * @param features to be added to list
     */
    public GenomeWideList(ChromosomeHandler handler, List<T> features) {
        this(handler);
        addAll(features);
    }

    /**
     * Basic methods/functions
     */

    /**
     * Initialize a genome wide list using an existing list (creates deep copy)
     *
     * @param gwList
     */
    public GenomeWideList(final GenomeWideList<T> gwList) {
        processLists(new FeatureFunction<T>() {
            @Override
            public void process(String chr, List<T> featureList) {
                if (gwList.containsKey(chr)) {
                    addAll(gwList.getFeatures(chr));
                }
            }
        });
    }

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
    public synchronized void setFeatures(String key, List<T> features) {
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
    public synchronized int size() {
        int val = 0;
        for (List<T> features : featureLists.values()) {
            val += features.size();
        }
        return val;
    }

    /**
     * @param features to be added to this list (deep copy)
     */
    @SuppressWarnings("unchecked")
    public synchronized void addAll(List<T> features) {
        for (T feature : features) {
            featureLists.get(feature.getKey()).add((T) feature.deepClone());
        }
    }

    /**
     * pass interface implementing a filter for anchors
     *
     * @param filter
     */
    public synchronized void filterLists(FeatureFilter<T> filter) {
        for (String chr : featureLists.keySet()) {
            featureLists.put(chr, filter.filter(chr, featureLists.get(chr)));
        }
    }

    /** methods to create copies **/

    /**
     * pass interface implementing a process for all anchors
     *
     * @param function
     */
    public synchronized void processLists(FeatureFunction<T> function) {
        for (String key : featureLists.keySet()) {
            function.process(key, featureLists.get(key));
        }
    }

    /**
     * @return deep copy of the anchor list
     */
    public GenomeWideList<T> deepClone() {
        GenomeWideList<T> clone = new GenomeWideList<>();
        for (String key : featureLists.keySet()) {
            clone.featureLists.put(key, cloneFeatureList(featureLists.get(key)));
        }
        return clone;
    }

    /**
     * @param features
     * @return deep copy of the list of features
     */
    @SuppressWarnings("unchecked")
    private List<T> cloneFeatureList(List<T> features) {
        List<T> clonedFeatures = new ArrayList<>();
        for (T feature : features) {
            clonedFeatures.add((T) feature.deepClone());//feature.<T>deepClone()
        }
        return clonedFeatures;
    }

    /**
     * @return set of keys for genome-wide regions (i.e. category/location keys)
     */
    public Set<String> keySet() {
        return featureLists.keySet();
    }

    /**
     * Add feature to genome-wide list with specified key
     *
     * @param key
     * @param feature
     */
    public synchronized void addFeature(String key, T feature) {
        if (featureLists.containsKey(key)) {
            featureLists.get(key).add(feature);
        } else {
            List<T> features = new ArrayList<>();
            features.add(feature);
            featureLists.put(key, features);
        }
    }

    public void simpleExport(final File file) {
        try {
            final FileWriter fw = new FileWriter(file);
            processLists(new FeatureFunction<T>() {
                @Override
                public void process(String chr, List<T> featureList) {
                    for (T t : featureList) {
                        try {
                            if (fw != null) fw.write(t.toString() + "\n");
                        } catch (IOException e) {
                            System.err.println("Unable to write to file for exporting GWList");
                        }
                    }
                }
            });
            try {
                fw.close();
            } catch (IOException e) {
                System.err.println("Unable to close file for exporting GWList");
            }
        } catch (IOException e) {
            System.err.println("Unable to open file for exporting GWList");
        }
    }
}
