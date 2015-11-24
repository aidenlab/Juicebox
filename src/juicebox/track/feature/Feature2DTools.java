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

import juicebox.tools.clt.juicer.CompareLists;
import juicebox.tools.utils.juicer.hiccups.HiCCUPSUtils;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by muhammadsaadshamim on 10/27/15.
 */
public class Feature2DTools {


    public static Feature2DList extractPeaksNearCentroids(final Feature2DList featureList, final Feature2DList centroids) {
        final Feature2DList peaks = new Feature2DList();

        featureList.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {

                if (centroids.containsKey(chr)) {

                    final Set<String> keys = new HashSet<String>();
                    for (Feature2D f : centroids.getFeatureList(chr)) {
                        keys.add(f.getLocationKey());
                    }


                    for (Feature2D f : feature2DList) {
                        if (keys.contains(f.getLocationKey())) {
                            //f.setAttribute(HiCCUPSUtils.nearCentroidAttr, "1");
                            peaks.addByKey(chr, f);
                        }
                    }
                } else {
                    System.err.println(chr + " key not found for centroids. NC. Possible error?");
                    System.err.println("Centroid: " + centroids.getKeySet());
                    System.err.println("Actual: " + featureList.getKeySet());
                }
            }
        });

        return peaks;
    }

    public static Feature2DList extractPeaksNotNearCentroids(final Feature2DList featureList, final Feature2DList centroids) {
        final Feature2DList peaks = new Feature2DList();

        featureList.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                if (centroids.containsKey(chr)) {
                    final Set<String> keys = new HashSet<String>();
                    for (Feature2D f : centroids.getFeatureList(chr)) {
                        keys.add(f.getLocationKey());
                    }

                    for (Feature2D f : feature2DList) {
                        if (!keys.contains(f.getLocationKey())) {
                            //f.setAttribute(HiCCUPSUtils.notNearCentroidAttr, "1");
                            peaks.addByKey(chr, f);
                        }
                    }
                } else {
                    System.err.println(chr + " key not found for centroids. NN. Possible error?");
                    System.err.println("Centroid: " + centroids.getKeySet());
                    System.err.println("Actual: " + featureList.getKeySet());
                }
            }
        });

        return peaks;
    }

    /**
     * Calculate FDR values for all peaks
     *
     * @param fdrLogBL
     * @param fdrLogDonut
     * @param fdrLogH
     * @param fdrLogV
     */
    public static void calculateFDR(Feature2DList features, final float[][] fdrLogBL, final float[][] fdrLogDonut,
                                    final float[][] fdrLogH, final float[][] fdrLogV) {
        features.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D feature : feature2DList) {
                    HiCCUPSUtils.calculateFDR(feature, fdrLogBL, fdrLogDonut, fdrLogH, fdrLogV);
                }
            }
        });
    }


    public static Feature2DList extractReproducibleCentroids(final Feature2DList firstFeatureList, Feature2DList secondFeatureList, final int radius) {

        final Feature2DList centroids = new Feature2DList();

        secondFeatureList.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> secondFeature2DList) {
                if (firstFeatureList.containsKey(chr)) {
                    List<Feature2D> base1FeatureList = firstFeatureList.getFeatureList(chr);
                    for (Feature2D f2 : secondFeature2DList) {
                        for (Feature2D f1 : base1FeatureList) {
                            int dx = f1.getStart1() - f2.getStart1();
                            int dy = f1.getStart2() - f2.getStart2();
                            double d = HiCCUPSUtils.hypotenuse(dx, dy);
                            if (d <= radius) {
                                //f2.setAttribute(HiCCUPSUtils.centroidAttr, "" + d);
                                centroids.addByKey(chr, f2);
                                break;
                            }
                        }
                    }
                }
            }
        });
        return centroids;
    }

    /**
     * @return peaks within radius of diagonal
     */
    public static Feature2DList getPeaksNearDiagonal(Feature2DList feature2DList, final int radius) {
        final Feature2DList peaks = new Feature2DList();
        feature2DList.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D f : feature2DList) {
                    int dist = Math.abs(f.getStart1() - f.getStart2());
                    if (dist < radius) {
                        //f.setAttribute(HiCCUPSUtils.nearDiagAttr, "1");
                        peaks.addByKey(chr, f);
                    }
                }
            }
        });
        return peaks;
    }

    /**
     * @return peaks with observed values exceeding limit
     */
    public static Feature2DList getStrongPeaks(Feature2DList feature2DList, final int limit) {
        final Feature2DList peaks = new Feature2DList();
        feature2DList.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D f : feature2DList) {
                    float obs = f.getFloatAttribute(HiCCUPSUtils.OBSERVED);
                    if (obs > limit) {
                        //f.setAttribute(HiCCUPSUtils.StrongAttr, "1");
                        peaks.addByKey(chr, f);
                    }
                }
            }
        });
        return peaks;
    }

    /**
     * @param listA
     * @param listB
     * @return feature list where duplicates/common features are removed and results are color coded
     */
    public static Feature2DList compareLists(final Feature2DList listA, final Feature2DList listB) {
        Feature2DList featuresUniqueToA = new Feature2DList(listA);
        Feature2DList featuresUniqueToB = new Feature2DList(listB);

        featuresUniqueToA.filterLists(new FeatureFilter() {
            @Override
            public List<Feature2D> filter(String chr, List<Feature2D> feature2DList) {
                if (listB.containsKey(chr)) {
                    feature2DList.removeAll(listB.getFeatureList(chr));
                }
                return feature2DList;
            }
        });

        featuresUniqueToB.filterLists(new FeatureFilter() {
            @Override
            public List<Feature2D> filter(String chr, List<Feature2D> feature2DList) {
                if (listA.containsKey(chr)) {
                    feature2DList.removeAll(listA.getFeatureList(chr));
                }
                return feature2DList;
            }
        });

        // color code results
        featuresUniqueToA.setColor(CompareLists.AAA);
        featuresUniqueToB.setColor(CompareLists.BBB);

        // also add an attribute in addition to color coding
        featuresUniqueToA.addAttributeFieldToAll("parent_list", "A");
        featuresUniqueToB.addAttributeFieldToAll("parent_list", "B");

        // combine into one list
        Feature2DList results = new Feature2DList(featuresUniqueToA);
        results.add(featuresUniqueToB);

        return results;
    }


    public static Feature2DList subtract(final Feature2DList listA, final Feature2DList listB) {
        Feature2DList result = new Feature2DList(listA);
        result.filterLists(new FeatureFilter() {
            @Override
            public List<Feature2D> filter(String chr, List<Feature2D> feature2DList) {
                if (listB.containsKey(chr)) {
                    feature2DList.removeAll(listB.getFeatureList(chr));
                }
                return feature2DList;
            }
        });
        result.removeDuplicates();
        return result;
    }
}
