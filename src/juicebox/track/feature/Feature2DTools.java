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

import juicebox.HiCGlobals;
import juicebox.tools.clt.juicer.CompareLists;
import juicebox.tools.utils.juicer.hiccups.HiCCUPSUtils;

import java.awt.*;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by muhammadsaadshamim on 10/27/15.
 */
public class Feature2DTools {


    public static Feature2DList extractPeaksNearCentroids(final Feature2DList featureList, final Feature2DList centroids,
                                                          final String errorMessage) {
        final Feature2DList peaks = new Feature2DList();

        centroids.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {

                if (featureList.containsKey(chr)) {

                    final Set<String> keys = new HashSet<String>();
                    for (Feature2D f : feature2DList) {
                        keys.add(f.getLocationKey());
                    }


                    for (Feature2D f : featureList.getFeatureList(chr)) {
                        if (keys.contains(f.getLocationKey())) {
                            peaks.addByKey(chr, f);
                        }
                    }
                } else if (HiCGlobals.printVerboseComments) {
                    System.err.println(chr + " key not found for centroids. Tag:NearCentroid-" + errorMessage +
                            "Invalid set of centroids must have been calculated");
                }
            }
        });

        return peaks;
    }

    public static Feature2DList extractPeaksNotNearCentroids(final Feature2DList featureList, final Feature2DList centroids,
                                                             final String errorMessage) {
        final Feature2DList peaks = new Feature2DList();

        featureList.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                if (centroids.containsKey(chr)) {
                    // there are centroids for this chr i.e. need to check if loops should be added

                    // get upper left corner location value
                    final Set<String> keys = new HashSet<String>();
                    for (Feature2D f : centroids.getFeatureList(chr)) {
                        keys.add(f.getLocationKey());
                    }

                    // add pixels not already the centroid
                    for (Feature2D f : feature2DList) {
                        if (!keys.contains(f.getLocationKey())) {
                            peaks.addByKey(chr, f);
                        }
                    }
                } else {
                    // no centroids for chr i.e. all of these loops should be added
                    peaks.addByKey(chr, feature2DList);
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

    public static Feature2DList extractReproducibleCentroids(final Feature2DList firstFeatureList, Feature2DList secondFeatureList, final int radius, final double fraction) {

        final Feature2DList centroids = new Feature2DList();

        secondFeatureList.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> secondFeature2DList) {
                if (firstFeatureList.containsKey(chr)) {
                    List<Feature2D> base1FeatureList = firstFeatureList.getFeatureList(chr);
                    for (Feature2D f2 : secondFeature2DList) {
                        double lowestDistance = -1;
                        Feature2D overlap = null;
                        for (Feature2D f1 : base1FeatureList) {
                            int dx = f1.getStart1() - f2.getStart1();
                            int dy = f1.getStart2() - f2.getStart2();
                            double d = HiCCUPSUtils.hypotenuse(dx, dy);
                            if (d < lowestDistance || lowestDistance == -1) {
                                overlap = f1;
                                lowestDistance = d;
                            }
                        }
                        if (lowestDistance != -1) {
                            double f = lowestDistance / (f2.getStart2() - f2.getStart1());
                            if (lowestDistance <= radius && f <= fraction) {
                                centroids.addByKey(chr, f2);
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
    public static Feature2DList compareLists(final Feature2DList listA, final Feature2DList listB, boolean colorCode) {
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

        if (colorCode) {
            // color code results
            featuresUniqueToA.setColor(CompareLists.AAA);
            featuresUniqueToB.setColor(CompareLists.BBB);

            // also add an attribute in addition to color coding
            featuresUniqueToA.addAttributeFieldToAll("parent_list", "A");
            featuresUniqueToB.addAttributeFieldToAll("parent_list", "B");
        }
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


    public static boolean loopIsUpstreamOfDomain(Feature2D loop, Feature2D domain, int threshold) {
        return loop.getEnd1() < domain.getStart1() - threshold &&
                loop.getEnd2() < domain.getStart2() - threshold;
    }

    public static boolean loopIsDownstreamOfDomain(Feature2D loop, Feature2D domain, int threshold) {
        return loop.getStart1() > domain.getEnd1() + threshold &&
                loop.getStart2() > domain.getEnd2() + threshold;
    }

    public static boolean domainContainsLoopWithinExpandedTolerance(Feature2D loop, Feature2D domain, int threshold) {

        Rectangle bounds = new Rectangle(domain.getStart1() - threshold, domain.getStart2() - threshold,
                domain.getWidth1() + 2 * threshold, domain.getWidth2() + 2 * threshold);
        Point point = new Point(loop.getMidPt1(), loop.getMidPt2());

        return bounds.contains(point);
    }

    /**
     * Compares a feature against all other features in list
     *
     * @param feature
     * @param existingFeatures
     * @return
     */
    public static boolean doesOverlap(Feature2D feature, List<Feature2D> existingFeatures) {
        boolean repeat = false;
        for (Feature2D existingFeature : existingFeatures) {
            if (existingFeature.overlapsWith(feature)) {
                repeat = true;
            }
        }
        return repeat;
    }

    public static boolean isResolutionPresent(final Feature2DList feature2DList, final int resolution) {

        final boolean[] returnValue = new boolean[1];
        returnValue[0] = false;
        feature2DList.processLists(new FeatureFunction() {
            @Override
            public void process(String chr, List<Feature2D> feature2DList) {
                for (Feature2D feature : feature2DList) {
                    if (feature.getWidth1() == resolution) {
                        returnValue[0] = true;
                        return;
                    }
                }
            }
        });
        return returnValue[0];

    }
}
