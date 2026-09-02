/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2022 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

package juicebox.tools.utils.juicer.hiccups;

import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.NormalizationVector;
import juicebox.data.basics.Chromosome;
import juicebox.tools.clt.juicer.HiCCUPS;
import juicebox.tools.clt.juicer.HiCCUPS2;
import juicebox.tools.utils.common.ArrayTools;
import juicebox.track.feature.*;
import juicebox.windowui.NormalizationType;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.*;

/**
 * Utility class for HiCCUPS
 * Created by muhammadsaadshamim on 6/2/15.
 */
public class HiCCUPSUtils {

    public static final String OBSERVED = "observed";
    // for debugging
    /*
    public static final String notNearCentroidAttr = "NotNearCentroid";
    public static final String centroidAttr = "Centroid";
    public static final String nearCentroidAttr = "NearCentroid";
    public static final String nearDiagAttr = "NearDiag";
    public static final String StrongAttr = "Strong";
    public static final String FilterStage = "Stage";
    */
    private static final String PEAK = "peak";
    private static final String EXPECTEDBL = "expectedBL";
    private static final String EXPECTEDDONUT = "expectedDonut";
    private static final String EXPECTEDH = "expectedH";
    private static final String EXPECTEDV = "expectedV";
    private static final String BINBL = "binBL";
    private static final String BINDONUT = "binDonut";
    private static final String BINH = "binH";
    private static final String BINV = "binV";
    private static final String FDRBL = "fdrBL";
    private static final String FDRDONUT = "fdrDonut";
    private static final String FDRH = "fdrH";
    private static final String FDRV = "fdrV";
    private static final String RADIUS = "radius";
    private static final String CENTROID1 = "centroid1";
    private static final String CENTROID2 = "centroid2";
    private static final String NUMCOLLAPSED = "numCollapsed";
    private static final String POST_PROCESSED = "postprocessed_pixels";
    private static final String MERGED = "merged_loops.bedpe";
    private static final String REQUESTED = "_from_requested_loops";
    private static final String MERGED_REQUESTED = "merged" + REQUESTED + ".bedpe";
    private static final String FDR_THRESHOLDS = "fdr_thresholds";
    private static final String ENRICHED_PIXELS = "enriched_pixels";
    private static final String REQUESTED_LIST = "requested_list";

    /**
     * @return a Feature2D peak for a possible peak location from hiccups
     */
    public static Feature2D generatePeak(String chrName, float observed, float peak, int rowPos, int colPos,
                                         float expectedBL, float expectedDonut, float expectedH, float expectedV,
                                         float binBL, float binDonut, float binH, float binV, int resolution) {

        Map<String, String> attributes = new HashMap<>();

        attributes.put(OBSERVED, String.valueOf(observed));
        attributes.put(PEAK, String.valueOf(peak));

        attributes.put(EXPECTEDBL, String.valueOf(expectedBL));
        attributes.put(EXPECTEDDONUT, String.valueOf(expectedDonut));
        attributes.put(EXPECTEDH, String.valueOf(expectedH));
        attributes.put(EXPECTEDV, String.valueOf(expectedV));

        attributes.put(BINBL, String.valueOf(binBL));
        attributes.put(BINDONUT, String.valueOf(binDonut));
        attributes.put(BINH, String.valueOf(binH));
        attributes.put(BINV, String.valueOf(binV));

        int pos1 = Math.min(rowPos, colPos);
        int pos2 = Math.max(rowPos, colPos);

        return new Feature2D(Feature2D.FeatureType.PEAK, chrName, pos1, pos1 + resolution, chrName, pos2, pos2 + resolution, Color.black, attributes);
    }

    public static Feature2D generatePeak(String chrName, float observed, float peak, int rowPos, int colPos,
                                         float expectedBL, float expectedDonut, float expectedH, float expectedV,
                                         float binBL, float binDonut, float binH, float binV, int resolution,
                                         float pvalBL, float pvalDonut, float pvalH, float pvalV) {
        Map<String, String> attributes = new HashMap<>();

        attributes.put(OBSERVED, String.valueOf(observed));
        attributes.put(PEAK, String.valueOf(peak));

        attributes.put(EXPECTEDBL, String.valueOf(expectedBL));
        attributes.put(EXPECTEDDONUT, String.valueOf(expectedDonut));
        attributes.put(EXPECTEDH, String.valueOf(expectedH));
        attributes.put(EXPECTEDV, String.valueOf(expectedV));

        attributes.put(BINBL, String.valueOf(binBL));
        attributes.put(BINDONUT, String.valueOf(binDonut));
        attributes.put(BINH, String.valueOf(binH));
        attributes.put(BINV, String.valueOf(binV));

        attributes.put(FDRBL, String.valueOf(pvalBL));
        attributes.put(FDRDONUT, String.valueOf(pvalDonut));
        attributes.put(FDRH, String.valueOf(pvalH));
        attributes.put(FDRV, String.valueOf(pvalV));


        int pos1 = Math.min(rowPos, colPos);
        int pos2 = Math.max(rowPos, colPos);

        return new Feature2D(Feature2D.FeatureType.PEAK, chrName, pos1, pos1 + resolution, chrName, pos2, pos2 + resolution, Color.black, attributes);

    }

    /**
     * Calculate fdr values for a given peak
     */
    public static void calculateFDR(Feature2D feature, float[][] fdrLogBL, float[][] fdrLogDonut, float[][] fdrLogH, float[][] fdrLogV) {

        int observed = (int) feature.getFloatAttribute(OBSERVED);
        int binBL = (int) feature.getFloatAttribute(BINBL);
        int binDonut = (int) feature.getFloatAttribute(BINDONUT);
        int binH = (int) feature.getFloatAttribute(BINH);
        int binV = (int) feature.getFloatAttribute(BINV);

        if (binBL >= 0 && binDonut >= 0 && binH >= 0 && binV >= 0 && observed >= 0) {
            feature.addFloatAttribute(FDRBL, (fdrLogBL[binBL][observed]));
            feature.addFloatAttribute(FDRDONUT, (fdrLogDonut[binDonut][observed]));
            feature.addFloatAttribute(FDRH, (fdrLogH[binH][observed]));
            feature.addFloatAttribute(FDRV, (fdrLogV[binV][observed]));
        } else {
            System.out.println("Error in calculateFDR binBL=" + binBL + " binDonut=" + binDonut + " binH=" + binH +
                    " binV=" + binV + " observed=" + observed);
        }

    }

    private static void removeLowMapQFeatures(Feature2DList list, final int resolution,
                                              final Dataset ds, final ChromosomeHandler chromosomeHandler,
                                              final NormalizationType norm) {

        final Map<String, Integer> chrNameToIndex = new HashMap<>();
        for (Chromosome chr : chromosomeHandler.getChromosomeArray()) {
            chrNameToIndex.put(Feature2DList.getKey(chr, chr), chr.getIndex());
        }
        if (HiCGlobals.printVerboseComments) {
            System.out.println("Initial: " + list.getNumTotalFeatures());
        }
        list.filterLists(new FeatureFilter() {
            @Override
            public List<Feature2D> filter(String chr, List<Feature2D> feature2DList) {
                try {
                    return removeLowMapQ(resolution, chrNameToIndex.get(chr), ds, feature2DList, norm);
                } catch (Exception e) {
                    System.err.println("Unable to remove low mapQ entries for " + chr);
                    //e.printStackTrace();
                }
                return new ArrayList<>();
            }
        });


    }

    private static void coalesceFeaturesToCentroid(Feature2DList list, final int resolution, final int centroidRadius) {
        list.filterLists(new FeatureFilter() {
            @Override
            public List<Feature2D> filter(String chr, List<Feature2D> feature2DList) {
                return coalescePixelsToCentroid(resolution, feature2DList, centroidRadius);
            }
        });
    }

    public static void filterOutFeaturesByEnrichment(Feature2DList list, final float maxEnrich) {
        list.filterLists(new FeatureFilter() {
            @Override
            public List<Feature2D> filter(String chr, List<Feature2D> feature2DList) {
                return enrichmentThreshold(feature2DList, maxEnrich);
            }
        });
    }

    private static List<Feature2D> enrichmentThreshold(List<Feature2D> feature2DList, final float maxEnrich) {
        List<Feature2D> filtered = new ArrayList<>();
        for (Feature2D feature : feature2DList) {
            if (enrichmentThresholdSatisfied(feature, maxEnrich))
                filtered.add(feature);
        }
        return filtered;
    }

    private static void doFinalCleanUp(Feature2DList list) {
        list.filterLists(new FeatureFilter() {
            @Override
            public List<Feature2D> filter(String chr, List<Feature2D> feature2DList) {
                return finalCleanUp(feature2DList);
            }
        });
    }

    private static List<Feature2D> finalCleanUp(List<Feature2D> feature2DList) {
        List<Feature2D> filtered = new ArrayList<>();
        for (Feature2D feature : feature2DList) {
            if (finalCleanSatisfied(feature))
                filtered.add(feature);
        }
        return filtered;
    }

    private static void filterOutFeaturesByFDR(Feature2DList list, int version) {
        list.filterLists(new FeatureFilter() {
            @Override
            public List<Feature2D> filter(String chr, List<Feature2D> feature2DList) {
                return fdrThreshold(feature2DList, version);
            }
        });
    }

    private static List<Feature2D> fdrThreshold(List<Feature2D> feature2DList, int version) {
        List<Feature2D> filtered = new ArrayList<>();
        for (Feature2D feature : feature2DList) {
            if (version == 1 ) {
                if (fdrThresholdsSatisfied(feature))
                    filtered.add(feature);
            } else if (version == 2) {
                if (fdrThresholds2Satisfied(feature))
                    filtered.add(feature);
            }
        }
        return filtered;
    }

    private static List<Feature2D> removeLowMapQ(int res, int chrIndex, Dataset ds, List<Feature2D> list, NormalizationType norm) throws IOException {

        List<Feature2D> features = new ArrayList<>();
        NormalizationVector normVectorContainer = ds.getNormalizationVector(chrIndex, ds.getZoomForBPResolution(res),
                norm);
        if (normVectorContainer == null) {
            HiCFileTools.triggerNormError(norm);
        } else {
            double[] normalizationVector = normVectorContainer.getData().getValues().get(0);
            for (Feature2D feature : list) {
                int index1 = (int) (feature.getStart1() / res);
                int index2 = (int) (feature.getStart2() / res);
                if (nearbyValuesClear(normalizationVector, index1) && nearbyValuesClear(normalizationVector, index2)) {
                    features.add(feature);
                }
            }
        }

        return features;
    }

    private static boolean nearbyValuesClear(double[] normalizationVector, int index) {
        for (int i = index - HiCCUPS.krNeighborhood; i <= index + HiCCUPS.krNeighborhood; i++) {
            if (Double.isNaN(normalizationVector[i]))
                return false;
        }
        return true;
    }

    /**
     * @return list of pixels coalesced to centroid of enriched region
     */
    private static List<Feature2D> coalescePixelsToCentroid(int resolution, List<Feature2D> feature2DList,
                                                            int originalClusterRadius) {
        // HashSet intermediate for removing duplicates; LinkedList used so that we can pop out highest obs values
        LinkedList<Feature2D> featureLL = new LinkedList<>(new HashSet<>(feature2DList));
        List<Feature2D> coalesced = new ArrayList<>();

        while (!featureLL.isEmpty()) {

            // See Feature2D
            Collections.sort(featureLL);
            Collections.reverse(featureLL);

            Feature2D pixel = featureLL.pollFirst();
            featureLL.remove(pixel);
            List<Feature2D> pixelList = new ArrayList<>();
            pixelList.add(pixel);

            int pixelListX = (int) pixel.getStart1();
            int pixelListY = (int) pixel.getStart2();
            double r = 0;
            double pixelClusterRadius = originalClusterRadius;

            for (Feature2D px : featureLL) {
                // TODO should likely reduce radius or at least start with default?
                //System.out.println("Radius " + HiCCUPS.pixelClusterRadius);
                if (hypotenuse(pixelListX - px.getStart1(), pixelListY - px.getStart2()) <= pixelClusterRadius) {
                    pixelList.add(px);
                    pixelListX = mean(pixelList, 1);
                    pixelListY = mean(pixelList, 2);

                    List<Double> distances = new ArrayList<>();
                    for (Feature2D px2 : pixelList) {
                        double dist = hypotenuse(pixelListX - px2.getStart1(), pixelListY - px2.getStart2());
                        if (Double.isNaN(dist) || dist < 0) {
                            System.err.println("Invalid distance while merging centroid");
                            System.exit(29);
                        }
                        distances.add(dist);
                    }
                    //System.out.println("Radii "+distances);
                    r = Math.round(Collections.max(distances));

                    pixelClusterRadius = originalClusterRadius + r;
                }
            }

            pixel.setEnd1((int) pixel.getStart1() + resolution);
            pixel.setEnd2((int) pixel.getStart2() + resolution);
            pixel.addIntAttribute(RADIUS, (int) Math.round(r));
            pixel.addIntAttribute(CENTROID1, (pixelListX + resolution / 2));
            pixel.addIntAttribute(CENTROID2, (pixelListY + resolution / 2));
            pixel.addIntAttribute(NUMCOLLAPSED, (pixelList.size()));
            setPixelColor(pixel);
            coalesced.add(pixel);

            featureLL.removeAll(pixelList);
        }

        return coalesced;
    }

    private static void setPixelColor(Feature2D pixel) {
        Color c = HiCCUPS.defaultPeakColor;
        if (HiCCUPS.shouldColorBeScaledByFDR) {
            double fdr = -Math.floor(Math.log10(
                    Math.max(pixel.getFloatAttribute(BINBL), pixel.getFloatAttribute(BINDONUT))));
            fdr = Math.max(Math.min(fdr, 10), 0) / 10;
            c = new Color((int) (fdr * c.getRed()), (int) (fdr * c.getGreen()), (int) (fdr * c.getBlue()));
        }
        pixel.setColor(c);
    }

    private static boolean fdrThresholdsSatisfied(Feature2D pixel) {
        double f = HiCCUPS.fdrsum;
        double t1 = HiCCUPS.oeThreshold1;
        double t2 = HiCCUPS.oeThreshold2;
        double t3 = HiCCUPS.oeThreshold3;

        int observed = Math.round(pixel.getFloatAttribute(OBSERVED));
        int numCollapsed = Math.round(pixel.getFloatAttribute(NUMCOLLAPSED));

        float expectedBL = pixel.getFloatAttribute(EXPECTEDBL);
        float expectedDonut = pixel.getFloatAttribute(EXPECTEDDONUT);
        float expectedH = pixel.getFloatAttribute(EXPECTEDH);
        float expectedV = pixel.getFloatAttribute(EXPECTEDV);

        float fdrBL = pixel.getFloatAttribute(FDRBL);
        float fdrDonut = pixel.getFloatAttribute(FDRDONUT);
        float fdrH = pixel.getFloatAttribute(FDRH);
        float fdrV = pixel.getFloatAttribute(FDRV);

        return observed > (t2 * expectedBL)
                && observed > (t2 * expectedDonut)
                && observed > (t1 * expectedH)
                && observed > (t1 * expectedV)
                && (observed > (t3 * expectedBL) || observed > (t3 * expectedDonut))
                && (numCollapsed > 1 || (fdrBL + fdrDonut + fdrH + fdrV) <= f);
    }

    private static boolean fdrThresholds2Satisfied(Feature2D pixel) {

        double f2 = HiCCUPS2.fdrsum2;
        double t1 = HiCCUPS2.oeThreshold1;
        double t2 = HiCCUPS2.oeThreshold2;
        double t3 = HiCCUPS2.oeThreshold3;




        int observed = Math.round(pixel.getFloatAttribute(OBSERVED));
        int numCollapsed = Math.round(pixel.getFloatAttribute(NUMCOLLAPSED));

        float expectedBL = pixel.getFloatAttribute(EXPECTEDBL);
        float expectedDonut = pixel.getFloatAttribute(EXPECTEDDONUT);
        float expectedH = pixel.getFloatAttribute(EXPECTEDH);
        float expectedV = pixel.getFloatAttribute(EXPECTEDV);

        float fdrBL = pixel.getFloatAttribute(FDRBL);
        float fdrDonut = pixel.getFloatAttribute(FDRDONUT);
        float fdrH = pixel.getFloatAttribute(FDRH);
        float fdrV = pixel.getFloatAttribute(FDRV);

        return observed > (t2 * expectedBL)
                && observed > (t2 * expectedDonut)
                && observed > (t1 * expectedH)
                && observed > (t1 * expectedV)
                && (observed > (t3 * expectedBL) || observed > (t3 * expectedDonut))
                && (numCollapsed > 1 || (fdrBL + fdrDonut + fdrH + fdrV) <= f2);

    }

    private static boolean finalCleanSatisfied(Feature2D pixel) {
        double f1 = HiCCUPS2.fdrsum1;
        double t4 = HiCCUPS2.oeThreshold4;

        long resolution = pixel.getWidth1();

        int observed = Math.round(pixel.getFloatAttribute(OBSERVED));
        int numCollapsed = Math.round(pixel.getFloatAttribute(NUMCOLLAPSED));

        float expectedBL = pixel.getFloatAttribute(EXPECTEDBL);
        float expectedDonut = pixel.getFloatAttribute(EXPECTEDDONUT);
        float expectedH = pixel.getFloatAttribute(EXPECTEDH);
        float expectedV = pixel.getFloatAttribute(EXPECTEDV);

        float fdrBL = pixel.getFloatAttribute(FDRBL);
        float fdrDonut = pixel.getFloatAttribute(FDRDONUT);
        float fdrH = pixel.getFloatAttribute(FDRH);
        float fdrV = pixel.getFloatAttribute(FDRV);

        if (resolution >= 5000) {
            return (observed < (t4 * expectedDonut) && observed < (t4 * expectedBL) && observed < (t4 * expectedH) && observed < (t4 * expectedV))
                    && (numCollapsed > 1 || (fdrBL + fdrDonut + fdrH + fdrV) <= f1);
        } else {
            return (observed < (t4 * expectedDonut) && observed < (t4 * expectedBL) && observed < (t4 * expectedH) && observed < (t4 * expectedV));
        }
    }

    private static boolean enrichmentThresholdSatisfied(Feature2D pixel, final float maxEnrich) {


        float observed = pixel.getFloatAttribute(OBSERVED);

        float expectedBL = pixel.getFloatAttribute(EXPECTEDBL);
        float expectedDonut = pixel.getFloatAttribute(EXPECTEDDONUT);
        float expectedH = pixel.getFloatAttribute(EXPECTEDH);
        float expectedV = pixel.getFloatAttribute(EXPECTEDV);

        return (observed < maxEnrich * expectedBL &&
                observed < maxEnrich * expectedDonut &&
                observed < maxEnrich * expectedH &&
                observed < maxEnrich * expectedV);
    }



    private static int mean(List<Feature2D> pixelList, int i) {
        int n = pixelList.size();
        double total = 0;
        for (Feature2D px : pixelList) {
            if (i == 1)
                total += px.getStart1();
            else if (i == 2)
                total += px.getStart2();
        }
        return (int) (total / n);
    }

    public static double hypotenuse(double x, double y) {
        return Math.sqrt(x * x + y * y);
    }

    /**
     * Code to merge high resolution calls (i.e. when <5kb resolution is used)
     *
     * @return loop list merged across resolutions
     */
    public static Feature2DList mergeHighResolutions(Map<Integer, Feature2DList> hiccupsLoopLists) {

        Feature2DList mergedList = new Feature2DList();

        if (hiccupsLoopLists.containsKey(1000) && hiccupsLoopLists.containsKey(2000) && hiccupsLoopLists.containsKey(5000)) {
            mergedList.add(handleGenericThreeListMerger(hiccupsLoopLists.get(1000), hiccupsLoopLists.get(2000),
                    hiccupsLoopLists.get(5000), 2000, 5000, 50000, 80000,
                    20, 40));
        } else if (hiccupsLoopLists.containsKey(1000) && hiccupsLoopLists.containsKey(2000)) {
            mergedList.add(handleGenericTwoListMerger(hiccupsLoopLists.get(1000), hiccupsLoopLists.get(2000), 2000,
                    50000, 20));
        } else if (hiccupsLoopLists.containsKey(1000) && hiccupsLoopLists.containsKey(5000)) {
            mergedList.add(handleGenericTwoListMerger(hiccupsLoopLists.get(1000), hiccupsLoopLists.get(5000), 5000,
                    80000, 20));
        } else if (hiccupsLoopLists.containsKey(2000) && hiccupsLoopLists.containsKey(5000)) {
            mergedList.add(handleGenericTwoListMerger(hiccupsLoopLists.get(2000), hiccupsLoopLists.get(5000), 5000,
                    80000, 40));
        }

        mergedList.removeDuplicates();

        if (hiccupsLoopLists.containsKey(10000)) {
            handleExistingGenericMerger(mergedList, hiccupsLoopLists.get(10000), 10000);
        }

        mergedList.removeDuplicates();

        return mergedList;
    }

    /**
     * Expected behavior is to merge 5, 10, and 25 kB lists as determined
     * If at least one of these resolutions are found, any other resolution (e.g. 50 kB) will be thrown out
     * If none of these expected resolutions are found, all remaining resolutions will be merged without any filtering
     *
     * @return loop list merged across resolutions
     */
    public static Feature2DList mergeAllResolutions(Map<Integer, Feature2DList> hiccupsLooplists) {

        Feature2DList mergedList = new Feature2DList();
        boolean listHasBeenAltered = false;

        if (hiccupsLooplists.containsKey(5000) || hiccupsLooplists.containsKey(10000)) {
            if (hiccupsLooplists.containsKey(5000) && hiccupsLooplists.containsKey(10000)) {

                if (HiCGlobals.printVerboseComments) {
                    System.out.println("Merge 5k and 10k res loops");
                }
                mergedList.add(handleFiveAndTenKBMerger(hiccupsLooplists.get(5000), hiccupsLooplists.get(10000)));

            } else if (hiccupsLooplists.containsKey(5000)) {

                if (HiCGlobals.printVerboseComments) {
                    System.out.println("Retrieve 5k res loops");
                }
                mergedList.add(hiccupsLooplists.get(5000));

            } else { // i.e. (hiccupsLooplists.containsKey(10000))

                if (HiCGlobals.printVerboseComments) {
                    System.out.println("Retrieve 10k res loops");
                }
                mergedList.add(hiccupsLooplists.get(10000));

            }
            listHasBeenAltered = true;
            mergedList.removeDuplicates();
        }

        if (hiccupsLooplists.containsKey(25000)) {
            if (listHasBeenAltered) {

                if (HiCGlobals.printVerboseComments) {
                    System.out.println("Merge with 25k res loops");
                }
                handleExistingMergerWithTwentyFiveKB(mergedList, hiccupsLooplists.get(25000));

            } else {

                if (HiCGlobals.printVerboseComments) {
                    System.out.println("Retrieve 25k res loops");
                }
                mergedList.add(hiccupsLooplists.get(25000));

            }
            mergedList.removeDuplicates();
            listHasBeenAltered = true;
        }

        // neither 25kB, 10kB, or 5kB list processed
        if (!listHasBeenAltered) {
            System.out.println("25kB, 10kB, or 5kB lists not found\nDefault lists being merged without filtering");
            for (Feature2DList list : hiccupsLooplists.values()) {
                mergedList.add(list);
            }
            mergedList.removeDuplicates();
        }

        return mergedList;
    }

    private static void handleExistingGenericMerger(Feature2DList mergedList, Feature2DList list2, int radius) {
        // add peaks unique to list 2
        Feature2DList centroidsList2 = Feature2DTools.extractReproducibleCentroids(mergedList, list2, 2 * radius);
        Feature2DList distantList2 = Feature2DTools.extractPeaksNotNearCentroids(list2, centroidsList2);
        mergedList.add(distantList2);
    }

    private static void handleExistingMergerWithTwentyFiveKB(Feature2DList mergedList, Feature2DList twentyFiveKBList) {
        // add peaks unique to 25 kB
        Feature2DList centroidsTwentyFiveKB = Feature2DTools.extractReproducibleCentroids(mergedList, twentyFiveKBList, 2 * 25000);
        Feature2DList distant25 = Feature2DTools.extractPeaksNotNearCentroids(twentyFiveKBList, centroidsTwentyFiveKB);
        mergedList.add(distant25);
    }

    private static Feature2DList handleGenericTwoListMerger(Feature2DList list1, Feature2DList list2, int radius1, int nearDiagonal1,
                                                            int observedLimit1) {
        // add peaks from list1 commonly found between list1 and list2 within 2*radius1
        Feature2DList centroidsList1List2 = Feature2DTools.extractReproducibleCentroids(list2, list1, 2 * radius1);
        Feature2DList mergedList = Feature2DTools.extractPeaksNearCentroids(list1, centroidsList1List2, "list12->centroids");

        // add peaks unique to list 2
        Feature2DList centroidsList2List1 = Feature2DTools.extractReproducibleCentroids(list1, list2, 2 * radius1);
        Feature2DList distantList2List1 = Feature2DTools.extractPeaksNotNearCentroids(list2, centroidsList2List1);
        mergedList.add(distantList2List1);

        // add peaks close to diagonal from list 1
        Feature2DList nearDiagList1 = Feature2DTools.getPeaksNearDiagonal(list1, nearDiagonal1);
        mergedList.add(nearDiagList1);

        // add particularly strong peaks from list 1
        Feature2DList strongList1 = Feature2DTools.getStrongPeaks(list1, observedLimit1);
        mergedList.add(strongList1);

        // TODO expand filter duplicates?
        mergedList.removeDuplicates();

        return mergedList;
    }

    private static Feature2DList handleGenericThreeListMerger(Feature2DList list1, Feature2DList list2, Feature2DList list3,
                                                              int radius1, int radius2, int nearDiagonal1, int nearDiagonal2,
                                                              int observedLimit1, int observedLimit2) {
        // add peaks from list1 commonly found between list1 and list2 within 2*radius1
        Feature2DList centroidsList1List2 = Feature2DTools.extractReproducibleCentroids(list2, list1, 2 * radius1);
        Feature2DList mergedList = Feature2DTools.extractPeaksNearCentroids(list1, centroidsList1List2, "list12->centroids");

        // add peaks from list1 commonly found between list1 and list3 with 2*radius2
        Feature2DList centroidsList1List3 = Feature2DTools.extractReproducibleCentroids(list3, list1, 2 * radius2);
        Feature2DList peaksList1List3 = Feature2DTools.extractPeaksNearCentroids(list1, centroidsList1List3, "list13->centroids");
        mergedList.add(peaksList1List3);

        // add peaks from list2 not found in list1, but common to list3 within 2*radius2
        Feature2DList centroidsList2List1 = Feature2DTools.extractReproducibleCentroids(list1, list2, 2 * radius1);
        Feature2DList distantList2List1 = Feature2DTools.extractPeaksNotNearCentroids(list2, centroidsList2List1);
        Feature2DList centroidsList2List3 = Feature2DTools.extractReproducibleCentroids(list3, distantList2List1, 2 * radius2);
        Feature2DList peaksList2List3 = Feature2DTools.extractPeaksNearCentroids(list2, centroidsList2List3, "list23->centroids");
        mergedList.add(peaksList2List3);

        // add peaks unique to list 3
        Feature2DList centroidsList3List1 = Feature2DTools.extractReproducibleCentroids(list1, list3, 2 * radius2);
        Feature2DList distantList3List1 = Feature2DTools.extractPeaksNotNearCentroids(list3, centroidsList3List1);
        Feature2DList centroidsList3List2 = Feature2DTools.extractReproducibleCentroids(list2, distantList3List1, 2 * radius2);
        Feature2DList distantList3List2 = Feature2DTools.extractPeaksNotNearCentroids(distantList3List1, centroidsList3List2);
        mergedList.add(distantList3List2);

        // add peaks close to diagonal from list 1
        Feature2DList nearDiagList1 = Feature2DTools.getPeaksNearDiagonal(list1, nearDiagonal1);
        mergedList.add(nearDiagList1);

        // add peaks close to diagonal from list 2
        Feature2DList nearDiagList2 = Feature2DTools.getPeaksNearDiagonal(distantList2List1, nearDiagonal2);
        mergedList.add(nearDiagList2);

        // add particularly strong peaks from list 1
        Feature2DList strongList1 = Feature2DTools.getStrongPeaks(list1, observedLimit1);
        mergedList.add(strongList1);

        // add particularly strong peaks from list 2
        Feature2DList strongList2 = Feature2DTools.getStrongPeaks(distantList2List1, observedLimit2);
        mergedList.add(strongList2);

        // TODO expand filter duplicates?
        mergedList.removeDuplicates();

        return mergedList;
    }

    private static Feature2DList handleFiveAndTenKBMerger(Feature2DList fiveKBList, Feature2DList tenKBList) {
        // add peaks commonly found between 5 and 10 kB
        Feature2DList centroidsFiveKB = Feature2DTools.extractReproducibleCentroids(tenKBList, fiveKBList, 2 * 10000);
        Feature2DList mergedList = Feature2DTools.extractPeaksNearCentroids(fiveKBList, centroidsFiveKB, "5->centroids");

        // add peaks unique to 10 kB
        Feature2DList centroidsTenKB = Feature2DTools.extractReproducibleCentroids(fiveKBList, tenKBList, 2 * 10000);
        Feature2DList distant10 = Feature2DTools.extractPeaksNotNearCentroids(tenKBList, centroidsTenKB);
        mergedList.add(distant10);

        // add peaks close to diagonal in 5kB
        Feature2DList nearDiag = Feature2DTools.getPeaksNearDiagonal(fiveKBList, 110000);
        mergedList.add(nearDiag);

        // add particularly strong peaks from 5kB
        Feature2DList strong = Feature2DTools.getStrongPeaks(fiveKBList, 100);
        mergedList.add(strong);

        // TODO expand filter duplicates?
        mergedList.removeDuplicates();

        return mergedList;
    }

    public static int[] extractIntegerValues(List<String> valList, int n) {
        if (valList != null && !valList.isEmpty()) {
            int[] result = ArrayTools.extractIntegers(valList);
            if (result.length == n) {
                return result;
            } else if (result.length == 1) {
                return ArrayTools.preInitializeIntArray(result[0], n);
            } else {
                System.err.println("Must pass " + n + " parameters in place of " + Arrays.toString(result));
                System.exit(30);
                return new int[0];
            }
        }
        return null;
    }

    public static double[] extractFDRValues(List<String> fdrStrings, int n, float defaultVal) {
        if (fdrStrings != null && !fdrStrings.isEmpty()) {
            double[] fdrValues = extractDoubleValues(fdrStrings, n, defaultVal);
            return ArrayTools.inverseArrayValues(fdrValues);
        }
        return null;
    }

    public static double[] extractDoubleValues(List<String> valList, int n, double defaultVal) {
        if (valList == null) {
            return ArrayTools.preInitializeDoubleArray(defaultVal, n);
        } else {
            // if < 0, just return whatever is extracted
            // if > 0, then verify lengths match up (to ensure 1-to-1 correspondence of resolution with other params)
            if (n < 0)
                return ArrayTools.extractDoubles(valList);
            else {
                double[] result = ArrayTools.extractDoubles(valList);
                if (result.length == n) {
                    return result;
                } else if (result.length == 1) {
                    return ArrayTools.preInitializeDoubleArray(result[0], n);
                } else {
                    System.err.println("Must pass " + n + " parameters in place of " + Arrays.toString(result));
                    System.exit(31);
                }
            }
        }
        return null;
    }

    public static void postProcess(Map<Integer, Feature2DList> looplists, Dataset ds,
                                            ChromosomeHandler chromosomeHandler, List<HiCCUPSConfiguration> configurations,
                                            NormalizationType norm, File outputDirectory, boolean isRequested, File outputFile, int version) {
        for (HiCCUPSConfiguration conf : configurations) {

            int res = conf.getResolution();
            removeLowMapQFeatures(looplists.get(res), res, ds, chromosomeHandler, norm);
            coalesceFeaturesToCentroid(looplists.get(res), res, conf.getClusterRadius());
            filterOutFeaturesByFDR(looplists.get(res), version);
            looplists.get(res).exportFeatureList(new File(outputDirectory, getPostprocessedLoopsFileName(res, isRequested)),
                    true, Feature2DList.ListFormat.FINAL);
        }

        Feature2DList mergedList;
        if (looplists.containsKey(1000) || looplists.containsKey(2000)) {
            mergedList = mergeHighResolutions(looplists);
        } else {
            mergedList = mergeAllResolutions(looplists);
        }
        if (version == 2) {
            doFinalCleanUp(mergedList);
        }
        mergedList.exportFeatureList(outputFile, true, Feature2DList.ListFormat.FINAL);
    }


    public static void calculateThresholdAndFDR(int index, int width, double fdr, float[] poissonPMF,
                                                long[][] rcsHist, float[] threshold, float[][] fdrLog) {
        if (rcsHist[index][0] > 0) {
            float[] expected = ArrayTools.scalarMultiplyArray(rcsHist[index][0], poissonPMF);
            float[] rcsExpected = ArrayTools.makeReverseCumulativeArray(expected);
            for (int j = 0; j < width; j++) {
                if (fdr * rcsExpected[j] <= rcsHist[index][j]) {
                    if (j == 0) {
                        threshold[index] = (width - 2);
                    } else {
                        threshold[index] = (j - 1);
                    }
                    break;
                }
            }

            for (int j = 0; j < width; j++) {
                float sum1 = rcsExpected[j];
                float sum2 = rcsHist[index][j];
                if (sum2 > 0) {
                    fdrLog[index][j] = sum1 / sum2;
                } else {
                    break;
                }
            }
        } else if (HiCGlobals.printVerboseComments) {
            System.out.println("poss err index: " + index + " rcsHist " + rcsHist[index][0]);
        }
    }

    public static long histogramPvals(Map<Long, Integer> kHist, long totalPValues, long mstar) {

        for (long i = totalPValues; i >= 1; i--) {
            if (kHist.get(i) != null) {
                mstar = mstar - kHist.get(i);
            }
            if (i == mstar) {
                return i;
            }
        }

        System.err.println("didn't converge");
        return 0;
    }

    public static Feature2DList filterOutFeaturelistByEnrichment(List<HiCCUPSConfiguration> configs, String folderPath, float maxEnrich, ChromosomeHandler commonChromosomesHandler) {
        Feature2DList results = new Feature2DList();
        for (HiCCUPSConfiguration config : configs) {

            String fname = folderPath + File.separator + getRequestedLoopsFileName(config.getResolution());
            Feature2DList requestedList = Feature2DParser.loadFeatures(fname, commonChromosomesHandler, true, null, false);
            HiCCUPSUtils.filterOutFeaturesByEnrichment(requestedList, maxEnrich);
            results.add(requestedList);
        }
        return results;
    }

    public static String getEnrichedPixelFileName(int resolution) {
        return ENRICHED_PIXELS + "_" + resolution + ".bedpe";
    }

    private static String getPostprocessedLoopsFileName(int resolution, boolean isRequested) {
        if (isRequested) {
            return POST_PROCESSED + "_" + getRequestedLoopsFileName(resolution);
        } else {
            return POST_PROCESSED + "_" + resolution + ".bedpe";
        }
    }

    public static String getRequestedLoopsFileName(int resolution) {
        return REQUESTED_LIST + "_" + resolution + ".bedpe";
    }

    public static String getMergedLoopsFileName() {
        return MERGED;
    }

    public static String getMergedRequestedLoopsFileName() {
        return MERGED_REQUESTED;
    }

    public static String getFDRThresholdsFilename(int resolution) {
        return FDR_THRESHOLDS + "_" + resolution;
    }
}
