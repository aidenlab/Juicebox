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

package juicebox.tools.utils.juicer.hiccups;

import juicebox.HiCGlobals;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.NormalizationVector;
import juicebox.tools.clt.juicer.HiCCUPS;
import juicebox.tools.utils.common.ArrayTools;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DTools;
import juicebox.track.feature.FeatureFilter;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

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

    /**
     * @return a Feature2D peak for a possible peak location from hiccups
     */
    public static Feature2D generatePeak(String chrName, float observed, float peak, int rowPos, int colPos,
                                         float expectedBL, float expectedDonut, float expectedH, float expectedV,
                                         float binBL, float binDonut, float binH, float binV, int resolution) {

        Map<String, String> attributes = new HashMap<String, String>();

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
                                              final Dataset ds, final List<Chromosome> chromosomes,
                                              final NormalizationType norm) {

        final Map<String, Integer> chrNameToIndex = new HashMap<String, Integer>();
        for (Chromosome chr : chromosomes) {
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
                return new ArrayList<Feature2D>();
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
        List<Feature2D> filtered = new ArrayList<Feature2D>();
        for (Feature2D feature : feature2DList) {
            if (enrichmentThresholdSatisfied(feature, maxEnrich))
                filtered.add(feature);
        }
        return filtered;
    }

    private static void filterOutFeaturesByFDR(Feature2DList list) {
        list.filterLists(new FeatureFilter() {
            @Override
            public List<Feature2D> filter(String chr, List<Feature2D> feature2DList) {
                return fdrThreshold(feature2DList);
            }
        });
    }

    private static List<Feature2D> fdrThreshold(List<Feature2D> feature2DList) {
        List<Feature2D> filtered = new ArrayList<Feature2D>();
        for (Feature2D feature : feature2DList) {
            if (fdrThresholdsSatisfied(feature))
                filtered.add(feature);
        }
        return filtered;
    }

    private static List<Feature2D> removeLowMapQ(int res, int chrIndex, Dataset ds, List<Feature2D> list, NormalizationType norm) throws IOException {

        List<Feature2D> features = new ArrayList<Feature2D>();
        NormalizationVector normVectorContainer = ds.getNormalizationVector(chrIndex, ds.getZoomForBPResolution(res),
                norm);
        if (normVectorContainer == null) {
            HiCFileTools.triggerNormError(norm);
        } else {
            double[] normalizationVector = normVectorContainer.getData();
            for (Feature2D feature : list) {
                int index1 = feature.getStart1() / res, index2 = feature.getStart2() / res;
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
        LinkedList<Feature2D> featureLL = new LinkedList<Feature2D>(new HashSet<Feature2D>(feature2DList));
        List<Feature2D> coalesced = new ArrayList<Feature2D>();

        while (!featureLL.isEmpty()) {

            // See Feature2D
            Collections.sort(featureLL);
            Collections.reverse(featureLL);

            Feature2D pixel = featureLL.pollFirst();
            featureLL.remove(pixel);
            List<Feature2D> pixelList = new ArrayList<Feature2D>();
            pixelList.add(pixel);

            int pixelListX = pixel.getStart1();
            int pixelListY = pixel.getStart2();
            double r = 0;
            double pixelClusterRadius = originalClusterRadius;

            for (Feature2D px : featureLL) {
                // TODO should likely reduce radius or at least start with default?
                //System.out.println("Radius " + HiCCUPS.pixelClusterRadius);
                if (hypotenuse(pixelListX - px.getStart1(), pixelListY - px.getStart2()) <= pixelClusterRadius) {
                    pixelList.add(px);
                    pixelListX = mean(pixelList, 1);
                    pixelListY = mean(pixelList, 2);

                    List<Double> distances = new ArrayList<Double>();
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

            pixel.setEnd1(pixel.getStart1() + resolution);
            pixel.setEnd2(pixel.getStart2() + resolution);
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

    private static void handleExistingMergerWithTwentyFiveKB(Feature2DList mergedList, Feature2DList twentyFiveKBList) {
        // add peaks unique to 25 kB
        Feature2DList centroidsTwentyFiveKB = Feature2DTools.extractReproducibleCentroids(mergedList, twentyFiveKBList, 2 * 25000);
        Feature2DList distant25 = Feature2DTools.extractPeaksNotNearCentroids(twentyFiveKBList, centroidsTwentyFiveKB, "25->centroids");
        mergedList.add(distant25);
    }

    private static Feature2DList handleFiveAndTenKBMerger(Feature2DList fiveKBList, Feature2DList tenKBList) {
        // add peaks commonly found between 5 and 10 kB
        Feature2DList centroidsFiveKB = Feature2DTools.extractReproducibleCentroids(tenKBList, fiveKBList, 2 * 10000);
        Feature2DList mergedList = Feature2DTools.extractPeaksNearCentroids(fiveKBList, centroidsFiveKB, "5->centroids");

        // add peaks unique to 10 kB
        Feature2DList centroidsTenKB = Feature2DTools.extractReproducibleCentroids(fiveKBList, tenKBList, 2 * 10000);
        Feature2DList distant10 = Feature2DTools.extractPeaksNotNearCentroids(tenKBList, centroidsTenKB, "10->centroids");
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

    public static Feature2DList postProcess(Map<Integer, Feature2DList> looplists, Dataset ds,
                                            List<Chromosome> commonChromosomes, List<HiCCUPSConfiguration> configurations,
                                            NormalizationType norm, File outputDirectory) {
        for (HiCCUPSConfiguration conf : configurations) {

            int res = conf.getResolution();
            removeLowMapQFeatures(looplists.get(res), res, ds, commonChromosomes, norm);
            coalesceFeaturesToCentroid(looplists.get(res), res, conf.getClusterRadius());
            filterOutFeaturesByFDR(looplists.get(res));
            looplists.get(res).exportFeatureList(new File(outputDirectory, POST_PROCESSED + "_" + res), true, Feature2DList.ListFormat.FINAL);
        }

        return mergeAllResolutions(looplists);
    }

    public static void calculateThresholdAndFDR(int index, int width, double fdr, float[] poissonPMF,
                                                int[][] rcsHist, float[] threshold, float[][] fdrLog) {
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
        }
    }
}
