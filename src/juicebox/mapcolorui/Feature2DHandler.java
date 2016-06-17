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

package juicebox.mapcolorui;

import gnu.trove.procedure.TIntProcedure;
import juicebox.data.MatrixZoomData;
import juicebox.track.HiCGridAxis;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.Feature2DParser;
import juicebox.track.feature.FeatureFunction;
import net.sf.jsi.SpatialIndex;
import net.sf.jsi.rtree.RTree;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;

import java.awt.*;
import java.util.*;
import java.util.List;

/**
 * Handles 2D features such as domains and peaks
 * Created by muhammadsaadshamim on 8/6/15.
 */
public class Feature2DHandler {

    //private static final float MAX_DIST_NEIGHBOR = 1000f;
    private static final int offsetPX = 4;
    public static boolean isTranslucentPlottingEnabled = false;
    public static int numberOfLoopsToFind = 1000;
    private static boolean sparseFeaturePlottingEnabled = false, enlargedFeaturePlottingEnabled = false;
    private static boolean showLoops = true;
    private final Map<String, Feature2DList> loopLists;
    private final Map<String, SpatialIndex> featureRtrees = new HashMap<String, SpatialIndex>();
    private final Map<String, List<Feature2D>> allFeaturesAcrossGenome = new HashMap<String, List<Feature2D>>();

    public Feature2DHandler() {
        loopLists = new HashMap<String, Feature2DList>();
        clearLists();
    }

    public static Rectangle rectangleFromFeature(HiCGridAxis xAxis, HiCGridAxis yAxis, Feature2D feature, double binOriginX,
                                                 double binOriginY, double scaleFactor) {

        int binStart1 = xAxis.getBinNumberForGenomicPosition(feature.getStart1());
        int binEnd1 = xAxis.getBinNumberForGenomicPosition(feature.getEnd1());
        int binStart2 = yAxis.getBinNumberForGenomicPosition(feature.getStart2());
        int binEnd2 = yAxis.getBinNumberForGenomicPosition(feature.getEnd2());

        // option to draw larger rectangles for ease of viewing
        int offset = 0, offsetDoubled = 0;
        if (enlargedFeaturePlottingEnabled) {
            offset = offsetPX;
            offsetDoubled = offsetPX + offsetPX;
        }

        int x = (int) ((binStart1 - binOriginX) * scaleFactor) - offset;
        int y = (int) ((binStart2 - binOriginY) * scaleFactor) - offset;
        int w = (int) Math.max(1, scaleFactor * (binEnd1 - binStart1)) + offsetDoubled;
        int h = (int) Math.max(1, scaleFactor * (binEnd2 - binStart2)) + offsetDoubled;


        return new Rectangle(x, y, w, h);
    }

    public static List<Pair<Rectangle, Feature2D>> featurePairs(List<Feature2D> features, MatrixZoomData zd,
                                                                double binOriginX, double binOriginY, double scale) {
        final List<Pair<Rectangle, Feature2D>> featurePairs = new ArrayList<Pair<Rectangle, Feature2D>>();

        final HiCGridAxis xAxis = zd.getXGridAxis();
        final HiCGridAxis yAxis = zd.getYGridAxis();

        for (Feature2D feature : features) {
            featurePairs.add(new Pair<Rectangle, Feature2D>(
                    rectangleFromFeature(xAxis, yAxis, feature, binOriginX, binOriginY, scale), feature));
        }

        return featurePairs;
    }

    public void clearLists() {
        loopLists.clear();
        showLoops = true;
        sparseFeaturePlottingEnabled = false;
        allFeaturesAcrossGenome.clear();
        featureRtrees.clear();
    }

    public void setShowLoops(boolean showLoops) {
        Feature2DHandler.showLoops = showLoops;
    }

    public void removeFeaturePath(String fileName) {
        Set<String> keysToRemove = new HashSet<String>();
        for (String key : loopLists.keySet()) {
            if (key.endsWith(fileName)) {
                setLoopsInvisible(key);
                keysToRemove.add(key);
            }
        }
        for (String key : keysToRemove)
            loopLists.remove(key);
    }

    public void setLoopsInvisible(String path) {
        if (loopLists.containsKey(path)) {
            loopLists.get(path).setVisible(false);
            remakeRTree();
        }
    }

    private void remakeRTree() {
        allFeaturesAcrossGenome.clear();
        featureRtrees.clear();
        for (Feature2DList list : loopLists.values()) {
            if (list.isVisible()) {
                list.processLists(new FeatureFunction() {
                    @Override
                    public void process(String chr, List<Feature2D> feature2DList) {

                        if (allFeaturesAcrossGenome.containsKey(chr)) {
                            allFeaturesAcrossGenome.get(chr).addAll(feature2DList);
                        } else {
                            allFeaturesAcrossGenome.put(chr, new ArrayList<Feature2D>(feature2DList));
                        }

                        // this part handles reflections, so if changes are needed to
                        // rendering upper right region, they should be made here
                        String[] parts = chr.split("_");
                        if (parts[0].equals(parts[1])) { // intrachromosomal
                            List<Feature2D> reflectedFeatures = new ArrayList<Feature2D>();
                            for (Feature2D feature : feature2DList) {
                                if (!feature.isOnDiagonal()) {
                                    reflectedFeatures.add(feature.reflectionAcrossDiagonal());
                                }
                            }
                            allFeaturesAcrossGenome.get(chr).addAll(reflectedFeatures);
                        }
                    }
                });
            }
        }

        for (String key : allFeaturesAcrossGenome.keySet()) {
            List<Feature2D> features = allFeaturesAcrossGenome.get(key);
            SpatialIndex si = new RTree();
            si.init(null);
            for (int i = 0; i < features.size(); i++) {
                //Rectangle rect = getRectFromFeature(features.get(i));
                //si.add(new net.sf.jsi.Rectangle((float)rect.getMinX(), (float)rect.getMinY(), (float)rect.getMaxX(), (float)rect.getMaxY()),i);
                Feature2D feature = features.get(i);
                si.add(new net.sf.jsi.Rectangle((float) feature.getStart1(), (float) feature.getStart2(),
                        (float) feature.getEnd1(), (float) feature.getEnd2()), i);
            }
            featureRtrees.put(key, si);
        }
    }

    public void loadLoopList(String path, List<Chromosome> chromosomes) {
        if (loopLists.get(path) != null) {
            loopLists.get(path).setVisible(true);
            System.out.println("Making " + path + " visible");
        } else {
            Feature2DList newList = Feature2DParser.loadFeatures(path, chromosomes, true, null, false);
            loopLists.put(path, newList);
        }
        remakeRTree();
    }

    public List<Feature2DList> getAllVisibleLoopLists() {
        List<Feature2DList> visibleLoopList = new ArrayList<Feature2DList>();
        if (showLoops) {
            for (Feature2DList list : loopLists.values()) {
                if (list.isVisible()) {
                    visibleLoopList.add(list);
                }
            }
        }
        return visibleLoopList;
    }

    public List<Feature2D> getVisibleFeatures(int chrIdx1, int chrIdx2) {
        List<Feature2D> visibleLoopList = new ArrayList<Feature2D>();
        if (showLoops) {
            for (Feature2DList list : loopLists.values()) {
                if (list.isVisible()) {
                    List<Feature2D> currList = list.get(chrIdx1, chrIdx2);
                    if (currList != null) {
                        for (Feature2D feature2D : currList) {
                            visibleLoopList.add(feature2D);
                        }
                    }
                }
            }
        }
        return visibleLoopList;
    }

    public List<Feature2D> findNearbyFeatures(MatrixZoomData zd, int chrIdx1, int chrIdx2, int x, int y, int n,
                                              double binOriginX, double binOriginY, double scale) {
        final List<Feature2D> foundFeatures = new ArrayList<Feature2D>();
        final String key = Feature2DList.getKey(chrIdx1, chrIdx2);

        if (featureRtrees.containsKey(key) && showLoops) {
            if (sparseFeaturePlottingEnabled) {
                final HiCGridAxis xAxis = zd.getXGridAxis();
                final HiCGridAxis yAxis = zd.getYGridAxis();


                featureRtrees.get(key).nearestN(
                        getGenomicPointFromXYCoordinate(x, y, xAxis, yAxis, binOriginX, binOriginY, scale),      // the point for which we want to find nearby rectangles
                        new TIntProcedure() {         // a procedure whose execute() method will be called with the results
                            public boolean execute(int i) {
                                Feature2D feature = allFeaturesAcrossGenome.get(key).get(i);
                                foundFeatures.add(feature);
                                return true;              // return true here to continue receiving results
                            }
                        },
                        n,                            // the number of nearby rectangles to find
                        Float.MAX_VALUE               // Don't bother searching further than this. MAX_VALUE means search everything
                );

            } else {
                foundFeatures.addAll(allFeaturesAcrossGenome.get(key));
            }
        }
        return foundFeatures;
    }

    public List<Pair<Rectangle, Feature2D>> findNearbyFeaturePairs(MatrixZoomData zd, int chrIdx1, int chrIdx2, int x, int y, int n,
                                                                   final double binOriginX, final double binOriginY, final double scale) {

        final List<Pair<Rectangle, Feature2D>> featurePairs = new ArrayList<Pair<Rectangle, Feature2D>>();

        if (showLoops) {
            final String key = Feature2DList.getKey(chrIdx1, chrIdx2);

            final HiCGridAxis xAxis = zd.getXGridAxis();
            final HiCGridAxis yAxis = zd.getYGridAxis();

            if (featureRtrees.containsKey(key)) {
                featureRtrees.get(key).nearestN(
                        getGenomicPointFromXYCoordinate(x, y, xAxis, yAxis, binOriginX, binOriginY, scale),      // the point for which we want to find nearby rectangles
                        new TIntProcedure() {         // a procedure whose execute() method will be called with the results
                            public boolean execute(int i) {
                                Feature2D feature = allFeaturesAcrossGenome.get(key).get(i);
                                featurePairs.add(new Pair<Rectangle, Feature2D>(rectangleFromFeature(xAxis, yAxis, feature, binOriginX, binOriginY, scale), feature));
                                return true;              // return true here to continue receiving results
                            }
                        },
                        n,  // the number of nearby rectangles to find
                        Float.MAX_VALUE  // Don't bother searching further than this. MAX_VALUE means search everything
                );
            }
        }

        return featurePairs;
    }

    public List<Feature2D> findContainedFeatures(int chrIdx1, int chrIdx2, net.sf.jsi.Rectangle currentWindow) {
        final List<Feature2D> foundFeatures = new ArrayList<Feature2D>();
        final String key = Feature2DList.getKey(chrIdx1, chrIdx2);

        if (featureRtrees.containsKey(key)) {

            featureRtrees.get(key).contains(
                    currentWindow,      // the window in which we want to find all rectangles
                    new TIntProcedure() {         // a procedure whose execute() method will be called with the results
                        public boolean execute(int i) {
                            Feature2D feature = allFeaturesAcrossGenome.get(key).get(i);
                            //System.out.println(feature.getChr1() + "\t" + feature.getStart1() + "\t" + feature.getStart2());
                            foundFeatures.add(feature);
                            return true;              // return true here to continue receiving results
                        }
                    }
            );

        } else {
            List<Feature2D> features = allFeaturesAcrossGenome.get(key);
            if (features != null) foundFeatures.addAll(features);
        }

        return foundFeatures;
    }

    private net.sf.jsi.Point getGenomicPointFromXYCoordinate(double x, double y, HiCGridAxis xAxis, HiCGridAxis yAxis,
                                                             double binOriginX, double binOriginY, double scale) {
        float x2 = (float) (((x / scale) + binOriginX) * xAxis.getBinSize());
        float y2 = (float) (((y / scale) + binOriginY) * yAxis.getBinSize());

        //System.out.println("x "+x2+" y "+y2);
        return new net.sf.jsi.Point(x2, y2);
    }

    public void setSparseFeaturePlotting(boolean status) {
        sparseFeaturePlottingEnabled = status;
    }

    public void enlarge2DFeaturePlotting(boolean status) {
        enlargedFeaturePlottingEnabled = status;
    }

    public void toggleFeatureOpacity(boolean status) {
        isTranslucentPlottingEnabled = status;
    }
}
