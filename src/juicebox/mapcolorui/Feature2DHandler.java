/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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
import juicebox.data.ChromosomeHandler;
import juicebox.data.MatrixZoomData;
import juicebox.gui.SuperAdapter;
import juicebox.track.HiCGridAxis;
import juicebox.track.feature.*;
import net.sf.jsi.SpatialIndex;
import net.sf.jsi.rtree.RTree;
import org.broad.igv.feature.Chromosome;

import java.awt.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Handles 2D features such as domains and peaks
 * Created by muhammadsaadshamim on 8/6/15.
 */
public class Feature2DHandler {

    //private static final float MAX_DIST_NEIGHBOR = 1000f;
    private static final int offsetPX = 4;
    public static int numberOfLoopsToFind = 1000;
    private final Map<String, SpatialIndex> featureRtrees = new HashMap<>();
    protected Feature2DList loopList;
    private boolean isTranslucentPlottingEnabled = false;
    private boolean sparseFeaturePlottingEnabled = false, isEnlargedPlottingEnabled = false;
    private boolean layerVisible = true;
    private String path = null;

    public Feature2DHandler() {
        loopList = new Feature2DList();
        clearLists();
    }

    public Feature2DHandler(Feature2DList inputList) {
        this();
        setLoopList(inputList);
    }

    public Rectangle getRectangleFromFeature(HiCGridAxis xAxis, HiCGridAxis yAxis, Feature2D feature, double binOriginX,
                                             double binOriginY, double scaleFactor) {

        int binStart1 = xAxis.getBinNumberForGenomicPosition(feature.getStart1());
        int binEnd1 = xAxis.getBinNumberForGenomicPosition(feature.getEnd1());
        int binStart2 = yAxis.getBinNumberForGenomicPosition(feature.getStart2());
        int binEnd2 = yAxis.getBinNumberForGenomicPosition(feature.getEnd2());

        // option to draw larger rectangles for ease of viewing
        int offset = 0, offsetDoubled = 0;
        if (isEnlargedPlottingEnabled) {
            offset = offsetPX;
            offsetDoubled = offsetPX + offsetPX;
        }

        int x = (int) ((binStart1 - binOriginX) * scaleFactor) - offset;
        int y = (int) ((binStart2 - binOriginY) * scaleFactor) - offset;
        int w = (int) Math.max(1, scaleFactor * (binEnd1 - binStart1)) + offsetDoubled;
        int h = (int) Math.max(1, scaleFactor * (binEnd2 - binStart2)) + offsetDoubled;

        return new Rectangle(x, y, w, h);
    }

    public List<Feature2DGuiContainer> convertFeaturesToFeaturePairs(AnnotationLayerHandler handler, List<Feature2D> features, MatrixZoomData zd,
                                                                     double binOriginX, double binOriginY, double scale) {
        final List<Feature2DGuiContainer> featurePairs = new ArrayList<>();

        final HiCGridAxis xAxis = zd.getXGridAxis();
        final HiCGridAxis yAxis = zd.getYGridAxis();

        for (Feature2D feature : features) {
            featurePairs.add(new Feature2DGuiContainer(
                    getRectangleFromFeature(xAxis, yAxis, feature, binOriginX, binOriginY, scale),
                    feature, handler));
        }

        return featurePairs;
    }

    protected void clearLists() {
        loopList = new Feature2DList();
        layerVisible = true;
        featureRtrees.clear();
    }

    public boolean getLayerVisibility() {
        return layerVisible;
    }

    public void setLayerVisibility(boolean showLoops) {
        this.layerVisible = showLoops;
    }


    protected void remakeRTree() {
        featureRtrees.clear();

        loopList.processLists(new FeatureFunction() {
            @Override
            public void process(String key, List<Feature2D> features) {

                SpatialIndex si = new RTree();
                si.init(null);
                for (int i = 0; i < features.size(); i++) {
                    Feature2D feature = features.get(i);
                    si.add(new net.sf.jsi.Rectangle((float) feature.getStart1(), (float) feature.getStart2(),
                            (float) feature.getEnd1(), (float) feature.getEnd2()), i);
                }
                featureRtrees.put(key, si);
            }
        });
        //}
    }

    public resultContainer setLoopList(String path, ChromosomeHandler chromosomeHandler) {
        int numFeaturesAdded = 0;
        ArrayList<String> attributes = null;
        Color color = null;
        if (this.path == null) {
            this.path = path;
            Feature2DList newList = Feature2DParser.loadFeatures(path, chromosomeHandler, true, null, false);
            numFeaturesAdded += newList.getNumTotalFeatures();
            color = newList.extractSingleFeature().getColor();
            attributes = newList.extractSingleFeature().getAttributeKeys();
            loopList = newList;
            Map<String, String> defaultAttributes = new HashMap<>(); //creates defaultAttributes map
            for (String attribute : attributes) {
                defaultAttributes.put(attribute, null);
            }
            loopList.setDefaultAttributes(defaultAttributes);
        }
        //loopLists.get(path).setVisible(true);
        remakeRTree();
        return new resultContainer(numFeaturesAdded, color, attributes);
    }

    public void createNewMergedLoopLists(List<Feature2DList> lists) {

        for (Feature2DList list : lists) {
            if (list.getNumTotalFeatures() > 0) {
                addToLoopList(list, false);
            }
        }

        remakeRTree();
    }

    public void setLoopList(Feature2DList feature2DList) {
        loopList = feature2DList;
        remakeRTree();
    }

    private void addToLoopList(Feature2DList feature2DList, boolean remakeTree) {
        loopList.add(feature2DList);
        if (remakeTree) {
            remakeRTree();
        }
    }

    public Feature2DList getAllVisibleLoops() {
        Feature2DList visibleLoopList = new Feature2DList();
        if (layerVisible) {
            visibleLoopList = loopList;
        }
        return visibleLoopList;
    }

    public List<Feature2D> getNearbyFeatures(MatrixZoomData zd, int chrIdx1, int chrIdx2, int x, int y, int n,
                                             final double binOriginX, final double binOriginY, final double scale) {
        final List<Feature2D> foundFeatures = new ArrayList<>();
        final String key = Feature2DList.getKey(chrIdx1, chrIdx2);
        final HiCGridAxis xAxis = zd.getXGridAxis();
        final HiCGridAxis yAxis = zd.getYGridAxis();


        if (featureRtrees.containsKey(key) && layerVisible) {
            if (sparseFeaturePlottingEnabled) {

                try {
                    featureRtrees.get(key).nearestN(
                            getGenomicPointFromXYCoordinate(x, y, xAxis, yAxis, binOriginX, binOriginY, scale),      // the point for which we want to find nearby rectangles
                            new TIntProcedure() {         // a procedure whose execute() method will be called with the results
                                public boolean execute(int i) {
                                    Feature2D feature = loopList.get(key).get(i);
                                    Rectangle rect = getRectangleFromFeature(xAxis, yAxis, feature, binOriginX, binOriginY, scale);
                                    if (!SuperAdapter.assemblyModeCurrentlyActive || (rect.getWidth() > 1 && rect.getHeight() > 1)) {
                                        foundFeatures.add(feature);
                                    }
                                    return true;              // return true here to continue receiving results
                                }
                            },
                            n,                            // the number of nearby rectangles to find
                            Float.MAX_VALUE               // Don't bother searching further than this. MAX_VALUE means search everything
                    );
                } catch (Exception e) {
                    System.err.println("Error encountered getting nearby features" + e.getLocalizedMessage());
                }
            } else {

                for (Feature2D feature : loopList.get(key)) {
                    Rectangle rect = getRectangleFromFeature(xAxis, yAxis, feature, binOriginX, binOriginY, scale);
                    if (!SuperAdapter.assemblyModeCurrentlyActive || (rect.getWidth() > 1 && rect.getHeight() > 1)) {
                        foundFeatures.add(feature);
                    }
                }
            }
        }
        return foundFeatures;
    }

    public List<Feature2D> getIntersectingFeatures(int chrIdx1, int chrIdx2, net.sf.jsi.Rectangle selectionWindow, boolean ignoreVisibility) {
        final List<Feature2D> foundFeatures = new ArrayList<>();
        final String key = Feature2DList.getKey(chrIdx1, chrIdx2);

        if (layerVisible || ignoreVisibility) {
            if (featureRtrees.containsKey(key)) {
                try {
                    featureRtrees.get(key).intersects(
                            selectionWindow,
                            new TIntProcedure() {     // a procedure whose execute() method will be called with the results
                                public boolean execute(int i) {
                                    Feature2D feature = loopList.get(key).get(i);
                                    foundFeatures.add(feature);
                                    return true;      // return true here to continue receiving results
                                }
                            });
                    } catch (Exception e) {
                        System.err.println("Error encountered getting intersecting features" + e.getLocalizedMessage());
                    }
            } else {
                System.err.println("returning all; didn't find " + key + " intersecting");
                List<Feature2D> features = loopList.get(key);
                if (features != null) foundFeatures.addAll(features);
            }
        }
        return foundFeatures;
    }

    public List<Feature2D> getContainedFeatures(int chrIdx1, int chrIdx2, net.sf.jsi.Rectangle currentWindow) {
        final List<Feature2D> foundFeatures = new ArrayList<>();
        final String key = Feature2DList.getKey(chrIdx1, chrIdx2);

        if (featureRtrees.containsKey(key)) {
            featureRtrees.get(key).contains(
                    currentWindow,      // the window in which we want to find all rectangles
                    new TIntProcedure() {         // a procedure whose execute() method will be called with the results
                        public boolean execute(int i) {
                            Feature2D feature = loopList.get(key).get(i);
                            //System.out.println(feature.getChr1() + "\t" + feature.getStart1() + "\t" + feature.getStart2());
                            foundFeatures.add(feature);
                            return true;              // return true here to continue receiving results
                        }
                    }
            );
        } else {
            System.err.println("returning all; key " + key + " not found contained");
            List<Feature2D> features = loopList.get(key);
            if (features != null) foundFeatures.addAll(features);
        }

        return foundFeatures;
    }

    private net.sf.jsi.Point getGenomicPointFromXYCoordinate(double x, double y, HiCGridAxis xAxis, HiCGridAxis yAxis,
                                                             double binOriginX, double binOriginY, double scale) {
        float x2 = (float) (((x / scale) + binOriginX) * xAxis.getBinSize());
        float y2 = (float) (((y / scale) + binOriginY) * yAxis.getBinSize());
        return new net.sf.jsi.Point(x2, y2);
    }

    public void setSparsePlottingEnabled(boolean status) {
        sparseFeaturePlottingEnabled = status;
    }

    public boolean getIsSparsePlottingEnabled() {
        return sparseFeaturePlottingEnabled;
    }

    public boolean getIsTransparent() {
        return isTranslucentPlottingEnabled;
    }

    public void setIsTransparent(boolean status) {
        isTranslucentPlottingEnabled = status;
    }

    public boolean getIsEnlarged() {
        return isEnlargedPlottingEnabled;
    }

    public void setIsEnlarged(boolean status) {
        isEnlargedPlottingEnabled = status;
    }

    public void setColorOfAllAnnotations(Color color) {
        loopList.setColor(color);

    }

    public Feature2DList getFeatureList() {
        return loopList;
    }

    public List<Feature2D> getContainedFeatures(Chromosome chrom, int rectULX, int rectULY, int rectLRX, int rectLRY, int resolution) {
        net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(
                rectULX * resolution,
                rectULY * resolution,
                rectLRX * resolution,
                rectLRY * resolution);
        return getContainedFeatures(chrom.getIndex(), chrom.getIndex(), currentWindow);
    }

    public class resultContainer {
        public final int n;
        public final Color color;
        final ArrayList<String> attributes;

        resultContainer(int n, Color color, ArrayList<String> attributes) {
            this.n = n;
            this.color = color;
            this.attributes = attributes;
        }
    }
}
