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

package juicebox.tools.utils.juicer.hiccups;

import juicebox.data.Dataset;
import juicebox.tools.clt.HiCCUPS;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.track.feature.FeatureFilter;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;

import java.awt.*;
import java.util.*;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 6/2/15.
 */
public class HiCCUPSUtils {

    public static final String OBSERVED = "observed";
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

    /**
     * Generate a Feature2D peak for a possible peak location from hiccups
     * @param chrName
     * @param observed
     * @param peak
     * @param rowPos
     * @param colPos
     * @param expectedBL
     * @param expectedDonut
     * @param expectedH
     * @param expectedV
     * @param binBL
     * @param binDonut
     * @param binH
     * @param binV
     * @return feature
     */
    public static Feature2D generatePeak(String chrName, float observed, float peak, int rowPos, int colPos,
                                         float expectedBL, float expectedDonut, float expectedH, float expectedV,
                                         float binBL, float binDonut, float binH, float binV) {

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

        return new Feature2D(Feature2D.peak, chrName, pos1, pos1+1, chrName, pos2, pos2 + 1, Color.black, attributes);
    }

    /**
     * Calculate fdr values for a given peak
     * @param feature
     * @param fdrLogBL
     * @param fdrLogDonut
     * @param fdrLogH
     * @param fdrLogV
     */
    public static void calculateFDR(Feature2D feature, float[][] fdrLogBL, float[][] fdrLogDonut, float[][] fdrLogH, float[][] fdrLogV) {

        int observed = (int) feature.getFloatAttribute(OBSERVED);
        int binBL = (int) feature.getFloatAttribute(BINBL);
        int binDonut = (int) feature.getFloatAttribute(BINDONUT);
        int binH = (int) feature.getFloatAttribute(BINH);
        int binV = (int) feature.getFloatAttribute(BINV);

        if(binBL >= 0  && binDonut >= 0  && binH >= 0  && binV >= 0  && observed >= 0) {
            feature.addAttribute(FDRBL, String.valueOf(fdrLogBL[binBL][observed]));
            feature.addAttribute(FDRDONUT, String.valueOf(fdrLogDonut[binDonut][observed]));
            feature.addAttribute(FDRH, String.valueOf(fdrLogH[binH][observed]));
            feature.addAttribute(FDRV, String.valueOf(fdrLogV[binV][observed]));
        }
        else{
            System.out.println("Error in calculateFDR binBL=" + binBL + " binDonut=" + binDonut +" binH=" + binH +
                    " binV="+ binV + " observed="+ observed);
        }

    }

    public static String oldOutput(Feature2D feature){
        return feature.getChr1()+"\t"+feature.getStart1()+"\t"+feature.getChr2()+"\t"+feature.getStart2()+"\t"+
                feature.getAttribute(OBSERVED)
                +"\t"+feature.getAttribute(EXPECTEDBL)
                +"\t"+feature.getAttribute(EXPECTEDDONUT)
                +"\t"+feature.getAttribute(EXPECTEDH)
                +"\t"+feature.getAttribute(EXPECTEDV)
                +"\t"+feature.getAttribute(BINBL)
                +"\t"+feature.getAttribute(BINDONUT)
                +"\t"+feature.getAttribute(BINH)
                +"\t"+feature.getAttribute(BINV)
                +"\t"+feature.getAttribute(FDRBL)
                +"\t"+feature.getAttribute(FDRDONUT)
                +"\t"+feature.getAttribute(FDRH)
                +"\t"+feature.getAttribute(FDRV);
    }

    public static void postProcessLoops(Feature2DList list, final int resolution,
                                        final Dataset ds, final List<Chromosome> chromosomes) {

        final Map<String, Integer> chrNameToIndex = new HashMap<String, Integer>();
        for (Chromosome chr : chromosomes) {
            chrNameToIndex.put(chr.getName(), chr.getIndex());
        }

        list.filterLists(new FeatureFilter() {
            @Override
            public List<Feature2D> filter(String chr, List<Feature2D> feature2DList) {
                return removeLowMapQ(resolution, chrNameToIndex.get(chr), ds, feature2DList);
            }
        });

        list.filterLists(new FeatureFilter() {
            @Override
            public List<Feature2D> filter(String chr, List<Feature2D> feature2DList) {
                return coalescePixelsToCentroid(resolution, feature2DList);
            }
        });
    }


    private static List<Feature2D> removeLowMapQ(int res, int chrIndex, Dataset ds, List<Feature2D> list) {

        double[] normalizationVector = ds.getNormalizationVector(chrIndex, ds.getZoomForBPResolution(res),
                NormalizationType.KR).getData();
        List<Feature2D> features = new ArrayList<Feature2D>();


        for (Feature2D feature : list) {
            int index1 = feature.getStart1() / res, index2 = feature.getStart2() / res;
            if (nearbyValuesClear(normalizationVector, index1) && nearbyValuesClear(normalizationVector, index2)) {
                features.add(feature);
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
     * @param resolution
     * @param feature2DList
     * @return
     */
    private static List<Feature2D> coalescePixelsToCentroid(int resolution, List<Feature2D> feature2DList) {

        LinkedList<Feature2D> featureLL = new LinkedList<Feature2D>(feature2DList);
        List<Feature2D> coalesced = new ArrayList<Feature2D>();

        while (!feature2DList.isEmpty()) {

            Collections.sort(featureLL);
            Collections.reverse(featureLL);

            Feature2D pixel = featureLL.pollFirst();
            List<Feature2D> pixelList = new ArrayList<Feature2D>();
            pixelList.add(pixel);

            int pixelListX = pixel.getStart1();
            int pixelListY = pixel.getStart2();

            int r = 0;
            for (Feature2D px : featureLL) {
                // TODO should likely reduce radius or at least start with default?
                if (hypotneuse(pixelListX - px.getStart1(), pixelListY - px.getStart2()) <= HiCCUPS.pixelClusterRadius) {
                    pixelList.add(px);
                    pixelListX = mean(pixelList, 1);
                    pixelListY = mean(pixelList, 2);

                    r = 0;
                    for (Feature2D px2 : pixelList) {
                        int rPrime = hypotneuse(pixelListX - px2.getStart1(), pixelListY - px2.getStart2());
                        if (rPrime > r)
                            r = rPrime;
                    }
                    HiCCUPS.pixelClusterRadius = HiCCUPS.originalPixelClusterRadius + r;
                }
            }

            pixel.setEnd1(pixel.getStart1() + resolution);
            pixel.setEnd2(pixel.getStart2() + resolution);
            pixel.addAttribute(RADIUS, String.valueOf(r));
            pixel.addAttribute(CENTROID1, String.valueOf(pixelListX + resolution / 2));
            pixel.addAttribute(CENTROID2, String.valueOf(pixelListY + resolution / 2));
            pixel.addAttribute(NUMCOLLAPSED, String.valueOf(pixelList.size()));

            for (Feature2D px : pixelList) {
                featureLL.remove(px);
            }

            setPixelColor(pixel);

            if (fdrThresholdsSatisfied(pixel))
                coalesced.add(pixel);
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

        int observed = (int) pixel.getFloatAttribute(OBSERVED);
        int numCollapsed = (int) pixel.getFloatAttribute(NUMCOLLAPSED);

        float expectedBL = pixel.getFloatAttribute(EXPECTEDBL);
        float expectedDonut = pixel.getFloatAttribute(EXPECTEDDONUT);
        float expectedH = pixel.getFloatAttribute(EXPECTEDH);
        float expectedV = pixel.getFloatAttribute(EXPECTEDV);

        float fdrBL = pixel.getFloatAttribute(FDRBL);
        float fdrDonut = pixel.getFloatAttribute(FDRDONUT);
        float fdrH = pixel.getFloatAttribute(FDRH);
        float fdrV = pixel.getFloatAttribute(FDRV);

        return observed > t2 * expectedBL
                && observed > t2 * expectedDonut
                && observed > t1 * expectedH
                && observed > t1 * expectedV
                && (observed > t3 * expectedBL || observed > t3 * expectedDonut)
                && (numCollapsed > 1 || (fdrBL + fdrDonut + fdrH + fdrV) <= f);
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

    private static int hypotneuse(int x, int y) {
        return (int) Math.sqrt(x * x + y * y);
    }

    public static Feature2DList mergeAllResolutions(List<Feature2DList> hiccupsLooplists) {
        if (hiccupsLooplists.size() == 1)
            return hiccupsLooplists.get(0);
        return null;
    }
}
