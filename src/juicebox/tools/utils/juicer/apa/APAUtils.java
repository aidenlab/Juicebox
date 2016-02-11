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

package juicebox.tools.utils.juicer.apa;

import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.common.StatPercentile;
import juicebox.track.feature.Feature2D;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Muhammad Shamim on 1/21/15.
 */
public class APAUtils {

    /**
     * @param filename
     * @param matrix
     */
    public static void saveMeasures(String filename, RealMatrix matrix, int currentRegionWidth) {

        Writer writer = null;

        APARegionStatistics apaStats = new APARegionStatistics(matrix, currentRegionWidth);

        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "utf-8"));
            writer.write("P2M" + '\t' + apaStats.getPeak2mean() + '\n');
            writer.write("P2UL" + '\t' + apaStats.getPeak2UL() + '\n');
            writer.write("P2UR" + '\t' + apaStats.getPeak2UR() + '\n');
            writer.write("P2LL" + '\t' + apaStats.getPeak2LL() + '\n');
            writer.write("P2LR" + '\t' + apaStats.getPeak2LR() + '\n');
            writer.write("ZscoreLL" + '\t' + apaStats.getZscoreLL());
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if (writer != null)
                    writer.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static void saveListText(String filename, List<Double> array) {
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "utf-8"));
            for (double val : array) {
                writer.write(val + " ");
            }
            writer.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if (writer != null)
                    writer.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static RealMatrix standardNormalization(RealMatrix matrix) {
        return matrix.copy().scalarMultiply(1. / Math.max(1., MatrixTools.mean(matrix)));
    }


    public static RealMatrix centerNormalization(RealMatrix matrix) {

        int center = matrix.getRowDimension() / 2;
        double centerVal = matrix.getEntry(center, center);

        if (centerVal == 0) {
            centerVal = MatrixTools.minimumPositive(matrix);
            if (centerVal == 0)
                centerVal = 1;
        }

        return matrix.copy().scalarMultiply(1. / centerVal);
    }

    public static double peakEnhancement(RealMatrix matrix) {
        int rows = matrix.getRowDimension();
        int center = rows / 2;
        double centerVal = matrix.getEntry(center, center);
        double remainingSum = APARegionStatistics.sum(matrix.getData()) - centerVal;
        double remainingAverage = remainingSum / (rows * rows - 1);
        return centerVal / remainingAverage;
    }


    /**
     * @param data
     * @return
     */
    public static RealMatrix rankPercentile(RealMatrix data) {
        int n = data.getColumnDimension();
        StatPercentile percentile = new StatPercentile(MatrixTools.flattenedRowMajorOrderMatrix(data));

        RealMatrix matrix = new Array2DRowRealMatrix(n, n);
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                double currValue = data.getEntry(r, c);
                if (currValue == 0) {
                    matrix.setEntry(r, c, 0);
                } else {
                    matrix.setEntry(r, c, percentile.evaluate(currValue));
                }
                //matrix.setEntry(r, c, percentile.evaluate());
            }
        }
        return matrix;
    }

    /**
     * Size filtering of loops
     *
     * @param features
     * @param minPeakDist
     * @param maxPeakDist
     * @return
     */
    public static ArrayList<Feature2D> filterFeaturesBySize(List<Feature2D> features,
                                                            double minPeakDist, double maxPeakDist, int resolution) {
        ArrayList<Feature2D> sizeFilteredFeatures = new ArrayList<Feature2D>();

        for (Feature2D feature : features) {
            double xMidPt = feature.getMidPt1();
            double yMidPt = feature.getMidPt2();
            int dist = (int) Math.round(Math.abs(xMidPt - yMidPt) / resolution);

            if (dist >= minPeakDist) {
                if (dist <= maxPeakDist) {
                    sizeFilteredFeatures.add(feature);
                }
            }
        }
        return new ArrayList<Feature2D>(sizeFilteredFeatures);
    }

    public static RealMatrix extractLocalizedData(MatrixZoomData zd, Feature2D loop,
                                                  int L, int resolution, int window, NormalizationType norm) throws IOException {
        int loopX = loop.getMidPt1() / resolution;
        int loopY = loop.getMidPt2() / resolution;
        int binXStart = loopX - window;
        int binXEnd = loopX + (window + 1);
        int binYStart = loopY - window;
        int binYEnd = loopY + (window + 1);

        return HiCFileTools.extractLocalBoundedRegion(zd, binXStart, binXEnd, binYStart, binYEnd, L, L, norm);
    }
}
