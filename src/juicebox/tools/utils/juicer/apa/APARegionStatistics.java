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

package juicebox.tools.utils.juicer.apa;

import juicebox.tools.clt.juicer.APA;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;

/**
 * Created by muhammadsaadshamim on 5/4/15.
 */
public class APARegionStatistics {

    //public static int regionWidth = 6;
    //width of boxes
    //int totWidth = x.getColumnDimension();
    //float widthp5 = width + .5f;
    //float half_width = width/2f;
    //private int mdpt, max;
    //private double centralVal;

    private final double peak2mean;
    private final double peak2UL;
    private final double peak2UR;
    private final double peak2LL;
    private final double peak2LR;
    private final double ZscoreLL;

    public APARegionStatistics(RealMatrix data) {
        int max = data.getColumnDimension();
        int midPoint = max / 2;
        double centralVal = data.getEntry(midPoint, midPoint);

        int regionWidth = APA.regionWidth;

        /** NOTE - indices are inclusive in java, but in python the second index is not inclusive */

        peak2mean = centralVal / ((sum(data.getData()) - centralVal) / (data.getColumnDimension() - 1));

        double avgUL = mean(data.getSubMatrix(0, regionWidth - 1, 0, regionWidth - 1).getData());
        peak2UL = centralVal / avgUL;

        double avgUR = mean(data.getSubMatrix(0, regionWidth - 1, max - regionWidth, max - 1).getData());
        peak2UR = centralVal / avgUR;

        double avgLL = mean(data.getSubMatrix(max - regionWidth, max - 1, 0, regionWidth - 1).getData());
        peak2LL = centralVal / avgLL;

        double avgLR = mean(data.getSubMatrix(max - regionWidth, max - 1, max - regionWidth, max - 1).getData());
        peak2LR = centralVal / avgLR;

        DescriptiveStatistics yStats = statistics(data.getSubMatrix(max - regionWidth, max - 1, 0, regionWidth - 1).getData());
        ZscoreLL = (centralVal - yStats.getMean()) / yStats.getStandardDeviation();
    }

    public static DescriptiveStatistics statistics(double[][] x) {
        DescriptiveStatistics stats = new DescriptiveStatistics();
        for (double[] row : x)
            for (double val : row)
                stats.addValue(val);
        return stats;
    }

    public static double sum(double[][] x) {
        double total = 0;
        for (double[] row : x)
            for (double val : row)
                total += val;
        return total;
    }

    private static double mean(double[][] x) {
        return statistics(x).getMean();
    }

    public double getPeak2mean() {
        return peak2mean;
    }

    public double getPeak2UL() {
        return peak2UL;
    }

    public double getPeak2UR() {
        return peak2UR;
    }

    public double getPeak2LL() {
        return peak2LL;
    }

    public double getPeak2LR() {
        return peak2LR;
    }

    public double getZscoreLL() {
        return ZscoreLL;
    }

    public double[] getRegionCornerValues() {
        return new double[]{peak2UL, peak2UR, peak2LL, peak2LR};
    }
}
