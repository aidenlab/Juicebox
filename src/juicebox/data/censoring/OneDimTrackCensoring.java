/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

package juicebox.data.censoring;

import juicebox.HiC;
import juicebox.data.anchor.MotifAnchor;
import juicebox.track.*;
import juicebox.windowui.HiCZoom;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.track.WindowFunction;
import org.broad.igv.util.Pair;

import java.util.ArrayList;
import java.util.List;

public class OneDimTrackCensoring {


    public static HiCDataPoint[] getFilteredData(HiCDataSource dataSource, HiC hic, Chromosome chromosome,
                                                 int startBin, int endBin, HiCGridAxis gridAxis,
                                                 double scaleFactor, WindowFunction windowFunction) {
        HiCZoom zoom;
        try {
            zoom = hic.getZd().getZoom();
        } catch (Exception e) {
            return null;
        }

        // x window
        int binSize = zoom.getBinSize();
        float gx1 = startBin * binSize;
        float gx2 = endBin * binSize;

        net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(gx1, gx1, gx2, gx2);
        List<Pair<MotifAnchor, MotifAnchor>> axisRegions = hic.getRTreeHandlerIntersectingFeatures(chromosome.getIndex(), currentWindow);

        List<HiCDataPoint[]> dataPointArrays = new ArrayList<>();
        for (Pair<MotifAnchor, MotifAnchor> regionPair : axisRegions) {

            MotifAnchor originalRegion = regionPair.getFirst();
            MotifAnchor translatedRegion = regionPair.getSecond();

            Chromosome orig = hic.getChromosomeHandler().getChromosomeFromIndex(originalRegion.getChr());
            HiCDataPoint[] array = dataSource.getData(orig, originalRegion.getX1() / binSize,
                    originalRegion.getX2() / binSize, gridAxis, scaleFactor, windowFunction);
            HiCDataPoint[] translatedArray = OneDimTrackCensoring.translateDataPointArray(zoom.getBinSize(), array, originalRegion, translatedRegion);
            dataPointArrays.add(translatedArray);
        }

        return OneDimTrackCensoring.mergeDataPoints(dataPointArrays);
    }

    private static HiCDataPoint[] translateDataPointArray(int binSize, HiCDataPoint[] array,
                                                          MotifAnchor originalRegion, MotifAnchor translatedRegion) {
        List<HiCDataPoint> translatedPoints = new ArrayList<>();

        if (array.length > 0 && array[0] instanceof HiCCoverageDataSource.CoverageDataPoint) {
            for (HiCDataPoint pointGen : array) {
                HiCCoverageDataSource.CoverageDataPoint point = (HiCCoverageDataSource.CoverageDataPoint) pointGen;
                if (point.genomicStart >= originalRegion.getX1() && point.genomicEnd <= originalRegion.getX2()) {
                    int newGStart = translatedRegion.getX1() + point.genomicStart - originalRegion.getX1();
                    int newGEnd = translatedRegion.getX1() + point.genomicEnd - originalRegion.getX1();
                    int newBinNum = newGStart / binSize;
                    translatedPoints.add(new HiCCoverageDataSource.CoverageDataPoint(newBinNum, newGStart, newGEnd, point.value));
                }
            }
        } else if (array.length > 0 && array[0] instanceof HiCDataAdapter.DataAccumulator) {
            for (HiCDataPoint pointGen : array) {
                HiCDataAdapter.DataAccumulator point = (HiCDataAdapter.DataAccumulator) pointGen;
                if (point.genomicStart >= originalRegion.getX1() && point.genomicEnd <= originalRegion.getX2()) {
                    int newGStart = translatedRegion.getX1() + point.genomicStart - originalRegion.getX1();
                    int newGEnd = translatedRegion.getX1() + point.genomicEnd - originalRegion.getX1();
                    int newBinNum = newGStart / binSize;
                    HiCDataAdapter.DataAccumulator accum = new HiCDataAdapter.DataAccumulator(newBinNum, point.width,
                            newGStart, newGEnd);
                    accum.nPts = point.nPts;
                    accum.weightedSum = point.weightedSum;
                    accum.max = point.max;
                    translatedPoints.add(accum);
                }
            }
        }

        HiCDataPoint[] points = new HiCDataPoint[translatedPoints.size()];
        for (int i = 0; i < points.length; i++) {
            points[i] = translatedPoints.get(i);
        }

        return points;
    }

    private static HiCDataPoint[] mergeDataPoints(List<HiCDataPoint[]> dataPointArrays) {
        int length = 0;
        for (HiCDataPoint[] array : dataPointArrays) {
            length += array.length;
        }

        HiCDataPoint[] mergedArray = new HiCDataPoint[length];
        int index = 0;
        for (HiCDataPoint[] array : dataPointArrays) {
            System.arraycopy(array, 0, mergedArray, index, array.length);
            index += array.length;
        }
        return mergedArray;
    }
}
