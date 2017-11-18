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

package juicebox.assembly;

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.track.HiCCoverageDataSource;
import juicebox.track.HiCDataPoint;
import juicebox.track.HiCDataSource;
import juicebox.track.HiCGridAxis;
import juicebox.windowui.HiCZoom;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.track.WindowFunction;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by dudcha on 11/17/17.
 */
public class OneDimAssemblyTrackLifter {

    public static HiCDataPoint[] liftDataArrayFromAsm(HiCDataSource dataSource, HiC hic, Chromosome chromosome, int binX1, int binX2, HiCGridAxis gridAxis, double scaleFactor, WindowFunction windowFunction) {
        HiCZoom zoom = hic.getZoom();
        // get aggregate scaffold handler
        AssemblyScaffoldHandler aFragHandler = AssemblyHeatmapHandler.getSuperAdapter().getAssemblyStateTracker().getAssemblyHandler();

        final int binSize = zoom.getBinSize();
        long actualBinSize = (long) binSize;
        if (chromosome.getIndex() == 0) {
            actualBinSize = 1000 * actualBinSize;
        }

        List<Scaffold> xAxisAggregateScaffolds = aFragHandler.getIntersectingAggregateFeatures(
                (long) (actualBinSize * binX1 * HiCGlobals.hicMapScale), (long) (actualBinSize * binX2 * HiCGlobals.hicMapScale));

        List<HiCDataPoint> modifiedDataPoints = new ArrayList<>();

        int x1pos, x2pos;
        for (Scaffold xScaffold : xAxisAggregateScaffolds) {
            x1pos = (int) (xScaffold.getOriginalStart() / HiCGlobals.hicMapScale);
            x2pos = (int) (xScaffold.getOriginalEnd() / HiCGlobals.hicMapScale);

            // have to case long because of thumbnail, maybe fix thumbnail instead

            if (xScaffold.getCurrentStart() < actualBinSize * binX1 * HiCGlobals.hicMapScale) {
                if (!xScaffold.getInvertedVsInitial()) {
                    x1pos = (int) ((xScaffold.getOriginalStart() + actualBinSize * binX1 * HiCGlobals.hicMapScale - xScaffold.getCurrentStart()) / HiCGlobals.hicMapScale);
                } else {
                    x2pos = (int) ((xScaffold.getOriginalStart() - actualBinSize * binX1 * HiCGlobals.hicMapScale + xScaffold.getCurrentEnd()) / HiCGlobals.hicMapScale);
                }
            }

            if (xScaffold.getCurrentEnd() > actualBinSize * binX2 * HiCGlobals.hicMapScale) {
                if (!xScaffold.getInvertedVsInitial()) {
                    x2pos = (int) ((xScaffold.getOriginalStart() + actualBinSize * binX2 * HiCGlobals.hicMapScale - xScaffold.getCurrentStart()) / HiCGlobals.hicMapScale);
                } else {
                    x1pos = (int) ((xScaffold.getOriginalStart() - actualBinSize * binX2 * HiCGlobals.hicMapScale + xScaffold.getCurrentEnd()) / HiCGlobals.hicMapScale);
                }
            }

            HiCDataPoint[] dataArray = dataSource.getData(chromosome, (int) (x1pos / actualBinSize), (int) (x2pos / actualBinSize), gridAxis, scaleFactor, windowFunction);

            for (HiCDataPoint point : dataArray) {
                int newStart;
                int newEnd;
                int newBin;
                if (!xScaffold.getInvertedVsInitial()) {
                    newStart = (int) ((xScaffold.getCurrentStart() + HiCGlobals.hicMapScale * point.getGenomicStart() - xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale);
                    newBin = (int) ((xScaffold.getCurrentStart() + HiCGlobals.hicMapScale * point.getBinNumber() * binSize - xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale / binSize);
                } else {
                    newStart = (int) ((xScaffold.getCurrentEnd() - HiCGlobals.hicMapScale * point.getGenomicEnd() + xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale);
                    newBin = (int) ((xScaffold.getCurrentEnd() - HiCGlobals.hicMapScale * point.getBinNumber() * binSize + xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale / binSize - 1);
                }
                newEnd = newStart + point.getGenomicEnd() - point.getGenomicStart();
//                newBin=newStart/binSize;
                if (point instanceof HiCCoverageDataSource.CoverageDataPoint) {
                    HiCCoverageDataSource.CoverageDataPoint covPoint = (HiCCoverageDataSource.CoverageDataPoint) point;
                    modifiedDataPoints.add(new HiCCoverageDataSource.CoverageDataPoint(newStart / binSize, newStart, newEnd, covPoint.value));
                }
//                else if (point instanceof HiCDataAdapter.DataAccumulator) { // not working quite as supposed to, seems like does not remove prev? confusing...
//                    HiCDataAdapter.DataAccumulator accumPoint = (HiCDataAdapter.DataAccumulator) point;
//                    HiCDataAdapter.DataAccumulator newAccumPoint = new HiCDataAdapter.DataAccumulator((double) newBin, accumPoint.width, newStart, newEnd);
//                    newAccumPoint.nPts = accumPoint.nPts;
//                    newAccumPoint.weightedSum = accumPoint.weightedSum;
//                    newAccumPoint.max = accumPoint.max;
//                    modifiedDataPoints.add(newAccumPoint);
//                    //if(accumPoint.getBinNumber()!=(double)newBin) System.out.println(accumPoint.getBinNumber()+" "+(double)newBin); // why do these not match?
//                }
            }
        }

        HiCDataPoint[] points = new HiCDataPoint[modifiedDataPoints.size()];
        for (int i = 0; i < points.length; i++) {
            points[i] = modifiedDataPoints.get(i);
        }
        return points;
    }
}