/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
import juicebox.data.basics.Chromosome;
import juicebox.track.*;
import juicebox.windowui.HiCZoom;
import org.broad.igv.feature.Exon;
import org.broad.igv.feature.IGVFeature;
import org.broad.igv.track.WindowFunction;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by dudcha on 11/17/17.
 */
public class OneDimAssemblyTrackLifter {

    public static HiCDataPoint[] liftDataArray(HiCDataSource dataSource,
                                               HiC hic,
                                               Chromosome chromosome,
                                               int binX1,
                                               int binX2,
                                               HiCGridAxis gridAxis,
                                               double scaleFactor,
                                               WindowFunction windowFunction) {
        HiCZoom zoom = hic.getZoom();
        // get aggregate scaffold handler
        AssemblyScaffoldHandler
                aFragHandler =
                AssemblyHeatmapHandler.getSuperAdapter().getAssemblyStateTracker().getAssemblyHandler();

        final int binSize = zoom.getBinSize();
        long actualBinSize = binSize;
        if (chromosome.getIndex() == 0) {
            actualBinSize = 1000 * actualBinSize;
        }

        List<Scaffold> xAxisAggregateScaffolds = aFragHandler.getIntersectingAggregateFeatures(
                (long) (actualBinSize * binX1 * HiCGlobals.hicMapScale),
                (long) (actualBinSize * binX2 * HiCGlobals.hicMapScale));

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
                    x2pos =
                            (int) ((xScaffold.getOriginalStart() + actualBinSize * binX2 * HiCGlobals.hicMapScale - xScaffold.getCurrentStart()) / HiCGlobals.hicMapScale);
                } else {
                    x1pos =
                            (int) ((xScaffold.getOriginalStart() - actualBinSize * binX2 * HiCGlobals.hicMapScale + xScaffold.getCurrentEnd()) / HiCGlobals.hicMapScale);
                }
            }

            HiCDataPoint[]
                    dataArray =
                    dataSource.getData(chromosome,
                            (int) (x1pos / actualBinSize),
                            (int) (x2pos / actualBinSize),
                            gridAxis,
                            scaleFactor,
                            windowFunction);

            for (HiCDataPoint point : dataArray) {
                // disregard points outside of the bin positions for this aggregate scaffold
                if (point.getBinNumber() < (int) (x1pos / actualBinSize) || point.getBinNumber() > (int) (x2pos / actualBinSize))
                    continue;
    
                long newStart;
                long newEnd;
                long newBin;
    
                if (!xScaffold.getInvertedVsInitial()) {
                    newStart = (long) ((xScaffold.getCurrentStart() + HiCGlobals.hicMapScale * point.getGenomicStart() - xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale);
                    newBin = (long) ((xScaffold.getCurrentStart() + HiCGlobals.hicMapScale * point.getBinNumber() * binSize - xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale / binSize);
                } else {
                    newStart = (long) ((xScaffold.getCurrentEnd() - HiCGlobals.hicMapScale * point.getGenomicEnd() + xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale);
                    newBin = (long) ((xScaffold.getCurrentEnd() - HiCGlobals.hicMapScale * point.getBinNumber() * binSize + xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale / binSize - 1);
                }
    
                newEnd = newStart + point.getGenomicEnd() - point.getGenomicStart();

                if (point instanceof HiCCoverageDataSource.CoverageDataPoint) {

                    HiCCoverageDataSource.CoverageDataPoint covPoint = (HiCCoverageDataSource.CoverageDataPoint) point;
                    modifiedDataPoints.add(new HiCCoverageDataSource.CoverageDataPoint((int) (newStart / binSize),
                            newStart,
                            newEnd,
                            covPoint.value));
                } else if (point instanceof HiCDataAdapter.DataAccumulator) {
                    HiCDataAdapter.DataAccumulator accumPoint = (HiCDataAdapter.DataAccumulator) point;
                    HiCDataAdapter.DataAccumulator
                            newAccumPoint =
                            new HiCDataAdapter.DataAccumulator(newBin, accumPoint.width, newStart, newEnd);
                    newAccumPoint.nPts = accumPoint.nPts;
                    newAccumPoint.weightedSum = accumPoint.weightedSum;
                    newAccumPoint.max = accumPoint.max;
                    modifiedDataPoints.add(newAccumPoint);
//          if(accumPoint.getBinNumber()!=(double)newBin)
//            System.out.println(accumPoint.getBinNumber()+" "+(double)newBin); // why do these not match?
                }

            }
        }
        HiCDataPoint[] points = new HiCDataPoint[modifiedDataPoints.size()];
        for (int i = 0; i < points.length; i++) {
            points[i] = modifiedDataPoints.get(i);
        }
        return points;
    }

    public static List<IGVFeatureCopy> liftIGVFeatures(
            HiC hic, Chromosome chromosome, int binX1, int binX2, HiCGridAxis gridAxis, ArrayList<IGVFeature> featureList, boolean isBed) {
        List<IGVFeatureCopy> newFeatureList = new ArrayList<>();

        // Initialize
        HiCZoom zoom = hic.getZoom();
        final double scaleFactor = hic.getScaleFactor();
        AssemblyScaffoldHandler aFragHandler = AssemblyHeatmapHandler.getSuperAdapter().getAssemblyStateTracker().getAssemblyHandler();

        final int binSize = zoom.getBinSize();
        long actualBinSize = binSize;
        if (chromosome.getIndex() == 0) {
            actualBinSize *= 1000;
        }

        List<Scaffold> xAxisAggregateScaffolds = aFragHandler.getIntersectingAggregateFeatures(
                (long) (actualBinSize * binX1 * HiCGlobals.hicMapScale), (long) (actualBinSize * binX2 * HiCGlobals.hicMapScale));

        // Determine positions of all aggregate scaffolds
        int x1pos, x2pos;
        HashMap<Scaffold, ArrayList<Integer>> scaffoldOriginalPositions = new HashMap<>();
        for (Scaffold xScaffold : xAxisAggregateScaffolds) {
            ArrayList<Integer> originalPositions = new ArrayList<>(2);
            x1pos = (int) (xScaffold.getOriginalStart() / HiCGlobals.hicMapScale);
            x2pos = (int) (xScaffold.getOriginalEnd() / HiCGlobals.hicMapScale);

            // Have to case long because of thumbnail, maybe fix thumbnail instead
            // Following results in "fragmentation" when feature is outside of window which may not be ideal if fragments are labeled in some matter
            // could fix by extending window boundaries to include intersecting feature boundaries
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

            originalPositions.add(x1pos);
            originalPositions.add(x2pos);
            scaffoldOriginalPositions.put(xScaffold, originalPositions);
        }

        // Iterate over features and over the aggregate scaffolds the feature spans
        for (IGVFeature feature : featureList) {
            double bin1 = HiCFeatureTrack.getFractionalBin(feature.getStart(), scaleFactor, gridAxis);
            double bin2 = HiCFeatureTrack.getFractionalBin(feature.getEnd(), scaleFactor, gridAxis);

            IGVFeatureCopy featureFraction;
            for (Scaffold xScaffold : xAxisAggregateScaffolds) {
              x1pos = scaffoldOriginalPositions.get(xScaffold).get(0);
              x2pos = scaffoldOriginalPositions.get(xScaffold).get(1);

              if (bin2 < (int) (x1pos / actualBinSize) || bin1 > (int) (x2pos / actualBinSize)) {
                  continue;
              }

              featureFraction = new IGVFeatureCopy(feature);

              if (feature.getStart()<x1pos){featureFraction.setStart(x1pos);}
              if (feature.getEnd()>x2pos){featureFraction.setEnd(x2pos);}

              int newStart, newEnd;

              if (!xScaffold.getInvertedVsInitial()) {
                  newStart = (int) ((xScaffold.getCurrentStart() + (HiCGlobals.hicMapScale * featureFraction.getStart()) - xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale);
              } else {
                  newStart = (int) ((xScaffold.getCurrentEnd() - (HiCGlobals.hicMapScale * featureFraction.getEnd()) + xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale);
              }

              newEnd = newStart + featureFraction.getEnd() - featureFraction.getStart();

              featureFraction.setStart(newStart);
              featureFraction.setEnd(newEnd);
              featureFraction.updateStrand(feature.getStrand(), xScaffold.getInvertedVsInitial(), isBed);

              // Update exons
              if (feature.getExons() != null) {
                List<Exon> newExons = new ArrayList<>();

                for (Exon exon : feature.getExons()) {

                  double exonBin1 = HiCFeatureTrack.getFractionalBin(exon.getStart(), scaleFactor, gridAxis);
                  double exonBin2 = HiCFeatureTrack.getFractionalBin(exon.getEnd(), scaleFactor, gridAxis);
                  if (exonBin2 < (int) (x1pos / actualBinSize) || exonBin1 > (x2pos / actualBinSize)) {
                    continue;
                  }

                  Exon newExon = new Exon(featureFraction.getChr(), exon.getStart(), exon.getEnd(), featureFraction.getStrand());

                  if (exon.getStart()<x1pos){newExon.setStart(x1pos);}
                  if (exon.getEnd()>x2pos){newExon.setEnd(x2pos);}

                  int newExonStart, newExonEnd;

                  if (!xScaffold.getInvertedVsInitial()) {
                    newExonStart = (int) ((xScaffold.getCurrentStart() + (HiCGlobals.hicMapScale * newExon.getStart()) - xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale);
                  } else {
                    newExonStart = (int) ((xScaffold.getCurrentEnd() - (HiCGlobals.hicMapScale * newExon.getEnd()) + xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale);
                  }

                  newExonEnd = newExonStart + newExon.getEnd() - newExon.getStart();
                  newExon.setStart(newExonStart);
                  newExon.setEnd(newExonEnd);

                  newExons.add(newExon);
                }

                featureFraction.updateExons(newExons);
              }

              newFeatureList.add(featureFraction);
            }
        }
        return newFeatureList;
    }
}