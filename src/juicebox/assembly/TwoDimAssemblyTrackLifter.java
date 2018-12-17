/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
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

import juicebox.HiCGlobals;
import juicebox.data.MatrixZoomData;
import juicebox.track.feature.AnnotationLayerHandler;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import juicebox.windowui.HiCZoom;

import java.util.List;

/**
 * Created by olga on 12/15/18.
 */
public class TwoDimAssemblyTrackLifter {

    public static void liftTwoDimAssemblyTrack(
            AnnotationLayerHandler handler,
            MatrixZoomData zd,
            int binX1,
            int binX2,
            int binY1,
            int binY2
    ) {

        //could have done this smarter by splitting in two rectangles...
        binX1 = Math.min(binX1, binY1);
        binY1 = binX1;
        binX2 = Math.max(binX2, binY2);
        binY2 = binX2;

        HiCZoom zoom = zd.getZoom();
        // get aggregate scaffold handler
        AssemblyScaffoldHandler
                aFragHandler =
                AssemblyHeatmapHandler.getSuperAdapter().getAssemblyStateTracker().getAssemblyHandler();

        final int binSize = zoom.getBinSize();
        long actualBinSize = (long) binSize;
        if (zd.getChr1().getIndex() == 0) {
            actualBinSize = 1000 * actualBinSize;
        }

        List<Scaffold> xAxisAggregateScaffolds = aFragHandler.getIntersectingAggregateFeatures(
                (long) (actualBinSize * binX1 * HiCGlobals.hicMapScale),
                (long) (actualBinSize * binX2 * HiCGlobals.hicMapScale));

        List<Scaffold> yAxisAggregateScaffolds = aFragHandler.getIntersectingAggregateFeatures(
                (long) (actualBinSize * binY1 * HiCGlobals.hicMapScale),
                (long) (actualBinSize * binY2 * HiCGlobals.hicMapScale));

        Feature2DList newFeature2DList = new Feature2DList();

        int x1pos, x2pos, y1pos, y2pos;
        for (Scaffold xScaffold : xAxisAggregateScaffolds) {

            x1pos = (int) (xScaffold.getOriginalStart() / HiCGlobals.hicMapScale);
            x2pos = (int) (xScaffold.getOriginalEnd() / HiCGlobals.hicMapScale);

            //Need to reconsider this due to lower left plotting

//            // have to case long because of thumbnail, maybe fix thumbnail instead
//            if (xScaffold.getCurrentStart() < actualBinSize * binX1 * HiCGlobals.hicMapScale) {
//                if (!xScaffold.getInvertedVsInitial()) {
//                    x1pos = (int) ((xScaffold.getOriginalStart() + actualBinSize * binX1 * HiCGlobals.hicMapScale - xScaffold.getCurrentStart()) / HiCGlobals.hicMapScale);
//                } else {
//                    x2pos = (int) ((xScaffold.getOriginalStart() - actualBinSize * binX1 * HiCGlobals.hicMapScale + xScaffold.getCurrentEnd()) / HiCGlobals.hicMapScale);
//                }
//            }
//
//            if (xScaffold.getCurrentEnd() > actualBinSize * binX2 * HiCGlobals.hicMapScale) {
//                if (!xScaffold.getInvertedVsInitial()) {
//                    x2pos =
//                            (int) ((xScaffold.getOriginalStart() + actualBinSize * binX2 * HiCGlobals.hicMapScale - xScaffold.getCurrentStart()) / HiCGlobals.hicMapScale);
//                } else {
//                    x1pos =
//                            (int) ((xScaffold.getOriginalStart() - actualBinSize * binX2 * HiCGlobals.hicMapScale + xScaffold.getCurrentEnd()) / HiCGlobals.hicMapScale);
//                }
//            }

            for (Scaffold yScaffold : yAxisAggregateScaffolds) {

                y1pos = (int) (yScaffold.getOriginalStart() / HiCGlobals.hicMapScale);
                y2pos = (int) (yScaffold.getOriginalEnd() / HiCGlobals.hicMapScale);

                //Need to reconsider this due to lower left plotting

//                // have to case long because of thumbnail, maybe fix thumbnail instead
//                if (yScaffold.getCurrentStart() < actualBinSize * binY1 * HiCGlobals.hicMapScale) {
//                    if (!yScaffold.getInvertedVsInitial()) {
//                        y1pos = (int) ((yScaffold.getOriginalStart() + actualBinSize * binY1 * HiCGlobals.hicMapScale - yScaffold.getCurrentStart()) / HiCGlobals.hicMapScale);
//                    } else {
//                        y2pos = (int) ((yScaffold.getOriginalStart() - actualBinSize * binY1 * HiCGlobals.hicMapScale + yScaffold.getCurrentEnd()) / HiCGlobals.hicMapScale);
//                    }
//                }
//
//                if (yScaffold.getCurrentEnd() > actualBinSize * binY2 * HiCGlobals.hicMapScale) {
//                    if (!yScaffold.getInvertedVsInitial()) {
//                        y2pos =
//                                (int) ((yScaffold.getOriginalStart() + actualBinSize * binY2 * HiCGlobals.hicMapScale - yScaffold.getCurrentStart()) / HiCGlobals.hicMapScale);
//                    } else {
//                        y1pos =
//                                (int) ((yScaffold.getOriginalStart() - actualBinSize * binY2 * HiCGlobals.hicMapScale + yScaffold.getCurrentEnd()) / HiCGlobals.hicMapScale);
//                    }
//                }


                if (y1pos < x1pos) {
                    continue;
                }

                //get pairs of features intersecting aggregate scaffold pair
                net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(x1pos, y1pos, x2pos, y2pos);

                List<Feature2D> intersectingFeatures = handler.getFeatureHandler().getOriginalContainedFeatures(1, 1, currentWindow);
                //List<Feature2D> intersectingFeatures = handler.getFeatureHandler().getIntersectingFeatures(1,1,currentWindow,false);
                //List<Feature2D> intersectingFeatures =handler.getAnnotationLayer().getIntersectingFeatures(1,1,currentWindow);


                for (Feature2D feature2D : intersectingFeatures) {

                    int newStart1;
                    int newEnd1;
                    int newStart2;
                    int newEnd2;

                    if (!xScaffold.getInvertedVsInitial()) {
                        newStart1 = (int) ((xScaffold.getCurrentStart() + HiCGlobals.hicMapScale * feature2D.getStart1() - xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale);
                    } else {
                        newStart1 = (int) ((xScaffold.getCurrentEnd() - HiCGlobals.hicMapScale * feature2D.getEnd1() + xScaffold.getOriginalStart()) / HiCGlobals.hicMapScale);
                    }

                    newEnd1 = newStart1 + feature2D.getEnd1() - feature2D.getStart1();

                    if (!yScaffold.getInvertedVsInitial()) {
                        newStart2 = (int) ((yScaffold.getCurrentStart() + HiCGlobals.hicMapScale * feature2D.getStart2() - yScaffold.getOriginalStart()) / HiCGlobals.hicMapScale);
                    } else {
                        newStart2 = (int) ((yScaffold.getCurrentEnd() - HiCGlobals.hicMapScale * feature2D.getEnd2() + yScaffold.getOriginalStart()) / HiCGlobals.hicMapScale);
                    }

                    newEnd2 = newStart2 + feature2D.getEnd2() - feature2D.getStart2();

                    Feature2D newFeature2D = feature2D.deepCopy();

                    newFeature2D.setStart1(newStart1);
                    newFeature2D.setEnd1(newEnd1);
                    newFeature2D.setStart2(newStart2);
                    newFeature2D.setEnd2(newEnd2);
                    newFeature2DList.add(1, 1, newFeature2D);
                }
            }
        }
        handler.getFeatureHandler().setLoopList(newFeature2DList);

        return;
    }
}
