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

package juicebox.mapcolorui;

import juicebox.data.MatrixZoomData;
import juicebox.tools.utils.Common.HiCFileTools;
import juicebox.track.Feature.Feature2D;
import juicebox.track.HiCGridAxis;
import org.broad.igv.util.Pair;

import java.awt.*;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 6/4/15.
 */
public class FeatureRenderer {

    // TODO make these variables accessible as user options
    // the other day, Erez mentioned his preferred was everything in lower left
    // can change for future.
    private static final boolean onlyPlotUpperRight = false;
    private static final boolean onlyPlotLowerLeft = false;
    private static final boolean allowUpperRightLoops = true;

    public static void render(Graphics2D loopGraphics, List<Feature2D> loops, MatrixZoomData zd,
                              double binOriginX, double binOriginY, double scaleFactor,
                              List<Pair<Rectangle, Feature2D>> drawnLoopFeatures,
                              Pair<Rectangle, Feature2D> highlightedFeature, boolean showFeatureHighlight,
                              int maxWidth, int maxHeight) {

        // Note: we're assuming feature.chr1 == zd.chr1, and that chr1 is on x-axis
        HiCGridAxis xAxis = zd.getXGridAxis();
        HiCGridAxis yAxis = zd.getYGridAxis();
        boolean sameChr = zd.getChr1Idx() == zd.getChr2Idx();

        if(loops != null) {
            for (Feature2D feature : loops) {

                loopGraphics.setColor(feature.getColor());

                // TODO this seems wrong. why is w added to y and not to x? bug/error?
                int binStart1 = xAxis.getBinNumberForGenomicPosition(feature.getStart1());
                int binEnd1 = xAxis.getBinNumberForGenomicPosition(feature.getEnd1());
                int binStart2 = yAxis.getBinNumberForGenomicPosition(feature.getStart2());
                int binEnd2 = yAxis.getBinNumberForGenomicPosition(feature.getEnd2());

                int x = (int) ((binStart1 - binOriginX) * scaleFactor);
                int y = (int) ((binStart2 - binOriginY) * scaleFactor);
                int w = (int) Math.max(1, scaleFactor * (binEnd1 - binStart1));
                int h = (int) Math.max(1, scaleFactor * (binEnd2 - binStart2));


                if (onlyPlotLowerLeft) {
                    //loopGraphics.setColor(Color.green);
                    loopGraphics.drawLine(x, y, x, y + h);
                    loopGraphics.drawLine(x, y + h, x + w, y + h);
                } else if (onlyPlotUpperRight) {
                    //loopGraphics.setColor(Color.blue);
                    loopGraphics.drawLine(x, y, x + w, y);
                    loopGraphics.drawLine(x + w, y, x + w, y + h);
                } else {
                    //loopGraphics.setColor(Color.yellow);
                    loopGraphics.drawRect(x, y, w, h);
                }
                //System.out.println(binStart1 + "-" + binEnd1);
                if (w > 5) {
                    // Thick line if there is room. TODO double check +/- 1
                    if (onlyPlotLowerLeft) {
                        loopGraphics.drawLine(x + 1, y + 1, x + 1, y + h + 1);
                        loopGraphics.drawLine(x + 1, y + h + 1, x + w + 1, y + h + 1);
                    } else if (onlyPlotUpperRight) {
                        loopGraphics.drawLine(x + 1, y + 1, x + w + 1, y + 1);
                        loopGraphics.drawLine(x + w + 1, y + 1, x + w + 1, y + h - 1);
                    } else {
                        loopGraphics.drawRect(x + 1, y + 1, w - 2, h - 2);
                    }
                } else {
                    loopGraphics.drawRect(x - 1, y - 1, w + 2, h + 2);
                }


                drawnLoopFeatures.add(new Pair<Rectangle, Feature2D>(new Rectangle(x - 1, y - 1, w + 2, h + 2), feature));

                feature.getClass();

                // TODO is there a reason for checking bounds and not just filtering by loop vs domain
                // TODO also are any features being missed y discard upper right
                // which is contained in
                if (allowUpperRightLoops && sameChr && !(binStart1 == binStart2 && binEnd1 == binEnd2)) {
                    x = (int) ((binStart2 - binOriginX) * scaleFactor);
                    y = (int) ((binStart1 - binOriginY) * scaleFactor);
                    w = (int) Math.max(1, scaleFactor * (binEnd2 - binStart2));
                    h = (int) Math.max(1, scaleFactor * (binEnd1 - binStart1));
                    //loopGraphics.setColor(Color.ORANGE);
                    loopGraphics.drawRect(x, y, w, h);
                    if (w > 5) {
                        //loopGraphics.setColor(Color.magenta);
                        loopGraphics.drawRect(x + 1, y + 1, w - 2, h - 2);
                    } else {
                        //loopGraphics.setColor(Color.CYAN);
                        loopGraphics.drawRect(x - 1, y - 1, w + 2, h + 2);
                    }
                    drawnLoopFeatures.add(new Pair<Rectangle, Feature2D>(new Rectangle(x - 1, y - 1, w + 2, h + 2), feature));
                }
            }
        }

        if(highlightedFeature != null && showFeatureHighlight){
            Feature2D feature = highlightedFeature.getSecond();
            loopGraphics.setColor(feature.getColor());

            int binStart1 = xAxis.getBinNumberForGenomicPosition(feature.getStart1());
            int binEnd1 = xAxis.getBinNumberForGenomicPosition(feature.getEnd1());
            int binStart2 = yAxis.getBinNumberForGenomicPosition(feature.getStart2());
            int binEnd2 = yAxis.getBinNumberForGenomicPosition(feature.getEnd2());

            loopGraphics.setColor(Color.BLACK);
            if(HiCFileTools.equivalentChromosome(feature.getChr1(),zd.getChr1())){
                int x = (int) ((binStart1 - binOriginX) * scaleFactor);
                int h = (int) Math.max(1, scaleFactor * (binEnd1 - binStart1));

                loopGraphics.drawLine(x, 0, x, maxHeight);
                loopGraphics.drawLine(x+h, 0, x+h, maxHeight);
            }
            if(HiCFileTools.equivalentChromosome(feature.getChr2(),zd.getChr2())){
                int y = (int) ((binStart2 - binOriginY) * scaleFactor);
                int w = (int) Math.max(1, scaleFactor * (binEnd2 - binStart2));

                loopGraphics.drawLine(0, y, maxWidth, y);
                loopGraphics.drawLine(0, y+w, maxWidth, y+w);
            }
        }
        loopGraphics.dispose();
    }
}
