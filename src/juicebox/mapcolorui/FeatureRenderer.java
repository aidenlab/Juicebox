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

import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.track.HiCGridAxis;
import juicebox.track.feature.Feature2D;

import java.awt.*;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 6/4/15.
 */
class FeatureRenderer {

    // TODO make these variables accessible as user options
    // the other day, Erez mentioned his preferred was everything in lower left
    // can change for future.
    private static final boolean onlyPlotUpperRight = false;
    private static final boolean onlyPlotLowerLeft = false;
    private static final boolean allowUpperRightLoops = true;

    public static void render(Graphics2D g2, List<Feature2D> loops, MatrixZoomData zd,
                              double binOriginX, double binOriginY, double scaleFactor,
                              Feature2D highlightedFeature, boolean showFeatureHighlight,
                              int maxWidth, int maxHeight) {

        // Note: we're assuming feature.chr1 == zd.chr1, and that chr1 is on x-axis
        HiCGridAxis xAxis = zd.getXGridAxis();
        HiCGridAxis yAxis = zd.getYGridAxis();
        boolean sameChr = zd.getChr1Idx() == zd.getChr2Idx();


        // plot circle center


        if (loops != null) {
            for (Feature2D feature : loops) {

                g2.setColor(feature.getColor());

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
                    g2.drawLine(x, y, x, y + h);
                    g2.drawLine(x, y + h, x + w, y + h);
                } else if (onlyPlotUpperRight) {
                    g2.drawLine(x, y, x + w, y);
                    g2.drawLine(x + w, y, x + w, y + h);
                } else {
                    //g2.setColor(Color.yellow);
                    g2.drawRect(x, y, w, h);
                }
                //System.out.println(binStart1 + "-" + binEnd1);
                if (w > 5) {
                    // Thick line if there is room. TODO double check +/- 1
                    if (onlyPlotLowerLeft) {
                        g2.drawLine(x + 1, y + 1, x + 1, y + h + 1);
                        g2.drawLine(x + 1, y + h + 1, x + w + 1, y + h + 1);
                    } else if (onlyPlotUpperRight) {
                        g2.drawLine(x + 1, y + 1, x + w + 1, y + 1);
                        g2.drawLine(x + w + 1, y + 1, x + w + 1, y + h - 1);
                    } else {
                        g2.drawRect(x + 1, y + 1, w - 2, h - 2);
                    }
                } else {
                    g2.drawRect(x - 1, y - 1, w + 2, h + 2);
                }
            }
        }

        if (highlightedFeature != null && showFeatureHighlight) {
            Feature2D feature = highlightedFeature;
            g2.setColor(feature.getColor());

            int binStart1 = xAxis.getBinNumberForGenomicPosition(feature.getStart1());
            int binEnd1 = xAxis.getBinNumberForGenomicPosition(feature.getEnd1());
            int binStart2 = yAxis.getBinNumberForGenomicPosition(feature.getStart2());
            int binEnd2 = yAxis.getBinNumberForGenomicPosition(feature.getEnd2());

            g2.setColor(Color.BLACK);
            if (HiCFileTools.equivalentChromosome(feature.getChr1(), zd.getChr1())) {
                int x = (int) ((binStart1 - binOriginX) * scaleFactor);
                int h = (int) Math.max(1, scaleFactor * (binEnd1 - binStart1));

                g2.drawLine(x, 0, x, maxHeight);
                g2.drawLine(x + h, 0, x + h, maxHeight);
            }
            if (HiCFileTools.equivalentChromosome(feature.getChr2(), zd.getChr2())) {
                int y = (int) ((binStart2 - binOriginY) * scaleFactor);
                int w = (int) Math.max(1, scaleFactor * (binEnd2 - binStart2));

                g2.drawLine(0, y, maxWidth, y);
                g2.drawLine(0, y + w, maxWidth, y + w);
            }
        }
        g2.dispose();
    }
}
