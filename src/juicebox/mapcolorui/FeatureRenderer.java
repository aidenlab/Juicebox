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
public class FeatureRenderer {

    public static PlottingOption enablePlottingOption = PlottingOption.ONLY_LOWER_LEFT;

    public static void render(Graphics2D g2, List<Feature2D> loops, MatrixZoomData zd,
                              double binOriginX, double binOriginY, double scaleFactor,
                              Feature2D highlightedFeature, boolean showFeatureHighlight,
                              int maxWidth, int maxHeight) {

        // Note: we're assuming feature.chr1 == zd.chr1, and that chr1 is on x-axis
        HiCGridAxis xAxis = zd.getXGridAxis();
        HiCGridAxis yAxis = zd.getYGridAxis();

        if (loops != null) {
            for (Feature2D feature : loops) {

                if (!feature.isOnDiagonal()) {
                    if (feature.isInLowerLeft()) {
                        if (enablePlottingOption == PlottingOption.ONLY_UPPER_RIGHT) {
                            continue;
                        }
                    } else if (feature.isInUpperRight()) {
                        if (enablePlottingOption == PlottingOption.ONLY_LOWER_LEFT) {
                            continue;
                        }
                    }
                }

                g2.setColor(feature.getColor());

                Rectangle rect = Feature2DHandler.rectangleFromFeature(xAxis, yAxis, feature, binOriginX, binOriginY, scaleFactor);
                int x = (int) rect.getX();
                int y = (int) rect.getY();
                int w = (int) rect.getWidth();
                int h = (int) rect.getHeight();

                if (feature.isOnDiagonal()) {
                    // contact domains
                    switch (enablePlottingOption) {
                        case ONLY_LOWER_LEFT:
                            g2.drawLine(x, y, x, y + h);
                            g2.drawLine(x, y + h, x + w, y + h);
                            if (w > 5) {
                                g2.drawLine(x + 1, y + 1, x + 1, y + h + 1);
                                g2.drawLine(x + 1, y + h + 1, x + w + 1, y + h + 1);
                            }
                            break;
                        case ONLY_UPPER_RIGHT:
                            g2.drawLine(x, y, x + w, y);
                            g2.drawLine(x + w, y, x + w, y + h);
                            if (w > 5) {
                                g2.drawLine(x + 1, y + 1, x + w + 1, y + 1);
                                g2.drawLine(x + w + 1, y + 1, x + w + 1, y + h - 1);
                            }
                            break;
                        case EVERYTHING:
                            g2.drawRect(x, y, w, h);
                            if (w > 5) {
                                g2.drawRect(x + 1, y + 1, w - 2, h - 2);
                            }
                            break;
                    }
                } else {
                    // loops, other features, etc
                    g2.drawRect(x, y, w, h);
                    if (w > 5) {
                        g2.drawRect(x + 1, y + 1, w - 2, h - 2);
                    } else {
                        g2.drawRect(x - 1, y - 1, w + 2, h + 2);
                    }
                }
            }
        }

        if (highlightedFeature != null && showFeatureHighlight) {
            g2.setColor(highlightedFeature.getColor());

            int binStart1 = xAxis.getBinNumberForGenomicPosition(highlightedFeature.getStart1());
            int binEnd1 = xAxis.getBinNumberForGenomicPosition(highlightedFeature.getEnd1());
            int binStart2 = yAxis.getBinNumberForGenomicPosition(highlightedFeature.getStart2());
            int binEnd2 = yAxis.getBinNumberForGenomicPosition(highlightedFeature.getEnd2());

            g2.setColor(Color.BLACK);
            if (HiCFileTools.equivalentChromosome(highlightedFeature.getChr1(), zd.getChr1())) {
                int x = (int) ((binStart1 - binOriginX) * scaleFactor);
                int h = (int) Math.max(1, scaleFactor * (binEnd1 - binStart1));

                g2.drawLine(x, 0, x, maxHeight);
                g2.drawLine(x + h, 0, x + h, maxHeight);
            }
            if (HiCFileTools.equivalentChromosome(highlightedFeature.getChr2(), zd.getChr2())) {
                int y = (int) ((binStart2 - binOriginY) * scaleFactor);
                int w = (int) Math.max(1, scaleFactor * (binEnd2 - binStart2));

                g2.drawLine(0, y, maxWidth, y);
                g2.drawLine(0, y + w, maxWidth, y + w);
            }
        }
        g2.dispose();
    }

    public enum PlottingOption {ONLY_LOWER_LEFT, ONLY_UPPER_RIGHT, EVERYTHING}
}
