/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.mapcolorui;

import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.CustomMatrixZoomData;
import juicebox.data.MatrixZoomData;

import java.awt.*;
import java.util.List;

public class BoundingBoxRenderer {

    private final HeatmapPanel parent;
    private long[] chromosomeBoundaries;

    public BoundingBoxRenderer(HeatmapPanel heatmapPanel) {
        parent = heatmapPanel;
    }

    public void drawAllByAllGrid(Graphics2D g, MatrixZoomData zd, boolean showGridLines,
                                 double binOriginX, double binOriginY, double scaleFactor) {
        if (HiCGlobals.isDarkulaModeEnabled) {
            g.setColor(Color.LIGHT_GRAY);
        } else {
            g.setColor(Color.DARK_GRAY);
        }

        long maxDimension = chromosomeBoundaries[chromosomeBoundaries.length - 1];
        int maxHeight = getGridLineHeightLimit(zd, maxDimension, scaleFactor);
        int maxWidth = getGridLineWidthLimit(zd, maxDimension, scaleFactor);

        g.drawLine(0, 0, 0, maxHeight);
        g.drawLine(0, 0, maxWidth, 0);
        g.drawLine(maxWidth, 0, maxWidth, maxHeight);
        g.drawLine(0, maxHeight, maxWidth, maxHeight);

        // Draw grid lines only if option is selected
        if (showGridLines) {
            for (long bound : chromosomeBoundaries) {
                // vertical lines
                int xBin = zd.getXGridAxis().getBinNumberForGenomicPosition(bound);
                int x = (int) ((xBin - binOriginX) * scaleFactor);
                g.drawLine(x, 0, x, maxHeight);

                // horizontal lines
                int yBin = zd.getYGridAxis().getBinNumberForGenomicPosition(bound);
                int y = (int) ((yBin - binOriginY) * scaleFactor);
                g.drawLine(0, y, maxWidth, y);
            }
        }

        //Cover gray background for the empty parts of the matrix:
        if (HiCGlobals.isDarkulaModeEnabled) {
            g.setColor(Color.darkGray);
        } else {
            g.setColor(Color.white);
        }

        int pHeight = parent.getHeight();
        int pWidth = parent.getWidth();
        g.fillRect(maxHeight, 0, pHeight, pWidth);
        g.fillRect(0, maxWidth, pHeight, pWidth);
        g.fillRect(maxHeight, maxWidth, pHeight, pWidth);
    }

    private int getGridLineWidthLimit(MatrixZoomData zd, long maxPosition, double scaleFactor) {
        if (parent.getWidth() < 50 || scaleFactor < 1e-10) {
            return 0;
        }
        int xBin = zd.getXGridAxis().getBinNumberForGenomicPosition(maxPosition);
        return (int) (xBin * scaleFactor);
    }

    private int getGridLineHeightLimit(MatrixZoomData zd, long maxPosition, double scaleFactor) {
        if (parent.getHeight() < 50 || scaleFactor < 1e-10) {
            return 0;
        }
        int yBin = zd.getYGridAxis().getBinNumberForGenomicPosition(maxPosition);
        return (int) (yBin * scaleFactor);
    }

    public void setChromosomeBoundaries(long[] chromosomeBoundaries) {
        this.chromosomeBoundaries = chromosomeBoundaries;
    }

    public void drawRegularGrid(Graphics2D g, MatrixZoomData zd, boolean showGridLines, ChromosomeHandler handler,
                                double binOriginX, double binOriginY, double scaleFactor) {
        if (showGridLines) {
            if (HiCGlobals.isDarkulaModeEnabled) {
                g.setColor(Color.LIGHT_GRAY);
            } else {
                g.setColor(Color.DARK_GRAY);
            }
            if (handler != null && zd != null) {
                if (handler.isCustomChromosome(zd.getChr1())) {
                    if (zd instanceof CustomMatrixZoomData) {
                        java.util.List<Long> xBins = ((CustomMatrixZoomData) zd).getBoundariesOfCustomChromosomeX();
                        //int maxSize = xBins.get(xBins.size() - 1);
                        int maxSize = (int) ((zd.getYGridAxis().getBinCount() - binOriginY) * scaleFactor);
                        for (long xBin : xBins) {
                            int x = (int) ((xBin - binOriginX) * scaleFactor);
                            g.drawLine(x, 0, x, maxSize);
                        }
                    }
                }
                if (handler.isCustomChromosome(zd.getChr2())) {
                    if (zd instanceof CustomMatrixZoomData) {
                        List<Long> yBins = ((CustomMatrixZoomData) zd).getBoundariesOfCustomChromosomeY();
                        //int maxSize = yBins.get(yBins.size() - 1);
                        int maxSize = (int) ((zd.getXGridAxis().getBinCount() - binOriginX) * scaleFactor);
                        for (long yBin : yBins) {
                            int y = (int) ((yBin - binOriginY) * scaleFactor);
                            g.drawLine(0, y, maxSize, y);
                        }
                    }
                }
            }
        }
    }
}
