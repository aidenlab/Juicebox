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

import juicebox.tools.utils.common.MatrixTools;
import org.apache.commons.math.linear.RealMatrix;
import org.tc33.jheatchart.HeatChart;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;

/**
 * Helper class to wrap heat map plotting and handle apa plots
 * The static plot method should be called all the necessary inputs.
 */
class APAPlotter {

    /**
     * apa heat map plots range between red (max value) and white (0)
     */
    private static final Color[] gradientColors = {Color.RED, Color.WHITE};
    private static final float[] gradientFractions = {0.0f, 1.0f};

    /**
     * used for comparisons
     */
    private static final double epsilon = 1e-6;

    /**
     * heat map dimension defaults
     */
    private static int fullHeight = 500;
    private static int heatmapWidth = 500;
    private static int colorScaleWidth = 40;
    private static int colorScaleHorizontalMargin = 10;
    private static int colorScaleVerticalMargin = 100;
    private static int extraWidthBuffer = 20;
    private static int fullWidth = heatmapWidth + colorScaleWidth + extraWidthBuffer;
    private static int numDivisions = 6;

    /**
     * Method for plotting apa data
     *
     * @param data       for heat map
     * @param axesRange  initial values and increments to annotate axes [x0, dx, y0, dy]
     * @param outputFile where image will saved
     */
    public static void plot(RealMatrix data,
                            int[] axesRange,
                            File outputFile,
                            String title, int currentRegionWidth,
                            boolean userDefinedColorScale, double colorMin, double colorMax) {

        // As noted in the Cell supplement:
        // "The color scale in all APA plots is set as follows.
        // The minimum of the color range is 0. The maximum is 5 x UR, where
        // UR is the mean value of the bins in the upper-right corner of the matrix.
        // The upper-right corner of the 10 kb resolution APA
        // plots is a 6 x 6 window (or 3 x 3 for 5 kb resolution APA plots)."
        // TODO

        Color lowColor;
        Color highColor;
        if (userDefinedColorScale) {
            double dataMin = MatrixTools.calculateMin(data);
            double dataMax = MatrixTools.calculateMax(data);
            if (dataMax > colorMax) {
                dataMax = colorMax;
            }
            if (dataMin < colorMin) {
                dataMin = colorMin;
            }
            MatrixTools.thresholdValuesDouble(data, colorMin, colorMax);
            int lowColorGB = (int) (255 - dataMin / colorMax * 255);
            int highColorGB = (int) (255 - dataMax / colorMax * 255);
            lowColor = new Color(255, lowColorGB, lowColorGB);
            highColor = new Color(255, highColorGB, highColorGB);
        } else {
            lowColor = Color.white;
            highColor = Color.red;
        }

        APARegionStatistics apaStats = new APARegionStatistics(data, currentRegionWidth);
        DecimalFormat df = new DecimalFormat("0.000");
        title += ", P2LL = " + df.format(apaStats.getPeak2LL());

        // initialize heat map
        HeatChart map = new HeatChart(data.getData());
        map.setXValues(axesRange[0], axesRange[1]);
        map.setYValues(axesRange[2], axesRange[3]);
        map.setTitle(title);

        map.setLowValueColour(lowColor);
        map.setHighValueColour(highColor);

        try {
            // calculate dimensions for plot wrapper
            initializeSizes(map);

            // create blank white image
            BufferedImage apaImage = new BufferedImage(fullWidth, fullHeight, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g2 = apaImage.createGraphics();
            g2.setBackground(Color.WHITE);
            g2.fillRect(0, 0, fullWidth, fullHeight);


            // plot in heat map, color bar, etc
            g2.drawImage(map.getChartImage(), 0, 0, heatmapWidth, fullHeight, null);
            drawHeatMapBorder(g2, map);
            plotColorScaleBar(g2);

            if (userDefinedColorScale) {
                plotSpecialColorScaleValues(g2, map, colorMin, colorMax);
            } else {
                plotColorScaleValues(g2, map);
            }

            // top left, top right, bottom left, bottom right values (from apa)

            drawCornerRegions(g2, map, new Dimension(currentRegionWidth, currentRegionWidth),
                    apaStats.getRegionCornerValues());

            // save data
            ImageIO.write(apaImage, "png", outputFile);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Initialize dimensions used for plotting apa data
     *
     * @param heatMap object
     */
    private static void initializeSizes(HeatChart heatMap) {

        Dimension mapDimensions = getImageDimensions(heatMap.getChartImage());
        //fullHeight = (int) (mapDimensions.height*((double)heatmapWidth)/mapDimensions.width);

        fullHeight = mapDimensions.height;
        heatmapWidth = mapDimensions.width;
        colorScaleWidth = 40;
        colorScaleHorizontalMargin = 10;
        colorScaleVerticalMargin = fullHeight / 5;
        extraWidthBuffer = 30;
        fullWidth = heatmapWidth + colorScaleWidth + extraWidthBuffer;
        numDivisions = calculateIdealNumDivisions(heatMap.getPermissiveIntRange());
    }

    /**
     * Calculate raw dimensions of image
     *
     * @param image
     * @return dimension (x,y) size of image in pixels
     */
    private static Dimension getImageDimensions(Image image) {
        ImageIcon icon = new ImageIcon(image);
        return new Dimension(icon.getIconWidth(), icon.getIconHeight());
    }

    /**
     * Calculate optimal number of color map divisions based on range
     *
     * @param range of heat map data (max - min)
     * @return optimal number of color map divisions
     */
    private static int calculateIdealNumDivisions(int range) {
        // 33 ~ minimum number of pixels per division
        for (int n = fullHeight / 33; n > 3; n--) {
            if (range % n == 0)
                return n;
        }
        return 5;
    }

    /**
     * @param g2 graphics2D object
     */
    private static void plotColorScaleBar(Graphics2D g2) {
        // calculate color scale bar dimensions & location
        Point csBarTL = new Point(heatmapWidth + colorScaleHorizontalMargin, colorScaleVerticalMargin);
        Point csBarBL = new Point(heatmapWidth + colorScaleHorizontalMargin, fullHeight - colorScaleVerticalMargin);
        Rectangle csBar = new Rectangle(csBarTL.x, csBarTL.y,
                colorScaleWidth - 2 * colorScaleHorizontalMargin, fullHeight - 2 * colorScaleVerticalMargin);

        // plot the color scale linear gradient
        LinearGradientPaint gradient = new LinearGradientPaint(csBarTL, csBarBL, gradientFractions, gradientColors);
        g2.setPaint(gradient);
        g2.fill(csBar);

        // plot a border around color scale
        g2.setColor(Color.black);
        g2.drawRect(csBar.x, csBar.y, csBar.width, csBar.height);
    }

    /**
     * Plot number value axis for color scale bar.
     *
     * @param g2      graphics2D object
     * @param heatMap object
     */
    private static void plotColorScaleValues(Graphics2D g2, HeatChart heatMap) {
        // size, increment calculations
        double valIncrement = Math.max(heatMap.getDataRange() / ((double) numDivisions), epsilon);
        double depthIncrement = ((double) (fullHeight - 2 * colorScaleVerticalMargin)) / ((double) numDivisions);
        int verticalDepth = fullHeight - colorScaleVerticalMargin;
        int csBarRightEdgeX = fullWidth - colorScaleHorizontalMargin - extraWidthBuffer;

        // formatting
        g2.setFont(heatMap.getAxisValuesFont());
        DecimalFormat df = new DecimalFormat("0.#");

        // draw each tick mark and its value
        for (double i = heatMap.getLowValue();
             i <= heatMap.getHighValue();
             i += valIncrement, verticalDepth -= depthIncrement) {

            if (i > heatMap.getHighValue() - epsilon)
                verticalDepth = colorScaleVerticalMargin;
            g2.drawString(df.format(i), csBarRightEdgeX + 5, verticalDepth); // value
            g2.drawLine(csBarRightEdgeX - 5, verticalDepth, csBarRightEdgeX, verticalDepth); // tick mark
        }
    }

    /**
     * Plot number value axis for color scale bar.
     *
     * @param g2      graphics2D object
     * @param heatMap object
     */
    private static void plotSpecialColorScaleValues(Graphics2D g2, HeatChart heatMap, double minColor, double maxColor) {
        // size, increment calculations
        double valIncrement = Math.max((maxColor - minColor) / ((double) numDivisions), epsilon);
        double depthIncrement = ((double) (fullHeight - 2 * colorScaleVerticalMargin)) / ((double) numDivisions);
        int verticalDepth = fullHeight - colorScaleVerticalMargin;
        int csBarRightEdgeX = fullWidth - colorScaleHorizontalMargin - extraWidthBuffer;

        // formatting
        g2.setFont(heatMap.getAxisValuesFont());
        DecimalFormat df = new DecimalFormat("0.#");


        // draw each tick mark and its value
        for (double i = minColor;
             i <= maxColor;
             i += valIncrement, verticalDepth -= depthIncrement) {

            if (i > maxColor - epsilon)
                verticalDepth = colorScaleVerticalMargin;
            g2.drawString(df.format(i), csBarRightEdgeX + 5, verticalDepth); // value
            g2.drawLine(csBarRightEdgeX - 5, verticalDepth, csBarRightEdgeX, verticalDepth); // tick mark
        }
    }

    /**
     * Draw black border around main heat map
     *
     * @param g2      graphics2D object
     * @param heatMap object
     */
    private static void drawHeatMapBorder(Graphics2D g2, HeatChart heatMap) {
        // calculate corners of heat map rectangle
        Point heatMapTL = heatMap.getHeatMapTL();
        Point heatMapBR = heatMap.getHeatMapBR();

        // plot border around heat map
        g2.setColor(Color.BLACK);
        g2.drawRect(heatMapTL.x, heatMapTL.y, heatMapBR.x - heatMapTL.x, heatMapBR.y - heatMapTL.y);
    }

    /**
     * Draw the corner boxes and their values as calculated by apa overlayed above the existing heat map
     *
     * @param g2                   graphics2D oObject
     * @param map                  heat map object
     * @param regionCellDimensions dimensions for the corner regions to be plotted in units of cells, not pixels
     *                             where each cell corresponds to a data point, usually 20px-by-20px (default)
     * @param regionAPAValues      apa results for each region in order of TL TR BL BR
     */
    private static void drawCornerRegions(Graphics2D g2, HeatChart map, Dimension regionCellDimensions,
                                          double[] regionAPAValues) {
        // retrieve corners of heat map
        Point topLeft = map.getHeatMapTL();
        Point topRight = map.getHeatMapTR();
        Point bottomLeft = map.getHeatMapBL();
        Point bottomRight = map.getHeatMapBR();

        // calculate dimensions of corner regions
        Dimension cellSize = map.getCellSize();
        int cornerWidth = regionCellDimensions.width * cellSize.width,
                cornerHeight = regionCellDimensions.height * cellSize.height;

        // slide to top left corner within each region
        topRight.translate(-cornerWidth, 0);
        bottomLeft.translate(0, -cornerHeight);
        bottomRight.translate(-cornerWidth, -cornerHeight);

        // plot the four region TL TR BL BR and their values
        Point[] points = {topLeft, topRight, bottomLeft, bottomRight};
        g2.setColor(Color.black);
        for (int i = 0; i < 4; i++) {
            // plot rectangle from upper left corner
            Point currPoint = points[i];
            g2.drawRect(currPoint.x, currPoint.y, cornerWidth, cornerHeight);

            // translate to center of rectangle
            currPoint.translate(cornerWidth / 2, cornerHeight / 2);
            drawCenteredDouble(g2, regionAPAValues[i], currPoint);
        }
    }

    /**
     * Plot double centered at a point (rather than from the upper left corner as is the default)
     *
     * @param g2       graphics2D object
     * @param value    to be plotted
     * @param position for value to be centered at
     */
    private static void drawCenteredDouble(Graphics2D g2, Double value, Point position) {
        DecimalFormat df = new DecimalFormat("0.000");
        drawCenteredString(g2, df.format(value), position);
    }

    /**
     * Plot text centered at a point (rather than from the upper left corner as is the default)
     *
     * @param g2       graphics2D object
     * @param text     to be plotted
     * @param position of where text will be centered at
     */
    private static void drawCenteredString(Graphics2D g2, String text, Point position) {
        FontMetrics fm = g2.getFontMetrics();
        int x2 = position.x - fm.stringWidth(text) / 2;
        int y2 = fm.getAscent() + (position.y - (fm.getAscent() + fm.getDescent()) / 2);
        g2.drawString(text, x2, y2);
    }
}
