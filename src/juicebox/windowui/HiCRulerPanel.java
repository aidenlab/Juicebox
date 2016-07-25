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

package juicebox.windowui;

import juicebox.Context;
import juicebox.HiC;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.track.HiCGridAxis;
import org.apache.log4j.Logger;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.ui.FontManager;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.io.Serializable;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.List;

/**
 * @author jrobinso
 */
public class HiCRulerPanel extends JPanel implements Serializable {

    private static final long serialVersionUID = 3754386054158787331L;
    private static Logger log = Logger.getLogger(HiCRulerPanel.class);
    private static boolean showOnlyEndPts = false;
    private static boolean showChromosomeFigure = true;
    private final Font tickFont = FontManager.getFont(Font.BOLD, 9);
    private final Font spanFont = FontManager.getFont(Font.BOLD, 12);
    private final HiC hic;
    private Orientation orientation;
    private Context context;

    public HiCRulerPanel(HiC hic) {
        this.hic = hic;
        setBackground(Color.white);
    }

    private static String formatNumber(double position) {

        if (showOnlyEndPts) {
            //Export Version
            NumberFormat df = NumberFormat.getInstance();
            df.setMinimumFractionDigits(2);
            df.setMaximumFractionDigits(2);
            df.setRoundingMode(RoundingMode.DOWN);
            //return f.valueToString(position);
            return df.format(position);
        } else {
            DecimalFormat formatter = new DecimalFormat();
            return formatter.format((int) position);
        }
    }

    private static TickSpacing findSpacing(long maxValue, int width, boolean scaleInKB) {

        if (maxValue < 10) {
            return new TickSpacing(1, HiC.Unit.BP.toString(), 1);
        }

        int maxNumberOfTickMarks = (int) Math.ceil(width / 25);

        // How many zeroes?
        int nZeroes = (int) Math.log10(maxValue);
        String majorUnit = scaleInKB ? "KB" : HiC.Unit.BP.toString();
        int unitMultiplier = 1;
        if (nZeroes > 9) {
            majorUnit = scaleInKB ? "TB" : "GB";
            unitMultiplier = (int) 1e9;
        }
        if (nZeroes > 6) {
            majorUnit = scaleInKB ? "GB" : "MB";
            unitMultiplier = (int) 1e6;
        } else if (nZeroes > 3) {
            majorUnit = scaleInKB ? "MB" : "KB";
            unitMultiplier = 1000;
        }

        int decrementIter = nZeroes - 1;
        while (decrementIter > -1) {

            int latestIncrement = (int) Math.pow(10, nZeroes - decrementIter);
            int nMajorTicks = (int) Math.ceil(maxValue / latestIncrement);

            if (nMajorTicks < maxNumberOfTickMarks) {
                return new TickSpacing(latestIncrement, majorUnit, unitMultiplier);
            }

            latestIncrement = (int) Math.pow(10, nZeroes - decrementIter + 1) / 2;
            nMajorTicks = (int) Math.ceil(maxValue / latestIncrement);

            if (nMajorTicks < maxNumberOfTickMarks) {
                return new TickSpacing(latestIncrement, majorUnit, unitMultiplier);
            }

            decrementIter--;
        }

        int spacing = (int) (maxValue / maxNumberOfTickMarks) / 250 * 250;

        return new TickSpacing(spacing, majorUnit, unitMultiplier);
    }

    public static boolean getShowOnlyEndPts() {
        return showOnlyEndPts;
    }

    public static void setShowOnlyEndPts(boolean toggled) {
        showOnlyEndPts = toggled;
    }

    public static boolean getShowChromosomeFigure() {
        return showChromosomeFigure;
    }

    public static void setShowChromosomeFigure(boolean toggled) {
        showChromosomeFigure = toggled;
    }

    public void setContext(Context frame, Orientation orientation) {
        this.context = frame;
        this.orientation = orientation;
    }

    @Override
    protected void paintComponent(Graphics g) {

        super.paintComponent(g);

        ((Graphics2D) g).setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        Graphics2D g2D = (Graphics2D) g;

        try {
            hic.getZd();
        } catch (Exception e) {
            return;
        }

        if (context == null) return;

        g.setColor(Color.black);

        AffineTransform t = g2D.getTransform();

        if (orientation == Orientation.VERTICAL) {
            AffineTransform rotateTransform = new AffineTransform();
            rotateTransform.quadrantRotate(-1);
            g2D.transform(rotateTransform);
        }

        // Clear panel
        drawTicks(g2D);
        drawChr(g2D);

        g2D.setTransform(t);


    }

    private void drawChr(Graphics g) {
        int w = isHorizontal() ? getWidth() : getHeight();
        int h = isHorizontal() ? getHeight() : getWidth();

        g.setFont(spanFont);

        Chromosome chromosome = context.getChromosome();

        if (chromosome != null) {
            if (!HiCFileTools.isAllChromosome(chromosome)) {
                String rangeString = chromosome.getName();
                int strWidth = g.getFontMetrics().stringWidth(rangeString);
                int strPosition = (w - strWidth) / 2;

                if (!isHorizontal()) strPosition = -strPosition;

                if (hic.getDisplayOption() == MatrixType.VS) {
                    if (isHorizontal()) {
                        rangeString = rangeString + " (control)";
                    } else {
                        rangeString = rangeString + " (observed)";
                    }
                }

                int vPos = h - 35;

                if (!showChromosomeFigure) {
                    g.drawString(rangeString, strPosition, vPos);
                }

            }
        }
    }

    private boolean isHorizontal() {
        return orientation == Orientation.HORIZONTAL;
    }

    private void drawTicks(Graphics g) {

        int w = isHorizontal() ? getWidth() : getHeight();
        int h = isHorizontal() ? getHeight() : getWidth();

        Color topTick = new Color(0, 0, 255);
        Color leftTick = new Color(0, 128, 0);


        if (w < 50 || hic.getScaleFactor() == 0) {
            return;
        }

        g.setFont(tickFont);

        Chromosome chromosome = context.getChromosome();

        if (chromosome == null) return;

        MatrixZoomData zd;
        try {
            zd = hic.getZd();
        } catch (Exception e) {
            return;
        }
        if (zd == null || zd.getXGridAxis() == null || zd.getYGridAxis() == null) return;

        if (HiCFileTools.isAllChromosome(chromosome)) {
            int x1 = 0;
            List<Chromosome> chromosomes = hic.getChromosomes();
            // Index 0 is whole genome
            int genomeCoord = 0;
            for (int i = 1; i < chromosomes.size(); i++) {
                Color tColor = isHorizontal() ? topTick : leftTick;
                g.setColor(tColor);


                double binOrigin = context.getBinOrigin();
                Chromosome c = chromosomes.get(i);
                genomeCoord += (c.getLength() / 1000);

                int xBin = zd.getXGridAxis().getBinNumberForGenomicPosition(genomeCoord);
                int x2 = (int) ((xBin - binOrigin) * hic.getScaleFactor());

                int x = (x1 + x2) / 2;
                int strWidth = g.getFontMetrics().stringWidth(c.getName());
                int strPosition = isHorizontal() ? x - strWidth / 2 : -x - strWidth / 2;
                g.drawString(c.getName(), strPosition, h - 15);

                int xpos = isHorizontal() ? x2 : -x2;

                g.drawLine(xpos, h - 10, xpos, h - 2);

                x1 = x2;
            }
        }

        else {
            HiCGridAxis axis = isHorizontal() ? zd.getXGridAxis() : zd.getYGridAxis();

            int binRange = (int) (w / hic.getScaleFactor());
            double binOrigin = context.getBinOrigin();     // <= by definition at left/top of panel

            int genomeOrigin = axis.getGenomicStart(binOrigin);
            int genomeEnd = axis.getGenomicEnd(binOrigin + binRange);
            int range = genomeEnd - genomeOrigin;

            TickSpacing ts = findSpacing(range, w, false);

            if (showOnlyEndPts) {

                // Hundredths decimal point
                int[] genomePositions = hic.getCurrentRegionWindowGenomicPositions();

                double startPosition = isHorizontal() ? genomePositions[0] : genomePositions[2];
                double endPosition = isHorizontal() ? genomePositions[1] : genomePositions[3];
                int endPositionBin = (int) (axis.getBinNumberForGenomicPosition((int) (endPosition - startPosition)) * hic.getScaleFactor());
                
                // actual strings to print and their widths
                String startPositionString = formatNumber(startPosition / ts.getUnitMultiplier()) + " " + ts.getMajorUnit();

                String endPositionString = formatNumber(endPosition / ts.getUnitMultiplier()) + " " + ts.getMajorUnit();
                int startPositionStringWidth = g.getFontMetrics().stringWidth(startPositionString);
                int endPositionStringWidth = g.getFontMetrics().stringWidth(endPositionString);

                //draw start
                int drawPositionStartString = isHorizontal() ? 0 : -startPositionStringWidth;
                g.drawString(startPositionString, drawPositionStartString, h - 15);
                g.drawLine(0, h - 10, 0, h - 2);

                //draw end
                if (!isHorizontal()) endPositionBin = -endPositionBin;
                int drawPositionEndString = isHorizontal() ? endPositionBin - endPositionStringWidth : endPositionBin;
                g.drawString(endPositionString, drawPositionEndString, h - 15);
                g.drawLine(endPositionBin, h - 10, endPositionBin, h - 2);

            } else {
                try {

                    int maxX = context.getChromosome().getLength();
                    double spacing = ts.getMajorTick();

                    // Find starting point closest to the current origin
                    int nTick = (int) (genomeOrigin / spacing) - 1;
                    int genomePosition = (int) (nTick * spacing);

                    int binNumber = axis.getBinNumberForGenomicPosition(genomePosition);
                    int x = (int) ((binNumber - binOrigin) * hic.getScaleFactor());

                    while (genomePosition < maxX && x < w) {
                        Color tColor = isHorizontal() ? topTick : leftTick;
                        g.setColor(tColor);

                        genomePosition = (int) (nTick * spacing);
                        binNumber = axis.getBinNumberForGenomicPosition(genomePosition);
                        x = (int) ((binNumber - binOrigin) * hic.getScaleFactor());

                        String chrPosition = formatNumber((double) genomePosition / ts.getUnitMultiplier()) + " " + ts.getMajorUnit();
                        int strWidth = g.getFontMetrics().stringWidth(chrPosition);
                        int strPosition = isHorizontal() ? x - strWidth / 2 : -x - strWidth / 2;

                        // prevent cut off near origin
                        if (isHorizontal()) {
                            if (binNumber == 0 && strPosition <= 0 && strPosition >= -strWidth / 2)
                                strPosition = 0;
                        } else {
                            if (binNumber == 0 && strPosition >= -strWidth && strPosition <= -strWidth / 2)
                                strPosition = -strWidth;
                        }

                        // todo bug or expected behavior?
                        // see chr1 of k562 mapq30 at fragment resolution
                        // axis is drawing overlapping positions onto each other
                        // traces to getFragmentNumberForGenomicPosition method
                        //System.out.println(genomePosition+"_"+chrPosition+"_"+strPosition);
                        if (nTick % 2 == 0) g.drawString(chrPosition, strPosition, h - 15);

                        int xpos = isHorizontal() ? x : -x;
                        g.drawLine(xpos, h - 10, xpos, h - 2);
                        nTick++;
                    }
                } catch (Exception e) {
                }
            }
        }
    }

    public enum Orientation {HORIZONTAL, VERTICAL}

    public static class TickSpacing {

        private final double majorTick;
        private final double minorTick;
        private String majorUnit = "";
        private int unitMultiplier = 1;

        TickSpacing(double majorTick, String majorUnit, int unitMultiplier) {
            this.majorTick = majorTick;
            this.minorTick = majorTick / 10.0;
            this.majorUnit = majorUnit;
            this.unitMultiplier = unitMultiplier;
        }

        public double getMajorTick() {
            return majorTick;
        }

        public double getMinorTick() {
            return minorTick;
        }

        public String getMajorUnit() {
            return majorUnit;
        }

        public void setMajorUnit(String majorUnit) {
            this.majorUnit = majorUnit;
        }

        public int getUnitMultiplier() {
            return unitMultiplier;
        }

        public void setUnitMultiplier(int unitMultiplier) {
            this.unitMultiplier = unitMultiplier;
        }
    }

// TODO -- possibly generalize?

    class ClickLink {

        final Rectangle region;
        final String value;
        final String tooltipText;

        ClickLink(Rectangle region, String value, String tooltipText) {
            this.region = region;
            this.value = value;
            this.tooltipText = tooltipText;
        }
    }
}
