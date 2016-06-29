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
    private static boolean showChromosomeFigure = false;
    private final Font tickFont = FontManager.getFont(Font.BOLD, 9);
    private final Font spanFont = FontManager.getFont(Font.BOLD, 12);
    private HiC hic;
    private Orientation orientation;
    private Context context;

    /**
     * Empty constructor for form builder
     */
    private HiCRulerPanel() {
    }

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

    private static TickSpacing findSpacing(long maxValue, boolean scaleInKB) {

        if (maxValue < 10) {
            return new TickSpacing(1, "bp", 1);
        }


        // Now man zeroes?
        int nZeroes = (int) Math.log10(maxValue);
        String majorUnit = scaleInKB ? "kb" : "bp";
        int unitMultiplier = 1;
        if (nZeroes > 9) {
            majorUnit = scaleInKB ? "tb" : "gb";
            unitMultiplier = 1000000000;
        }
        if (nZeroes > 6) {
            majorUnit = scaleInKB ? "gb" : "mb";
            unitMultiplier = 1000000;
        } else if (nZeroes > 3) {
            majorUnit = scaleInKB ? "mb" : "kb";
            unitMultiplier = 1000;
        }

        double nMajorTicks = maxValue / Math.pow(10, nZeroes - 1);
        if (nMajorTicks < 25) {
            return new TickSpacing(Math.pow(10, nZeroes - 1), majorUnit, unitMultiplier);
        } else {
            return new TickSpacing(Math.pow(10, nZeroes) / 2, majorUnit, unitMultiplier);
        }
    }

    public void showOnlyEndPts(boolean toggled) {
        showOnlyEndPts = toggled;
    }

    public void showChromosomeFigure(boolean toggled) {
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
        drawChr(g2D, orientation);

        g2D.setTransform(t);


    }

    private void drawChr(Graphics g, Orientation orientation) {
        int w = isHorizontal() ? getWidth() : getHeight();
        int h = isHorizontal() ? getHeight() : getWidth();

        g.setFont(spanFont);

        Chromosome chromosome = context.getChromosome();

        if (chromosome != null) {
            if (!chromosome.getName().equals("All")) {
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

        if (chromosome.getName().equals("All")) {
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
            if (showOnlyEndPts) {

                HiCGridAxis axis = isHorizontal() ? zd.getXGridAxis() : zd.getYGridAxis();

                int binRange = (int) (w / hic.getScaleFactor());
                double binOrigin = context.getBinOrigin();

                int genomeOrigin = axis.getGenomicStart(binOrigin);

                int genomeEnd = axis.getGenomicEnd(binOrigin + binRange);

                int range = genomeEnd - genomeOrigin;

                TickSpacing ts = findSpacing(range, false);

                // Hundredths decimal point
                int[] genomePositions = hic.getCurrentRegionWindowGenomicPositions();

                String genomeStartX = formatNumber((float) (genomePositions[0] * 1.0) / ts.getUnitMultiplier()) + " " + ts.getMajorUnit();
                String genomeStartY = formatNumber((float) (genomePositions[2] * 1.0) / ts.getUnitMultiplier()) + " " + ts.getMajorUnit();
                String genomeEndX = formatNumber((float) (genomePositions[1] * 1.0) / ts.getUnitMultiplier()) + " " + ts.getMajorUnit();
                String genomeEndY = formatNumber((float) (genomePositions[3] * 1.0) / ts.getUnitMultiplier()) + " " + ts.getMajorUnit();

                if (isHorizontal()) {
                    int maxX = context.getChromosome().getLength();
                    int x = (int) (axis.getBinNumberForGenomicPosition(maxX) * hic.getScaleFactor());

                    int endbinNumber = (genomePositions[1] > maxX) ? x : w;
                    // hic.getScaleFactor

                    //Horizontal Start
                    g.drawString(genomeStartX, 10, h - 15);
                    g.drawLine(0, h - 10, 0, h - 2);
                    //Horizontal End
                    g.drawString(genomeEndX, endbinNumber - g.getFontMetrics().stringWidth(genomeEndX) - 5, h - 15);
                    g.drawLine(endbinNumber - 5, h - 10, endbinNumber - 5, h - 2);
                } else {
                    //Vertical Start
                    g.drawString(genomeStartY, -g.getFontMetrics().stringWidth(genomeEndX) - 5, h - 15);
                    g.drawLine(0, h - 10, 0, h - 2);
                    //Vertical End
                    g.drawString(genomeEndY, -w + 10, h - 15);
                    g.drawLine(-(w - 5), h - 10, -(w - 5), h - 2);

                }
            } else {
                try {
                    HiCGridAxis axis = isHorizontal() ? zd.getXGridAxis() : zd.getYGridAxis();

                    int binRange = (int) (w / hic.getScaleFactor());
                    double binOrigin = context.getBinOrigin();     // <= by definition at left/top of panel

                    int genomeOrigin = axis.getGenomicStart(binOrigin);

                    int genomeEnd = axis.getGenomicEnd(binOrigin + binRange);

                    int range = genomeEnd - genomeOrigin;


                    TickSpacing ts = findSpacing(range, false);
                    double spacing = ts.getMajorTick();

                    // Find starting point closest to the current origin
                    int maxX = context.getChromosome().getLength();
                    int nTick = (int) (genomeOrigin / spacing) - 1;
                    int genomePosition = (int) (nTick * spacing);

                    //int x = frame.getScreenPosition(genomeTickNumber);
                    int binNUmber = axis.getBinNumberForGenomicPosition(genomePosition);

                    int x = (int) ((binNUmber - binOrigin) * hic.getScaleFactor());

                    while (genomePosition < maxX && x < w) {
                        Color tColor = (orientation == Orientation.HORIZONTAL ? topTick : leftTick);
                        g.setColor(tColor);

                        genomePosition = (int) (nTick * spacing);

                        // x = frame.getScreenPosition(genomeTickNumber);
                        binNUmber = axis.getBinNumberForGenomicPosition(genomePosition);

                        x = (int) ((binNUmber - binOrigin) * hic.getScaleFactor());

                        String chrPosition = formatNumber((double) genomePosition / ts.getUnitMultiplier()) + " " + ts.getMajorUnit();
                        int strWidth = g.getFontMetrics().stringWidth(chrPosition);
                        int strPosition = isHorizontal() ? x - strWidth / 2 : -x - strWidth / 2;
                        //if (strPosition > strEnd) {

                        if (nTick % 2 == 0) {
                            g.drawString(chrPosition, strPosition, h - 15);
                        }
                        //strEnd = strPosition + strWidth;
                        //}

                        int xpos = (orientation == Orientation.HORIZONTAL ? x : -x);

                        g.drawLine(xpos, h - 10, xpos, h - 2);
                        nTick++;
                    }
                } catch (Exception e) {
                    return;

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
