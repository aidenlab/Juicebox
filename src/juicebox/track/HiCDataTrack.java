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

package juicebox.track;

import juicebox.Context;
import juicebox.HiC;
import juicebox.gui.SuperAdapter;
import org.broad.igv.renderer.DataRange;
import org.broad.igv.track.WindowFunction;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;

/**
 * @author jrobinso
 *         Date: 11/2/12
 *         Time: 9:41 AM
 */
public class HiCDataTrack extends HiCTrack {

    private static final int TRACK_MARGIN = 2;
    private final HiC hic;
    private final HiCDataSource dataSource;
    private final NumberFormat formatter = NumberFormat.getInstance();
    private final float[] dash = {8.0f};
    private final BasicStroke dashedStroke = new BasicStroke(0.5f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 2.0f, dash, 0.0f);
    private final Color dashColor = new Color(120, 120, 120);
    private boolean logScale = false;
    private HiCDataPoint[] data;
    private WindowFunction windowFunction = WindowFunction.mean;


    public HiCDataTrack(HiC hic, ResourceLocator locator, HiCDataSource da) {
        super(locator);
        this.hic = hic;
        this.dataSource = da;
        this.logScale = dataSource.isLog();
    }

    @Override
    public void render(Graphics2D g, Context context, Rectangle rect, TrackPanel.Orientation orientation, HiCGridAxis gridAxis) {


        int height = orientation == TrackPanel.Orientation.X ? rect.height : rect.width;
        int width = orientation == TrackPanel.Orientation.X ? rect.width : rect.height;

        int y = orientation == TrackPanel.Orientation.X ? rect.y : rect.x;
        int x = orientation == TrackPanel.Orientation.X ? rect.x : rect.y;

        Graphics2D g2d = (Graphics2D) g.create(x, y, width, height);
        y = 0;
        x = 0;

        String chr = context.getChromosome().getName();
        double startBin = context.getBinOrigin();
        double endBin = startBin + (width / hic.getScaleFactor());
        data = dataSource.getData(context.getChromosome(), (int) startBin, (int) endBin + 1,
                gridAxis, hic.getScaleFactor(), windowFunction);

        if (data == null) return;

        Color posColor = dataSource.getColor();
        Color negColor = dataSource.getAltColor();

        // Get the Y axis definition, consisting of minimum, maximum, and base value.  Often
        // the base value is == min value which is == 0.
        DataRange dataRange = dataSource.getDataRange();
        float maxValue = dataRange.getMaximum();
        float baseValue = dataRange.getBaseline();
        float minValue = dataRange.getMinimum();
        boolean isLog = dataRange.isLog();

        if (isLog) {
            minValue = (float) (minValue == 0 ? 0 : Math.log10(minValue));
            maxValue = (float) Math.log10(maxValue);
        }

        // Calculate the Y scale factor.

        double delta = (maxValue - minValue);
        double yScaleFactor = (height - TRACK_MARGIN) / delta;

        // Calculate the Y position in pixels of the base value.  Clip to bounds of rectangle
        double baseDelta = maxValue - baseValue;
        int baseY = (int) (y + baseDelta * yScaleFactor);
        if (baseY < y) {
            baseY = y;
        } else if (baseY > y + (height - TRACK_MARGIN)) {
            baseY = y + (height - TRACK_MARGIN);
        }

        //for (int i = 0; i < data.length; i++) {
        for (HiCDataPoint d : data) {

            //HiCDataPoint d = data[i];
            if (d == null) continue;

            double bin = d.getBinNumber() - startBin;
            double widthInBIns = d.getWithInBins();
            int xPixelLeft = x + (int) Math.round(bin * hic.getScaleFactor()); //context.getScreenPosition (genomicPosition);
            int dx = (int) Math.max(1, widthInBIns * hic.getScaleFactor());
            int xPixelRight = xPixelLeft + dx;

            if (xPixelRight < x) {
                continue;
            } else if (xPixelLeft > x + width) {
                break;
            }

            double dataY = d.getValue(windowFunction);
            if (isLog && dataY <= 0) {
                continue;
            }

            if (!Double.isNaN(dataY)) {

                // Compute the pixel y location.  Clip to bounds of rectangle.
                double dy = isLog ? Math.log10(dataY) - baseValue : (dataY - baseValue);
                int pY = baseY - (int) (dy * yScaleFactor);
                if (pY < y) {
                    pY = y;
                } else if (pY > y + (height - TRACK_MARGIN)) {
                    pY = y + (height - TRACK_MARGIN);
                }

                Color color = (dataY >= baseValue) ? posColor : negColor;
                g2d.setColor(color);

                if (dx <= 1) {
                    g2d.drawLine(xPixelLeft, baseY, xPixelLeft, pY);
                } else {
                    if (pY > baseY) {
                        g2d.fillRect(xPixelLeft, baseY, dx, pY - baseY);

                    } else {
                        g2d.fillRect(xPixelLeft, pY, dx, baseY - pY);
                    }
                }

            }
        }

        // If min is < 0 draw a line
        if (minValue < 0) {
            g2d.setColor(dashColor);
            g2d.setStroke(dashedStroke);
            g2d.drawLine(0, baseY, width, baseY);
        }
    }

    public WindowFunction getWindowFunction() {
        return windowFunction;
    }

    public void setWindowFunction(WindowFunction windowFunction) {
        this.windowFunction = windowFunction;
    }

    public String getName() {
        return dataSource.getName();
    }

    @Override
    public void setName(String text) {
        dataSource.setName(text);
    }

    @Override
    public Color getPosColor() {
        return dataSource.getColor();
    }

    @Override
    public String getToolTipText(int x, int y, TrackPanel.Orientation orientation) {
        StringBuilder txt = new StringBuilder();


        txt.append("<span style='color:red; font-family: arial; font-size: 12pt;'>");
        txt.append(getName());
        txt.append("</span>");

        if (data == null) return txt.toString();

        Context context = orientation == TrackPanel.Orientation.X ? hic.getXContext() : hic.getYContext();

        double binOrigin = context.getBinOrigin();
        final double scaleFactor = hic.getScaleFactor();
        final double substepSize = 1.0 / scaleFactor;
        double bin = (binOrigin + (x / scaleFactor));

        HiCDataPoint target = new HiCDataAdapter.DataAccumulator(bin);
        int idx = Arrays.binarySearch(data, target, new Comparator<HiCDataPoint>() {
            @Override
            public int compare(HiCDataPoint weightedSum, HiCDataPoint weightedSum1) {
                final double binNumber = weightedSum.getBinNumber();
                final double binNumber1 = weightedSum1.getBinNumber();

                int bin = (int) binNumber;
                int bin1 = (int) binNumber1;
                if (bin == bin1) {
                    double rem = binNumber - bin;
                    double rem1 = binNumber1 - bin;
                    if (Math.abs(rem - rem1) < substepSize) {
                        return 0;
                    } else {
                        if (rem > rem1) return 1;
                        else return -1;
                    }
                } else {
                    return bin - bin1;
                }

            }
        });


        txt.append("<span style='font-family: arial; font-size: 12pt;'>");
        if (idx < 0) {
            txt.append("<br>bin: ").append(formatter.format((int) bin));
        } else {
            HiCDataPoint ws = data[idx];
            if (ws == null) return null;

            txt.append("<br>").append(formatter.format(ws.getGenomicStart()))
                    .append("-")
                    .append(formatter.format(ws.getGenomicEnd()))
                    .append("<br>bin: ")
                    .append(formatter.format(ws.getBinNumber()))
                    .append("<br>value: ")
                    .append(formatter.format(ws.getValue(windowFunction)));
        }
        txt.append("</span>");
        return txt.toString();
    }

    @Override
    public void setColor(Color selectedColor) {
        dataSource.setColor(selectedColor);
    }

    @Override
    public JPopupMenu getPopupMenu(final TrackPanel trackPanel, final SuperAdapter superAdapter) {

        JPopupMenu menu = super.getPopupMenu(trackPanel, superAdapter);
        menu.addSeparator();

        JMenuItem menuItem = new JMenuItem("Configure track...");
        menuItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                final TrackConfigDialog trackConfigDialog = new TrackConfigDialog(superAdapter.getMainWindow(), HiCDataTrack.this);
                trackConfigDialog.setVisible(true);
                if (!trackConfigDialog.isCanceled()) {
                    superAdapter.updateTrackPanel();
                }
            }
        });
        menu.add(menuItem);
//
//        JMenuItem menuItem = new JMenuItem("Set Y axis range...");
//        menuItem.addActionListener(new ActionListener() {
//            @Override
//            public void actionPerformed(ActionEvent e) {
//                DataRange dataRange = dataSource.getDataRange();
//                //String newValue = JOptionPane.showInputDialog(trackPanel, "Y scale maximum: " + dataRange.getMaximum());
//                try {
//                    DataRangeDialog dlg = new DataRangeDialog(MainWindow.getInstance(), dataRange);
//                    dlg.setVisible(true);
//                    if (!dlg.isCanceled()) {
//                        float min = Math.min(dlg.getMin(), dlg.getMax());
//                        float max = Math.max(dlg.getMin(), dlg.getMax());
//                        float mid = dlg.getBase();
//                        if (mid < min) mid = min;
//                        else if (mid > max) mid = max;
//                        dataRange = new DataRange(min, mid, max);
//                        dataRange.setType(dlg.getDataRangeType());
//                        dataSource.setDataRange(dataRange);
//                        MainWindow.getInstance().repaint();
//                    }
//
//                } catch (NumberFormatException nfe) {
//                    JOptionPane.showMessageDialog(trackPanel, "Must enter a number.");
//                }
//            }
//        });
//        menu.add(menuItem);

        return menu;
    }

    /*
     useless at present
    @Override
    public void mouseClicked(int x, int y, Context context, TrackPanel.Orientation orientation) {

    }
    */

    public DataRange getDataRange() {
        return dataSource.getDataRange();  //To change body of created methods use File | Settings | File Templates.
    }

    public void setDataRange(DataRange dataRange) {
        dataSource.setDataRange(dataRange);
    }

    public Color getAltColor() {
        return dataSource.getAltColor();
    }

    @Override
    public void setAltColor(Color selectedColor) {
        dataSource.setAltColor(selectedColor);
    }

    public Collection<WindowFunction> getAvailableWindowFunctions() {

        return dataSource.getAvailableWindowFunctions();
    }
}
