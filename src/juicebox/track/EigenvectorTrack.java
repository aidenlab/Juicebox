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

package juicebox.track;

import juicebox.Context;
import juicebox.HiC;
import juicebox.data.MatrixZoomData;
import juicebox.gui.SuperAdapter;
import juicebox.tools.utils.common.ArrayTools;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.broad.igv.renderer.GraphicUtils;
import org.broad.igv.renderer.Renderer;
import org.broad.igv.ui.FontManager;
import org.broad.igv.util.ResourceLocator;
import org.broad.igv.util.collections.LRUCache;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.AffineTransform;
import java.util.HashMap;
import java.util.Map;

/**
 * @author Jim Robinson
 * @since 4/13/12
 */
public class EigenvectorTrack extends HiCTrack {

    // actual dataset object saves more; we will just save 6 most recent ones here
    private final LRUCache<String, double[]> dataCache = new LRUCache<>(6);
    //private final Map<String, double[]> dataCache = new HashMap<>();
    private final Map<String, Double> dataMaxCache = new HashMap<>();
    private final Map<String, Double> medianCache = new HashMap<>();
    private final Map<String, Integer> flippingRecordCache = new HashMap<>();
    private final HiC hic;

    private int currentZoomBinSize = -1;
    private String name;
    private boolean isControl = false;


    public EigenvectorTrack(String id, String name, HiC hic, boolean isControl) {
        super(new ResourceLocator(id));
        this.hic = hic;
        this.name = name;
        this.isControl = isControl;
    }

    private double[] loadData(int chrIdxPreCtrlInt, int zoomBinSize) {

        double[] data = hic.getEigenvector(chrIdxPreCtrlInt, 0, isControl);
        currentZoomBinSize = zoomBinSize;
        String cacheKey = getCacheKey(chrIdxPreCtrlInt, zoomBinSize);

        if (data != null && data.length > 0) {
            int flipVal = 1;
            if (flippingRecordCache.containsKey(cacheKey)) {
                flipVal = flippingRecordCache.get(cacheKey);
                if (flipVal == -1) {
                    data = ArrayTools.flipArrayValues(data);
                }
            } else {
                flippingRecordCache.put(cacheKey, flipVal);
            }

            dataCache.put(cacheKey, data);

            DescriptiveStatistics stats = new DescriptiveStatistics();

            for (double datum : data) {
                if (!Double.isNaN(datum)) {
                    stats.addValue(datum);
                }
            }

            medianCache.put(cacheKey, stats.getPercentile(50));

            double max = 0;
            for (double aData : stats.getValues()) {
                if (Math.abs(aData) > max) max = Math.abs(aData);
            }
            dataMaxCache.put(cacheKey, max);
        }
        return data;
    }

    private String getCacheKey(int chrIdx, int zoomBinSize) {
        String key = "observed_";
        if (isControl) {
            key = "control_";
        }
        return key + chrIdx + "_" + zoomBinSize;
    }

    @Override
    public String getToolTipText(int x, int y, TrackPanel.Orientation orientation) {

        return "";
//        if (data == null) return null;
//
//        int binOrigin = hic.xContext.getBinOrigin();
//        int bin = binOrigin + (int) (x / hic.xContext.getScaleFactor());
//
//        return bin < data.length ? String.valueOf(data[bin]) : null;

    }

    @Override
    public JPopupMenu getPopupMenu(final TrackPanel trackPanel, final SuperAdapter superAdapter, final TrackPanel.Orientation orientation) {

        JPopupMenu menu = super.getPopupMenu(trackPanel, superAdapter, orientation);
        menu.addSeparator();

        JMenuItem menuItem = new JMenuItem("Flip Eigenvector");
        menuItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                MatrixZoomData zd = getAppropriateZD();
                if (zd == null) return;

                int chrIdx = orientation == TrackPanel.Orientation.X ? zd.getChr1Idx() : zd.getChr2Idx();
                flipEigenvector(chrIdx, superAdapter.getHiC().getZoom().getBinSize());
                hic.refreshEigenvectorTrackIfExists();
            }
        });
        menu.add(menuItem);

        return menu;

    }

    protected void flipEigenvector(int chrIdx, int zoomBinSize) {
        String cacheKey = getCacheKey(chrIdx, zoomBinSize);
        int currFlipVal = flippingRecordCache.get(cacheKey);
        flippingRecordCache.put(cacheKey, -1 * currFlipVal);
    }


    /**
     * Render the track in the supplied rectangle.  It is the responsibility of the track to draw within the
     * bounds of the rectangle.
     *
     * @param g      the graphics context
     * @param rect     the track bounds, relative to the enclosing DataPanel bounds.
     * @param gridAxis
     */

    @Override
    public void render(Graphics g, Context context, Rectangle rect, TrackPanel.Orientation orientation, HiCGridAxis gridAxis) {

        g.setColor(getPosColor());

        int height = orientation == TrackPanel.Orientation.X ? rect.height : rect.width;
        int width = orientation == TrackPanel.Orientation.X ? rect.width : rect.height;
        int y = orientation == TrackPanel.Orientation.X ? rect.y : rect.x;
        int x = orientation == TrackPanel.Orientation.X ? rect.x : rect.y;

        MatrixZoomData zd = getAppropriateZD();
        if (zd == null) return;

        int zoomBinSize = hic.getZoom().getBinSize();

        int chrIdx = orientation == TrackPanel.Orientation.X ? zd.getChr1Idx() : zd.getChr2Idx();
        String cacheKey = getCacheKey(chrIdx, zoomBinSize);
        double[] eigen = dataCache.get(cacheKey);
        if (eigen == null) {
            eigen = loadData(chrIdx, zoomBinSize);
        }


        if (eigen == null || eigen.length == 0) {
            Font original = g.getFont();
            g.setFont(FontManager.getFont(12));

            if (orientation == TrackPanel.Orientation.X) {
                GraphicUtils.drawCenteredText("Eigenvector not available at this resolution", rect, g);
            } else {
                drawRotatedString((Graphics2D) g, "Eigenvector not available at this resolution", (2 * rect.height) / 3, rect.x + 15);
            }

            g.setFont(original);
            return;
        }

        double dataMax = dataMaxCache.get(cacheKey);
        double median = medianCache.get(cacheKey);

        int h = height / 2;

        for (int bin = (int) context.getBinOrigin(); bin < eigen.length; bin++) {

            if (Double.isNaN(eigen[bin])) continue;

            int xPixelLeft = x + (int) ((bin - context.getBinOrigin()) * hic.getScaleFactor());
            int xPixelRight = x + (int) ((bin + 1 - context.getBinOrigin()) * hic.getScaleFactor());

            if (xPixelRight < x) {
                continue;
            } else if (xPixelLeft > x + width) {
                break;
            }

            double x2 = eigen[bin] - median;
            double max = dataMax - median;

            int myh = (int) ((x2 / max) * h);
            if (x2 > 0) {
                g.setColor(getPosColor());
                g.fillRect(xPixelLeft, y + h - myh, (xPixelRight - xPixelLeft), myh);
            } else {
                g.setColor(getNegColor());
                g.fillRect(xPixelLeft, y + h, (xPixelRight - xPixelLeft), -myh);
            }
        }


    }

    private MatrixZoomData getAppropriateZD() {
        try {
            if (isControl) {
                return hic.getControlZd();
            } else {
                return hic.getZd();
            }
        } catch (Exception e) {
            return null;
        }
    }

    public String getName() {
        return name;
    }

    @Override
    public void setName(String text) {
        this.name = text;
    }

    public Renderer<?> getRenderer() {
        return null;  //TODO change body of implemented methods use File | Settings | File Templates.
    }

    public void forceRefreshCache() {
        currentZoomBinSize = -1;
        dataCache.clear();
        dataMaxCache.clear();
        medianCache.clear();
    }

    private void drawRotatedString(Graphics2D g2, String string, float x, float y) {
        AffineTransform orig = g2.getTransform();
        g2.rotate(0);
        g2.setColor(Color.BLUE);
        g2.translate(x, 0);
        g2.scale(-1, 1);
        g2.translate(-x, 0);
        g2.drawString(string, x, y);
        g2.setTransform(orig);
        //g2.translate(0,0);
        //g2.scale(1,1);
    }
}
