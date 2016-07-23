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

package juicebox.track;

import juicebox.Context;
import juicebox.HiC;
import juicebox.data.MatrixZoomData;
import org.apache.commons.math.stat.StatUtils;
import org.broad.igv.renderer.GraphicUtils;
import org.broad.igv.renderer.Renderer;
import org.broad.igv.ui.FontManager;
import org.broad.igv.util.ResourceLocator;
import org.broad.igv.util.collections.DoubleArrayList;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.util.HashMap;
import java.util.Map;

/**
 * @author Jim Robinson
 * @since 4/13/12
 */
public class EigenvectorTrack extends HiCTrack {


    private final Map<Integer, double[]> dataCache = new HashMap<Integer, double[]>();
    private final Map<Integer, Double> dataMaxCache = new HashMap<Integer, Double>();
    private final Map<Integer, Double> medianCache = new HashMap<Integer, Double>();
    private final HiC hic;
    private Color color = Color.blue.darker();
    private Color altColor = Color.red.darker();
    private int currentZoom = -1;
    private String name;
    private boolean isControl = false;
    private int isCtrlInt = 0;


    public EigenvectorTrack(String id, String name, HiC hic, boolean isControl) {
        super(new ResourceLocator(id));
        this.hic = hic;
        this.name = name;
        this.isControl = isControl;
        if (isControl) {
            isCtrlInt = 1000;
            // there aren't any organisms I'm aware of with 1000 chromosomes, we should be safe with this offset
            // could probably multiply by -1 as well, would not work for all by all (-0 = 0)
            // but genomewide doesn't have an eigenvector todo consider if *(-1) is a better option
        }

    }

    private void setData(int chrIdx, double[] data) {

        if (data != null && data.length > 0) {
            DoubleArrayList tmp = new DoubleArrayList(data.length);

            for (double datum : data) {
                if (!Double.isNaN(datum)) {
                    tmp.add(datum);
                }
            }


            /*
            for (int i = 0; i < data.length; i++) {
                if (!Double.isNaN(data[i])) {
                    tmp.add(data[i]);
                }
            }
            */
            double[] tmpArray = tmp.toArray();
            medianCache.put(chrIdx + isCtrlInt, StatUtils.percentile(tmpArray, 50));
            double max = 0;
            for (double aData : tmpArray) {
                if (Math.abs(aData) > max) max = Math.abs(aData);
            }
            dataMaxCache.put(chrIdx + isCtrlInt, max);
        }
    }

    private void clearDataCache() {
        dataCache.clear();
        dataMaxCache.clear();
        medianCache.clear();
    }

    public Color getPosColor() {
        return color;
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
    public void setColor(Color selectedColor) {
        this.color = selectedColor;
    }

    @Override
    public void setAltColor(Color selectedColor) {
        this.altColor = selectedColor;
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

        g.setColor(color);

        int height = orientation == TrackPanel.Orientation.X ? rect.height : rect.width;
        int width = orientation == TrackPanel.Orientation.X ? rect.width : rect.height;
        int y = orientation == TrackPanel.Orientation.X ? rect.y : rect.x;
        int x = orientation == TrackPanel.Orientation.X ? rect.x : rect.y;

        MatrixZoomData zd;
        try {
            if (isControl) {
                zd = hic.getControlZd();
            } else {
                zd = hic.getZd();
            }
        } catch (Exception e) {
            return;
        }

        int zoom = zd.getZoom().getBinSize();
        if (zoom != currentZoom) {
            clearDataCache();
        }

        int chrIdx = orientation == TrackPanel.Orientation.X ? zd.getChr1Idx() : zd.getChr2Idx();
        double[] eigen = dataCache.get(chrIdx + isCtrlInt);
        if (eigen == null) {
            eigen = hic.getEigenvector(chrIdx, 0, isControl);
            currentZoom = zoom;
            setData(chrIdx, eigen);
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

        double dataMax = dataMaxCache.get(chrIdx + isCtrlInt);
        double median = medianCache.get(chrIdx + isCtrlInt);


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
                g.fillRect(xPixelLeft, y + h - myh, (xPixelRight - xPixelLeft), myh);
            } else {
                g.fillRect(xPixelLeft, y + h, (xPixelRight - xPixelLeft), -myh);
            }
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

    public void forceRefresh() {
        currentZoom = -1;
        clearDataCache();
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
