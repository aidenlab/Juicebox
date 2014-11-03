package juicebox.track;

import org.apache.commons.math.stat.StatUtils;
import juicebox.Context;
import juicebox.HiC;
import org.broad.igv.renderer.GraphicUtils;
import org.broad.igv.renderer.Renderer;
import org.broad.igv.ui.FontManager;
import org.broad.igv.util.ResourceLocator;
import org.broad.igv.util.collections.DoubleArrayList;

import java.awt.*;
import java.util.HashMap;
import java.util.Map;

/**
 * @author Jim Robinson
 * @date 4/13/12
 */
public class EigenvectorTrack extends HiCTrack {


    private Color color = Color.blue.darker();
    private Color altColor = Color.red.darker();
    Map<Integer, double[]> dataCache = new HashMap<Integer, double[]>();
    private Map<Integer, Double> dataMaxCache = new HashMap<Integer, Double>();
    private Map<Integer, Double> medianCache = new HashMap<Integer, Double>();
    HiC hic;
    int currentZoom = -1;
    private String name = "eigenvector";

    public EigenvectorTrack(String id, String name, HiC hic) {
        super(new ResourceLocator(id));
        this.hic = hic;
    }

    private void setData(int chrIdx, double[] data) {

        if (data != null && data.length > 0) {
            DoubleArrayList tmp = new DoubleArrayList(data.length);
            for (int i = 0; i < data.length; i++) {
                if (!Double.isNaN(data[i])) {
                    tmp.add(data[i]);
                }
            }
            double[] tmpArray = tmp.toArray();
            medianCache.put(chrIdx, StatUtils.percentile(tmpArray, 50));
            double max = 0;
            for (double aData : tmpArray) {
                if (Math.abs(aData) > max) max = Math.abs(aData);
            }
            dataMaxCache.put(chrIdx, max);
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
    public void setName(String text) {
        this.name = text;
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
     * @param g2d      the graphics context
     * @param rect     the track bounds, relative to the enclosing DataPanel bounds.
     * @param gridAxis
     */

    @Override
    public void render(Graphics2D g2d, Context context, Rectangle rect, TrackPanel.Orientation orientation, HiCGridAxis gridAxis) {

        g2d.setColor(color);

        int height = orientation == TrackPanel.Orientation.X ? rect.height : rect.width;
        int width = orientation == TrackPanel.Orientation.X ? rect.width : rect.height;
        int y = orientation == TrackPanel.Orientation.X ? rect.y : rect.x;
        int x = orientation == TrackPanel.Orientation.X ? rect.x : rect.y;

        int zoom = hic.getZd().getZoom().getBinSize();
        if (zoom != currentZoom) {
            clearDataCache();
        }

        int chrIdx = orientation == TrackPanel.Orientation.X ? hic.getZd().getChr1Idx() : hic.getZd().getChr2Idx();
        double[] eigen = dataCache.get(chrIdx);
        if (eigen == null) {
            eigen = hic.getEigenvector(chrIdx, 0);
            currentZoom = zoom;
            setData(chrIdx, eigen);
        }

        if (eigen == null || eigen.length == 0) {
            g2d.setFont(FontManager.getFont(8));
            GraphicUtils.drawCenteredText("Eigenvector not available at this resolution", rect, g2d);
            return;  // No data available
        }

        double dataMax = dataMaxCache.get(chrIdx);
        double median = medianCache.get(chrIdx);


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
                g2d.fillRect(xPixelLeft, y + h - myh, (xPixelRight - xPixelLeft), myh);
            } else {
                g2d.fillRect(xPixelLeft, y + h, (xPixelRight - xPixelLeft), -myh);
            }
        }


    }

    public String getName() {
        return name;
    }

    public Renderer getRenderer() {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public void forceRefresh() {
        currentZoom = -1;
        clearDataCache();
    }
}
