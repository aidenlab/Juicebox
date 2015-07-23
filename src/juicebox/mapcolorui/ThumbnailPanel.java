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

import juicebox.HiC;
import juicebox.MainWindow;
import juicebox.data.MatrixZoomData;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.geom.*;
import java.io.Serializable;

/**
 * @author jrobinso
 * @date Aug 2, 2010
 */
public class ThumbnailPanel extends JComponent implements Serializable {


    private static final long serialVersionUID = -3856114428388478494L;
    private static final AlphaComposite ALPHA_COMP = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.75f);
    private final MainWindow mainWindow;
    private final HiC hic;
    private Image image;
    private Point lastPoint = null;

    private Rectangle innerRectangle;


    public ThumbnailPanel(MainWindow mainWindow, HiC model) {

        this.mainWindow = mainWindow;
        this.hic = model;

        addMouseListener(new MouseAdapter() {


            @Override
            public void mouseClicked(MouseEvent mouseEvent) {
                if (mouseEvent.getClickCount() >= 1) {

                    try {
                        int xBP = (int) (mouseEvent.getX() * xScale());
                        int yBP = (int) (mouseEvent.getY() * yScale());

                        hic.center(xBP, yBP);
                    } catch (Exception e) {
                        System.out.println("Error when thumbnail clicked");
                        e.printStackTrace();
                    }
                }
            }

            @Override
            public void mousePressed(MouseEvent mouseEvent) {
                if (innerRectangle != null && innerRectangle.contains(mouseEvent.getPoint())) {
                    lastPoint = mouseEvent.getPoint();
                    setCursor(MainWindow.fistCursor);
                }

            }

            @Override
            public void mouseReleased(MouseEvent mouseEvent) {
                lastPoint = null;
                setCursor(Cursor.getDefaultCursor());
            }
        });

        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent mouseEvent) {
                if (lastPoint != null) {
                    int dxBP = ((int) ((mouseEvent.getX() - lastPoint.x) * xScale()));
                    int dyBP = ((int) ((mouseEvent.getY() - lastPoint.y) * yScale()));
                    hic.moveBy(dxBP, dyBP);
                    lastPoint = mouseEvent.getPoint();
                }


            }
        });
    }

    private double xScale() {
        MatrixZoomData matrixZoomData;
        if(hic.getZd() == null && mainWindow.isReloadState()){
            matrixZoomData = hic.getZoomDataForReloadState();
        }
        else {
            matrixZoomData = hic.getZd();
        }

        return (double) matrixZoomData.getXGridAxis().getBinCount() / getWidth();

    }

    private double yScale() {
        MatrixZoomData matrixZoomData;
        if(hic.getZd() == null && mainWindow.isReloadState()){
            matrixZoomData = hic.getZoomDataForReloadState();
        }
        else {
            matrixZoomData = hic.getZd();
        }
        return (double) matrixZoomData.getYGridAxis().getBinCount() / getHeight();

    }


    public void setImage(Image image) {
        this.image = image;
    }

    public String getName() {
        return null;
    }

    public void setName(String nm) {

    }

    @Override
    protected void paintComponent(Graphics g) {

        ((Graphics2D) g).setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        if (image != null) {
            g.drawImage(image, 0, 0, null);
            renderVisibleWindow((Graphics2D) g);
            //TODO******   UNCOMMENT  ******
        }
    }

    private void renderVisibleWindow(Graphics2D g) {


        if (hic != null && hic.getXContext() != null) {

            Rectangle outerRectangle = new Rectangle(0, 0, getBounds().width, getBounds().height);

            int wPixels = mainWindow.getHeatmapPanel().getWidth();
            int hPixels = mainWindow.getHeatmapPanel().getHeight();

            double originX = hic.getXContext().getBinOrigin();
            int x = (int) (originX / xScale());

            double originY = hic.getYContext().getBinOrigin();
            int y = (int) (originY / yScale());

            double wBins = wPixels / hic.getScaleFactor();
            int w = (int) (wBins / xScale());

            double yBins = hPixels / hic.getScaleFactor();
            int h = (int) (yBins / yScale());

            if (w < 4) {
                int delta = 4 - w;
                x -= delta / 2;
                w = 4;
            }
            if (h < 4) {
                int delta = 4 - h;
                y -= delta / 2;
                h = 4;
            }

            innerRectangle = new Rectangle(x, y, w, h);
            Shape shape = new SquareDonut(outerRectangle, innerRectangle);

            g.setColor(Color.gray);
            g.setComposite(ALPHA_COMP);
            g.fill(shape);

            g.draw(innerRectangle);
        }
    }

    private static class SquareDonut implements Shape {
        private final Area area;

        public SquareDonut(Rectangle outerRectangle, Rectangle innerRectangle) {
            this.area = new Area(outerRectangle);
            area.subtract(new Area(innerRectangle));
        }

        public Rectangle getBounds() {
            return area.getBounds();
        }

        public Rectangle2D getBounds2D() {
            return area.getBounds2D();
        }

        public boolean contains(double v, double v1) {
            return area.contains(v, v1);
        }

        public boolean contains(Point2D point2D) {
            return area.contains(point2D);
        }

        public boolean intersects(double v, double v1, double v2, double v3) {
            return area.intersects(v, v1, v2, v3);
        }

        public boolean intersects(Rectangle2D rectangle2D) {
            return area.intersects(rectangle2D);
        }

        public boolean contains(double v, double v1, double v2, double v3) {
            return area.contains(v, v1, v2, v3);
        }

        public boolean contains(Rectangle2D rectangle2D) {
            return area.contains(rectangle2D);
        }

        public PathIterator getPathIterator(AffineTransform affineTransform) {
            return area.getPathIterator(affineTransform);
        }

        public PathIterator getPathIterator(AffineTransform affineTransform, double v) {
            return area.getPathIterator(affineTransform, v);
        }
    }
}
