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
import juicebox.MainWindow;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.ui.FontManager;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.*;
import java.io.Serializable;


/**
 * @author Jay Jung Hee Ryu
 * @date 5/26/16.
 */

// extends to JScrollPane or ScrollBar//
// Load chromosome figure shape from GapSizes from tool.

public class HiCChromosomeFigPanel extends JComponent implements Serializable {

    private static final long serialVersionUID = 123798L;
    private final Font spanFont = FontManager.getFont(Font.BOLD, 12);
    private HiC hic;
    private Orientation orientation;
    private Context context;
    private Point lastPoint = null;
    private int chrFigStart = 0;
    private int chrFigEnd = 0;


    /**
     * Empty constructor for form builder
     */
    private HiCChromosomeFigPanel() {
    }

    public HiCChromosomeFigPanel(final HiC hic) {
        this.hic = hic;
        setBackground(Color.WHITE);

        addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent mouseEvent) {
                if (mouseEvent.getClickCount() >= 1) {
                    int dxBP;
                    int dyBP;
                    Point2D.Double scale = isHorizontal() ? getHiCScale(HiCChromosomeFigPanel.this.getWidth(), HiCChromosomeFigPanel.this.getHeight()) :
                            getHiCScale(HiCChromosomeFigPanel.this.getHeight(), HiCChromosomeFigPanel.this.getWidth());
                    if (isHorizontal()) {
                        try {
                            dxBP = (int) ((mouseEvent.getX() - (chrFigStart + chrFigEnd) / 2) * scale.getX());
                            dyBP = 0;
                            hic.moveBy(dxBP, dyBP);
                        } catch (Exception e) {
                            System.out.println("Error when region clicked");
                            e.printStackTrace();
                        }
                    } else {
                        try {
                            dxBP = 0;
                            dyBP = (int) ((mouseEvent.getY() + (chrFigStart + chrFigEnd) / 2) * scale.getX());
                            hic.moveBy(dxBP, dyBP);
                        } catch (Exception e) {
                            System.out.println("Error when region clicked");
                            e.printStackTrace();
                        }
                    }
                }
            }

            @Override
            public void mousePressed(MouseEvent e) {
                if (isHorizontal()) {
                    if (e.getX() >= chrFigStart && e.getX() <= chrFigEnd) {
                        lastPoint = e.getPoint();
                        setCursor(MainWindow.fistCursor);
                    }
                } else {
                    if (e.getY() >= -chrFigStart && e.getY() <= -chrFigEnd) {
                        lastPoint = e.getPoint();
                        setCursor(MainWindow.fistCursor);
                    }
                }
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                lastPoint = null;
                setCursor(Cursor.getDefaultCursor());
            }
        });

        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (lastPoint != null) {
                    int dxBP;
                    int dyBP;
                    Point2D.Double scale = isHorizontal() ? getHiCScale(HiCChromosomeFigPanel.this.getWidth(), HiCChromosomeFigPanel.this.getHeight()) :
                            getHiCScale(HiCChromosomeFigPanel.this.getHeight(), HiCChromosomeFigPanel.this.getWidth());
                    if (isHorizontal()) {
                        dxBP = ((int) ((e.getX() - lastPoint.x) * scale.getX()));
                        dyBP = 0;
                    } else {
                        dxBP = 0;
                        dyBP = (int) ((e.getY() - lastPoint.y) * scale.getX());
                    }
                    hic.moveBy(dxBP, dyBP);
                    lastPoint = e.getPoint();
                }
            }
        });

        addMouseWheelListener(new MouseWheelListener() {
            @Override
            public void mouseWheelMoved(MouseWheelEvent e) {
                int scroll = e.getWheelRotation();
                int dxBP;
                int dyBP;
                if (isHorizontal()) {
                    dxBP = scroll;
                    dyBP = 0;
                } else {
                    dxBP = 0;
                    dyBP = scroll;
                }
                hic.moveBy(dxBP, dyBP);
            }
        });

        //Chromosome change pop-up menu
    }

    public Point2D.Double getHiCScale(int width, int height) {
        try {
            return new Point2D.Double((double) hic.getZd().getXGridAxis().getBinCount() / width,
                    (double) hic.getZd().getYGridAxis().getBinCount() / height);
        } catch (Exception e) {
            return null;
        }
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
        drawChr(g2D);
        g2D.setTransform(t);

    }

    private void drawChr(Graphics2D g) {
        int w = isHorizontal() ? getWidth() : getHeight();
        int h = isHorizontal() ? getHeight() : getWidth();

        g.setFont(spanFont);

        Chromosome chromosome = context.getChromosome();

        if (chromosome != null) {
            if (!chromosome.getName().equals("All")) {
                //Draw Chromosome Name
                String rangeString = chromosome.getName();

                int strWidth = g.getFontMetrics().stringWidth(rangeString);
                int strPosition = (w - strWidth) / 2;

                if (!isHorizontal()) strPosition = -strPosition;
                int vPos = h / 2 + 3;
                //Draw Chromosome Figure
                if (isHorizontal()) {
                    drawRegion(g, w, h);
                } else {
                    drawRegion(g, w, h);
                }

                Rectangle r = new Rectangle(strPosition, h / 4, strWidth, h / 2);
                g.setClip(r);
                g.setColor(Color.BLACK);
                g.drawString(rangeString, strPosition, vPos);
            }
        }
    }

    private int genomeLength() {
        return context.getChromosome().getLength();
    }

    private int[] genomePositions() {
        return hic.getCurrentRegionWindowGenomicPositions();
    }

    private void drawRegion(Graphics2D g, int w, int h) {
        Color chrContour = new Color(116, 173, 212);
        Color chrFillIn = new Color(163, 202, 187);
        Color chrInside = new Color(222, 222, 222);

        int genomeLength = genomeLength();

        int[] genomePositions;
        try {
            genomePositions = genomePositions();
        } catch (Exception e) {
            return;
        }

        float chrFigLength = w - 2;

        if (isHorizontal()) {
            chrFigStart = Math.round((chrFigLength * genomePositions[0]) / genomeLength + 1);
            chrFigEnd = genomePositions[1] > genomeLength ? w - 1 : Math.round(chrFigLength * genomePositions[1] / genomeLength) + 1;
            int chrFigRegion = chrFigEnd - chrFigStart;
            g.setColor(chrContour);
            g.drawRoundRect(1, h / 4, w - 2, h / 2, h / 2, h / 2);
            // Lines
            g.drawLine(chrFigStart, h / 2, chrFigStart, h / 4 - 3);
            g.drawLine(0, 0, 0, 3);
            g.drawLine(chrFigStart, h / 4 - 3, 0, 3);

            g.drawLine(chrFigEnd, h / 2, chrFigEnd, h / 4 - 3);
            g.drawLine(w - 1, 0, w - 1, 3);
            g.drawLine(chrFigEnd, h / 4 - 3, w - 1, 3);

            // Later implement shape to create a chromosome shape
            RoundRectangle2D chrFig = new RoundRectangle2D.Double(1, h / 4, w - 2, h / 2, h / 2, h / 2);
            g.setClip(chrFig);
            g.setColor(chrInside);
            g.fillRect(1, h / 4, w - 2, h / 2);

            // Chromosome region
            Rectangle region = new Rectangle(chrFigStart, 0, chrFigRegion, h);
            g.clip(region);
            g.setColor(chrFillIn);
            g.fillRect(chrFigStart, 0, chrFigRegion, h);

        } else {
            chrFigStart = -Math.round((chrFigLength * genomePositions[2]) / genomeLength) - 1;
            chrFigEnd = genomePositions[3] > genomeLength ? -w + 1 : -Math.round(chrFigLength * genomePositions[3] / genomeLength) - 1;
            int chrFigRegion = -chrFigEnd + chrFigStart;
            g.setColor(chrContour);
            g.drawRoundRect(-w + 1, h / 4, w - 2, h / 2, h / 2, h / 2);
            // lines
            g.setColor(chrContour);
            g.drawLine(chrFigStart, h / 2, chrFigStart, h / 4 - 3);
            g.drawLine(0, 0, 0, 3);
            g.drawLine(chrFigStart, h / 4 - 3, 0, 3);

            g.drawLine(chrFigEnd, h / 2, chrFigEnd, h / 4 - 3);
            g.drawLine(-w + 1, 0, -w + 1, 3);
            g.drawLine(chrFigEnd, h / 4 - 3, -w + 1, 3);

            // Later implement shape to create a chromosome shape
            RoundRectangle2D chrFig = new RoundRectangle2D.Double(-w + 1, h / 4, w - 2, h / 2, h / 2, h / 2);
            g.setClip(chrFig);
            g.setColor(chrInside);
            g.fillRect(-w + 1, h / 4, w - 2, h / 2);

            // Chromosome region
            Rectangle region = new Rectangle(chrFigEnd, 0, chrFigRegion, h);
            g.clip(region);
            g.setColor(chrFillIn);
            g.fillRect(chrFigEnd, 0, chrFigRegion, h);
        }
    }

    private boolean isHorizontal() {
        return orientation == Orientation.HORIZONTAL;
    }

    public enum Orientation {HORIZONTAL, VERTICAL}

    private static class Realchromosome implements Shape {
        //Two round rectangle sticked together
        // a * (w-2) width & (1-a) * (w-2) width round rect
        // 0 as a default (no centromere)

        private final Area area;

        public Realchromosome(Rectangle outerRectangle, Rectangle innerRectangle) {
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

