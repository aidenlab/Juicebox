/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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
import juicebox.HiCGlobals;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.gui.SuperAdapter;
import juicebox.track.feature.Feature2D;
import org.broad.igv.util.Pair;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: neva
 * Date: 4/3/12
 * Time: 4:08 PM
 * To change this template use File | Settings | File Templates.
 */
public class TrackPanel extends JPanel {

    private static final long serialVersionUID = -1195744055137430563L;
    //private MouseAdapter mouseAdapter;
    private final HiC hic;
    private final Orientation orientation;
    // HiCTrack eigenvectorTrack;
    private final Collection<Pair<Rectangle, HiCTrack>> trackRectangles;
    private final SuperAdapter superAdapter;

    public TrackPanel(SuperAdapter superAdapter, HiC hiC, Orientation orientation) {
        this.superAdapter = superAdapter;
        this.hic = hiC;
        this.orientation = orientation;
        setAutoscrolls(true);
        trackRectangles = new ArrayList<>();
        if (HiCGlobals.isDarkulaModeEnabled) {
            setBackground(Color.black);
        } else {
            setBackground(Color.white);
        }
        addMouseAdapter(superAdapter);

        //setToolTipText("");   // Has side affect of turning on tt text
    }

    public void removeTrack(HiCTrack track) {
        hic.removeTrack(track);
        superAdapter.revalidate();
        superAdapter.repaint();
        superAdapter.getLayersPanel().redraw1DLayerPanels(superAdapter);
    }

    public void moveTrackUp(HiCTrack track) {
        // Move up the track in the array by 1.
        hic.moveTrack(track, true);
        superAdapter.revalidate();
        superAdapter.repaint();
    }

    public void moveTrackDown(HiCTrack track) {
        // Move down the track in the array by 1.
        hic.moveTrack(track, false);
        superAdapter.revalidate();
        superAdapter.repaint();

    }

    public List<HiCTrack> getTrackList() {
        return hic.getLoadedTracks();
    }

    private void addMouseAdapter(final SuperAdapter superAdapter) {
        MouseAdapter mouseAdapter = new MouseAdapter() {

            @Override
            public void mouseMoved(MouseEvent e) {
                TrackPanel.this.superAdapter.updateMainViewPanelToolTipText(tooltipText(e.getX(), e.getY(), true));
            }

            @Override
            public void mouseReleased(MouseEvent mouseEvent) {
                if (mouseEvent.isPopupTrigger()) {
                    handlePopupEvent(mouseEvent);
                }
            }

            @Override
            public void mouseClicked(MouseEvent mouseEvent) {

                Context context = orientation == Orientation.X ? hic.getXContext() : hic.getYContext();

                if (mouseEvent.isPopupTrigger()) {
                    handlePopupEvent(mouseEvent);
                } else if (mouseEvent.getClickCount() > 1) {

                    int x = mouseEvent.getX();
                    int y = mouseEvent.getY();
                    if (orientation == Orientation.Y) {
                        y = mouseEvent.getX();
                        x = mouseEvent.getY();

                    }
                    for (Pair<Rectangle, HiCTrack> p : trackRectangles) {
                        Rectangle r = p.getFirst();
                        if (y >= r.y && y < r.y + r.height) {
                            HiCTrack track = p.getSecond();
                            track.mouseClicked(x, y, context, orientation);
                        }
                    }
                }

            }

            @Override
            public void mousePressed(MouseEvent mouseEvent) {
                if (mouseEvent.isPopupTrigger()) {
                    handlePopupEvent(mouseEvent);
                }
            }

            private void handlePopupEvent(MouseEvent mouseEvent) {
                for (Pair<Rectangle, HiCTrack> p : trackRectangles) {
                    Rectangle r = p.getFirst();
                    if (r.contains(mouseEvent.getPoint())) {

                        HiCTrack track = p.getSecond();
                        JPopupMenu menu = track.getPopupMenu(TrackPanel.this, superAdapter, orientation);
                        menu.show(mouseEvent.getComponent(), mouseEvent.getX(), mouseEvent.getY());
                        repaint();

//                        Collection<Track> selectedTracks = Arrays.asList(p.getSecond());
//                        TrackClickEvent te = new TrackClickEvent(mouseEvent, null);
//                        IGVPopupMenu menu = TrackMenuUtils.getPopupMenu(selectedTracks, "", te);
//                        menu.show(mouseEvent.getComponent(), mouseEvent.getX(), mouseEvent.getY());
                    }
                }
            }

        };

        this.addMouseListener(mouseAdapter);
        this.addMouseMotionListener(mouseAdapter);
    }

    /**
     * Returns the current height of this component.
     * This method is preferable to writing
     * <code>component.getBounds().height</code>, or
     * <code>component.getSize().height</code> because it doesn't cause any
     * heap allocations.
     *
     * @return the current height of this component
     */
    @Override
    public int getHeight() {
        if (orientation == Orientation.X) {
            int h = 0;
            for (HiCTrack t : hic.getLoadedTracks()) {
                h += t.getHeight();
            }

            return h;
        } else {
            return super.getHeight();
        }
    }

    @Override
    public int getWidth() {
        if (orientation == Orientation.Y) {
            int h = 0;
            for (HiCTrack t : hic.getLoadedTracks()) {
                h += t.getHeight();
            }

            return h;
        } else {
            return super.getWidth();
        }
    }

    /**
     * If the <code>preferredSize</code> has been set to a
     * non-<code>null</code> value just returns it.
     * If the UI delegate's <code>getPreferredSize</code>
     * method returns a non <code>null</code> value then return that;
     * otherwise defer to the component's layout manager.
     *
     * @return the value of the <code>preferredSize</code> property
     * @see #setPreferredSize
     * @see javax.swing.plaf.ComponentUI
     */
    @Override
    public Dimension getPreferredSize() {
        return new Dimension(getWidth(), getHeight());
    }

    protected void paintComponent(Graphics g) {

        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        AffineTransform originalTransform = g2d.getTransform();
        if (orientation == Orientation.Y) {
            AffineTransform rotateTransform = new AffineTransform();
            rotateTransform.quadrantRotate(1);
            rotateTransform.scale(1, -1);
            g2d.transform(rotateTransform);
        }

        trackRectangles.clear();
        java.util.List<HiCTrack> tracks = new ArrayList<>(hic.getLoadedTracks());
        if (tracks.isEmpty()) {
            return;
        }


        Rectangle rect = getBounds();
        g.setColor(getBackground());
        g.fillRect(rect.x, rect.y, rect.width, rect.height);

        //int rectBottom = orientation == Orientation.X ? rect.y + rect.height : rect.x + rect.width;
        int y = orientation == Orientation.X ? rect.y : rect.x;

        try {
            HiCGridAxis gridAxis = orientation == Orientation.X ? hic.getZd().getXGridAxis() : hic.getZd().getYGridAxis();

            for (HiCTrack hicTrack : tracks) {
                if (hicTrack.getHeight() > 0) {
                    int h = hicTrack.getHeight();

                    Rectangle trackRectangle;
                    if (orientation == Orientation.X) {
                        trackRectangle = new Rectangle(rect.x, y, rect.width, h);
                    } else {
                        //noinspection SuspiciousNameCombination
                        trackRectangle = new Rectangle(y, rect.y, h, rect.height);
                    }

                    if (getContext() != null) {

                        hicTrack.render(g, getContext(), trackRectangle, orientation, gridAxis);
                        y += h;

                        trackRectangles.add(new Pair<>(trackRectangle, hicTrack));
                    }


                }
            }
        } catch (Exception e) {
            if (HiCGlobals.printVerboseComments)
                e.printStackTrace();
        }

        g2d.setTransform(originalTransform);
        Point cursorPoint = hic.getCursorPoint();
        if (cursorPoint == null) {
            cursorPoint = hic.getDiagonalCursorPoint();
        }

        if (cursorPoint != null) {
            if (HiCGlobals.isDarkulaModeEnabled) {
                g.setColor(HiCGlobals.DARKULA_RULER_LINE_COLOR);
            } else {
                g.setColor(HiCGlobals.RULER_LINE_COLOR);
            }
            if (orientation == Orientation.X) {
                g.drawLine(cursorPoint.x, 0, cursorPoint.x, getHeight());
            } else {
                g.drawLine(0, cursorPoint.y, getWidth(), cursorPoint.y);
            }
        }

        try {
            Feature2D highlight = hic.getHighlightedFeature();
            if (highlight != null) {
                g.setColor(HiCGlobals.HIGHLIGHT_COLOR);
                MatrixZoomData zd = hic.getZd();
                HiCGridAxis xAxis = zd.getXGridAxis();
                HiCGridAxis yAxis = zd.getYGridAxis();
                double binOriginX = hic.getXContext().getBinOrigin();
                double binOriginY = hic.getYContext().getBinOrigin();
                int binStart1 = xAxis.getBinNumberForGenomicPosition(highlight.getStart1());
                int binEnd1 = xAxis.getBinNumberForGenomicPosition(highlight.getEnd1());
                int binStart2 = yAxis.getBinNumberForGenomicPosition(highlight.getStart2());
                int binEnd2 = yAxis.getBinNumberForGenomicPosition(highlight.getEnd2());
                double scaleFactor = hic.getScaleFactor();

                if (orientation == Orientation.X) {
                    if (HiCFileTools.equivalentChromosome(highlight.getChr1(), zd.getChr1())) {
                        int x3 = (int) ((binStart1 - binOriginX) * scaleFactor);
                        int h3 = (int) Math.max(1, scaleFactor * (binEnd1 - binStart1));

                        g.drawLine(x3, 0, x3, getHeight());
                        g.drawLine(x3 + h3, 0, x3 + h3, getHeight());
                    }
                } else if (HiCFileTools.equivalentChromosome(highlight.getChr2(), zd.getChr2())) {
                    int y3 = (int) ((binStart2 - binOriginY) * scaleFactor);
                    int w3 = (int) Math.max(1, scaleFactor * (binEnd2 - binStart2));

                    g.drawLine(0, y3, getWidth(), y3);
                    g.drawLine(0, y3 + w3, getWidth(), y3 + w3);
                }
            }
        } catch (Exception e2) {
            //
        }
    }

    private Context getContext() {
        return orientation == Orientation.X ? hic.getXContext() : hic.getYContext();
    }

    public String tooltipText(int mx, int my, boolean isMouseDirectlyOnTrack) {

        if (isMouseDirectlyOnTrack) {
            int x = mx;
            int y = my;
            if (orientation == Orientation.Y) {
                y = mx;
                x = my;
            }
            for (Pair<Rectangle, HiCTrack> p : trackRectangles) {
                Rectangle r = p.getFirst();
                if (r.contains(mx, my)) {
                    return p.getSecond().getToolTipText(x, y, orientation);
                }
            }
        } else {
            StringBuilder toolTipText = new StringBuilder();

            if (orientation == Orientation.X) {
                for (Pair<Rectangle, HiCTrack> p : trackRectangles) {
                    Rectangle r = p.getFirst();
                    int y = r.y + r.height / 2;
                    if (r.contains(mx, y)) {
                        String tempText = p.getSecond().getToolTipText(mx, y, orientation);
                        if (tempText.length() > 0) toolTipText.append("<br>").append(tempText);
                    }
                }
            } else {
                for (Pair<Rectangle, HiCTrack> p : trackRectangles) {
                    Rectangle r = p.getFirst();
                    int x = r.x + r.width / 2;
                    if (r.contains(x, my)) {
                        String tempText = p.getSecond().getToolTipText(my, x, orientation);
                        if (tempText.length() > 0) toolTipText.append("<br>").append(tempText);
                    }
                }
            }
            if (toolTipText.length() > 5) return "<br>" + toolTipText;
        }
        return null;
    }


//    @Override
//    public String getToolTipText(MouseEvent event) {
//        return tooltipText(event.getX(), event.getY());
//    }

    public enum Orientation {X, Y}
}
