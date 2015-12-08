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
import juicebox.HiCGlobals;
import juicebox.gui.SuperAdapter;
import org.broad.igv.util.Pair;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;
import java.util.Collection;

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
        trackRectangles = new ArrayList<Pair<Rectangle, HiCTrack>>();
        //setBackground(new Color(238, 238, 238));
        setBackground(Color.white);
        addMouseAdapter(superAdapter);

        //setToolTipText("");   // Has side affect of turning on tt text
    }

    public void removeTrack(HiCTrack track) {
        hic.removeTrack(track);
        superAdapter.revalidate();
        //this.revalidate();
        superAdapter.repaint();
    }

    private void addMouseAdapter(final SuperAdapter superAdapter) {
        MouseAdapter mouseAdapter = new MouseAdapter() {

            @Override
            public void mouseMoved(MouseEvent e) {
                TrackPanel.this.superAdapter.updateToolTipText(tooltipText(e.getX(), e.getY()));
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
                        JPopupMenu menu = track.getPopupMenu(TrackPanel.this, superAdapter);
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
        Graphics2D graphics = (Graphics2D) g;
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        AffineTransform t = graphics.getTransform();
        if (orientation == Orientation.Y) {
            AffineTransform rotateTransform = new AffineTransform();
            rotateTransform.quadrantRotate(1);
            rotateTransform.scale(1, -1);
            graphics.transform(rotateTransform);
        }

        trackRectangles.clear();
        java.util.List<HiCTrack> tracks = new ArrayList<HiCTrack>(hic.getLoadedTracks());
        if (tracks.isEmpty()) {
            return;
        }


        Rectangle rect = getBounds();
        graphics.setColor(getBackground());
        graphics.fillRect(rect.x, rect.y, rect.width, rect.height);

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

                        hicTrack.render(graphics, getContext(), trackRectangle, orientation, gridAxis);
                        y += h;

                        trackRectangles.add(new Pair<Rectangle, HiCTrack>(trackRectangle, hicTrack));
                    }


                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        graphics.setTransform(t);
        Point cursorPoint = hic.getCursorPoint();
        if (cursorPoint != null) {
            graphics.setColor(HiCGlobals.RULER_LINE_COLOR);
            if (orientation == Orientation.X) {
                graphics.drawLine(cursorPoint.x, 0, cursorPoint.x, getHeight());
            } else {
                graphics.drawLine(0, cursorPoint.y, getWidth(), cursorPoint.y);
            }
        }

    }

    private Context getContext() {
        return orientation == Orientation.X ? hic.getXContext() : hic.getYContext();
    }

    private String tooltipText(int mx, int my) {

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
        return null;
    }


//    @Override
//    public String getToolTipText(MouseEvent event) {
//        return tooltipText(event.getX(), event.getY());
//    }

    public enum Orientation {X, Y}
}
