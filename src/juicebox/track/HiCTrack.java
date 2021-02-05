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
import juicebox.gui.SuperAdapter;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;


/**
 * @author jrobinso
 *         Date: 9/10/12
 *         Time: 3:15 PM
 */
public abstract class HiCTrack{

    private static int height = 25;
    private final ResourceLocator locator;
    private Color posColor = Color.blue.darker();
    private Color negColor = Color.red.darker();

    HiCTrack(ResourceLocator locator) {
        this.locator = locator;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        juicebox.track.HiCTrack.height = height;
    }

    public ResourceLocator getLocator() {
        return locator;
    }

    public void mouseClicked(int x, int y, Context context, TrackPanel.Orientation orientation) {
        // Ignore by default, override in subclasses
    }

    public JPopupMenu getPopupMenu(final TrackPanel trackPanel, final SuperAdapter superAdapter, TrackPanel.Orientation orientation) {
        JPopupMenu menu = new JPopupMenu(getName());

        JMenuItem menuItem = new JMenuItem("Remove track");
        menuItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                trackPanel.removeTrack(HiCTrack.this);
            }
        });
        menu.add(menuItem);

        JMenuItem menuItem2 = new JMenuItem("Move up...");
        menuItem2.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                trackPanel.moveTrackUp(HiCTrack.this);
            }
        });

        //if track is on the top don't add to the menu
        if (trackPanel.getTrackList().indexOf(HiCTrack.this) != 0) {
            menu.add(menuItem2);
        }

        JMenuItem menuItem3 = new JMenuItem("Move down...");
        menuItem3.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                trackPanel.moveTrackDown(HiCTrack.this);
            }
        });

        //if track is on the bottom don't add to the menu
        if (trackPanel.getTrackList().indexOf(HiCTrack.this) != trackPanel.getTrackList().size() - 1) {
            menu.add(menuItem3);
        }

        return menu;
    }

    public abstract String getName();

    public abstract void setName(String text);

    public Color getPosColor() {
        return posColor;
    }

    public void setPosColor(Color posColor) {
        this.posColor = posColor;
    }

    public Color getNegColor() {
        return negColor;
    }

    public void setNegColor(Color negColor) {
        this.negColor = negColor;
    }

    public abstract void render(Graphics g2d,
                                Context context,
                                Rectangle trackRectangle,
                                TrackPanel.Orientation orientation,
                                HiCGridAxis gridAxis);

    public abstract String getToolTipText(int x, int y, TrackPanel.Orientation orientation);
}
