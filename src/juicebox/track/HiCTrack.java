/*
 * Copyright (C) 2011-2014 Aiden Lab - All Rights Reserved
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Aiden Lab All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. Aiden Lab is not responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
 */

package juicebox.track;

import juicebox.Context;
import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

//import juicebox.MainWindow;
//import org.broad.igv.renderer.DataRange;
//import org.broad.igv.track.RenderContext;

/**
 * @author jrobinso
 *         Date: 9/10/12
 *         Time: 3:15 PM
 */
public abstract class HiCTrack {

    private static final int height = 25;
    private final ResourceLocator locator;

    public HiCTrack(ResourceLocator locator) {
        this.locator = locator;
    }

    public int getHeight() {
        return height;
    }

    public ResourceLocator getLocator() {
        return locator;
    }


    public void mouseClicked(int x, int y, Context context, TrackPanel.Orientation orientation) {
        // Ignore by default, override in subclasses
    }

    public JPopupMenu getPopupMenu(final TrackPanel trackPanel) {
        JPopupMenu menu = new JPopupMenu(getName());

        JMenuItem menuItem = new JMenuItem("Remove track");
        menuItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                trackPanel.removeTrack(HiCTrack.this);
            }
        });
        menu.add(menuItem);

        return menu;
    }

    public abstract String getName();

    public abstract void setName(String text);

    public abstract Color getPosColor();

    public abstract void render(Graphics2D g2d,
                                Context context,
                                Rectangle trackRectangle,
                                TrackPanel.Orientation orientation,
                                HiCGridAxis gridAxis);

    public abstract String getToolTipText(int x, int y, TrackPanel.Orientation orientation);

    public abstract void setColor(Color selectedColor);

    public abstract void setAltColor(Color selectedColor);
}
