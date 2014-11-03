package juicebox.track;

import juicebox.Context;
import juicebox.MainWindow;
import org.broad.igv.renderer.DataRange;
import org.broad.igv.track.RenderContext;
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
public abstract class HiCTrack {

    private int height = 25;
    protected ResourceLocator locator;

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

    public abstract Color getPosColor();

    public abstract void render(Graphics2D g2d,
                                Context context,
                                Rectangle trackRectangle,
                                TrackPanel.Orientation orientation,
                                HiCGridAxis gridAxis);

    public abstract String getToolTipText(int x, int y, TrackPanel.Orientation orientation);

    public abstract void setName(String text);

    public abstract void setColor(Color selectedColor);

    public abstract void setAltColor(Color selectedColor);
}
