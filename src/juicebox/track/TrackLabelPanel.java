package juicebox.track;

import juicebox.Context;
import juicebox.HiC;
import org.broad.igv.renderer.GraphicUtils;
import org.broad.igv.ui.FontManager;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;

/**
 * @author jrobinso
 *         Date: 8/3/13
 *         Time: 9:36 PM
 */
public class TrackLabelPanel extends JComponent {

    HiC hic;
    HiCTrack eigenvectorTrack;

    public TrackLabelPanel(HiC hic) {
        this.hic = hic;
    }




    @Override
    protected void paintComponent(Graphics g) {

        if(hic.getDataset() == null) return;

        super.paintComponent(g);

        Graphics2D graphics = (Graphics2D) g;

        Rectangle rect = getBounds();

        graphics.setColor(getBackground());
        graphics.fillRect(rect.x, rect.y, rect.width, rect.height);

        int y = rect.y;

        java.util.List<HiCTrack> tracks = new ArrayList<HiCTrack>(hic.getLoadedTracks());
        if (tracks.isEmpty() && eigenvectorTrack == null) {
            return;
        }

        for (HiCTrack hicTrack : tracks) {
            if (hicTrack.getHeight() > 0) {
                int h = hicTrack.getHeight();

                // write track name in upper left hand corner
                graphics.setFont(FontManager.getFont(Font.BOLD, 8));
                graphics.setColor(Color.black);
                graphics.drawString(hicTrack.getName(), rect.x + 10, y + 20);
                //GraphicUtils.drawRightJustifiedText(hicTrack.getName(), rect.width - 10, y + 20, graphics);
                verifyTextWillFitInPanel(hicTrack.getName(), graphics);

                Context context = hic.getXContext();
                if (context != null) {
                    y += h;
                }
            }
        }
    }

    private void verifyTextWillFitInPanel(String name, Graphics2D graphics){

        FontMetrics fontMetrics = graphics.getFontMetrics();
        Rectangle2D textBounds = fontMetrics.getStringBounds(name, graphics);
        int labelTextSize = (int) textBounds.getWidth();

        if(this.getSize().width < labelTextSize)
            this.setSize(labelTextSize, this.getSize().height);
    }
}
