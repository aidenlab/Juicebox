package juicebox.track;

import juicebox.Context;
import juicebox.HiC;

import javax.swing.*;
import java.awt.*;
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

        int rectBottom = rect.y + rect.height;
        int y = rect.y;

        HiCGridAxis gridAxis = hic.getZd().getXGridAxis();
        java.util.List<HiCTrack> tracks = new ArrayList<HiCTrack>(hic.getLoadedTracks());
        if ((tracks == null || tracks.isEmpty()) && eigenvectorTrack == null) {
            return;
        }

        for (HiCTrack hicTrack : tracks) {
            if (hicTrack.getHeight() > 0) {
                int h = hicTrack.getHeight();

                Rectangle trackRectangle;
                trackRectangle = new Rectangle(rect.x, y, rect.width, h);

                graphics.drawString(hicTrack.getName(), 10, y);

                Context context = hic.getXContext();
                if (context != null) {
                    y += h;
                }


            }
        }
        if (eigenvectorTrack != null) {
            int h = rectBottom - y;
            Rectangle trackRectangle;
            trackRectangle = new Rectangle(rect.x, y, rect.width, h);
            graphics.drawString("Eigenvector", 10, y);
        }
    }

    public void setEigenvectorTrack(HiCTrack eigenvectorTrack) {
        this.eigenvectorTrack = eigenvectorTrack;
    }
}
