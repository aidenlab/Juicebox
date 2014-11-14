package juicebox.track;

import juicebox.Context;
import juicebox.HiC;
import org.broad.igv.renderer.GraphicUtils;
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

        int y = rect.y;

        java.util.List<HiCTrack> tracks = new ArrayList<HiCTrack>(hic.getLoadedTracks());
        if (tracks.isEmpty() && eigenvectorTrack == null) {
            return;
        }

        for (HiCTrack hicTrack : tracks) {
            if (hicTrack.getHeight() > 0) {
                int h = hicTrack.getHeight();

                // write track name in upper left hand corner
                graphics.setFont(new Font("default", Font.BOLD, 10));
                graphics.setColor(Color.black);
                GraphicUtils.drawRightJustifiedText(hicTrack.getName(), rect.width - 10, y + 20, graphics);

                Context context = hic.getXContext();
                if (context != null) {
                    y += h;
                }
            }
        }
    }
}
