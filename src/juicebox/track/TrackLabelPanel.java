package juicebox.track;

//import juicebox.Context;
import juicebox.HiC;

//import java.util.HashMap;
//import java.util.Map;
//import org.broad.igv.renderer.GraphicUtils;
import org.broad.igv.ui.FontManager;

import javax.swing.*;
import java.awt.*;
//import java.awt.geom.Rectangle2D;
import java.util.ArrayList;

/**
 * @author jrobinso
 *         Date: 8/3/13
 *         Time: 9:36 PM
 */
public class TrackLabelPanel extends JPanel {

    private static final long serialVersionUID = 1627813915602134471L;
    final HiC hic;
    HiCTrack eigenvectorTrack;
    private int numExtraBufferLinesSpaces = 2;
    //private java.util.List<HiCTrack> tracks;
    JLabel textLabel;

    String multiLineText = "";

    public TrackLabelPanel(HiC hic) {
        this.hic = hic;
        setLayout(new GridLayout(0, 1));
    }

    public void updateLabels(){

        removeAll();

        if(hic.getDataset() == null){
            return;
        }

        java.util.List<HiCTrack> tracks = new ArrayList<HiCTrack>(hic.getLoadedTracks());
        if (tracks.isEmpty() && eigenvectorTrack == null) {
            return;
        }

        multiLineText="";

        for (HiCTrack hicTrack : tracks) {
            multiLineText += hicTrack.getName() + "<br><br>";
        }

        multiLineText = "<html>"+multiLineText+"</html>";
        //System.out.println(multiLineText);

        textLabel = getTrackLabel(multiLineText, false);
        add(textLabel);

    }

    private JLabel getTrackLabel(String name, boolean addToolTip){
        JLabel label = new JLabel(name, SwingConstants.RIGHT);
        label.setVerticalAlignment(SwingConstants.TOP);
        label.setFont(FontManager.getFont(Font.BOLD, 10));
        if(addToolTip)
            label.setToolTipText(name);
        return label;
    }
}
