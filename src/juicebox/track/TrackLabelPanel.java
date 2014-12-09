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

//import juicebox.Context;

import juicebox.HiC;
import org.broad.igv.ui.FontManager;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;

//import java.util.HashMap;
//import java.util.Map;
//import org.broad.igv.renderer.GraphicUtils;
//import java.awt.geom.Rectangle2D;

/**
 * @author jrobinso
 *         Date: 8/3/13
 *         Time: 9:36 PM
 */
public class TrackLabelPanel extends JPanel {

    private static final long serialVersionUID = 1627813915602134471L;
    private final HiC hic;
    private HiCTrack eigenvectorTrack;
    private int numExtraBufferLinesSpaces = 2;

    public TrackLabelPanel(HiC hic) {
        this.hic = hic;
        setLayout(new GridLayout(0, 1));
    }

    public void updateLabels() {

        removeAll();

        if (hic.getDataset() == null) {
            return;
        }

        java.util.List<HiCTrack> tracks = new ArrayList<HiCTrack>(hic.getLoadedTracks());
        if (tracks.isEmpty() && eigenvectorTrack == null) {
            return;
        }

        String multiLineText = "";

        for (HiCTrack hicTrack : tracks) {
            multiLineText += hicTrack.getName() + "<br><br>";
        }

        multiLineText = "<html>" + multiLineText + "</html>";
        //System.out.println(multiLineText);

        JLabel textLabel = getTrackLabel(multiLineText, false);
        add(textLabel);

    }

    private JLabel getTrackLabel(String name, boolean addToolTip) {
        JLabel label = new JLabel(name, SwingConstants.RIGHT);
        label.setVerticalAlignment(SwingConstants.TOP);
        label.setFont(FontManager.getFont(Font.BOLD, 10));
        if (addToolTip)
            label.setToolTipText(name);
        return label;
    }
}
