/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2014 Broad Institute, Aiden Lab
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
        int trackCount = tracks.size() + (eigenvectorTrack == null ? 0 : 1);
        if (trackCount < 2) {
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
