/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

//import org.broad.igv.track.Track;

import org.broad.igv.util.ResourceLocator;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.*;
import java.util.List;

/**
 * @author Jim Robinson
 * @date 5/8/12
 */
class HiCLoadDialog extends JDialog {

    private static final long serialVersionUID = -7529973146086845915L;
    private final Collection<String> selectedTracks = new HashSet<String>();
    private boolean canceled = false;


    public HiCLoadDialog(Frame parent, Map<String, List<ResourceLocator>> locators, List<HiCTrack> tracks) {
        super(parent);
        initComponents(locators, tracks);
        setModal(true);
        this.setSize(750, 800);
    }

    private void initComponents(Map<String, List<ResourceLocator>> locators, List<HiCTrack> tracks) {

        Container contentPane = getContentPane();
        contentPane.setLayout(new BorderLayout());


        //======== dialogPane ========
        JPanel dialogPane = new JPanel();
        dialogPane.setBorder(new EmptyBorder(12, 12, 12, 12));
        dialogPane.setLayout(new BorderLayout());

        final Box mainPanel = new Box(BoxLayout.Y_AXIS);
        mainPanel.setAlignmentX(LEFT_ALIGNMENT);

        Set<String> loadedTrackNames = new HashSet<String>(tracks.size());
        for (HiCTrack t : tracks) {
            loadedTrackNames.add(t.getName());
        }

        for (Map.Entry<String, List<ResourceLocator>> entry : locators.entrySet()) {
            String catName = entry.getKey();
            List<ResourceLocator> locatorList = entry.getValue();
            mainPanel.add(new CategoryPanel(catName, locatorList, loadedTrackNames));
        }

        JScrollPane sp = new JScrollPane(mainPanel);
        sp.setBackground(mainPanel.getBackground());
        dialogPane.add(sp, BorderLayout.CENTER);
        contentPane.add(dialogPane, BorderLayout.CENTER);


        JPanel buttonBar = new JPanel();
        buttonBar.setBorder(new EmptyBorder(12, 0, 0, 0));
        buttonBar.setLayout(new GridBagLayout());
        ((GridBagLayout) buttonBar.getLayout()).columnWidths = new int[]{0, 85, 80};
        ((GridBagLayout) buttonBar.getLayout()).columnWeights = new double[]{1.0, 0.0, 0.0};

        //---- okButton ----
        JButton okButton = new JButton("OK");
        okButton.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                canceled = false;
                for (Component c : mainPanel.getComponents()) {
                    if (c instanceof CategoryPanel) {
                        selectedTracks.addAll(((CategoryPanel) c).getSelectedTracks());
                    }
                }
                setVisible(false);
            }
        });


        //---- cancelButton ----
        JButton cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                canceled = true;
                setVisible(false);
            }
        });

        buttonBar.add(cancelButton, new GridBagConstraints(2, 0, 1, 1, 0.0, 0.0,
                GridBagConstraints.CENTER, GridBagConstraints.BOTH,
                new Insets(0, 0, 0, 0), 0, 0));
        buttonBar.add(okButton, new GridBagConstraints(1, 0, 1, 1, 0.0, 0.0,
                GridBagConstraints.CENTER, GridBagConstraints.BOTH,
                new Insets(0, 0, 0, 5), 0, 0));


        contentPane.add(buttonBar, BorderLayout.SOUTH);
        pack();
        setLocationRelativeTo(getOwner());
    }

    public boolean isCanceled() {
        return canceled;
    }

    public Collection<String> getSelectedTracks() {
        return selectedTracks;
    }
}
