/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

package juicebox.tools.dev;

import juicebox.HiC;
import juicebox.data.Dataset;
import juicebox.gui.SuperAdapter;

import javax.swing.*;
import java.awt.*;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 9/6/16.
 * Probably should just be deleted, but keep for now.
 */
public class GenomeSelectionPanel extends JPanel {

    private static final long serialVersionUID = 817133921738L;

    /**
     */
    public GenomeSelectionPanel(SuperAdapter superAdapter) {
        super(new BorderLayout());

        JPanel genomePanel = generateGenomeActivationPanel(superAdapter, "Active Genomewide Chromosomes:");

        add(genomePanel, BorderLayout.CENTER);
    }

    /**
     * @param superAdapter
     */
    public static void launchMapSubsetGUI(SuperAdapter superAdapter) {
        HiC hic = superAdapter.getHiC();
        Dataset reader = hic.getDataset();
        Dataset controlReader = hic.getControlDataset();
        if (reader != null) {
            JFrame frame = new JFrame("Map Selection Panel");
            MapSelectionPanel newContentPane = new MapSelectionPanel(superAdapter, reader, controlReader);
            newContentPane.setOpaque(true);
            frame.setContentPane(newContentPane);
            frame.pack();
            frame.setVisible(true);
        }
    }

    /**
     * @param superAdapter
     * @param title
     * @return
     */
    private JPanel generateGenomeActivationPanel(final SuperAdapter superAdapter, String title) {
        final JButton showItButton = new JButton("Update View");
        try {
            /*
            final List<Chromosome> chromosomes = superAdapter.getHiC().getChromosomes();
            if (chromosomes != null && chromosomes.size() > 0) {

                final List<JCheckBox> checkBoxes = new ArrayList<JCheckBox>();
                for (Chromosome chromosome : chromosomes) {
                    checkBoxes.add(new JCheckBox(chromosome.getName()));
                }

                showItButton.addActionListener(new ActionListener() {
                    public void actionPerformed(ActionEvent e) {
                        List<Chromosome> activeChromosomes = new ArrayList<Chromosome>();
                        for (int i = 0; i < chromosomes.size(); i++) {
                            if (checkBoxes.get(i).isSelected()) {
                                activeChromosomes.add(chromosomes.get(i));
                            }
                        }
                        //Private.updateActiveGenome(activeChromosomes);
                        // because cache keys currently don't account for map activation
                        HiCGlobals.useCache = false;
                        superAdapter.refresh();
                    }
                });
                return createPane(title, checkBoxes, showItButton);

            }
            */
        } catch (Exception e) {
            //e.printStackTrace();
        }

        return null;
    }

    /**
     * @param description
     * @param checkBoxes
     * @param showButton
     * @return
     */
    private JPanel createPane(String description, List<JCheckBox> checkBoxes, JButton showButton) {

        JLabel label = new JLabel(description);

        JPanel box = new JPanel();
        box.setLayout(new BoxLayout(box, BoxLayout.PAGE_AXIS));
        for (JCheckBox checkBox : checkBoxes) {
            box.add(checkBox);
        }
        JScrollPane scrollPane = new JScrollPane(box);

        JPanel pane = new JPanel(new BorderLayout());
        pane.add(label, BorderLayout.PAGE_START);
        pane.add(scrollPane, BorderLayout.CENTER);
        pane.add(showButton, BorderLayout.PAGE_END);

        return pane;
    }
}
