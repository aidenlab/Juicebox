/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.gui;

import juicebox.mapcolorui.HeatmapRenderer;

import javax.swing.*;
import javax.swing.border.Border;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class PseudoCountEditor extends JDialog {

    private static final long serialVersionUID = 9000026;

    public PseudoCountEditor(final SuperAdapter superAdapter) {
        super(superAdapter.getMainWindow());

        Container contentPane = getContentPane();
        contentPane.setLayout(new BorderLayout());
        Border padding = BorderFactory.createEmptyBorder(20, 20, 5, 20);

        setModal(true);

        JLabel labelPseudocount = new JLabel("Pseudocount");
        // todo error if called when pearson not loaded yet
        final JTextField textPseudocount = new JTextField("" + HeatmapRenderer.PSEUDO_COUNT);

        JPanel box = new JPanel();
        box.setLayout(new GridLayout(0, 2));
        box.add(labelPseudocount);
        box.add(textPseudocount);

        JButton updateButton = new JButton("Update Values");
        updateButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                HeatmapRenderer.PSEUDO_COUNT = Float.parseFloat(textPseudocount.getText());
                superAdapter.refresh();
                PseudoCountEditor.this.dispose();
            }
        });
        JButton resetButton = new JButton("Reset Values");
        resetButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                HeatmapRenderer.PSEUDO_COUNT = 1f;
                superAdapter.refresh();
                PseudoCountEditor.this.dispose();
            }
        });

        JPanel buttonPanel = new JPanel();
        buttonPanel.setBorder(padding);
        buttonPanel.setLayout(new GridLayout(0, 2));
        buttonPanel.add(updateButton);
        buttonPanel.add(resetButton);


        contentPane.add(box, BorderLayout.CENTER);
        contentPane.add(buttonPanel, BorderLayout.PAGE_END);
        box.setBorder(padding);
        pack();
        setLocationRelativeTo(getOwner());
        setVisible(true);
    }
}
