/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

package juicebox.mapcolorui;

import juicebox.HiCGlobals;
import juicebox.gui.SuperAdapter;

import javax.swing.*;
import javax.swing.border.Border;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * Created by muhammadsaadshamim on 1/25/17.
 */
public class PearsonColorScaleEditor extends JDialog {

    private static final long serialVersionUID = 81248513331738L;

    public PearsonColorScaleEditor(final SuperAdapter superAdapter, final PearsonColorScale pearsonColorScale) {
        super(superAdapter.getMainWindow());

        Container contentPane = getContentPane();
        contentPane.setLayout(new BorderLayout());
        Border padding = BorderFactory.createEmptyBorder(20, 20, 5, 20);

        setModal(true);

        final String key = superAdapter.getHiC().getColorScaleKey();

        JLabel labelPosMax = new JLabel("Positive Max");
        JLabel labelPosMin = new JLabel("Positive Min");
        JLabel labelNegMax = new JLabel("Negative Max");
        JLabel labelNegMin = new JLabel("Negative Min");
        final JTextField textPosMax = new JTextField("" + pearsonColorScale.getPosMax(key));
        final JTextField textPosMin = new JTextField("" + pearsonColorScale.getPosMin(key));
        final JTextField textNegMax = new JTextField("" + pearsonColorScale.getNegMax(key));
        final JTextField textNegMin = new JTextField("" + pearsonColorScale.getNegMin(key));

        JPanel box = new JPanel();
        box.setLayout(new GridLayout(0, 2));
        box.add(labelPosMax);
        box.add(textPosMax);
        box.add(labelPosMin);
        box.add(textPosMin);
        box.add(labelNegMax);
        box.add(textNegMax);
        box.add(labelNegMin);
        box.add(textNegMin);

        JButton updateButton = new JButton("Update Values");
        updateButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Color mainBackgroundColor = HiCGlobals.isDarkulaModeEnabled ? Color.DARK_GRAY : Color.WHITE;
                textPosMax.setBackground(mainBackgroundColor);
                textPosMin.setBackground(mainBackgroundColor);
                textNegMax.setBackground(mainBackgroundColor);
                textNegMin.setBackground(mainBackgroundColor);


                float maxA = Float.parseFloat(textPosMax.getText());
                float minA = Float.parseFloat(textPosMin.getText());
                float maxB = Float.parseFloat(textNegMax.getText());
                float minB = Float.parseFloat(textNegMin.getText());

                if (minA < 0) {
                    textPosMin.setBackground(Color.RED);
                } else if (minA >= maxA) {
                    textPosMax.setBackground(Color.RED);
                    textPosMin.setBackground(Color.RED);
                } else if (maxB > 0) {
                    textNegMax.setBackground(Color.RED);
                } else if (minB >= maxB) {
                    textNegMax.setBackground(Color.RED);
                    textNegMin.setBackground(Color.RED);
                } else {
                    pearsonColorScale.setMinMax(key, minB, maxB, minA, maxA);
                    superAdapter.refresh();
                }
            }
        });
        JButton resetButton = new JButton("Reset Values");
        resetButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                pearsonColorScale.resetValues(key);
                superAdapter.refresh();
                PearsonColorScaleEditor.this.dispose();
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
