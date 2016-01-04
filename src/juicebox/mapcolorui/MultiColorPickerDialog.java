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

package juicebox.mapcolorui;

import juicebox.gui.MainViewPanel;

import javax.swing.*;
import javax.swing.colorchooser.AbstractColorChooserPanel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

class MultiColorPickerDialog extends JDialog {
    private static final long serialVersionUID = -678567876;

    private final JButton[] bColor = new JButton[24];
    private final JButton[] bChoose = new JButton[24];
    private final JButton[] bDelete = new JButton[24];
    private final JColorChooser chooser = new JColorChooser();

    private final JPanel preview = new JPanel();
    private final JPanel prvPanel1 = new JPanel();
    private final JPanel prvPanel2 = new JPanel();
    private final JPanel prvPanel3 = new JPanel();

    private final JPanel chooserPanel = new JPanel();
    private final JButton bOk = new JButton("OK");
    private final JButton bCancel = new JButton("Cancel");

    public MultiColorPickerDialog() {
        super();
        setResizable(false);
        setLayout(new BoxLayout(getContentPane(), 1));
        final Color defaultColor = getBackground();

        chooser.setSize(new Dimension(690, 270));
        chooserPanel.setMaximumSize(new Dimension(690, 270));

        AbstractColorChooserPanel[] accp = chooser.getChooserPanels();
        chooser.removeChooserPanel(accp[0]);
        chooser.removeChooserPanel(accp[1]);
        chooser.removeChooserPanel(accp[2]);
        chooser.removeChooserPanel(accp[4]);

        chooser.setPreviewPanel(new JPanel());

        chooserPanel.add(chooser);

        prvPanel1.add(new JLabel("RGB "));
        prvPanel2.add(new JLabel("Pick "));
        prvPanel3.add(new JLabel("Clear"));

        for (int idx = 0; idx < 24; idx++) {
            final int x = idx;
            bColor[x] = new JButton();
            bColor[x].setBackground(defaultColor);
            bColor[x].setBorder(BorderFactory.createLineBorder(Color.DARK_GRAY, 1));
            bColor[x].setOpaque(true);
            bChoose[x] = new JButton("+");
            bDelete[x] = new JButton("-");

            bColor[x].setPreferredSize(new Dimension(15, 15));
            prvPanel1.add(bColor[x]);

            bChoose[x].setPreferredSize(new Dimension(15, 15));
            bDelete[x].setPreferredSize(new Dimension(15, 15));

            bColor[x].addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    chooser.setColor(bColor[x].getBackground());
                }
            });
            bChoose[x].addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    //bColor[x].setVisible(true);
                    bColor[x].setBackground(chooser.getColor());
                }
            });
            bDelete[x].addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    //bColor[x].setVisible(false);
                    bColor[x].setBackground(defaultColor);
                }
            });

            prvPanel2.add(bChoose[x]);

            prvPanel3.add(bDelete[x]);
        }

        prvPanel1.setPreferredSize(new Dimension(600, 30));
        prvPanel2.setPreferredSize(new Dimension(600, 30));
        prvPanel3.setPreferredSize(new Dimension(600, 30));

        getContentPane().add(chooserPanel);

        preview.add(prvPanel1);
        preview.add(prvPanel2);
        preview.add(prvPanel3);
        add(preview);

        JPanel okCancel = new JPanel();

        okCancel.add(bOk);
        bOk.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

                MainViewPanel.preDefMapColorGradient.clear();
                MainViewPanel.preDefMapColorFractions.clear();

                //todo - make the bColor add/remove behavior instead.
                for (JButton aBColor : bColor) {
                    if (aBColor.getBackground() != defaultColor)
                        MainViewPanel.preDefMapColorGradient.add(aBColor.getBackground());
                }

                float tmpfraction = 0.0f;
                int tmpSize = MainViewPanel.preDefMapColorGradient.size();
                float tmpGap = 0;
                if (tmpSize > 0) {
                    tmpGap = (1.0f / MainViewPanel.preDefMapColorGradient.size());
                }

                for (int i = 0; i < tmpSize; i++) {
                    MainViewPanel.preDefMapColorFractions.add(tmpfraction);
                    tmpfraction += tmpGap;
                }
                dispose();
            }
        });

        bCancel.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dispose();
            }
        });
        okCancel.add(bCancel);
        add(okCancel);

        setSize(new Dimension(690, 500));
        setVisible(true);

        setLocationRelativeTo(getOwner());
    }

    public void initValue(Color[] colorArray) {
        for (int cIdx = 0; cIdx < colorArray.length && cIdx < bColor.length; cIdx++) {
            bColor[cIdx].setBackground(colorArray[cIdx]);
        }
    }
}
