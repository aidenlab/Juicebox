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

import com.jidesoft.swing.JideButton;
import juicebox.MainWindow;
import juicebox.gui.MainViewPanel;
import juicebox.gui.SuperAdapter;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.text.DecimalFormat;
import java.text.ParseException;

/**
 * @author Jim Robinson
 */
class ColorRangeDialog extends JDialog {

    private static final long serialVersionUID = -2570891809264626823L;
    private static MultiColorPickerDialog gradientPick;
    private static JideButton plusButton;
    private static JideButton minusButton;
    private final RangeSlider colorSlider;
    private final double colorRangeFactor;
    private final DecimalFormat df1;
    private final DecimalFormat df2;
    public Color[] tmpCol = new Color[24];
    private JTextField minimumField;
    private JTextField maximumField;

    public ColorRangeDialog(SuperAdapter superAdapter, JColorRangePanel colorRangePanel,
                            RangeSlider colorSlider, double colorRangeFactor, boolean isObserved) {
        super(superAdapter.getMainWindow());
        initComponents(superAdapter, colorRangePanel, isObserved);
        this.colorSlider = colorSlider;
        if (!isObserved) colorRangeFactor = 8;
        this.colorRangeFactor = colorRangeFactor;


        df1 = new DecimalFormat("#,###,###,##0");
        df2 = new DecimalFormat("##.##");

        if (isObserved) {
            minimumField.setText(df1.format(colorSlider.getMinimum() / colorRangeFactor));
            maximumField.setText(df1.format(colorSlider.getMaximum() / colorRangeFactor));
        } else {
            minimumField.setText(df2.format(1 / (colorSlider.getMaximum() / colorRangeFactor)));
            maximumField.setText(df2.format(colorSlider.getMaximum() / colorRangeFactor));
        }
        //tickSpacingField.setText(df.format(colorSlider.getMajorTickSpacing() / colorRangeFactor));
        maximumField.requestFocusInWindow();
    }

    private void initComponents(final SuperAdapter superAdapter, final JColorRangePanel colorRangePanel,
                                final boolean isObserved) {

        JPanel dialogPane = new JPanel();
        JPanel panel3 = new JPanel();
        JLabel label1 = new JLabel();
        JPanel contentPanel = new JPanel();
        JPanel panel2 = new JPanel();
        JLabel label4 = new JLabel();

        final JPanel panel5 = new JPanel();
        JLabel label5 = new JLabel();

        final JPanel panel6 = new JPanel();

        minimumField = new JTextField();
        JPanel panel1 = new JPanel();
        JLabel label2 = new JLabel();
        maximumField = new JTextField();

        JPanel buttonBar = new JPanel();
        JButton okButton = new JButton();
        JButton cancelButton = new JButton();

        Container contentPane = getContentPane();
        contentPane.setLayout(new BorderLayout());


        dialogPane.setBorder(new EmptyBorder(12, 12, 12, 12));
        dialogPane.setLayout(new BorderLayout());


        panel3.setLayout(new FlowLayout(FlowLayout.LEFT, 5, 25));

        //---- label1 ----
        label1.setText("Set color slider control range");
        panel3.add(label1);

        dialogPane.add(panel3, BorderLayout.NORTH);


        contentPanel.setLayout(new BoxLayout(contentPanel, BoxLayout.Y_AXIS));

        //======== panel2 ========
        {
            panel2.setLayout(new FlowLayout(FlowLayout.LEFT));

            //---- label4 ----
            label4.setText("Minimum:");
            panel2.add(label4);

            //---- minimumField ----
            minimumField.setMaximumSize(new Dimension(100, 20));
            minimumField.setMinimumSize(new Dimension(100, 20));
            minimumField.setPreferredSize(new Dimension(100, 20));
            panel2.add(minimumField);
        }
        if (!isObserved) {
            minimumField.setEnabled(false);

        }
        contentPanel.add(panel2);

        panel1.setLayout(new FlowLayout(FlowLayout.LEADING));

        //---- label2 ----
        label2.setText("Maximum");
        panel1.add(label2);

        //---- maximumField ----
        maximumField.setMaximumSize(new Dimension(100, 20));
        maximumField.setMinimumSize(new Dimension(100, 20));
        maximumField.setPreferredSize(new Dimension(100, 20));
        panel1.add(maximumField);
        maximumField.addFocusListener(new FocusListener() {
            @Override
            public void focusGained(FocusEvent focusEvent) {
                //To change body of implemented methods use File | Settings | File Templates.
            }

            @Override
            public void focusLost(FocusEvent focusEvent) {
                if (!isObserved) {
                    double max;
                    try {
                        max = df2.parse(maximumField.getText()).doubleValue();
                    } catch (ParseException error) {
                        return;
                    }
                    minimumField.setText(df2.format(1 / max));
                }
            }
        });

        contentPanel.add(panel1);


        //======== panel5 ========
        panel5.setLayout(new FlowLayout(FlowLayout.LEADING));

        final JButton colorChooserButton = new JButton("Map color");
        colorChooserButton.setSize(50, 50);
        colorChooserButton.setForeground(MainWindow.hicMapColor);

        colorChooserButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Color color = JColorChooser.showDialog(panel5, "Map color", colorChooserButton.getForeground());
                colorChooserButton.setForeground(color);
                MainWindow.hicMapColor = color;
                //System.out.println("The selected color was:" + color);
            }
        });

//        colorChooserButton.setEnabled(!MainViewPanel.preDefMapColor);
//        contentPanel.add(panel5);

        final JButton paletteChooserButton = new JButton("Create gradient");
        paletteChooserButton.setSize(50, 50);
        paletteChooserButton.setForeground(MainWindow.hicMapColor);

        paletteChooserButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                gradientPick = new MultiColorPickerDialog();
                gradientPick.initValue(MainViewPanel.preDefMapColorGradient.toArray(new Color[MainViewPanel.preDefMapColorGradient.size()]));
            }
        });

        paletteChooserButton.setEnabled(MainViewPanel.preDefMapColor);

        //======== panel6 ========

        JRadioButton rColor = new JRadioButton(new AbstractAction("Color") {
            private static final long serialVersionUID = 4201L;

            @Override
            public void actionPerformed(ActionEvent e) {
                JRadioButton rLog = (JRadioButton) e.getSource();
                boolean val = !rLog.isSelected();
                MainViewPanel.preDefMapColor = val;
                colorChooserButton.setEnabled(!val);
                paletteChooserButton.setEnabled(val);
            }
        });
        rColor.setSelected(true);
        JRadioButton rGradient = new JRadioButton(new AbstractAction("Gradient") {
            private static final long serialVersionUID = 4202L;

            @Override
            public void actionPerformed(ActionEvent e) {
                JRadioButton rLog = (JRadioButton) e.getSource();
                boolean val = rLog.isSelected();
                MainViewPanel.preDefMapColor = val;
                colorChooserButton.setEnabled(!val);
                paletteChooserButton.setEnabled(val);
            }
        });
//        ButtonGroup group = new ButtonGroup();
//        group.add(rColor);
//        group.add(rGradient);

//        panel5.add(rColor);
//        panel5.add(colorChooserButton);
//        panel6.add(rGradient);

//        panel6.setLayout(new FlowLayout(FlowLayout.LEADING));
//        panel6.add(paletteChooserButton);
//        contentPanel.add(panel6);

        dialogPane.add(contentPanel, BorderLayout.CENTER);

        buttonBar.setBorder(new EmptyBorder(12, 0, 0, 0));
        buttonBar.setLayout(new GridBagLayout());
        ((GridBagLayout) buttonBar.getLayout()).columnWidths = new int[]{0, 85, 80};
        ((GridBagLayout) buttonBar.getLayout()).columnWeights = new double[]{1.0, 0.0, 0.0};

        //---- okButton ----
        okButton.setText("OK");
        okButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                okButtonActionPerformed(e, superAdapter, colorRangePanel, isObserved);
            }
        });
        buttonBar.add(okButton, new GridBagConstraints(1, 0, 1, 1, 0.0, 0.0,
                GridBagConstraints.CENTER, GridBagConstraints.BOTH,
                new Insets(0, 0, 0, 5), 0, 0));

        //---- cancelButton ----
        cancelButton.setText("Cancel");
        cancelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                colorRangePanel.setColorRangeSliderVisible(true, superAdapter);
                if (superAdapter.getMainViewPanel().setResolutionSliderVisible(true, superAdapter)) {
                    // TODO succeeded
                } else {
                    // TODO failed
                }
                setVisible(false);
            }
        });
        buttonBar.add(cancelButton, new GridBagConstraints(2, 0, 1, 1, 0.0, 0.0,
                GridBagConstraints.CENTER, GridBagConstraints.BOTH,
                new Insets(0, 0, 0, 0), 0, 0));

        dialogPane.add(buttonBar, BorderLayout.SOUTH);

        contentPane.add(dialogPane, BorderLayout.CENTER);
        pack();
        setLocationRelativeTo(getOwner());
        //maximumField.requestFocusInWindow();
    }

    private void okButtonActionPerformed(ActionEvent e, SuperAdapter superAdapter,
                                         JColorRangePanel colorRangePanel, boolean isObserved) {
        double max, min = 0;

        try {
            if (isObserved) {
                max = df1.parse(maximumField.getText()).doubleValue();
                min = df1.parse(minimumField.getText()).doubleValue();
            } else {
                max = df2.parse(maximumField.getText()).doubleValue();
            }
        } catch (ParseException error) {
            JOptionPane.showMessageDialog(this, "Must enter a number", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }
        if (max < min) {
            JOptionPane.showMessageDialog(this, "Maximum may not be less than minimum", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        int iMin = (int) (colorRangeFactor * min);
        int iMax = (int) (colorRangeFactor * max);
        if (!isObserved) {
            iMax = (int) (max * colorRangeFactor);
            iMin = (int) (colorRangeFactor / max);
        }
        colorSlider.setMinimum(iMin);
        colorSlider.setMaximum(iMax);
        colorRangePanel.setColorRangeSliderVisible(true, superAdapter);
        if (superAdapter.getMainViewPanel().setResolutionSliderVisible(true, superAdapter)) {
            // TODO succeeded
        } else {
            // TODO failed
        }
        setVisible(false);
    }

}
