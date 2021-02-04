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

package juicebox.mapcolorui;

import juicebox.gui.SuperAdapter;
import juicebox.windowui.MatrixType;

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

    private static final long serialVersionUID = 9000027;
    private final DecimalFormat df2;
    private final JTextField minimumField = new JTextField();
    private final JTextField maximumField = new JTextField();
    private final JTextField lowerField = new JTextField();
    private final JTextField upperField = new JTextField();
    private final JLabel minTextLabel = new JLabel("Minimum:");
    private final JLabel lowerValTextLabel = new JLabel("Lower value:");
    private final JLabel maxTextLabel = new JLabel("Maximum:");
    private final JLabel higherValTextLabel = new JLabel("Higher value:");
    
    public ColorRangeDialog(SuperAdapter superAdapter, JColorRangePanel colorRangePanel,
                            RangeSlider colorSlider, double colorRangeFactor, MatrixType option) {
        super(superAdapter.getMainWindow());
        
        boolean isOEColorScaleType = MatrixType.isOEColorScaleType(option);
        initComponents(superAdapter, colorRangePanel, isOEColorScaleType);
        
        df2 = new DecimalFormat("####.###");

        minimumField.setText(df2.format(colorSlider.getMinimum() / colorRangeFactor));
        maximumField.setText(df2.format(colorSlider.getMaximum() / colorRangeFactor));
        lowerField.setText(df2.format(colorSlider.getLowerValue() / colorRangeFactor));
        upperField.setText(df2.format(colorSlider.getUpperValue() / colorRangeFactor));

        maximumField.requestFocusInWindow();
    }

    private void initComponents(final SuperAdapter superAdapter, final JColorRangePanel colorRangePanel,
                                final boolean isOEColorScaleType) {

        JPanel dialogPane = new JPanel();
        dialogPane.setBorder(new EmptyBorder(12, 12, 12, 12));
        dialogPane.setLayout(new BorderLayout());


        JPanel titlePanel = new JPanel();
        titlePanel.setLayout(new FlowLayout(FlowLayout.LEFT, 5, 25));
        dialogPane.add(titlePanel, BorderLayout.NORTH);

        JLabel titleLabel = new JLabel("Set color slider control range");
        if (isOEColorScaleType) {
            titleLabel.setText("Set log-scaled color slider control range");
        }
        titlePanel.add(titleLabel);

        JPanel contentPanel = new JPanel();
        contentPanel.setLayout(new BoxLayout(contentPanel, BoxLayout.Y_AXIS));

        Container contentPane = getContentPane();
        contentPane.setLayout(new BorderLayout());

        JPanel minInfoEntryPanel = new JPanel();
        minInfoEntryPanel.setLayout(new FlowLayout(FlowLayout.LEFT));

        minInfoEntryPanel.add(minTextLabel);
        setDimension(minimumField);
        minInfoEntryPanel.add(minimumField);

        minInfoEntryPanel.add(lowerValTextLabel);
        setDimension(lowerField);
        minInfoEntryPanel.add(lowerField);

        if (isOEColorScaleType) {
            minimumField.setEnabled(false);
        }

        contentPanel.add(minInfoEntryPanel);

        JPanel maxInfoEntryPanel = new JPanel(new FlowLayout(FlowLayout.LEADING));

        maxInfoEntryPanel.add(maxTextLabel);
        setDimension(maximumField);
        maxInfoEntryPanel.add(maximumField);
        maximumField.addFocusListener(new FocusListener() {
            @Override
            public void focusGained(FocusEvent focusEvent) {
                //To change body of implemented methods use File | Settings | File Templates.
            }

            @Override
            public void focusLost(FocusEvent focusEvent) {
                if (isOEColorScaleType) {
                    double max;
                    try {
                        max = df2.parse(maximumField.getText()).doubleValue();
                    } catch (ParseException error) {
                        return;
                    }
                    minimumField.setText(df2.format(-max));
                }
            }
        });

        maxInfoEntryPanel.add(higherValTextLabel);
        setDimension(upperField);
        maxInfoEntryPanel.add(upperField);

        upperField.addFocusListener(new FocusListener() {
            @Override
            public void focusGained(FocusEvent focusEvent) {
                //To change body of implemented methods use File | Settings | File Templates.
            }

            @Override
            public void focusLost(FocusEvent focusEvent) {
                if (isOEColorScaleType) {
                    double max;
                    try {
                        max = df2.parse(upperField.getText()).doubleValue();
                    } catch (ParseException error) {
                        return;
                    }
                    lowerField.setText(df2.format(-max));
                }
            }
        });

        contentPanel.add(maxInfoEntryPanel);

        dialogPane.add(contentPanel, BorderLayout.CENTER);

        JPanel buttonBar = new JPanel(new GridBagLayout());
        buttonBar.setBorder(new EmptyBorder(12, 0, 0, 0));
        ((GridBagLayout) buttonBar.getLayout()).columnWidths = new int[]{0, 85, 80};
        ((GridBagLayout) buttonBar.getLayout()).columnWeights = new double[]{1.0, 0.0, 0.0};

        //---- okButton ----
        JButton okButton = new JButton("OK");
        okButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                okButtonActionPerformed(e, superAdapter, colorRangePanel, isOEColorScaleType);
            }
        });
        buttonBar.add(okButton, new GridBagConstraints(1, 0, 1, 1, 0.0, 0.0,
                GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 5), 0, 0));

        JButton cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                colorRangePanel.setColorRangeSliderVisible(true, superAdapter);
                if (!superAdapter.getMainViewPanel().setResolutionSliderVisible(true, superAdapter)) {
                    System.err.println("Something went wrong");
                }
                setVisible(false);
            }
        });

        buttonBar.add(cancelButton, new GridBagConstraints(2, 0, 1, 1, 0.0, 0.0,
                GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));

        dialogPane.add(buttonBar, BorderLayout.SOUTH);
        contentPane.add(dialogPane, BorderLayout.CENTER);
        pack();
        setLocationRelativeTo(getOwner());
    }

    private void setDimension(JTextField field) {
        field.setMaximumSize(new Dimension(100, 20));
        field.setMinimumSize(new Dimension(100, 20));
        field.setPreferredSize(new Dimension(100, 20));
    }

    private void okButtonActionPerformed(ActionEvent e, SuperAdapter superAdapter,
                                         JColorRangePanel colorRangePanel, boolean isOEColorScaleType) {
        double max = 0, min = 0;
        double lower = 0, upper = 0;

        try {
            if (isOEColorScaleType) {
                max = df2.parse(maximumField.getText()).doubleValue();
                //upper = df2.parse(upperField.getText()).doubleValue();
            }
            max = df2.parse(maximumField.getText()).doubleValue();
            min = df2.parse(minimumField.getText()).doubleValue();
            upper = df2.parse(upperField.getText()).doubleValue();
            lower = df2.parse(lowerField.getText()).doubleValue();

        } catch (ParseException error) {
            JOptionPane.showMessageDialog(this, "Must enter a number", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }
        if (max <= min || lower < min || upper <= min || upper <= lower || max <= lower || max < upper) {
            JOptionPane.showMessageDialog(this, "Values not appropriate", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        int iMax = (int) max;
        int iLower = (int) lower;
        int iUpper = (int) upper;

        if (isOEColorScaleType) {
            colorRangePanel.updateRatioColorSlider(superAdapter.getHiC(), iMax, iUpper);
        } else {
            colorRangePanel.updateColorSlider(superAdapter.getHiC(), iLower, iUpper, iMax);
        }

        colorRangePanel.setColorRangeSliderVisible(true, superAdapter);
        if (!superAdapter.getMainViewPanel().setResolutionSliderVisible(true, superAdapter)) {
            System.err.println("Error encountered 12134");
        }
        setVisible(false);
    }
}
