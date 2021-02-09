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

package juicebox.track;

import juicebox.gui.SuperAdapter;
import org.broad.igv.renderer.DataRange;
import org.broad.igv.track.WindowFunction;
import org.broad.igv.ui.color.ColorChooserPanel;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;

/**
 * @author Jim Robinson
 */
public class TrackConfigPanel extends JPanel {

    private static final long serialVersionUID = 900007;
    private final HiCTrack track;
	private final boolean canceled = false;
	private ColorChooserPanel posColorChooser;
    private ColorChooserPanel altColorChooser;
    private JCheckBox logScaleCB;
    private JTextField minYField;
    private JTextField maxYField;
    private JRadioButton meanRB;
    private JRadioButton maxRB;
    private JTextField nameField;

    public TrackConfigPanel(SuperAdapter superAdapter, HiCTrack track) {
        super(new FlowLayout());

        this.track = track;
        initComponents(superAdapter);

        if (track instanceof EigenvectorTrack) {
            altColorChooser.setEnabled(true);
        }

        if (track instanceof HiCDataTrack) {
            minYField.setEnabled(true);
            maxYField.setEnabled(true);
            logScaleCB.setEnabled(true);
            altColorChooser.setEnabled(true);
            maxRB.setEnabled(true);
            meanRB.setEnabled(true);

            HiCDataTrack dataTrack = (HiCDataTrack) track;
            minYField.setText(String.valueOf(dataTrack.getDataRange().getMinimum()));
            maxYField.setText(String.valueOf(dataTrack.getDataRange().getMaximum()));
            logScaleCB.setSelected(dataTrack.getDataRange().isLog());
            altColorChooser.setSelectedColor(dataTrack.getNegColor());

            if (!dataTrack.getAvailableWindowFunctions().contains(WindowFunction.max)) {
                maxRB.setEnabled(false);
            }

            if (dataTrack.getWindowFunction() == WindowFunction.max) {
                maxRB.setSelected(true);
            } else {
                meanRB.setSelected(true);
            }

        } else {
            minYField.setEnabled(false);
            maxYField.setEnabled(false);
            logScaleCB.setEnabled(false);
            altColorChooser.setEnabled(false);
            maxRB.setEnabled(false);
            meanRB.setEnabled(false);
        }
    }

    private void minYFieldFocusLost(FocusEvent e) {
        if (track instanceof HiCDataTrack) {
            if (!validateNumeric(minYField.getText())) {
                DataRange dr = ((HiCDataTrack) track).getDataRange();
                minYField.setText(String.valueOf(dr.getMinimum()));
            }
        }
    }

    private void maxYFieldFocusLost(FocusEvent e) {
        if (track instanceof HiCDataTrack) {
            if (!validateNumeric(maxYField.getText())) {
                DataRange dr = ((HiCDataTrack) track).getDataRange();
                minYField.setText(String.valueOf(dr.getMaximum()));
            }
        }
    }

    private void initComponents(final SuperAdapter superAdapter) {
        // JFormDesigner - Component initialization - DO NOT MODIFY  //GEN-BEGIN:initComponents
        // Generated using JFormDesigner non-commercial license


        nameField = new JTextField(track.getName(), 10);
        nameField.setToolTipText("Change the name for this annotation: " + nameField.getText());
        nameField.setMaximumSize(new Dimension(100, 30));
        nameField.getDocument().addDocumentListener(new DocumentListener() {

            private void action() {
                track.setName(nameField.getText());
                nameField.setToolTipText("Change the name for this annotation: " + nameField.getText());
                superAdapter.updateTrackPanel();
                superAdapter.repaintTrackPanels();
                superAdapter.repaint();
            }

            @Override
            public void insertUpdate(DocumentEvent e) {
                action();
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                action();
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                action();
            }
        });
        add(nameField);

        add(new JLabel("Positive:"));
        posColorChooser = new ColorChooserPanel();
        posColorChooser.setToolTipText("Change color for positive values");
        posColorChooser.setSelectedColor(track.getPosColor());
        posColorChooser.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Color color = posColorChooser.getSelectedColor();
                if (color != null) {
                    track.setPosColor(color);
                    superAdapter.updateTrackPanel();
                    superAdapter.repaintTrackPanels();
                }
            }
        });
        add(posColorChooser);

        add(new JLabel("Negative:"));
        altColorChooser = new ColorChooserPanel();
        altColorChooser.setToolTipText("Change color for negative values");
        altColorChooser.setSelectedColor(track.getNegColor());
        altColorChooser.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Color color = altColorChooser.getSelectedColor();
                if (color != null) {
                    track.setNegColor(color);
                    superAdapter.updateTrackPanel();
                    superAdapter.repaintTrackPanels();
                }
            }
        });
        add(altColorChooser);

        add(new JLabel("Min:"));
        minYField = new JTextField("", 3);
        /*minYField.addFocusListener(new FocusAdapter() {
            @Override
            public void focusLost(FocusEvent e) {
                minYFieldFocusLost(e);
                superAdapter.updateTrackPanel();
                superAdapter.repaintTrackPanels();
            }
        });*/
        add(minYField);
        minYField.getDocument().addDocumentListener(dataActionDocument(minYField, superAdapter));

        add(new JLabel("Max:"));
        maxYField = new JTextField("", 3);
        /*maxYField.addFocusListener(new FocusAdapter() {
            @Override
            public void focusLost(FocusEvent e) {
                maxYFieldFocusLost(e);
                superAdapter.updateTrackPanel();
                superAdapter.repaintTrackPanels();
            }
        });*/
        add(maxYField);
        maxYField.getDocument().addDocumentListener(dataActionDocument(maxYField, superAdapter));

        logScaleCB = new JCheckBox();
        logScaleCB.setText("Log");
        logScaleCB.setToolTipText("Set logarithmic scaling");
        logScaleCB.setEnabled(false);
        add(logScaleCB);
        logScaleCB.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                performDataAction(superAdapter);
            }
        });


        add(new JLabel("DRF"));
        meanRB = new JRadioButton("Mean");
        meanRB.setToolTipText("Data Reduction Function");
        maxRB = new JRadioButton("Max");
        maxRB.setToolTipText("Data Reduction Function");
        ButtonGroup dataReductionGroup = new ButtonGroup();
        dataReductionGroup.add(meanRB);
        dataReductionGroup.add(maxRB);
        add(meanRB);
        add(maxRB);
        meanRB.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (track instanceof HiCDataTrack) {
                    WindowFunction wf = maxRB.isSelected() ? WindowFunction.max : WindowFunction.mean;
                    ((HiCDataTrack) track).setWindowFunction(wf);
                    superAdapter.updateTrackPanel();
                    superAdapter.repaintTrackPanels();
                }
            }
        });
        maxRB.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (track instanceof HiCDataTrack) {
                    WindowFunction wf = maxRB.isSelected() ? WindowFunction.max : WindowFunction.mean;
                    ((HiCDataTrack) track).setWindowFunction(wf);
                    superAdapter.updateTrackPanel();
                    superAdapter.repaintTrackPanels();
                }
            }
        });
    }

    private DocumentListener dataActionDocument(final JTextField field, final SuperAdapter superAdapter) {
        return new DocumentListener() {

            private void action() {
                field.setToolTipText("Change: " + field.getText());
                if (isReasonableText(field.getText())) {
                    performDataAction(superAdapter);
                }
            }

            @Override
            public void insertUpdate(DocumentEvent e) {
                action();
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                action();
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                action();
            }
        };
    }

    private void performDataAction(SuperAdapter superAdapter) {
        if (track instanceof HiCDataTrack && validateNumeric(minYField.getText())
                && validateNumeric(maxYField.getText())) {

            float newMin = Float.parseFloat(minYField.getText());
            float newMax = Float.parseFloat(maxYField.getText());
            DataRange newDataRange = new DataRange(newMin, newMax);
            if (newMin < 0 && newMax > 0) {
                newDataRange = new DataRange(newMin, 0f, newMax);
            }
            newDataRange.setType(logScaleCB.isSelected() ? DataRange.Type.LOG : DataRange.Type.LINEAR);
            ((HiCDataTrack) track).setDataRange(newDataRange);

            superAdapter.updateTrackPanel();
            superAdapter.repaintTrackPanels();
        }
    }

    private boolean isReasonableText(String text) {
        return text.length() > 0 && !text.equals(".") && !text.equals("-");
    }

    private boolean validateNumeric(String text) {
        try {
            if (isReasonableText(text)) {
                Double.parseDouble(text);
                return true;
            }
        } catch (NumberFormatException e) {
            JOptionPane.showMessageDialog(this, "Error: numeric value required (" + text + ")");
        }
        return false;
    }
}
