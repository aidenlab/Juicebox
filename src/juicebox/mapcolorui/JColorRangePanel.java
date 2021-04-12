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

import com.jidesoft.swing.JideButton;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.gui.SuperAdapter;
import juicebox.windowui.MatrixType;
import org.broad.igv.ui.FontManager;

import javax.swing.*;
import javax.swing.border.LineBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.font.TextAttribute;
import java.util.Hashtable;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 8/3/15.
 */
public class JColorRangePanel extends JPanel {

    private static final long serialVersionUID = 9000029;
    private static RangeSlider colorRangeSlider;
    private static JLabel colorRangeLabel;
    private static JButton plusButton, minusButton;
    private double colorRangeScaleFactor = 1;
    private final HeatmapPanel heatmapPanel;

    public JColorRangePanel(final SuperAdapter superAdapter, final HeatmapPanel heatmapPanel) {
        super();
        setLayout(new BorderLayout());
        JPanel sliderPanel = new JPanel();
        sliderPanel.setLayout(new BoxLayout(sliderPanel, BoxLayout.X_AXIS));
        this.heatmapPanel = heatmapPanel;

        colorRangeSlider = new RangeSlider();

        colorRangeSlider.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseEntered(MouseEvent mouseEvent) {
                super.mouseEntered(mouseEvent);
                colorRangeSliderUpdateToolTip(superAdapter.getHiC().getDisplayOption());
            }
        });
        colorRangeSlider.setEnabled(false);
        colorRangeSlider.setDisplayToBlank(true);

        //---- colorRangeLabel ----
        colorRangeLabel = new JLabel("Color Range");
        colorRangeLabel.addMouseListener(new MouseAdapter() {
            private Font original;

            @SuppressWarnings({"unchecked", "rawtypes"})
            @Override
            public void mouseEntered(MouseEvent e) {
                if (colorRangeSlider.isEnabled()) {
                    original = e.getComponent().getFont();
                    Map attributes = original.getAttributes();
                    attributes.put(TextAttribute.UNDERLINE, TextAttribute.UNDERLINE_ON);
                    e.getComponent().setFont(original.deriveFont(attributes));
                }
            }

            @Override
            public void mouseExited(MouseEvent e) {
                //if (colorRangeSlider.isEnabled())
                e.getComponent().setFont(original);
            }

        });

        colorRangeLabel.setHorizontalAlignment(SwingConstants.CENTER);
        colorRangeLabel.setToolTipText("Range of color scale in counts per mega-base squared.");
        colorRangeLabel.setHorizontalTextPosition(SwingConstants.CENTER);

        colorRangeLabel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (e.isPopupTrigger() && colorRangeSlider.isEnabled()) {
                    processClick();
                }
            }

            @Override
            public void mouseClicked(MouseEvent e) {
                //No double click here...
                if (e.getClickCount() == 1 && colorRangeSlider.isEnabled()) {
                    processClick();
                }
            }

            private void processClick() {
                ColorRangeDialog rangeDialog = new ColorRangeDialog(superAdapter, JColorRangePanel.this,
                        colorRangeSlider, colorRangeScaleFactor, superAdapter.getHiC().getDisplayOption());
                setColorRangeSliderVisible(false, superAdapter);
                if (!superAdapter.getMainViewPanel().setResolutionSliderVisible(false, superAdapter)) {
                    System.err.println("error 2984");
                }
                rangeDialog.setVisible(true);
            }
        });

        JPanel colorLabelPanel = new JPanel(new BorderLayout());
        colorLabelPanel.setBackground(HiCGlobals.backgroundColor); //set color to gray
        colorLabelPanel.add(colorRangeLabel, BorderLayout.CENTER);
        add(colorLabelPanel, BorderLayout.PAGE_START);

        //---- colorRangeSlider ----
        colorRangeSlider.setPaintTicks(false);
        colorRangeSlider.setPaintLabels(false);
        colorRangeSlider.setMaximumSize(new Dimension(32767, 52));
        colorRangeSlider.setPreferredSize(new Dimension(200, 52));
        colorRangeSlider.setMinimumSize(new Dimension(36, 52));

        colorRangeSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                double min = colorRangeSlider.getLowerValue() / colorRangeScaleFactor;
                double max = colorRangeSlider.getUpperValue() / colorRangeScaleFactor;

                HiC hic = superAdapter.getHiC();

                String key = "";
                try {
                    if (hic != null && hic.getZd() != null && hic.getDisplayOption() != null) {
                        key = HeatmapRenderer.getColorScaleCacheKey(hic.getZd(), hic.getDisplayOption(), hic.getObsNormalizationType(), hic.getControlNormalizationType());
                    }
                } catch (Exception e2) {
                    if (HiCGlobals.printVerboseComments) {
                        e2.printStackTrace();
                    }
                }

                heatmapPanel.setNewDisplayRange(hic.getDisplayOption(), min, max, key);
                colorRangeSliderUpdateToolTip(hic.getDisplayOption());
            }
        });
        sliderPanel.add(colorRangeSlider);
        JPanel plusMinusPanel = new JPanel();
        plusMinusPanel.setLayout(new BoxLayout(plusMinusPanel, BoxLayout.Y_AXIS));

        plusButton = new JideButton();
        plusButton.setIcon(new ImageIcon(getClass().getResource("/images/zoom-plus.png")));
        plusButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

                HiC hic = superAdapter.getHiC();
                double newMax = (colorRangeSlider.getMaximum() / colorRangeScaleFactor) * 2;
                double newHigh = colorRangeSlider.getUpperValue() / colorRangeScaleFactor;
                double newLow = colorRangeSlider.getLowerValue() / colorRangeScaleFactor;

                if (MatrixType.isOEColorScaleType(hic.getDisplayOption())) {
                    updateRatioColorSlider(hic, newMax, newHigh);
                } else {
                    updateColorSlider(hic, newLow, newHigh, newMax);
                }
            }
        });

        minusButton = new JideButton();
        minusButton.setIcon(new ImageIcon(getClass().getResource("/images/zoom-minus.png")));
        minusButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //Set limit to maximum range:
                HiC hic = superAdapter.getHiC();
                double newMax = (colorRangeSlider.getMaximum() / colorRangeScaleFactor) / 2;
                double newHigh = Math.min(newMax, colorRangeSlider.getUpperValue() / colorRangeScaleFactor);
                double newLow = Math.min(newMax - 1, colorRangeSlider.getLowerValue() / colorRangeScaleFactor);

                if (MatrixType.isOEColorScaleType(hic.getDisplayOption())) {
                    updateRatioColorSlider(hic, newMax, newHigh);
                } else {
                    updateColorSlider(hic, newLow, newHigh, newMax);
                }
            }
        });

        colorRangeSlider.setUpperValue(1200);
        colorRangeSlider.setDisplayToBlank(true);

        plusMinusPanel.add(plusButton);
        plusMinusPanel.add(minusButton);
        plusButton.setEnabled(false);
        minusButton.setEnabled(false);
        sliderPanel.add(plusMinusPanel);
        add(sliderPanel, BorderLayout.PAGE_END);


        setBorder(LineBorder.createGrayLineBorder());
        setMinimumSize(new Dimension(96, 70));
        setPreferredSize(new Dimension(202, 70));
        setMaximumSize(new Dimension(32769, 70));
    }


    public boolean setColorRangeSliderVisible(boolean state, SuperAdapter superAdapter) {
        plusButton.setEnabled(state);
        minusButton.setEnabled(state);
        colorRangeSlider.setEnabled(state);
        if (state) {
            colorRangeLabel.setForeground(Color.BLUE);
        } else {
            colorRangeLabel.setForeground(Color.BLACK);
        }
        return true;
        //why are we calling this?  why is this method a boolean method at all?
        //return superAdapter.safeDisplayOptionComboBoxActionPerformed();
    }

    public void updateColorSlider(HiC hic, double lower, double upper, double max) {
        if (max < 1) {
            max = 5;
        } // map going into zero state?
        if (upper == 0) {
            upper = 1;
        }
        if (upper == lower) {
            lower = upper - 1;
        }

        // We need to scale min and max to integers for the slider to work.  Scale such that there are
        // 100 divisions between max and 0
        colorRangeScaleFactor = 100.0 / max;

        colorRangeSlider.setPaintTicks(true);
        colorRangeSlider.setPaintLabels(true);

        int iMax = (int) (colorRangeScaleFactor * max);
        int lValue = (int) (colorRangeScaleFactor * lower);
        int uValue = (int) (colorRangeScaleFactor * upper);

        colorRangeSlider.setMinimum(0);
        colorRangeSlider.setMaximum(iMax);
        colorRangeSlider.setLowerValue(lValue);
        colorRangeSlider.setUpperValue(uValue);

        Hashtable<Integer, JLabel> labelTable = new Hashtable<>();

        Font f = FontManager.getFont(8);

        final JLabel minTickLabel = new JLabel(String.valueOf(0));
        minTickLabel.setFont(f);
        final JLabel maxTickLabel = new JLabel(String.valueOf((int) max));
        maxTickLabel.setFont(f);
        final JLabel LoTickLabel = new JLabel(String.valueOf((int) lower));
        LoTickLabel.setFont(f);
        final JLabel UpTickLabel = new JLabel(String.valueOf((int) upper));
        UpTickLabel.setFont(f);

        labelTable.put(0, minTickLabel);
        labelTable.put(iMax, maxTickLabel);
        labelTable.put(lValue, LoTickLabel);
        labelTable.put(uValue, UpTickLabel);

        colorRangeSlider.setLabelTable(labelTable);

        String key = HeatmapRenderer.getColorScaleCacheKey(hic.getZd(), hic.getDisplayOption(), hic.getObsNormalizationType(), hic.getControlNormalizationType());
        heatmapPanel.setNewDisplayRange(hic.getDisplayOption(), lower, upper, key);
        colorRangeSliderUpdateToolTip(hic.getDisplayOption());
    }

    public void updateRatioColorSlider(HiC hic, double max, double val) {
        // We need to scale min and max to integers for the slider to work.  Scale such that there are
        // 100 divisions between max and 0
        if (max < 1) {
            max = 5;
        } // map going into zero state?
        if (val == 0) {
            val = 1;
        }

        colorRangeScaleFactor = 100.0 / (2 * max);

        colorRangeSlider.setPaintTicks(true);
        colorRangeSlider.setPaintLabels(true);

        int iMin = (int) (colorRangeScaleFactor * -max);
        int iMax = (int) (colorRangeScaleFactor * max);
        int lValue = (int) (colorRangeScaleFactor * -val);
        int uValue = (int) (colorRangeScaleFactor * val);

        colorRangeSlider.setMinimum(iMin);
        colorRangeSlider.setMaximum(iMax);
        colorRangeSlider.setLowerValue(lValue);
        colorRangeSlider.setUpperValue(uValue);

        Hashtable<Integer, JLabel> labelTable = new Hashtable<>();

        Font f = FontManager.getFont(8);

        final JLabel minTickLabel = new JLabel(String.valueOf((int) -max));
        minTickLabel.setFont(f);
        final JLabel maxTickLabel = new JLabel(String.valueOf((int) max));
        maxTickLabel.setFont(f);
        final JLabel LoTickLabel = new JLabel(String.valueOf((int) -val));
        LoTickLabel.setFont(f);
        final JLabel UpTickLabel = new JLabel(String.valueOf((int) val));
        UpTickLabel.setFont(f);

        labelTable.put(iMin, minTickLabel);
        labelTable.put(iMax, maxTickLabel);
        labelTable.put(lValue, LoTickLabel);
        labelTable.put(uValue, UpTickLabel);

        colorRangeSlider.setLabelTable(labelTable);

        String key = HeatmapRenderer.getColorScaleCacheKey(hic.getZd(), hic.getDisplayOption(), hic.getObsNormalizationType(), hic.getControlNormalizationType());
        heatmapPanel.setNewDisplayRange(hic.getDisplayOption(), -val, val, key);
        colorRangeSliderUpdateToolTip(hic.getDisplayOption());
    }

    public String getColorRangeValues() {

        int iMin = colorRangeSlider.getMinimum();
        int lowValue = colorRangeSlider.getLowerValue();
        int upValue = colorRangeSlider.getUpperValue();
        int iMax = colorRangeSlider.getMaximum();

        return iMin + "$$" + lowValue + "$$" + upValue + "$$" + iMax;// + "$$" + dScaleFactor;

    }

    public double getColorRangeScaleFactor() {
        return colorRangeScaleFactor;
    }

    private void colorRangeSliderUpdateToolTip(MatrixType displayOption) {
        int iMin = colorRangeSlider.getMinimum();
        int lValue = colorRangeSlider.getLowerValue();
        int uValue = colorRangeSlider.getUpperValue();
        int iMax = colorRangeSlider.getMaximum();

        colorRangeSlider.setToolTipText("<html>Range: " + (int) (iMin / colorRangeScaleFactor) + " " + (int) (iMax / colorRangeScaleFactor) +
                "<br>Showing: " + (int) (lValue / colorRangeScaleFactor) + " - " + (int) (uValue / colorRangeScaleFactor) + "</html>");

        Font f = FontManager.getFont(8);

        Hashtable<Integer, JLabel> labelTable = new Hashtable<>();

        if (MatrixType.isOEColorScaleType(displayOption)) {
            colorRangeSlider.setToolTipText("Log Enrichment Values");
        } else {
            colorRangeSlider.setToolTipText("Observed Counts");
        }

        final JLabel minTickLabel = new JLabel(String.valueOf((int) (iMin / colorRangeScaleFactor)));
        minTickLabel.setFont(f);
        final JLabel LoTickLabel = new JLabel(String.valueOf((int) (lValue / colorRangeScaleFactor)));
        LoTickLabel.setFont(f);
        final JLabel UpTickLabel = new JLabel(String.valueOf((int) (uValue / colorRangeScaleFactor)));
        UpTickLabel.setFont(f);
        final JLabel maxTickLabel = new JLabel(String.valueOf((int) (iMax / colorRangeScaleFactor)));
        maxTickLabel.setFont(f);

        labelTable.put(iMin, minTickLabel);
        labelTable.put(lValue, LoTickLabel);
        labelTable.put(uValue, UpTickLabel);
        labelTable.put(iMax, maxTickLabel);

        colorRangeSlider.setLabelTable(labelTable);
    }

    public void setElementsVisible(boolean val, SuperAdapter superAdapter) {
        setColorRangeSliderVisible(val, superAdapter);
        colorRangeSlider.setDisplayToBlank(!val);
        plusButton.setEnabled(val);
        minusButton.setEnabled(val);
    }

    public void handleNewFileLoading(MatrixType option) {
        boolean isColorScaleType = MatrixType.isColorScaleType(option);
        colorRangeSlider.setEnabled(isColorScaleType);
        colorRangeSlider.setDisplayToOE(MatrixType.isOEColorScaleType(option));
        plusButton.setEnabled(isColorScaleType);
        minusButton.setEnabled(isColorScaleType);
    }
}
