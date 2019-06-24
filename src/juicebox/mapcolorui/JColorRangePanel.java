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

    private static final long serialVersionUID = -1656738668931765037L;
    private static RangeSlider colorRangeSlider;
    private static JLabel colorRangeLabel;
    private static JButton plusButton, minusButton;
    private double colorRangeScaleFactor = 1;

    public JColorRangePanel(final SuperAdapter superAdapter, final HeatmapPanel heatmapPanel, boolean activatePreDef) {
        super();
        setLayout(new BorderLayout());
        JPanel sliderPanel = new JPanel();
        sliderPanel.setLayout(new BoxLayout(sliderPanel, BoxLayout.X_AXIS));

        colorRangeSlider = new RangeSlider();

        colorRangeSlider.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseEntered(MouseEvent mouseEvent) {
                super.mouseEntered(mouseEvent);
                colorRangeSliderUpdateToolTip(superAdapter.getHiC());
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
                if (superAdapter.getMainViewPanel().setResolutionSliderVisible(false, superAdapter)) {
                    // TODO succeeded
                } else {
                    // TODO failed
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
                } catch (Exception ignored) {
                    if (HiCGlobals.printVerboseComments) {
                        ignored.printStackTrace();
                    }
                }

                heatmapPanel.setNewDisplayRange(hic.getDisplayOption(), min, max, key);
                colorRangeSliderUpdateToolTip(hic);
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

                colorRangeSlider.setMaximum(Math.min(Math.max(colorRangeSlider.getMaximum() * 2, 1), (Integer.MAX_VALUE)));
                HiC hic = superAdapter.getHiC();

                if (MatrixType.isComparisonType(hic.getDisplayOption())) {
                    colorRangeSlider.setMinimum(-colorRangeSlider.getMaximum());
                    colorRangeSlider.setLowerValue(-colorRangeSlider.getUpperValue());
                }
                colorRangeSliderUpdateToolTip(hic);
            }
        });

        minusButton = new JideButton();
        minusButton.setIcon(new ImageIcon(getClass().getResource("/images/zoom-minus.png")));
        minusButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //Set limit to maximum range:
                HiC hic = superAdapter.getHiC();
                int newMax = colorRangeSlider.getMaximum() / 2;
                if (newMax > 0) {
                    colorRangeSlider.setMaximum(newMax);
                    if (MatrixType.isComparisonType(hic.getDisplayOption())) {
                        colorRangeSlider.setMinimum(-newMax);
                        colorRangeSlider.setLowerValue(-colorRangeSlider.getUpperValue());
                    }
                }
                colorRangeSliderUpdateToolTip(hic);
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

    public void updateColorSlider(HiC hic, double min, double lower, double upper, double max) {
        if (max == 0) {
            max = 1;
        } // map going into zero state?
        double scaleFactor = 100.0 / max;
        updateColorSlider(hic, min, lower, upper, max, scaleFactor);
    }

    private void updateColorSlider(HiC hic, double min, double lower, double upper, double max, double scaleFactor) {
        // We need to scale min and max to integers for the slider to work.  Scale such that there are
        // 100 divisions between max and 0

        colorRangeScaleFactor = scaleFactor;

        colorRangeSlider.setPaintTicks(true);
        //colorRangeSlider.setSnapToTicks(true);
        colorRangeSlider.setPaintLabels(true);

        int iMin = (int) (colorRangeScaleFactor * min);
        int iMax = (int) (colorRangeScaleFactor * max);
        int lValue = (int) (colorRangeScaleFactor * lower);
        int uValue = (int) (colorRangeScaleFactor * upper);

        colorRangeSlider.setMinimum(iMin);
        colorRangeSlider.setLowerValue(lValue);
        colorRangeSlider.setUpperValue(uValue);
        colorRangeSlider.setMaximum(iMax);

        Hashtable<Integer, JLabel> labelTable = new Hashtable<>();

        Font f = FontManager.getFont(8);

        final JLabel minTickLabel = new JLabel(String.valueOf((int) min));
        minTickLabel.setFont(f);
        final JLabel maxTickLabel = new JLabel(String.valueOf((int) max));
        maxTickLabel.setFont(f);
        final JLabel LoTickLabel = new JLabel(String.valueOf((int) lower));
        LoTickLabel.setFont(f);
        final JLabel UpTickLabel = new JLabel(String.valueOf((int) upper));
        UpTickLabel.setFont(f);

        labelTable.put(iMin, minTickLabel);
        labelTable.put(iMax, maxTickLabel);
        labelTable.put(lValue, LoTickLabel);
        labelTable.put(uValue, UpTickLabel);


        colorRangeSlider.setLabelTable(labelTable);
        colorRangeSliderUpdateToolTip(hic);
    }

    public void updateRatioColorSlider(HiC hic, double max, double val) {
        // We need to scale min and max to integers for the slider to work.  Scale such that there are
        // 100 divisions between max and 0

        colorRangeScaleFactor = 100.0 / (2 * max);

        colorRangeSlider.setPaintTicks(true);
        colorRangeSlider.setPaintLabels(true);

        int iMin = (int) (colorRangeScaleFactor * -max);
        int iMax = (int) (colorRangeScaleFactor * max);
        int lValue = (int) (colorRangeScaleFactor * -val);
        int uValue = (int) (colorRangeScaleFactor * val);

        colorRangeSlider.setMinimum(iMin);
        colorRangeSlider.setLowerValue(lValue);
        colorRangeSlider.setUpperValue(uValue);
        colorRangeSlider.setMaximum(iMax);

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
        colorRangeSliderUpdateToolTip(hic);
    }

    public String getColorRangeValues() {

        int iMin = colorRangeSlider.getMinimum();
        int lowValue = colorRangeSlider.getLowerValue();
        int upValue = colorRangeSlider.getUpperValue();
        int iMax = colorRangeSlider.getMaximum();
        //double dScaleFactor = colorRangeScaleFactor;

        return iMin + "$$" + lowValue + "$$" + upValue + "$$" + iMax;// + "$$" + dScaleFactor;

    }

    public double getColorRangeScaleFactor() {
        return colorRangeScaleFactor;
    }

    private void colorRangeSliderUpdateToolTip(HiC hic) {

        if (MatrixType.isColorScaleType(hic.getDisplayOption())) {

            int iMin = colorRangeSlider.getMinimum();
            int lValue = colorRangeSlider.getLowerValue();
            int uValue = colorRangeSlider.getUpperValue();
            int iMax = colorRangeSlider.getMaximum();

            /*
            colorRangeSlider.setToolTipText("<html>Range: " + (int) (iMin / colorRangeScaleFactor) + " "

                    + (int) (iMax / colorRangeScaleFactor) + "<br>Showing: " +
                    (int) (lValue / colorRangeScaleFactor) + " "
                    + (int) (uValue / colorRangeScaleFactor)
                    + "</html>");
            */

            Font f = FontManager.getFont(8);

            Hashtable<Integer, JLabel> labelTable = new Hashtable<>();


            if (MatrixType.isComparisonType(hic.getDisplayOption())) {
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

    }

    public void setElementsVisible(boolean val, SuperAdapter superAdapter) {
        setColorRangeSliderVisible(val, superAdapter);
        colorRangeSlider.setDisplayToBlank(!val);
        plusButton.setEnabled(val);
        minusButton.setEnabled(val);
    }

    public void handleNewFileLoading(MatrixType option, boolean activatePreDef) {
        boolean isColorScaleType = MatrixType.isColorScaleType(option);
        colorRangeSlider.setEnabled(isColorScaleType || activatePreDef);
        colorRangeSlider.setDisplayToOE(MatrixType.isComparisonType(option));
        colorRangeSlider.setDisplayToPreDef(activatePreDef);
        plusButton.setEnabled(isColorScaleType);
        minusButton.setEnabled(isColorScaleType);
    }
}
