/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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
import juicebox.gui.MainViewPanel;
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
    private int[] colorValuesToRestore = null;

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
                        colorRangeSlider, colorRangeScaleFactor, superAdapter.getHiC().getDisplayOption() == MatrixType.OBSERVED);
                setColorRangeSliderVisible(false, superAdapter);
                superAdapter.getMainViewPanel().setResolutionSliderVisible(false, superAdapter);
                rangeDialog.setVisible(true);
            }
        });

        JPanel colorLabelPanel = new JPanel();
        colorLabelPanel.setBackground(HiCGlobals.backgroundColor); //set color to gray
        colorLabelPanel.setLayout(new BorderLayout());
        colorLabelPanel.add(colorRangeLabel, BorderLayout.CENTER);

        add(colorLabelPanel, BorderLayout.PAGE_START);

        //---- colorRangeSlider ----
        colorRangeSlider.setPaintTicks(false);
        colorRangeSlider.setPaintLabels(false);
        colorRangeSlider.setMaximumSize(new Dimension(32767, 52));
        colorRangeSlider.setPreferredSize(new Dimension(200, 52));
        colorRangeSlider.setMinimumSize(new Dimension(36, 52));
        resetRegularColorRangeSlider(activatePreDef);

        colorRangeSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                double min = colorRangeSlider.getLowerValue() / colorRangeScaleFactor;
                double max = colorRangeSlider.getUpperValue() / colorRangeScaleFactor;

                HiC hic = superAdapter.getHiC();
                if (hic.getDisplayOption() == MatrixType.OE || hic.getDisplayOption() == MatrixType.RATIO) {
                    //System.out.println(colorRangeSlider.getUpperValue());
                    heatmapPanel.setOEMax(colorRangeSlider.getUpperValue());
                } else if (MainViewPanel.preDefMapColor) {
                    heatmapPanel.setPreDefRange(min, max);
                } else {
                    heatmapPanel.setObservedRange(min, max);
                }
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

                if (hic.getDisplayOption() == MatrixType.OE || hic.getDisplayOption() == MatrixType.RATIO) {
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
                    if (hic.getDisplayOption() == MatrixType.OE || hic.getDisplayOption() == MatrixType.RATIO) {
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


    public void setColorRangeSliderVisible(boolean state, SuperAdapter superAdapter) {
        plusButton.setEnabled(state);
        minusButton.setEnabled(state);
        colorRangeSlider.setEnabled(state);
        if (state) {
            colorRangeLabel.setForeground(Color.BLUE);
        } else {
            colorRangeLabel.setForeground(Color.BLACK);
        }
        superAdapter.safeDisplayOptionComboBoxActionPerformed();
    }

    public void updateColorSlider(HiC hic, double min, double lower, double upper, double max) {
        // We need to scale min and max to integers for the slider to work.  Scale such that there are
        // 100 divisions between max and 0

        colorRangeScaleFactor = 100.0 / max;

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

        Hashtable<Integer, JLabel> labelTable = new Hashtable<Integer, JLabel>();

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
        //TODO******   UNCOMMENT  ******
        colorRangeSliderUpdateToolTip(hic);

    }

    private void resetRegularColorRangeSlider(boolean preDefMapColor) {
        if (colorValuesToRestore != null) {
            //refreshChromosomes();
            //setInitialZoom();
            colorRangeSlider.setDisplayToBlank(false);
            if (preDefMapColor) {
                colorRangeSlider.setDisplayToPreDef(true);
                colorRangeSlider.setDisplayToOE(false);
            } else {
                colorRangeSlider.setDisplayToPreDef(false);
                colorRangeSlider.setDisplayToOE(false);
            }
            colorRangeSlider.setMinimum(colorValuesToRestore[0]);
            colorRangeSlider.setMaximum(colorValuesToRestore[1]);
            colorRangeSlider.setLowerValue(colorValuesToRestore[2]);
            colorRangeSlider.setUpperValue(colorValuesToRestore[3]);
            colorRangeScaleFactor = colorValuesToRestore[4];

            //refresh();
            colorValuesToRestore = null;
        }
    }

    private void resetOEColorRangeSlider() {

        colorRangeSlider.setDisplayToBlank(false);
        if (colorValuesToRestore == null) {
            colorValuesToRestore = new int[5];
            colorValuesToRestore[0] = colorRangeSlider.getMinimum();
            colorValuesToRestore[1] = colorRangeSlider.getMaximum();
            colorValuesToRestore[2] = colorRangeSlider.getLowerValue();
            colorValuesToRestore[3] = colorRangeSlider.getUpperValue();
            colorValuesToRestore[4] = (int) colorRangeScaleFactor;
        }

        colorRangeSlider.setMinimum(-20);
        colorRangeSlider.setMaximum(20);
        colorRangeSlider.setLowerValue(-5);
        colorRangeSlider.setUpperValue(5);

    }

    public String getColorRangeValues() {

        int iMin = colorRangeSlider.getMinimum();
        int lowValue = colorRangeSlider.getLowerValue();
        int upValue = colorRangeSlider.getUpperValue();
        int iMax = colorRangeSlider.getMaximum();

        return iMin + "$$" + lowValue + "$$" + upValue + "$$" + iMax;

    }

    private void colorRangeSliderUpdateToolTip(HiC hic) {
        if (hic.getDisplayOption() == MatrixType.OBSERVED ||
                hic.getDisplayOption() == MatrixType.CONTROL ||
                hic.getDisplayOption() == MatrixType.OE || hic.getDisplayOption() == MatrixType.RATIO) {

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

            Hashtable<Integer, JLabel> labelTable = new Hashtable<Integer, JLabel>();


            if (hic.getDisplayOption() == MatrixType.OE || hic.getDisplayOption() == MatrixType.RATIO) {
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
        // ((ColorRangeModel)colorRangeSlider.getModel()).setObserved(option == MatrixType.OBSERVED || option == MatrixType.CONTROL || option == MatrixType.EXPECTED);
        boolean activateOE = option == MatrixType.OE || option == MatrixType.RATIO;
        boolean isObservedOrControl = option == MatrixType.OBSERVED || option == MatrixType.CONTROL;

        colorRangeSlider.setEnabled(option == MatrixType.OBSERVED || option == MatrixType.CONTROL || activateOE || activatePreDef);
        colorRangeSlider.setDisplayToOE(activateOE);
        colorRangeSlider.setDisplayToPreDef(activatePreDef);

        if (activateOE) {
            resetOEColorRangeSlider();
        } else {
            resetRegularColorRangeSlider(activatePreDef);
        }

        plusButton.setEnabled(activateOE || isObservedOrControl);
        minusButton.setEnabled(activateOE || isObservedOrControl);
    }

    public void resetPreFileLoad() {
        colorValuesToRestore = null;
    }
}
