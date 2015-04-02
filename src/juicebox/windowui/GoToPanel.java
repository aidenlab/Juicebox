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

package juicebox.windowui;

import com.jidesoft.swing.JideButton;
import juicebox.HiC;
import juicebox.MainWindow;

import javax.swing.*;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * Created by nchernia on 4/2/15.
 */
public class GoToPanel extends JPanel implements ActionListener {
    private static JideButton goButton;
    private static JTextField positionChrLeft;
    private static JTextField positionChrTop;
    private HiC hic;

    public GoToPanel(HiC hic) {
        super();
        this.hic = hic;
        setBackground(new Color(238, 238, 238));
        setBorder(LineBorder.createGrayLineBorder());
        setLayout(new BorderLayout());
        JPanel goLabelPanel = new JPanel();
        goLabelPanel.setBackground(new Color(204, 204, 204));
        goLabelPanel.setLayout(new BorderLayout());
        JLabel goLabel = new JLabel("Goto");
        goLabel.setHorizontalAlignment(SwingConstants.CENTER);
        goLabelPanel.add(goLabel, BorderLayout.CENTER);
        add(goLabelPanel, BorderLayout.PAGE_START);
        JPanel goButtonPanel = new JPanel();
        goButtonPanel.setBackground(new Color(238, 238, 238));
        goButtonPanel.setLayout(new BoxLayout(goButtonPanel, BoxLayout.X_AXIS));
        positionChrTop = new JTextField();
        positionChrTop.setFont(new Font("Arial", Font.ITALIC, 10));
        positionChrTop.setEnabled(false);
        positionChrTop.addActionListener(this);
        //positionChrTop.setPreferredSize(new Dimension(10, 10));


        positionChrLeft = new JTextField();
        positionChrLeft.setFont(new Font("Arial", Font.ITALIC, 10));
        positionChrLeft.setEnabled(false);
        positionChrLeft.addActionListener(this);
        //positionChrLeft.setPreferredSize(new Dimension(10, 10));
        JPanel goPositionPanel = new JPanel();
        goPositionPanel.setLayout(new BorderLayout());


        goPositionPanel.add(positionChrTop, BorderLayout.PAGE_START);
        goPositionPanel.add(positionChrLeft, BorderLayout.PAGE_END);

        goButtonPanel.add(goPositionPanel, BorderLayout.PAGE_START);
        goButton = new JideButton();
        goButton.setEnabled(false);
        goButton.setIcon(new ImageIcon(getClass().getResource("/toolbarButtonGraphics/general/Refresh24.gif")));
        goButton.addActionListener(this);
        goButtonPanel.add(goButton);

        add(goButtonPanel);

        setMinimumSize(new Dimension(100, 70));
        setPreferredSize(new Dimension(120, 70));
        setMaximumSize(new Dimension(200, 70));
    }


    public void setEnabled(boolean enabled) {
        super.setEnabled(enabled);
        positionChrTop.setEnabled(enabled);
        positionChrLeft.setEnabled(enabled);
        goButton.setEnabled(enabled);
    }

    public void actionPerformed(ActionEvent event) {
        if (positionChrLeft.getText().isEmpty()) {
            positionChrLeft.setText(positionChrTop.getText());
        }
        parsePositionText();
    }

    public void setPositionChrLeft(String newPositionDate) {
        positionChrLeft.setText(newPositionDate);
    }

    public void setPositionChrTop(String newPositionDate) {
        positionChrTop.setText(newPositionDate);
    }


    public void parsePositionText() {
        //Expected format: <chr>:<start>-<end>:<resolution>

        String delimiters = "\\s+|:\\s*|\\-\\s*";
        String dashDelimiters = "\\s+|\\-\\s*";
        java.lang.Integer outBinSize = 0;
        Long outBinLeft = 0L;
        Long outBinTop = 0L;
        Long topStart = 0L;
        Long topEnd = 0L;
        Long leftStart = 0L;
        Long leftEnd = 0L;

        String[] leftChrTokens = positionChrLeft.getText().split(delimiters);
        String[] topChrTokens = positionChrTop.getText().split(delimiters);
        String[] leftDashChrTokens = positionChrLeft.getText().split(dashDelimiters);
        String[] topDashChrTokens = positionChrTop.getText().split(dashDelimiters);

        String resolutionUnits = "BP";
        String LeftChrName = "";
        String TopChrName = "";
        int LeftChrInt = 0;
        int TopChrInt = 0;

        //Read Chromosomes:
        //First chromosome:
        if (topChrTokens.length > 0) {
            if (topChrTokens[0].toLowerCase().contains("chr")) {
                TopChrName = topChrTokens[0].substring(3);
            } else {
                TopChrName = topChrTokens[0].toLowerCase();
            }
        } else {
            positionChrTop.setBackground(Color.yellow);
            return;
        }
        try {
            TopChrInt = Integer.parseInt(TopChrName);
            //todo - replace with actual chromosome range (won't work right now on genomes outside of human+mouse)
            if (TopChrInt > 22) {
                positionChrTop.setBackground(Color.yellow);
                return;
            }

        } catch (Exception e) {
            if (TopChrName.toLowerCase().equals("x")) {
                TopChrName = "X";
            } else if (TopChrName.toLowerCase().equals("y")) {
                TopChrName = "Y";
            } else if (TopChrName.toLowerCase().equals("mt") || TopChrName.toLowerCase().equals("m")) {
                TopChrName = "MT";
            } else {

                positionChrTop.setBackground(Color.yellow);
                return;
            }
        }

        //Second chromosome:
        if (leftChrTokens.length > 0) {
            if (leftChrTokens[0].toLowerCase().contains("chr")) {
                LeftChrName = leftChrTokens[0].substring(3);
            } else {
                LeftChrName = leftChrTokens[0].toLowerCase();
            }
        } else {
            positionChrLeft.setBackground(Color.yellow);
            return;
        }
        try {
            LeftChrInt = Integer.parseInt(LeftChrName);

            //todo - replace with actual chromosome range (won't work right now on genomes outside of human+mouse)
            if (LeftChrInt > 22) {
                positionChrLeft.setBackground(Color.yellow);
                return;
            }
        } catch (Exception e) {
            if (LeftChrName.toLowerCase().equals("x")) {
                LeftChrName = "X";
            } else if (LeftChrName.toLowerCase().equals("y")) {
                LeftChrName = "Y";
            } else if (LeftChrName.toLowerCase().equals("mt") || LeftChrName.toLowerCase().equals("m")) {
                LeftChrName = "MT";
            } else {
                positionChrLeft.setBackground(Color.yellow);
                return;
            }
        }

        //Read positions:
        if (topChrTokens.length > 2 && topDashChrTokens.length > 1) {
            //Make sure values are numerical:
            try {
                Long.parseLong(topChrTokens[1].replaceAll(",", ""));
            } catch (Exception e) {
                positionChrTop.setBackground(Color.yellow);
                return;
            }
            try {
                Long.parseLong(topChrTokens[2].replaceAll(",", ""));
            } catch (Exception e) {
                positionChrLeft.setBackground(Color.yellow);
                return;
            }
            topStart = Long.min(Long.valueOf(topChrTokens[1].replaceAll(",", "")), Long.valueOf(topChrTokens[2].replaceAll(",", "")));
            topEnd = Long.max(Long.valueOf(topChrTokens[1].replaceAll(",", "")), Long.valueOf(topChrTokens[2].replaceAll(",", "")));
            outBinTop = topStart + ((topEnd - topStart) / 2);

        } else if (topChrTokens.length > 1) {
            outBinTop = Long.valueOf(topChrTokens[1].replaceAll(",", ""));
        }


        if (leftChrTokens.length > 2 && leftDashChrTokens.length > 1) {
            leftStart = Long.min(Long.valueOf(leftChrTokens[1].replaceAll(",", "")), Long.valueOf(leftChrTokens[2].replaceAll(",", "")));
            leftEnd = Long.max(Long.valueOf(leftChrTokens[1].replaceAll(",", "")), Long.valueOf(leftChrTokens[2].replaceAll(",", "")));
            outBinLeft = leftStart + ((leftEnd - leftStart) / 2);
        } else if (topChrTokens.length > 1) {
            //Make sure values are numerical: TODO parsing topChr but later getting value of leftChr - is this correct? seems like a typo
            try {
                Long.parseLong(topChrTokens[1].replaceAll(",", ""));
            } catch (Exception e) {
                positionChrTop.setBackground(Color.yellow);
                return;
            }
            outBinLeft = Long.valueOf(leftChrTokens[1].replaceAll(",", ""));
        }

        //Read resolution:
        if (topChrTokens.length > 3 || (topDashChrTokens.length == 1 && topChrTokens.length > 2)) {
            if (topDashChrTokens.length == 1) {
                outBinSize = hic.validateBinSize(topChrTokens[2].toLowerCase());
                if (outBinSize != null && topChrTokens[2].toLowerCase().contains("f")) {
                    resolutionUnits = "FRAG";
                } else {
                    positionChrTop.setBackground(Color.yellow);
                    return;
                }
            } else if (topChrTokens.length > 3) {
                outBinSize = hic.validateBinSize(topChrTokens[3].toLowerCase());
                if (outBinSize != null && topChrTokens[3].toLowerCase().contains("f")) {
                    resolutionUnits = "FRAG";
                } else if (outBinSize == null) {
                    positionChrTop.setBackground(Color.yellow);
                    return;
                }
            }
        } else if (leftChrTokens.length > 3 || (leftDashChrTokens.length == 1 && leftChrTokens.length > 2)) {
            if (leftDashChrTokens.length == 1) {
                outBinSize = hic.validateBinSize(leftChrTokens[2].toLowerCase());
                if (outBinSize != null && leftChrTokens[2].toLowerCase().contains("f")) {
                    resolutionUnits = "FRAG";
                } else if (outBinSize == null) {
                    positionChrLeft.setBackground(Color.yellow);
                    return;
                }
            } else if (leftChrTokens.length > 3) {
                outBinSize = hic.validateBinSize(leftChrTokens[3].toLowerCase());
                if (outBinSize != null && leftChrTokens[3].toLowerCase().contains("f")) {
                    resolutionUnits = "FRAG";
                } else {
                    positionChrLeft.setBackground(Color.yellow);
                    return;
                }
            }
        } else if (hic.getZoom().getBinSize() != 0) {
            outBinSize = hic.validateBinSize(String.valueOf(hic.getZoom().getBinSize()));
            if (outBinSize != null) {
                resolutionUnits = hic.getZoom().getUnit().toString();
            }
        }

        positionChrTop.setBackground(Color.white);
        positionChrLeft.setBackground(Color.white);

        if (outBinSize == null) {
            // If bin size is not valid, set to max bin size:
            outBinSize = 250000;
        }

        hic.setState(TopChrName, LeftChrName, resolutionUnits, outBinSize, 0, 0, hic.getScaleFactor());
        if (outBinTop > 0 && outBinLeft > 0) {
            hic.centerBP(Math.round(outBinTop), Math.round(outBinLeft));
        }

        //We might end with ALL->All view, make sure normalization state is updates accordingly...
        MainWindow.getInstance().setNormalizationDisplayState();
    }

}
