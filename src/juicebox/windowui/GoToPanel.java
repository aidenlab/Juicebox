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

package juicebox.windowui;

import com.jidesoft.swing.JideButton;
import htsjdk.samtools.seekablestream.SeekableHTTPStream;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.gui.SuperAdapter;
import org.apache.log4j.Logger;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.ui.util.MessageUtils;

import javax.swing.*;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.HashMap;

/**
 * Created by nchernia on 4/2/15.
 */
public class GoToPanel extends JPanel implements ActionListener, FocusListener {
    private static final long serialVersionUID = -6639157254305571236L;
    private static final Logger log = Logger.getLogger(GoToPanel.class);
    private static JideButton goButton;
    private static JTextField positionChrLeft;
    private static JTextField positionChrTop;
    private final HiC hic;
    private final SuperAdapter superAdapter;
    private String genomeID;
    private HashMap<String, GeneLocation> geneLocationHashMap = null;

    public GoToPanel(SuperAdapter superAdapter) {
        super();
        this.hic = superAdapter.getHiC();
        this.superAdapter = superAdapter;

        JLabel goLabel = new JLabel("Goto");
        goLabel.setHorizontalAlignment(SwingConstants.CENTER);

        JPanel goLabelPanel = new JPanel();
        goLabelPanel.setBackground(HiCGlobals.backgroundColor);
        goLabelPanel.setLayout(new BorderLayout());
        goLabelPanel.add(goLabel, BorderLayout.CENTER);

        positionChrTop = initializeGoToTextField();
        positionChrLeft = initializeGoToTextField();

        JPanel goPositionPanel = new JPanel();
        goPositionPanel.setLayout(new BorderLayout());
        goPositionPanel.add(positionChrTop, BorderLayout.PAGE_START);
        goPositionPanel.add(positionChrLeft, BorderLayout.PAGE_END);

        goButton = new JideButton();
        goButton.setEnabled(false);
        goButton.setIcon(new ImageIcon(getClass().getResource("/toolbarButtonGraphics/general/Refresh24.gif")));
        goButton.addActionListener(this);

        JPanel goButtonPanel = new JPanel();
        goButtonPanel.setBackground(new Color(238, 238, 238));
        goButtonPanel.setLayout(new BoxLayout(goButtonPanel, BoxLayout.X_AXIS));
        goButtonPanel.add(goPositionPanel, BorderLayout.PAGE_START);
        goButtonPanel.add(goButton);

        setBackground(new Color(238, 238, 238));
        setBorder(LineBorder.createGrayLineBorder());
        setLayout(new BorderLayout());
        add(goLabelPanel, BorderLayout.PAGE_START);
        add(goButtonPanel);
        setMinimumSize(new Dimension(100, 70));
        setPreferredSize(new Dimension(120, 70));
        setMaximumSize(new Dimension(200, 70));
    }

    public JideButton getGoButton() {
        return goButton;
    }

    public JTextField getPositionChrTop() {
        return positionChrTop;
    }

    public void setPositionChrTop(String newPositionDate) {
        positionChrTop.setText(newPositionDate);
    }

    public JTextField getPositionChrLeft() {
        return positionChrLeft;
    }

    public void setPositionChrLeft(String newPositionDate) {
        positionChrLeft.setText(newPositionDate);
    }

    private JTextField initializeGoToTextField() {
        JTextField textField = new JTextField();
        textField.setFont(new Font("Arial", Font.ITALIC, 10));
        textField.setEnabled(false);
        textField.addActionListener(this);
        textField.addFocusListener(this);
        return textField;
    }

    public void setEnabled(boolean enabled) {
        super.setEnabled(enabled);
        positionChrTop.setEnabled(enabled);
        positionChrLeft.setEnabled(enabled);
        goButton.setEnabled(enabled);
    }

    public void actionPerformed(ActionEvent event) {
        if (positionChrLeft.getText().isEmpty() && positionChrTop.getText().isEmpty()) {
            positionChrTop.setBackground(Color.yellow);
            positionChrLeft.setBackground(Color.yellow);
        } else if (positionChrLeft.getText().isEmpty()) {
            positionChrLeft.setText(positionChrTop.getText());
        } else if (positionChrTop.getText().isEmpty()) {
            positionChrTop.setText(positionChrLeft.getText());
        }
        parsePositionText();
    }

    private void parsePositionText() {
        //Expected format 1: <chr>:<start>-<end>:<resolution>
        //Expected format 2: <chr>:<midpt>:<resolution>

        String delimiters = "\\s+|:\\s*|\\-\\s*";
        String dashDelimiters = "\\s+|\\-\\s*";

        String[] leftChrTokens = positionChrLeft.getText().split(delimiters);
        String[] topChrTokens = positionChrTop.getText().split(delimiters);
        String[] leftDashChrTokens = positionChrLeft.getText().split(dashDelimiters);
        String[] topDashChrTokens = positionChrTop.getText().split(dashDelimiters);

        if (topChrTokens.length == 1 || leftChrTokens.length == 1) {
            parseGenePositionText();
            return;
        }

        //Read Chromosomes:
        HashMap<String, Chromosome> chromosomeMap = new HashMap<String, Chromosome>();
        for (Chromosome c : hic.getDataset().getChromosomes()) {
            chromosomeMap.put(c.getName().toLowerCase(), c);
            chromosomeMap.put("chr" + c.getName().toLowerCase(), c);
            if (c.getName().equals("MT")) chromosomeMap.put("chrm", c);
        }

        Chromosome topChr = chromosomeMap.get(topChrTokens[0].toLowerCase());
        if (topChr == null) {
            positionChrTop.setBackground(Color.yellow);
            log.error("Cannot find " + topChrTokens[0] + " in dataset's chromosome list");
            return;
        }

        Chromosome leftChr = chromosomeMap.get(leftChrTokens[0].toLowerCase());
        if (leftChr == null) {
            positionChrLeft.setBackground(Color.yellow);
            log.error("Cannot find " + leftChrTokens[0] + " in dataset's chromosome list");
            return;
        }

        // chrPositions {start, end, outBin, estimatedOutBinSize}
        int[] topChrPositions;
        try {
            topChrPositions = extractParametersFromTokens(topChrTokens, topDashChrTokens, positionChrTop);
        } catch (Exception e) {
            return;
        }

        int[] leftChrPositions;
        try {
            leftChrPositions = extractParametersFromTokens(leftChrTokens, leftDashChrTokens, positionChrLeft);
        } catch (Exception e) {
            return;
        }

        //Read resolution:
        int outBinSize = 0;
        String resolutionUnits = "BP";
        int estimatedOutBinSize = Math.max(topChrPositions[3], leftChrPositions[3]);

        if (topChrTokens.length > 3 || (topDashChrTokens.length == 1 && topChrTokens.length > 2)) {
            try {
                int[] resolutionParameters = extractResolutionParametersFromTokens(topChrTokens, topDashChrTokens, positionChrTop);
                outBinSize = resolutionParameters[0];
                if (resolutionParameters[1] < 0) {
                    resolutionUnits = "FRAG";
                }
            } catch (Exception e) {
                return;
            }
        } else if (leftChrTokens.length > 3 || (leftDashChrTokens.length == 1 && leftChrTokens.length > 2)) {
            try {
                int[] resolutionParameters = extractResolutionParametersFromTokens(leftChrTokens, leftDashChrTokens, positionChrLeft);
                outBinSize = resolutionParameters[0];
                if (resolutionParameters[1] < 0) {
                    resolutionUnits = "FRAG";
                }
            } catch (Exception e) {
                return;
            }
        } else if (estimatedOutBinSize > 0) {
            outBinSize = estimatedOutBinSize;
        } else if (hic.getZoom().getBinSize() != 0) { //no resolution specified, not at whole genome view
            outBinSize = hic.validateBinSize(String.valueOf(hic.getZoom().getBinSize()));
            if (outBinSize != Integer.MIN_VALUE) {
                resolutionUnits = hic.getZoom().getUnit().toString();
            }
        }

        positionChrTop.setBackground(Color.white);
        positionChrLeft.setBackground(Color.white);

        if (outBinSize == Integer.MIN_VALUE) {
            outBinSize = 250000; // If bin size is not valid, set to max bin size
        }

        hic.setLocation(topChr.getName(), leftChr.getName(), resolutionUnits, outBinSize, Math.max(topChrPositions[2], 0),
                Math.max(leftChrPositions[2], 0), hic.getScaleFactor(), HiC.ZoomCallType.STANDARD, "Goto", true);

    }

    private int[] extractResolutionParametersFromTokens(String[] chrTokens, String[] dashChrTokens, JTextField positionChr) {
        int outBinSize = 0;
        int resolutionUnits = 1;//BP

        if (dashChrTokens.length == 1) {
            outBinSize = hic.validateBinSize(chrTokens[2].toLowerCase());
            if (outBinSize != Integer.MIN_VALUE && chrTokens[2].toLowerCase().contains("f")) {
                resolutionUnits = -1; //FRAG
            } else if (outBinSize == Integer.MIN_VALUE) {
                positionChr.setBackground(Color.yellow);
                log.error("Invalid resolution " + chrTokens[2].toLowerCase());
            }
        } else if (chrTokens.length > 3) {
            outBinSize = hic.validateBinSize(chrTokens[3].toLowerCase());
            if (outBinSize != Integer.MIN_VALUE && chrTokens[3].toLowerCase().contains("f")) {
                resolutionUnits = -1; //FRAG
            } else if (outBinSize == Integer.MIN_VALUE) {
                positionChr.setBackground(Color.yellow);
                log.error("Invalid resolution " + chrTokens[3].toLowerCase());
            }
        }
        return new int[]{outBinSize, resolutionUnits};
    }

    private int[] extractParametersFromTokens(String[] chrTokens, String[] dashChrTokens, JTextField positionChr) throws IOException {
        //Read positions:
        int start = 0, end = 0, outBin = 0, estimatedOutBinSize = 0;
        if (chrTokens.length > 2 && dashChrTokens.length > 1) {
            //Make sure values are numerical:
            try {
                start = Integer.min(cleanUpNumber(chrTokens[1]), cleanUpNumber(chrTokens[2]));
                end = Integer.max(cleanUpNumber(chrTokens[1]), cleanUpNumber(chrTokens[2]));
            } catch (Exception e) {
                log.error("Cannot parse " + chrTokens[1] + " or " + chrTokens[2] + ". Expecting int");
                positionChr.setBackground(Color.yellow);
                throw new IOException();
            }
            outBin = start + ((end - start) / 2);
            int diff = end - start;
            estimatedOutBinSize = getEstimationOfAppropriateZoomLevel(diff);

        } else if (chrTokens.length > 1) {
            //Make sure values are numerical:
            try {
                outBin = cleanUpNumber(chrTokens[1]);
            } catch (Exception e) {
                log.error("Cannot parse " + chrTokens[1] + ". Expecting int");
                positionChr.setBackground(Color.yellow);
                throw new IOException();
            }
        }
        return new int[]{start, end, outBin, estimatedOutBinSize};
    }

    // TODO this should get map keys from official list of resolutions, sort the list, then return appropriately
    private int getEstimationOfAppropriateZoomLevel(int diff0) {
        // divide because the width from x1 to x2 in chromosome should be significantly bigger then the resolution
        int diff = diff0 / 1000;
        if (diff >= 2500000) {
            return 2500000;
        } else if (diff >= 1000000) {
            return 1000000;
        } else if (diff >= 500000) {
            return 500000;
        } else if (diff >= 100000) {
            return 100000;
        } else if (diff >= 50000) {
            return 50000;
        } else if (diff >= 25000) {
            return 25000;
        } else if (diff >= 10000) {
            return 10000;
        } else {
            return 5000;
        }
    }

    private int cleanUpNumber(String number) {
        return Integer.valueOf(number.toLowerCase()
                .replaceAll(",", "")
                .replaceAll("m", "000000")
                .replaceAll("k", "000"));
    }

    private void parseGenePositionText() {
        String genomeID = hic.getDataset().getGenomeId();
        // Currently only human and mouse, not worrying about small differences in location between genomes
        if (genomeID.equals("b37") || genomeID.equals("hg38") || genomeID.equals("hg18")) genomeID = "hg19";
        if (genomeID.equals("mm10")) genomeID = "mm9";
        if (geneLocationHashMap == null || !genomeID.equals(this.genomeID)) {
            initializeGeneHashMap(genomeID);
        } else {
            extractGeneLocation();
        }
    }

    private void initializeGeneHashMap(String genomeID) {
        if (genomeID.equals("hg19") || genomeID.equals("hg38") || genomeID.equals("mm9") || genomeID.equals("mm10")) {
            final String gID = genomeID;
            Runnable runnable = new Runnable() {
                @Override
                public void run() {
                    unsafeInitializeGeneHashMap(gID);
                }
            };
            superAdapter.executeLongRunningTask(runnable, "Initialize Gene Hash Map");
        } else {
            MessageUtils.showErrorMessage("Cannot find genes for " + genomeID, null);
            positionChrTop.setBackground(Color.yellow);
            geneLocationHashMap = null;
        }
    }

    private void unsafeInitializeGeneHashMap(String genomeID) {
        // Custom format parsed from ref Gene file.
        // Name1 Name2 chromosome position (where position is midpoint of transcription start and end)
        String path = "http://hicfiles.s3.amazonaws.com/internal/" + genomeID + "_refGene.txt";
        BufferedReader reader;
        try {
            SeekableHTTPStream stream = new SeekableHTTPStream(new URL(path));

            reader = new BufferedReader(new InputStreamReader(stream), HiCGlobals.bufferSize);
            MessageUtils.showMessage("Loading gene database for " + genomeID + ".\nIt might take few minutes. ");
        } catch (Exception error) {
            MessageUtils.showErrorMessage("Failed to read gene database", error);
            positionChrTop.setBackground(Color.yellow);
            geneLocationHashMap = null;
            return;
        }

        geneLocationHashMap = new HashMap<String, GeneLocation>();
        String nextLine;

        try {
            while ((nextLine = reader.readLine()) != null) {
                String[] values = nextLine.split(" ");
                GeneLocation location = new GeneLocation(values[2].trim(), Integer.valueOf(values[3].trim()));
                geneLocationHashMap.put(values[0].trim(), location);
                geneLocationHashMap.put(values[1].trim(), location);
            }
        } catch (Exception error) {
            MessageUtils.showErrorMessage("Failed to parse gene database", error);
            positionChrTop.setBackground(Color.yellow);
            geneLocationHashMap = null;

        }
        if (geneLocationHashMap != null) this.genomeID = genomeID;
        extractGeneLocation();
    }

    private void extractGeneLocation() {
        GeneLocation location1 = geneLocationHashMap.get(positionChrTop.getText().trim());
        GeneLocation location2 = geneLocationHashMap.get(positionChrLeft.getText().trim());
        if (location1 == null) {
            positionChrTop.setBackground(Color.yellow);
            MessageUtils.showMessage("Gene location map doesn't contain " + positionChrTop.getText().trim());
            return;
        }
        if (location2 == null) {
            positionChrLeft.setBackground(Color.yellow);
            MessageUtils.showMessage("Gene location map doesn't contain " + positionChrLeft.getText().trim());
            return;
        }
        hic.setLocation(location1.chromosome, location2.chromosome, "BP", hic.getZoom().getBinSize(), location1.centerPosition,
                location2.centerPosition, hic.getScaleFactor(), HiC.ZoomCallType.STANDARD, "Gene Goto", true);

        superAdapter.setNormalizationDisplayState();
    }

    public void focusGained(FocusEvent event) {
        if (event.getSource() == positionChrLeft) positionChrLeft.setBackground(Color.white);
        else if (event.getSource() == positionChrTop) positionChrTop.setBackground(Color.white);
    }

    public void focusLost(FocusEvent event) {

    }

    private class GeneLocation {
        private final String chromosome;
        private final int centerPosition;

        private GeneLocation(String chromosome, int centerPosition) {
            this.chromosome = chromosome;
            this.centerPosition = centerPosition;
        }
    }

}
