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
import htsjdk.samtools.seekablestream.SeekableHTTPStream;
import juicebox.HiC;
import juicebox.MainWindow;
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
import java.io.*;
import java.net.URL;
import java.util.HashMap;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * Created by nchernia on 4/2/15.
 */
public class GoToPanel extends JPanel implements ActionListener, FocusListener {
    private static JideButton goButton;
    private static JTextField positionChrLeft;
    private static JTextField positionChrTop;
    private HiC hic;
    private String genomeID;
    private HashMap<String, GeneLocation> geneLocationHashMap = null;
    private static final Logger log = Logger.getLogger(GoToPanel.class);

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
        positionChrTop.addFocusListener(this);
        //positionChrTop.setPreferredSize(new Dimension(10, 10));


        positionChrLeft = new JTextField();
        positionChrLeft.setFont(new Font("Arial", Font.ITALIC, 10));
        positionChrLeft.setEnabled(false);
        positionChrLeft.addActionListener(this);
        positionChrLeft.addFocusListener(this);
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
        if (positionChrLeft.getText().isEmpty() && positionChrTop.getText().isEmpty()) {
            positionChrTop.setBackground(Color.yellow);
            positionChrLeft.setBackground(Color.yellow);
        }
        else if (positionChrLeft.getText().isEmpty()) {
            positionChrLeft.setText(positionChrTop.getText());
        }
        else if (positionChrTop.getText().isEmpty()) {
            positionChrTop.setText(positionChrLeft.getText());
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
        int outBinSize = 0;
        int outBinLeft = 0;
        int outBinTop = 0;
        int topStart = 0;
        int topEnd = 0;
        int leftStart = 0;
        int leftEnd = 0;

        String[] leftChrTokens = positionChrLeft.getText().split(delimiters);
        String[] topChrTokens = positionChrTop.getText().split(delimiters);
        String[] leftDashChrTokens = positionChrLeft.getText().split(dashDelimiters);
        String[] topDashChrTokens = positionChrTop.getText().split(dashDelimiters);

        String resolutionUnits = "BP";
        Chromosome leftChr;
        Chromosome topChr;

        if (topChrTokens.length == 1) {
            parseGenePositionText();
            return;
        }
        //Read Chromosomes:
        //First chromosome:
        List<Chromosome> chromosomeList = hic.getDataset().getChromosomes();

        HashMap<String, Chromosome> chromosomeMap = new HashMap<String, Chromosome>();
        for (Chromosome c : chromosomeList) {
            chromosomeMap.put(c.getName().toLowerCase(), c);
            chromosomeMap.put("chr" + c.getName().toLowerCase(), c);
            if (c.getName().equals("MT")) chromosomeMap.put("chrm", c);
        }


        topChr = chromosomeMap.get(topChrTokens[0].toLowerCase());
        if (topChr == null) {
            positionChrTop.setBackground(Color.yellow);
            log.error("Cannot find " + topChrTokens[0] + " in dataset's chromosome list");
            return;
        }
        leftChr = chromosomeMap.get(leftChrTokens[0].toLowerCase());
        if (leftChr == null) {
            positionChrLeft.setBackground(Color.yellow);
            log.error("Cannot find " + leftChrTokens[0] + " in dataset's chromosome list");
            return;
        }

        //Read positions:
        if (topChrTokens.length > 2 && topDashChrTokens.length > 1) {
            //Make sure values are numerical:
            try {
                topStart = Integer.min(Integer.valueOf(topChrTokens[1].replaceAll(",", "")), Integer.valueOf(topChrTokens[2].replaceAll(",", "")));
                topEnd = Integer.max(Integer.valueOf(topChrTokens[1].replaceAll(",", "")), Integer.valueOf(topChrTokens[2].replaceAll(",", "")));
            } catch (Exception e) {
                positionChrTop.setBackground(Color.yellow);
                log.error("Cannot parse " + topChrTokens[1] + " or " +  topChrTokens[2] + ". Expecting int");
                return;
            }
            outBinTop = topStart + ((topEnd - topStart) / 2);
        } else if (topChrTokens.length > 1) {
            //Make sure values are numerical:
            try {
                outBinTop = Integer.valueOf(topChrTokens[1].replaceAll(",", ""));
            } catch (Exception e) {
                positionChrTop.setBackground(Color.yellow);
                log.error("Cannot parse " + topChrTokens[1] + ". Expecting int");
                return;
            }

        }

        if (leftChrTokens.length > 2 && leftDashChrTokens.length > 1) {
            //Make sure values are numerical:
            try {
                leftStart = Integer.min(Integer.valueOf(leftChrTokens[1].replaceAll(",", "")), Integer.valueOf(leftChrTokens[2].replaceAll(",", "")));
                leftEnd = Integer.max(Integer.valueOf(leftChrTokens[1].replaceAll(",", "")), Integer.valueOf(leftChrTokens[2].replaceAll(",", "")));
            } catch (Exception e) {
                positionChrLeft.setBackground(Color.yellow);
                log.error("Cannot parse " + leftChrTokens[1] + " or " +  leftChrTokens[2] + ". Expecting int");
                return;
            }
            outBinLeft = leftStart + ((leftEnd - leftStart) / 2);
        } else if (leftChrTokens.length > 1) {
            //Make sure values are numerical:
            try {
                outBinLeft = Integer.valueOf(leftChrTokens[1].replaceAll(",", ""));
            } catch (Exception e) {
                positionChrLeft.setBackground(Color.yellow);
                log.error("Cannot parse " + leftChrTokens[1] + ". Expecting int");
                return;
            }
        }

        //Read resolution:
        if (topChrTokens.length > 3 || (topDashChrTokens.length == 1 && topChrTokens.length > 2)) {
            if (topDashChrTokens.length == 1) {
                outBinSize = hic.validateBinSize(topChrTokens[2].toLowerCase());
                if (outBinSize != Integer.MIN_VALUE && topChrTokens[2].toLowerCase().contains("f")) {
                    resolutionUnits = "FRAG";
                } else {
                    positionChrTop.setBackground(Color.yellow);
                    log.error("Invalid resolution " + topChrTokens[2].toLowerCase());
                    return;
                }
            } else if (topChrTokens.length > 3) {
                outBinSize = hic.validateBinSize(topChrTokens[3].toLowerCase());
                if (outBinSize != Integer.MIN_VALUE && topChrTokens[3].toLowerCase().contains("f")) {
                    resolutionUnits = "FRAG";
                } else if (outBinSize == Integer.MIN_VALUE) {
                    positionChrTop.setBackground(Color.yellow);
                    log.error("Invalid resolution " + topChrTokens[3].toLowerCase());
                    return;
                }
            }
        } else if (leftChrTokens.length > 3 || (leftDashChrTokens.length == 1 && leftChrTokens.length > 2)) {
            if (leftDashChrTokens.length == 1) {
                outBinSize = hic.validateBinSize(leftChrTokens[2].toLowerCase());
                if (outBinSize != Integer.MIN_VALUE && leftChrTokens[2].toLowerCase().contains("f")) {
                    resolutionUnits = "FRAG";
                } else if (outBinSize == Integer.MIN_VALUE) {
                    positionChrLeft.setBackground(Color.yellow);
                    log.error("Invalid resolution " + leftChrTokens[2].toLowerCase());
                    return;
                }
            } else if (leftChrTokens.length > 3) {
                outBinSize = hic.validateBinSize(leftChrTokens[3].toLowerCase());
                if (outBinSize != Integer.MIN_VALUE && leftChrTokens[3].toLowerCase().contains("f")) {
                    resolutionUnits = "FRAG";
                } else {
                    positionChrLeft.setBackground(Color.yellow);
                    log.error("Invalid resolution " + leftChrTokens[3].toLowerCase());
                    return;
                }
            }
        } else if (hic.getZoom().getBinSize() != 0) {
            outBinSize = hic.validateBinSize(String.valueOf(hic.getZoom().getBinSize()));
            if (outBinSize != Integer.MIN_VALUE) {
                resolutionUnits = hic.getZoom().getUnit().toString();
            }
        }

        positionChrTop.setBackground(Color.white);
        positionChrLeft.setBackground(Color.white);

        if (outBinSize == Integer.MIN_VALUE) {
            // If bin size is not valid, set to max bin size:
            outBinSize = 250000;
        }

        hic.setState(topChr.getName(), leftChr.getName(), resolutionUnits, outBinSize, 0, 0, hic.getScaleFactor());
        if (outBinTop > 0 && outBinLeft > 0) {
            hic.centerBP(outBinTop, outBinLeft);
        }

        //We might end with ALL->All view, make sure normalization state is updates accordingly...
        MainWindow.getInstance().setNormalizationDisplayState();
    }

    private void parseGenePositionText() {
        String genomeID = hic.getDataset().getGenomeId();
        // Currently only human and mouse, not worrying about small differences in location between genomes
        if (genomeID.equals("b37") || genomeID.equals("hg38")) genomeID = "hg19";
        if (genomeID.equals("mm10")) genomeID = "mm9";
        if (geneLocationHashMap == null || !genomeID.equals(this.genomeID)) {
            initializeGeneHashMap(genomeID);
        }
        else {
            GeneLocation location1 = geneLocationHashMap.get(positionChrTop.getText().trim());
            GeneLocation location2 = geneLocationHashMap.get(positionChrLeft.getText().trim());
            if (location1 == null) {
                positionChrTop.setBackground(Color.yellow);
                log.error("Gene location map doesn't contain " + positionChrTop.getText().trim());
                return;
            }
            if (location2 == null) {
                positionChrLeft.setBackground(Color.yellow);
                log.error("Gene location map doesn't contain " + positionChrLeft.getText().trim());
                return;
            }
            hic.setState(location1.chromosome, location2.chromosome, "BP", 5000, 0, 0, hic.getScaleFactor());
            hic.centerBP(location1.centerPosition, location2.centerPosition);
            MainWindow.getInstance().setNormalizationDisplayState();
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
            MainWindow.getInstance().executeLongRunningTask(runnable, "Initialize Gene Hash Map");
        }
        else {
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

            reader = new BufferedReader(new InputStreamReader(stream));
        }
        catch (Exception error) {
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
        }
        catch (Exception error) {
            MessageUtils.showErrorMessage("Failed to parse gene database", error);
            positionChrTop.setBackground(Color.yellow);
            geneLocationHashMap = null;

        }
        if (geneLocationHashMap != null) this.genomeID = genomeID;
        GeneLocation location1 = geneLocationHashMap.get(positionChrTop.getText().trim());
        GeneLocation location2 = geneLocationHashMap.get(positionChrLeft.getText().trim());
        if (location1 == null) {
            positionChrTop.setBackground(Color.yellow);
            log.error("Gene location map doesn't contain " + positionChrTop.getText().trim());
            return;
        }
        if (location2 == null) {
            positionChrLeft.setBackground(Color.yellow);
            log.error("Gene location map doesn't contain " + positionChrLeft.getText().trim());
            return;
        }
        hic.setState(location1.chromosome, location2.chromosome, "BP", 5000, 0, 0, hic.getScaleFactor());
        hic.centerBP(location1.centerPosition, location2.centerPosition);
        MainWindow.getInstance().setNormalizationDisplayState();
    }

    private class GeneLocation {
        private String chromosome;
        private int centerPosition;

        private GeneLocation(String chromosome, int centerPosition) {
            this.chromosome = chromosome;
            this.centerPosition = centerPosition;
        }
    }

    public void focusGained(FocusEvent event) {
        if (event.getSource() == positionChrLeft) positionChrLeft.setBackground(Color.white);
        else if (event.getSource() == positionChrTop) positionChrTop.setBackground(Color.white);
    }
    public void focusLost(FocusEvent event) {

    }

}
