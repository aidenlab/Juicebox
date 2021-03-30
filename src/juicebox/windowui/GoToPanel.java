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

package juicebox.windowui;

import com.jidesoft.swing.JideButton;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.assembly.Scaffold;
import juicebox.data.ChromosomeHandler;
import juicebox.data.GeneLocation;
import juicebox.data.basics.Chromosome;
import juicebox.gui.SuperAdapter;
import juicebox.tools.utils.juicer.GeneTools;
import juicebox.track.HiCTrack;

import javax.swing.*;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Created by nchernia on 4/2/15.
 */
public class GoToPanel extends JPanel implements ActionListener, FocusListener {
    private static final long serialVersionUID = 9000041;
    private static JideButton goButton;
    private static JTextField positionChrLeft;
    private static JTextField positionChrTop;
    private final HiC hic;
    private final SuperAdapter superAdapter;
    private String genomeID;
    private Map<String, GeneLocation> geneLocationHashMap = null;

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
        goButtonPanel.setBackground(HiCGlobals.diffGrayColor);
        goButtonPanel.setLayout(new BoxLayout(goButtonPanel, BoxLayout.X_AXIS));
        goButtonPanel.add(goPositionPanel, BorderLayout.PAGE_START);
        goButtonPanel.add(goButton);

        setBackground(HiCGlobals.diffGrayColor);
        setBorder(LineBorder.createGrayLineBorder());
        setLayout(new BorderLayout());
        add(goLabelPanel, BorderLayout.PAGE_START);
        add(goButtonPanel);
        setMinimumSize(new Dimension(100, 70));
        setPreferredSize(new Dimension(120, 70));
        setMaximumSize(new Dimension(200, 70));
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

    public void setPositionChrLeft(String newPositionDate) {
        positionChrLeft.setText(newPositionDate);
    }

    public void setPositionChrTop(String newPositionDate) {
        positionChrTop.setText(newPositionDate);
    }

    private void parsePositionText() {

        if (SuperAdapter.assemblyModeCurrentlyActive) {
            goToScaffoldName(positionChrLeft.getText(), positionChrTop.getText());
            return;
        }

        //Expected format 1: <chr>:<start>-<end>:<resolution>
        //Expected format 2: <chr>:<midpt>:<resolution>

        List<String[]> resultLeft = parse(positionChrLeft);
        String[] leftChrTokens = resultLeft.get(0);
        String[] leftDashChrTokens = resultLeft.get(1);

        List<String[]> resultTop = parse(positionChrTop);
        String[] topChrTokens = resultTop.get(0);
        String[] topDashChrTokens = resultTop.get(1);

        if (topChrTokens.length == 1 || leftChrTokens.length == 1) {
            parseGenePositionText();
            return;
        }

        ChromosomeHandler handler = hic.getDataset().getChromosomeHandler();

        Chromosome topChr = handler.getChromosomeFromName(topChrTokens[0]);
        if (topChr == null) {
            positionChrTop.setBackground(Color.yellow);
            System.err.println("Cannot find " + topChrTokens[0] + " in dataset's chromosome list");
            return;
        }

        Chromosome leftChr = handler.getChromosomeFromName(leftChrTokens[0]);
        if (leftChr == null) {
            positionChrLeft.setBackground(Color.yellow);
            System.err.println("Cannot find " + leftChrTokens[0] + " in dataset's chromosome list");
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
        HiC.Unit resolutionUnits = HiC.Unit.BP;
        int estimatedOutBinSize = Math.max(topChrPositions[3], leftChrPositions[3]);

        if (topChrTokens.length > 3 || (topDashChrTokens.length == 1 && topChrTokens.length > 2)) {
            try {
                int[] resolutionParameters = extractResolutionParametersFromTokens(topChrTokens, topDashChrTokens, positionChrTop);
                outBinSize = resolutionParameters[0];
                if (resolutionParameters[1] < 0) {
                    resolutionUnits = HiC.Unit.FRAG;
                }
            } catch (Exception e) {
                return;
            }
        } else if (leftChrTokens.length > 3 || (leftDashChrTokens.length == 1 && leftChrTokens.length > 2)) {
            try {
                int[] resolutionParameters = extractResolutionParametersFromTokens(leftChrTokens, leftDashChrTokens, positionChrLeft);
                outBinSize = resolutionParameters[0];
                if (resolutionParameters[1] < 0) {
                    resolutionUnits = HiC.Unit.FRAG;
                }
            } catch (Exception e) {
                return;
            }
        } else if (estimatedOutBinSize > 0) {
            outBinSize = estimatedOutBinSize;
        } else if (hic.getZoom().getBinSize() != 0) { //no resolution specified, not at whole genome view
            outBinSize = hic.getZoom().getBinSize();
            if (outBinSize != Integer.MIN_VALUE) {
                resolutionUnits = hic.getZoom().getUnit();
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

    private List<String[]> parse(JTextField textField) {
        String dashDelimiters = "\\s+|\\-\\s*";
        String[] tmpChrTokens = textField.getText().split(":");
        String[] chrTokens = new String[0];
        String[] dashChrTokens = new String[0];

        if (tmpChrTokens.length == 1) {
            chrTokens = tmpChrTokens;
        } else if (tmpChrTokens.length == 2) {
            dashChrTokens = textField.getText().substring(tmpChrTokens[0].length() + 1).split(dashDelimiters);
            chrTokens = new String[dashChrTokens.length + 1];
            chrTokens[0] = tmpChrTokens[0];
            for (int i = 0; i < dashChrTokens.length; i++) {
                chrTokens[i + 1] = dashChrTokens[i];
            }
        } else if (tmpChrTokens.length == 3) {
            dashChrTokens = textField.getText().substring(tmpChrTokens[0].length() + 1, tmpChrTokens[0].length() + tmpChrTokens[1].length() + 1).split(dashDelimiters);
            chrTokens = new String[dashChrTokens.length + 2];
            chrTokens[0] = tmpChrTokens[0];
            int i;
            for (i = 0; i < dashChrTokens.length; i++) {
                chrTokens[i + 1] = dashChrTokens[i];
            }
            System.out.println(i);
            chrTokens[i + 1] = tmpChrTokens[2];
        }

        List<String[]> result = new ArrayList<>();
        result.add(chrTokens);
        result.add(dashChrTokens);
        return result;
    }

    private int[] extractResolutionParametersFromTokens(String[] chrTokens, String[] dashChrTokens, JTextField positionChr) {
        int outBinSize = 0;
        int resolutionUnits = 1;//BP

        try {
            if (dashChrTokens.length == 1) {
                outBinSize = cleanUpNumber(chrTokens[2]);
            } else if (chrTokens.length > 3) {
                outBinSize = cleanUpNumber(chrTokens[3]);
            }
            System.out.println("Out bin size " + outBinSize);
        } catch (Exception e) {
            positionChr.setBackground(Color.yellow);
            System.err.println("Invalid resolution " + chrTokens[3]);
        }
        return new int[]{outBinSize, resolutionUnits};
    }

    private int[] extractParametersFromTokens(String[] chrTokens, String[] dashChrTokens, JTextField positionChr) throws IOException {
        //Read positions:
        int start = 0, end = 0, outBin = 0, estimatedOutBinSize = 0;
        if (chrTokens.length > 2 && dashChrTokens.length > 1) {
            //Make sure values are numerical:
            try {
                start = Math.min(cleanUpNumber(chrTokens[1]), cleanUpNumber(chrTokens[2]));
                end = Math.max(cleanUpNumber(chrTokens[1]), cleanUpNumber(chrTokens[2]));
            } catch (Exception e) {
                System.err.println("Cannot parse " + chrTokens[1] + " or " + chrTokens[2] + ". Expecting int");
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
                System.err.println("Cannot parse " + chrTokens[1] + ". Expecting int");
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

        for (HiCZoom zoom : hic.getDataset().getBpZooms()) {
            int res = zoom.getBinSize();
            if (diff >= res)
                return res;
        }
        return 5000;
    }

    private int cleanUpNumber(String number) {
        return (int) (Long.parseLong(number.toLowerCase()
                .replaceAll(",", "")
                .replaceAll("m", "000000")
                .replaceAll("k", "000")) / HiCGlobals.hicMapScale);
    }

    private void parseGenePositionText() {
        String genomeID = hic.getDataset().getGenomeId();
        // Currently only human and mouse, not worrying about small differences in location between genomes
        if (genomeID.equals("b37")) genomeID = "hg19";
        //if (geneLocationHashMap == null || !genomeID.equals(this.genomeID)) { //don't understand the genomeID check
        if (geneLocationHashMap == null || this.genomeID == null) {
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
            for (HiCTrack track : hic.getLoadedTracks()) {
                if (track.getName().contains("refGene")) {
                    Runnable runnable = new Runnable() {
                        @Override
                        public void run() {
                            unsafeInitializeGeneHashMap(track.getLocator().getPath());
                        }
                    };
                    superAdapter.executeLongRunningTask(runnable, "Initialize Gene Hash Map");
                }
            }
        }
    }

    private void unsafeInitializeGeneHashMap(String genomeID) {
        // Custom format parsed from ref Gene file.
        // Name1 Name2 chromosome position (where position is midpoint of transcription start and end)
        BufferedReader reader;
        try {
            reader = GeneTools.getStreamToGeneFile(genomeID);
        } catch (Exception error) {
            SuperAdapter.showMessageDialog("Failed to read gene database");
            positionChrTop.setBackground(Color.yellow);
            geneLocationHashMap = null;
            return;
        }

        try {
            ChromosomeHandler handler = hic.getChromosomeHandler();
            geneLocationHashMap = GeneTools.getLocationMap(reader, handler);
        } catch (Exception error) {
            SuperAdapter.showMessageDialog("Failed to parse gene database");
            positionChrTop.setBackground(Color.yellow);
            geneLocationHashMap = null;
        }
        if (geneLocationHashMap != null) this.genomeID = genomeID; //why was this needed?
        extractGeneLocation();
    }

    private void goToScaffoldName(String scafName1, String scafName2) {
        long location1 = -1;
        long location2 = -1;
        String chr1Name = "";
        String chr2Name = "";
        for (Scaffold scaffold : superAdapter.getAssemblyStateTracker().getAssemblyHandler().getListOfScaffolds()) {
            if (scaffold.name.equals(scafName1)) {
                chr1Name = scaffold.chrName;
                location1 = scaffold.getCurrentFeature2D().getMidPt1();
            }
            if (scaffold.name.equals(scafName2)) {
                chr2Name = scaffold.chrName;
                location2 = scaffold.getCurrentFeature2D().getMidPt2();
            }
        }
        try {
            hic.setLocation(chr1Name, chr2Name, HiC.Unit.BP, hic.getZd().getBinSize(),
                    location1, location2, hic.getScaleFactor(),
                    HiC.ZoomCallType.STANDARD, "Assembly Goto", true);
            superAdapter.setNormalizationDisplayState();
        } catch (NullPointerException e) {
            System.err.println("Cannot recognize scaffold name");
        }

    }

    private void extractGeneLocation() {
        GeneLocation location1 = geneLocationHashMap.get(positionChrTop.getText().trim().toLowerCase());
        GeneLocation location2 = geneLocationHashMap.get(positionChrLeft.getText().trim().toLowerCase());
        if (location1 == null) {
            positionChrTop.setBackground(Color.yellow);
            SuperAdapter.showMessageDialog("Gene location map doesn't contain " + positionChrTop.getText().trim());
            return;
        }
        if (location2 == null) {
            positionChrLeft.setBackground(Color.yellow);
            SuperAdapter.showMessageDialog("Gene location map doesn't contain " + positionChrLeft.getText().trim());
            return;
        }
        // Note that the global BP resolutions might not be what's stored in the file
        List<HiCZoom> zooms = hic.getDataset().getBpZooms();

        List<Integer> bpResolutions = new ArrayList<>();
        for (HiCZoom zoom:zooms){
            bpResolutions.add(zoom.getBinSize());
        }
        int geneZoomResolution = hic.getZoom().getBinSize();
        if (!bpResolutions.contains(geneZoomResolution)) {
            geneZoomResolution = Collections.min(bpResolutions);
        }

        hic.setLocation(location1.getChromosome().getName(), location2.getChromosome().getName(), HiC.Unit.BP, geneZoomResolution,
                location1.getCenterPosition(), location2.getCenterPosition(), hic.getScaleFactor(),
                HiC.ZoomCallType.STANDARD, "Gene Goto", true);

        superAdapter.setNormalizationDisplayState();
    }

    public void focusGained(FocusEvent event) {
        if (event.getSource() == positionChrLeft) positionChrLeft.setBackground(Color.white);
        else if (event.getSource() == positionChrTop) positionChrTop.setBackground(Color.white);
    }

    public void focusLost(FocusEvent event) {

    }
}
