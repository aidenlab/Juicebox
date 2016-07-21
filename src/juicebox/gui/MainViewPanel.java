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

package juicebox.gui;

import com.jidesoft.swing.JideButton;
import juicebox.Context;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.mapcolorui.HeatmapPanel;
import juicebox.mapcolorui.JColorRangePanel;
import juicebox.mapcolorui.ResolutionControl;
import juicebox.mapcolorui.ThumbnailPanel;
import juicebox.track.TrackLabelPanel;
import juicebox.track.TrackPanel;
import juicebox.windowui.*;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 8/4/15.
 */
public class MainViewPanel {

    public static final List<Color> preDefMapColorGradient = HiCGlobals.createNewPreDefMapColorGradient();
    public static final List<Float> preDefMapColorFractions = new ArrayList<Float>();
    public static boolean preDefMapColor = false;
    private static JComboBox<Chromosome> chrBox1;
    private static JComboBox<Chromosome> chrBox2;
    private static JideButton refreshButton;
    private static JComboBox<String> normalizationComboBox;
    private static JComboBox<MatrixType> displayOptionComboBox;
    private static JColorRangePanel colorRangePanel;
    private static ResolutionControl resolutionSlider;
    private static TrackPanel trackPanelX;
    private static TrackPanel trackPanelY;
    private static TrackLabelPanel trackLabelPanel;
    private static HiCRulerPanel rulerPanelX;
    private static HeatmapPanel heatmapPanel;
    private static HiCRulerPanel rulerPanelY;
    private static ThumbnailPanel thumbnailPanel;
    private static JEditorPane mouseHoverTextPanel;
    private static GoToPanel goPanel;
    private static JPanel hiCPanel;
    private static HiCChromosomeFigPanel chromosomePanelX;
    private static HiCChromosomeFigPanel chromosomePanelY;
    private static JPanel bottomChromosomeFigPanel;
    private static JPanel chrSidePanel;
    private static JPanel chrSidePanel3;
    private boolean tooltipAllowedToUpdated = true;
    private boolean ignoreUpdateThumbnail = false;


    public void setIgnoreUpdateThumbnail(boolean flag) {ignoreUpdateThumbnail = flag;}

    public JComboBox<Chromosome> getChrBox2() {
        return chrBox2;
    }

    public JComboBox<Chromosome> getChrBox1() {
        return chrBox1;
    }

    public void initializeMainView(final SuperAdapter superAdapter, Container contentPane,
                                   Dimension bigPanelDim, Dimension panelDim) {
        contentPane.setLayout(new BorderLayout());

        final JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BorderLayout());
        contentPane.add(mainPanel, BorderLayout.CENTER);
        mainPanel.setBackground(Color.white);

        final JPanel toolbarPanel = new JPanel();
        toolbarPanel.setBorder(null);

        toolbarPanel.setLayout(new GridBagLayout());
        mainPanel.add(toolbarPanel, BorderLayout.NORTH);

        JPanel bigPanel = new JPanel();
        bigPanel.setLayout(new BorderLayout());
        bigPanel.setBackground(Color.white);

        bigPanel.setPreferredSize(new Dimension(bigPanelDim));
        bigPanel.setMaximumSize(new Dimension(bigPanelDim));
        bigPanel.setMinimumSize(new Dimension(bigPanelDim));

        JPanel bottomPanel = new JPanel();
        bottomPanel.setBackground(Color.white);


        JMenuBar menuBar = null;
        try {
            menuBar = superAdapter.createMenuBar();
        } catch (Exception e) {
            e.printStackTrace();
        }
        assert menuBar != null;
        contentPane.add(menuBar, BorderLayout.NORTH);

        GridBagConstraints toolbarConstraints = new GridBagConstraints();
        toolbarConstraints.anchor = GridBagConstraints.LINE_START;
        toolbarConstraints.fill = GridBagConstraints.HORIZONTAL;
        toolbarConstraints.gridx = 0;
        toolbarConstraints.gridy = 0;
        toolbarConstraints.weightx = 0.1;

        // --- Chromosome panel ---
        JPanel chrSelectionPanel = new JPanel();
        toolbarPanel.add(chrSelectionPanel, toolbarConstraints);

        chrSelectionPanel.setBorder(LineBorder.createGrayLineBorder());

        chrSelectionPanel.setLayout(new BorderLayout());

        JPanel chrLabelPanel = new JPanel();
        JLabel chrLabel = new JLabel("Chromosomes");
        chrLabel.setHorizontalAlignment(SwingConstants.CENTER);
        chrLabelPanel.setBackground(HiCGlobals.backgroundColor);
        chrLabelPanel.setLayout(new BorderLayout());
        chrLabelPanel.add(chrLabel, BorderLayout.CENTER);
        chrSelectionPanel.add(chrLabelPanel, BorderLayout.PAGE_START);

        JPanel chrButtonPanel = new JPanel();
        chrButtonPanel.setBackground(new Color(238, 238, 238));
        chrButtonPanel.setLayout(new BoxLayout(chrButtonPanel, BoxLayout.X_AXIS));

        //---- chrBox1 ----
        chrBox1 = new JComboBox<Chromosome>();
        chrBox1.setModel(new DefaultComboBoxModel<Chromosome>(new Chromosome[]{new Chromosome(0, HiCFileTools.ALL_CHROMOSOME, 0)}));
        chrBox1.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                chrBox1ActionPerformed(e);
            }
        });
        chrBox1.setPreferredSize(new Dimension(95, 70));
        chrButtonPanel.add(chrBox1);

        //---- chrBox2 ----
        chrBox2 = new JComboBox<Chromosome>();
        chrBox2.setModel(new DefaultComboBoxModel<Chromosome>(new Chromosome[]{new Chromosome(0, HiCFileTools.ALL_CHROMOSOME, 0)}));
        chrBox2.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                chrBox2ActionPerformed(e);
            }
        });
        chrBox2.setPreferredSize(new Dimension(95, 70));
        chrButtonPanel.add(chrBox2);


        //---- refreshButton ----
        refreshButton = new JideButton();
        refreshButton.setIcon(new ImageIcon(getClass().getResource("/toolbarButtonGraphics/general/Refresh24.gif")));
        refreshButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                superAdapter.safeRefreshButtonActionPerformed();
            }
        });
        refreshButton.setPreferredSize(new Dimension(24, 24));
        chrButtonPanel.add(refreshButton);

        chrBox1.setEnabled(false);
        chrBox2.setEnabled(false);
        refreshButton.setEnabled(false);
        chrSelectionPanel.add(chrButtonPanel, BorderLayout.CENTER);

        chrSelectionPanel.setMinimumSize(new Dimension(200, 70));
        chrSelectionPanel.setPreferredSize(new Dimension(210, 70));

        //======== Display Option Panel ========
        JPanel displayOptionPanel = new JPanel();
        displayOptionPanel.setBackground(new Color(238, 238, 238));
        displayOptionPanel.setBorder(LineBorder.createGrayLineBorder());
        displayOptionPanel.setLayout(new BorderLayout());
        JPanel displayOptionLabelPanel = new JPanel();
        displayOptionLabelPanel.setBackground(HiCGlobals.backgroundColor);
        displayOptionLabelPanel.setLayout(new BorderLayout());

        JLabel displayOptionLabel = new JLabel("Show");
        displayOptionLabel.setHorizontalAlignment(SwingConstants.CENTER);
        displayOptionLabelPanel.add(displayOptionLabel, BorderLayout.CENTER);
        displayOptionPanel.add(displayOptionLabelPanel, BorderLayout.PAGE_START);
        JPanel displayOptionButtonPanel = new JPanel();
        displayOptionButtonPanel.setBorder(new EmptyBorder(0, 10, 0, 10));
        displayOptionButtonPanel.setLayout(new GridLayout(1, 0, 20, 0));
        displayOptionComboBox = new JComboBox<MatrixType>();
        displayOptionComboBox.setModel(new DefaultComboBoxModel<MatrixType>(new MatrixType[]{MatrixType.OBSERVED}));
        displayOptionComboBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                superAdapter.safeDisplayOptionComboBoxActionPerformed();
                normalizationComboBox.setEnabled(!isWholeGenome());
            }
        });
        displayOptionButtonPanel.add(displayOptionComboBox);
        displayOptionPanel.add(displayOptionButtonPanel, BorderLayout.CENTER);
        displayOptionPanel.setMinimumSize(new Dimension(140, 70));
        displayOptionPanel.setPreferredSize(new Dimension(140, 70));
        displayOptionPanel.setMaximumSize(new Dimension(140, 70));

        toolbarConstraints.gridx = 1;
        toolbarConstraints.weightx = 0.1;
        toolbarPanel.add(displayOptionPanel, toolbarConstraints);
        displayOptionComboBox.setEnabled(false);

        //======== Normalization Panel ========
        JPanel normalizationPanel = new JPanel();
        normalizationPanel.setBackground(new Color(238, 238, 238));
        normalizationPanel.setBorder(LineBorder.createGrayLineBorder());
        normalizationPanel.setLayout(new BorderLayout());

        JPanel normalizationLabelPanel = new JPanel();
        normalizationLabelPanel.setBackground(HiCGlobals.backgroundColor);
        normalizationLabelPanel.setLayout(new BorderLayout());

        JLabel normalizationLabel = new JLabel("Normalization");
        normalizationLabel.setHorizontalAlignment(SwingConstants.CENTER);
        normalizationLabelPanel.add(normalizationLabel, BorderLayout.CENTER);
        normalizationPanel.add(normalizationLabelPanel, BorderLayout.PAGE_START);

        JPanel normalizationButtonPanel = new JPanel();
        normalizationButtonPanel.setBorder(new EmptyBorder(0, 10, 0, 10));
        normalizationButtonPanel.setLayout(new GridLayout(1, 0, 20, 0));
        normalizationComboBox = new JComboBox<String>();
        normalizationComboBox.setModel(new DefaultComboBoxModel<String>(new String[]{NormalizationType.NONE.getLabel()}));
        normalizationComboBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                superAdapter.safeNormalizationComboBoxActionPerformed(e);
            }
        });
        normalizationButtonPanel.add(normalizationComboBox);
        normalizationPanel.add(normalizationButtonPanel, BorderLayout.CENTER);
        normalizationPanel.setPreferredSize(new Dimension(180, 70));
        normalizationPanel.setMinimumSize(new Dimension(140, 70));


        toolbarConstraints.gridx = 2;
        toolbarConstraints.weightx = 0.1;
        toolbarPanel.add(normalizationPanel, toolbarConstraints);
        normalizationComboBox.setEnabled(false);

        //======== Resolution Panel ========
        hiCPanel = new JPanel();
        hiCPanel.setBackground(Color.white);
        hiCPanel.setLayout(new HiCLayout());
        bigPanel.add(hiCPanel, BorderLayout.CENTER);

        JPanel wrapGapPanel = new JPanel();
        wrapGapPanel.setBackground(Color.white);
        wrapGapPanel.setMaximumSize(new Dimension(5, 5));
        wrapGapPanel.setMinimumSize(new Dimension(5, 5));
        wrapGapPanel.setPreferredSize(new Dimension(5, 5));
        wrapGapPanel.setBorder(LineBorder.createBlackLineBorder());
        bigPanel.add(wrapGapPanel, BorderLayout.EAST);


        // splitPanel.insertPane(hiCPanel, 0);
        // splitPanel.setBackground(Color.white);

        //---- rulerPanel2 ----
        JPanel topPanel = new JPanel();
        topPanel.setBackground(Color.white);
        topPanel.setLayout(new BorderLayout());
        hiCPanel.add(topPanel, BorderLayout.NORTH);
        trackLabelPanel = new TrackLabelPanel(superAdapter.getHiC());
        trackLabelPanel.setBackground(Color.white);
        hiCPanel.add(trackLabelPanel, HiCLayout.NORTH_WEST);

        JPanel topCenterPanel = new JPanel();
        topCenterPanel.setBackground(Color.BLUE);
        topCenterPanel.setLayout(new BorderLayout());
        topPanel.add(topCenterPanel, BorderLayout.CENTER);

        trackPanelX = new TrackPanel(superAdapter, superAdapter.getHiC(), TrackPanel.Orientation.X);
        trackPanelX.setMaximumSize(new Dimension(4000, 50));
        trackPanelX.setPreferredSize(new Dimension(1, 50));
        trackPanelX.setMinimumSize(new Dimension(1, 50));
        trackPanelX.setVisible(false);
        topCenterPanel.add(trackPanelX, BorderLayout.NORTH);

        rulerPanelX = new HiCRulerPanel(superAdapter.getHiC());
        rulerPanelX.setMaximumSize(new Dimension(4000, 50));
        rulerPanelX.setMinimumSize(new Dimension(1, 50));
        rulerPanelX.setPreferredSize(new Dimension(1, 50));
        rulerPanelX.setBorder(null);
        topCenterPanel.add(rulerPanelX, BorderLayout.SOUTH);

        //---- rulerPanel1 ----
        JPanel leftPanel = new JPanel();
        leftPanel.setBackground(Color.white);
        leftPanel.setLayout(new BorderLayout());
        hiCPanel.add(leftPanel, BorderLayout.WEST);

        trackPanelY = new TrackPanel(superAdapter, superAdapter.getHiC(), TrackPanel.Orientation.Y);
        trackPanelY.setMaximumSize(new Dimension(50, 4000));
        trackPanelY.setPreferredSize(new Dimension(50, 1));
        trackPanelY.setMinimumSize(new Dimension(50, 1));
        trackPanelY.setVisible(false);
        leftPanel.add(trackPanelY, BorderLayout.WEST);

        rulerPanelY = new HiCRulerPanel(superAdapter.getHiC());
        rulerPanelY.setMaximumSize(new Dimension(50, 4000));
        rulerPanelY.setPreferredSize(new Dimension(50, 800));
        rulerPanelY.setBorder(null);
        rulerPanelY.setMinimumSize(new Dimension(50, 1));
        leftPanel.add(rulerPanelY, BorderLayout.EAST);


        //==== Chromosome Context Toggled ====
        //---- chromosomeSidePanel ----
        chrSidePanel = new JPanel();
        chrSidePanel.setBackground(Color.white);
        chrSidePanel.setLayout(new BorderLayout());
        chrSidePanel.setMaximumSize(new Dimension(4000, 50));
        chrSidePanel.setPreferredSize(new Dimension(50, 50));
        chrSidePanel.setMinimumSize(new Dimension(50, 50));
        chrSidePanel.setVisible(true);

        JPanel chrSidePanel2 = new JPanel();
        chrSidePanel2.setBackground(Color.white);
        chrSidePanel2.setLayout(new BorderLayout());
        chrSidePanel2.setMaximumSize(new Dimension(50, 50));
        chrSidePanel2.setPreferredSize(new Dimension(50, 50));
        chrSidePanel2.setMinimumSize(new Dimension(50, 50));

        chrSidePanel3 = new JPanel();
        chrSidePanel3.setBackground(Color.white);
        chrSidePanel3.setLayout(new BorderLayout());
        chrSidePanel3.setMaximumSize(new Dimension(50, 4000));
        chrSidePanel3.setPreferredSize(new Dimension(50, 50));
        chrSidePanel3.setMinimumSize(new Dimension(50, 50));
        chrSidePanel3.setVisible(true);

        //---- chromosomeFigPanel2 ----
        bottomChromosomeFigPanel = new JPanel();
        bottomChromosomeFigPanel.setBackground(Color.white);
        bottomChromosomeFigPanel.setLayout(new BorderLayout());
        //bottomChromosomeFigPanel.setVisible(true);

        chromosomePanelX = new HiCChromosomeFigPanel(superAdapter.getHiC());
        chromosomePanelX.setMaximumSize(new Dimension(4000, 50));
        chromosomePanelX.setPreferredSize(new Dimension(1, 50));
        chromosomePanelX.setMinimumSize(new Dimension(1, 50));
        bottomChromosomeFigPanel.add(chromosomePanelX, BorderLayout.CENTER);
        bottomChromosomeFigPanel.add(chrSidePanel2, BorderLayout.EAST);
        bottomChromosomeFigPanel.setVisible(true);

        leftPanel.add(chrSidePanel, BorderLayout.SOUTH);
        topPanel.add(chrSidePanel3, BorderLayout.EAST);

        //---- chromosomeFigPanel1 ----
        chromosomePanelY = new HiCChromosomeFigPanel(superAdapter.getHiC());
        chromosomePanelY.setMaximumSize(new Dimension(50, 4000));
        chromosomePanelY.setPreferredSize(new Dimension(50, 1));
        chromosomePanelY.setMinimumSize(new Dimension(50, 1));
        chromosomePanelY.setVisible(true);


        //---- heatmapPanel ----
        //Dimension screenDimension = Toolkit.getDefaultToolkit().getScreenSize();
        //int panelSize = screenDimension.height - 210;


        int panelWidth = (int) panelDim.getWidth();
        int panelHeight = (int) panelDim.getHeight();
        System.err.println("Window W: " + panelWidth + " H" + panelHeight);

        JPanel wrapHeatmapPanel = new JPanel(new BorderLayout());
        wrapHeatmapPanel.setMaximumSize(new Dimension(panelDim));
        wrapHeatmapPanel.setMinimumSize(new Dimension(panelDim));
        wrapHeatmapPanel.setPreferredSize(new Dimension(panelDim));
        wrapHeatmapPanel.setBackground(Color.white);
        wrapHeatmapPanel.setVisible(true);

        heatmapPanel = new HeatmapPanel(superAdapter);
        heatmapPanel.setMaximumSize(new Dimension(panelWidth - 5, panelHeight - 5));
        heatmapPanel.setMinimumSize(new Dimension(panelWidth - 5, panelHeight - 5));
        heatmapPanel.setPreferredSize(new Dimension(panelWidth - 5, panelHeight - 5));
        heatmapPanel.setBackground(Color.white);

        wrapHeatmapPanel.add(heatmapPanel, BorderLayout.CENTER);

        //Chromosome Context Toggled
        wrapHeatmapPanel.add(bottomChromosomeFigPanel, BorderLayout.SOUTH);
        wrapHeatmapPanel.add(chromosomePanelY, BorderLayout.EAST);

        //hiCPanel.setMaximumSize(new Dimension(panelWidth, panelHeight));
        //hiCPanel.setMinimumSize(new Dimension(panelWidth, panelHeight));
        //hiCPanel.setPreferredSize(new Dimension(panelWidth, panelHeight));

        hiCPanel.add(wrapHeatmapPanel, BorderLayout.CENTER);

        //======== Resolution Slider Panel ========

        // Resolution  panel
        resolutionSlider = new ResolutionControl(superAdapter);
        resolutionSlider.setPreferredSize(new Dimension(200, 70));
        resolutionSlider.setMinimumSize(new Dimension(150, 70));

        toolbarConstraints.gridx = 3;
        toolbarConstraints.weightx = 0.1;
        toolbarPanel.add(resolutionSlider, toolbarConstraints);

        //======== Color Range Panel ========
        colorRangePanel = new JColorRangePanel(superAdapter, heatmapPanel, preDefMapColor);

        toolbarConstraints.gridx = 4;
        toolbarConstraints.weightx = 0.5;
        toolbarPanel.add(colorRangePanel, toolbarConstraints);

        goPanel = new GoToPanel(superAdapter);
        toolbarConstraints.gridx = 5;
        toolbarConstraints.weightx = 0.25;
        toolbarPanel.add(goPanel, toolbarConstraints);
        // not sure this is working
        //toolbarPanel.setPreferredSize(new Dimension(panelHeight,100));
        toolbarPanel.setEnabled(false);


        //======== Right side panel ========

        JPanel rightSidePanel = new JPanel(new BorderLayout());//(new BorderLayout());
        rightSidePanel.setBackground(Color.white);
        rightSidePanel.setPreferredSize(new Dimension(210, 1000));
        rightSidePanel.setMaximumSize(new Dimension(10000, 10000));

        //======== Bird's view mini map ========

        JPanel thumbPanel = new JPanel();
        thumbPanel.setLayout(new BorderLayout());

        //---- thumbnailPanel ----
        thumbnailPanel = new ThumbnailPanel(superAdapter);
        thumbnailPanel.setBackground(Color.white);
        thumbnailPanel.setMaximumSize(new Dimension(210, 210));
        thumbnailPanel.setMinimumSize(new Dimension(210, 210));
        thumbnailPanel.setPreferredSize(new Dimension(210, 210));

//        JPanel gapPanel = new JPanel();
//        gapPanel.setMaximumSize(new Dimension(1, 1));
//        rightSidePanel.add(gapPanel,BorderLayout.WEST);
        thumbPanel.add(thumbnailPanel, BorderLayout.CENTER);
        thumbPanel.setBackground(Color.white);
        rightSidePanel.add(thumbPanel, BorderLayout.NORTH);

        //========= mouse hover text ======
        JPanel tooltipPanel = new JPanel(new BorderLayout());
        tooltipPanel.setBackground(Color.white);
        tooltipPanel.setPreferredSize(new Dimension(210, 490));
        mouseHoverTextPanel = new JEditorPane();
        mouseHoverTextPanel.setEditable(false);
        mouseHoverTextPanel.setContentType("text/html");
        mouseHoverTextPanel.setFont(new Font("sans-serif", 0, 20));
        mouseHoverTextPanel.setBackground(Color.white);
        mouseHoverTextPanel.setBorder(null);
        int mouseTextY = rightSidePanel.getBounds().y + rightSidePanel.getBounds().height;

        //*Dimension prefSize = new Dimension(210, 490);
        Dimension prefSize = new Dimension(210, 390);
        mouseHoverTextPanel.setPreferredSize(prefSize);

        JScrollPane tooltipScroller = new JScrollPane(mouseHoverTextPanel);
        tooltipScroller.setBackground(Color.white);
        tooltipScroller.setBorder(null);

        tooltipPanel.setPreferredSize(new Dimension(210, 500));
        tooltipPanel.add(tooltipScroller);
        tooltipPanel.setBounds(new Rectangle(new Point(0, mouseTextY), prefSize));
        tooltipPanel.setBackground(Color.white);
        tooltipPanel.setBorder(null);

        rightSidePanel.add(tooltipPanel, BorderLayout.CENTER);

        // compute preferred size
        Dimension preferredSize = new Dimension();
        for (int i = 0; i < rightSidePanel.getComponentCount(); i++) {
            Rectangle bounds = rightSidePanel.getComponent(i).getBounds();
            preferredSize.width = Math.max(bounds.x + bounds.width, preferredSize.width);
            preferredSize.height = Math.max(bounds.y + bounds.height, preferredSize.height);
        }
        Insets insets = rightSidePanel.getInsets();
        preferredSize.width += insets.right + 20;
        preferredSize.height += insets.bottom;
        rightSidePanel.setMinimumSize(preferredSize);
        rightSidePanel.setPreferredSize(preferredSize);
        mainPanel.add(bigPanel, BorderLayout.CENTER);
        mainPanel.add(rightSidePanel, BorderLayout.EAST);
    }

    public JPanel getHiCPanel() {
        return hiCPanel;
    }

    public void updateToolTipText(String s) {
        if (tooltipAllowedToUpdated)
            mouseHoverTextPanel.setText(s);
        mouseHoverTextPanel.setCaretPosition(0);
    }

    public boolean isResolutionLocked() {
        return resolutionSlider.isResolutionLocked();
    }

    public HeatmapPanel getHeatmapPanel() {
        return heatmapPanel;
    }

    public void updateZoom(HiCZoom newZoom) {
        resolutionSlider.setZoom(newZoom);
    }

    public void updateAndResetZoom(HiCZoom newZoom) {
        resolutionSlider.setZoom(newZoom);
        //resolutionSlider.reset();
    }

    /*
     * Only accessed from within another unsafe method in Heatmap Panel class,
     * which in turn is encapsulated (i.e. made safe)
     */
    public void unsafeSetSelectedChromosomes(SuperAdapter superAdapter, Chromosome xChrom, Chromosome yChrom) {
        chrBox1.setSelectedIndex(yChrom.getIndex());
        chrBox2.setSelectedIndex(xChrom.getIndex());
        unsafeRefreshChromosomes(superAdapter);
    }

    public void unsafeRefreshChromosomes(SuperAdapter superAdapter) {

        if (chrBox1.getSelectedIndex() == 0 || chrBox2.getSelectedIndex() == 0) {
            chrBox1.setSelectedIndex(0);
            chrBox2.setSelectedIndex(0);
        }

        Chromosome chr1 = (Chromosome) chrBox1.getSelectedItem();
        Chromosome chr2 = (Chromosome) chrBox2.getSelectedItem();

        Chromosome chrX = chr1.getIndex() < chr2.getIndex() ? chr1 : chr2;
        Chromosome chrY = chr1.getIndex() < chr2.getIndex() ? chr2 : chr1;

        superAdapter.unsafeUpdateHiCChromosomes(chrX, chrY);
        setNormalizationDisplayState(superAdapter.getHiC());

        updateThumbnail(superAdapter.getHiC());
    }

    public void setSelectedChromosomesNoRefresh(Chromosome xChrom, Chromosome yChrom, Context xContext, Context yContext) {
        chrBox1.setSelectedIndex(yChrom.getIndex());
        chrBox2.setSelectedIndex(xChrom.getIndex());
        rulerPanelX.setContext(xContext, HiCRulerPanel.Orientation.HORIZONTAL);
        rulerPanelY.setContext(yContext, HiCRulerPanel.Orientation.VERTICAL);
        chromosomePanelX.setContext(xContext, HiCChromosomeFigPanel.Orientation.HORIZONTAL);
        chromosomePanelY.setContext(yContext, HiCChromosomeFigPanel.Orientation.VERTICAL);
        resolutionSlider.setEnabled(!xChrom.getName().equals(Globals.CHR_ALL));
    }

    /**
     * Chromosome "0" is whole genome
     *
     * @param chromosomes list of chromosomes
     */
    void setChromosomes(List<Chromosome> chromosomes) {
        int[] chromosomeBoundaries = new int[chromosomes.size() - 1];
        long bound = 0;
        for (int i = 1; i < chromosomes.size(); i++) {
            Chromosome c = chromosomes.get(i);
            bound += (c.getLength() / 1000);
            chromosomeBoundaries[i - 1] = (int) bound;
        }
        heatmapPanel.setChromosomeBoundaries(chromosomeBoundaries);

        chrBox1.setModel(new DefaultComboBoxModel<Chromosome>(chromosomes.toArray(new Chromosome[chromosomes.size()])));
        chrBox2.setModel(new DefaultComboBoxModel<Chromosome>(chromosomes.toArray(new Chromosome[chromosomes.size()])));

    }

    private boolean isInterChromosomal() {
        Chromosome chr1 = (Chromosome) chrBox1.getSelectedItem();
        Chromosome chr2 = (Chromosome) chrBox2.getSelectedItem();
        return chr1.getIndex() != chr2.getIndex();
    }

    /**
     * Note that both versions of isWholeGenome are needed otherwise we get
     * a bug when partial states have changed
     */
    private boolean isWholeGenome() {
        Chromosome chr1 = (Chromosome) chrBox1.getSelectedItem();
        Chromosome chr2 = (Chromosome) chrBox2.getSelectedItem();
        return HiCFileTools.isAllChromosome(chr1) || HiCFileTools.isAllChromosome(chr2);
    }

    private boolean isWholeGenome(HiC hic) {
        Chromosome chr1 = hic.getXContext().getChromosome();
        Chromosome chr2 = hic.getYContext().getChromosome();
        return HiCFileTools.isAllChromosome(chr1) || HiCFileTools.isAllChromosome(chr2);
    }

    public void setNormalizationDisplayState(HiC hic) {

        // Test for new dataset ("All"),  or change in chromosome
        if (isWholeGenome()) { // for now only allow observed
            hic.setDisplayOption(MatrixType.OBSERVED);
            displayOptionComboBox.setSelectedIndex(0);
            normalizationComboBox.setSelectedIndex(0);
        } else if (isInterChromosomal()) {
            if (MatrixType.isOnlyIntrachromosomalType(hic.getDisplayOption())) {
                hic.setDisplayOption(MatrixType.OBSERVED);
                displayOptionComboBox.setSelectedIndex(0);
            }
        }

        normalizationComboBox.setEnabled(!isWholeGenome(hic));
        displayOptionComboBox.setEnabled(true);
    }

    public void repaintTrackPanels() {
        trackPanelX.repaint();
        trackPanelY.repaint();
    }

    public void updateThumbnail(HiC hic) {
        if (ignoreUpdateThumbnail) return;
        //new Exception().printStackTrace();

        if (hic.getMatrix() != null) {

            //   MatrixZoomData zd0 = initialZoom == null ? hic.getMatrix().getFirstZoomData(hic.getZoom().getUnit()) :
            //           hic.getMatrix().getZoomData(initialZoom);
            MatrixZoomData zd0 = hic.getMatrix().getFirstZoomData(hic.getZoom().getUnit());
            MatrixZoomData zdControl = null;
            if (hic.getControlMatrix() != null)
                zdControl = hic.getControlMatrix().getFirstZoomData(hic.getZoom().getUnit());
            Image thumbnail = heatmapPanel.getThumbnailImage(zd0, zdControl,
                    thumbnailPanel.getWidth(), thumbnailPanel.getHeight(),
                    hic.getDisplayOption(), hic.getNormalizationType());
            if (thumbnail != null) {
                thumbnailPanel.setImage(thumbnail);
                thumbnailPanel.repaint();
            }
        } else {
            thumbnailPanel.setImage(null);
        }
    }

    private void chrBox1ActionPerformed(ActionEvent e) {
        if (chrBox1.getSelectedIndex() == 0) {
            chrBox2.setSelectedIndex(0);
        }
    }

    private void chrBox2ActionPerformed(ActionEvent e) {
        if (chrBox2.getSelectedIndex() == 0) {
            chrBox1.setSelectedIndex(0);
        }
    }

    public boolean setResolutionSliderVisible(boolean state, SuperAdapter superAdapter) {

        // Test for new dataset ("All"),  or change in chromosome
        boolean makeResVisible = state && !isWholeGenome();

        resolutionSlider.setEnabled(makeResVisible);
        if (makeResVisible) {
            resolutionSlider.setForeground(Color.BLUE);
        } else {
            resolutionSlider.setForeground(Color.BLACK);
        }
        return true;
        // why are we calling this?  why is this a boolean method?
        //return superAdapter.safeDisplayOptionComboBoxActionPerformed();
    }

    public void updateTrackPanel(boolean hasTracks) {

        trackLabelPanel.updateLabels();

        if (hasTracks) {
            if (!trackPanelX.isVisible()) {
                trackPanelX.setVisible(true);
                trackLabelPanel.setVisible(true);
            }
            if (!trackPanelY.isVisible()) {
                trackPanelY.setVisible(true);
            }
        } else {
            if (trackPanelX.isVisible()) {
                trackPanelX.setVisible(false);
                trackLabelPanel.setVisible(false);
            }
            if (trackPanelY.isVisible()) {
                trackPanelY.setVisible(false);
            }
        }

        trackPanelX.invalidate();
        trackLabelPanel.invalidate();
        trackPanelY.invalidate();
    }

    public void setShowChromosomeFig(boolean showFigure) {

        if (showFigure) {
            if (!bottomChromosomeFigPanel.isVisible()) {
                bottomChromosomeFigPanel.setVisible(true);
            }
            if (!chromosomePanelY.isVisible()) {
                chromosomePanelY.setVisible(true);
            }
            if (!chrSidePanel.isVisible()) {
                chrSidePanel.setVisible(true);
            }
            if (!chrSidePanel3.isVisible()) {
                chrSidePanel3.setVisible(true);
            }
        } else {
            if (bottomChromosomeFigPanel.isVisible()) {
                bottomChromosomeFigPanel.setVisible(false);
            }
            if (chromosomePanelY.isVisible()) {
                chromosomePanelY.setVisible(false);
            }
            if (chrSidePanel.isVisible()) {
                chrSidePanel.setVisible(false);
            }
            if (chrSidePanel3.isVisible()) {
                chrSidePanel3.setVisible(false);
            }
        }
        HiCRulerPanel.setShowChromosomeFigure(showFigure);
        chromosomePanelY.invalidate();
        bottomChromosomeFigPanel.invalidate();
        chrSidePanel.invalidate();
        chrSidePanel3.invalidate();
    }

    public String getToolTip() {
        return mouseHoverTextPanel.getText();
    }

    public void setDisplayBox(int indx) {
        displayOptionComboBox.setSelectedIndex(indx);
    }

    public void setNormalizationBox(int indx) {
        normalizationComboBox.setSelectedIndex(indx);
    }

    public void setNormalizationEnabledForReload() {
        //normalizationComboBox.setEnabled(true);
        normalizationComboBox.setEnabled(!isWholeGenome());
    }

    public void setPositionChrLeft(String newPositionDate) {
        goPanel.setPositionChrLeft(newPositionDate);
    }

    public void setPositionChrTop(String newPositionDate) {
        goPanel.setPositionChrTop(newPositionDate);
    }

    public void setEnableForAllElements(SuperAdapter superAdapter, boolean status) {
        chrBox1.setEnabled(status);
        chrBox2.setEnabled(status);
        refreshButton.setEnabled(status);
        colorRangePanel.setElementsVisible(status, superAdapter);
        if (setResolutionSliderVisible(status, superAdapter)) {
            // TODO succeeded
        } else {
            // TODO failed
        }
        goPanel.setEnabled(status);
    }

    public String getColorRangeValues() {
        return colorRangePanel.getColorRangeValues();
    }

    public double getColorRangeScaleFactor() {
        return colorRangePanel.getColorRangeScaleFactor();
    }

    public void updateRatioColorSlider(HiC hic, double maxColor, double upColor) {
        colorRangePanel.updateRatioColorSlider(hic, maxColor, upColor);
    }

    public void updateColorSlider(HiC hic, double minColor, double lowColor, double upColor, double maxColor) {
        colorRangePanel.updateColorSlider(hic, minColor, lowColor, upColor, maxColor);
    }

    public void updateColorSlider(HiC hic, double minColor, double lowColor, double upColor, double maxColor, double scalefactor) {
        colorRangePanel.updateColorSlider(hic, minColor, lowColor, upColor, maxColor);//scalefactor);
    }

    public void setEnabledForNormalization(String[] normalizationOptions, boolean status) {
        if (normalizationOptions.length == 1) {
            normalizationComboBox.setEnabled(false);
        } else {
            normalizationComboBox.setModel(new DefaultComboBoxModel<String>(normalizationOptions));
            normalizationComboBox.setSelectedIndex(0);
            normalizationComboBox.setEnabled(status && !isWholeGenome());
        }
    }

    public JComboBox<MatrixType> getDisplayOptionComboBox() {
        return displayOptionComboBox;
    }

    public void resetResolutionSlider() {
        resolutionSlider.unit = HiC.Unit.BP;
        resolutionSlider.reset();
    }

    public void setSelectedDisplayOption(MatrixType[] options, boolean control) {
        if (control) {
            MatrixType originalMatrixType = (MatrixType) displayOptionComboBox.getSelectedItem();
            displayOptionComboBox.setModel(new DefaultComboBoxModel<MatrixType>(options));
            int indx = 0;
            for (int i = 0; i < displayOptionComboBox.getItemCount(); i++) {
                if (originalMatrixType.equals(displayOptionComboBox.getItemAt(i))) {
                    indx = i;
                    break;
                }
            }
            displayOptionComboBox.setSelectedIndex(indx);
        } else {
            displayOptionComboBox.setModel(new DefaultComboBoxModel<MatrixType>(options));
            displayOptionComboBox.setSelectedIndex(0);
        }
    }

    public JEditorPane getMouseHoverTextPanel() {
        return mouseHoverTextPanel;
    }

    public ResolutionControl getResolutionSlider() {
        return resolutionSlider;
    }

    public JColorRangePanel getColorRangePanel() {
        return colorRangePanel;
    }

    public boolean isTooltipAllowedToUpdated() {
        return tooltipAllowedToUpdated;
    }

    public void toggleToolTipUpdates(boolean tooltipAllowedToUpdated) {
        this.tooltipAllowedToUpdated = tooltipAllowedToUpdated;
    }

    public JComboBox<String> getNormalizationComboBox() {
        return normalizationComboBox;
    }

    public HiCRulerPanel getRulerPanelY() {
        return rulerPanelY;
    }

    public HiCRulerPanel getRulerPanelX() {
        return rulerPanelX;
    }

    public HiCChromosomeFigPanel getChromosomeFigPanelY() {
        return chromosomePanelY;
    }

    public HiCChromosomeFigPanel getChromosomeFigPanelX() {
        return chromosomePanelX;
    }


    /*public boolean isPearsonDisplayed() {
        return displayOptionComboBox.getSelectedItem() == MatrixType.PEARSON;
    }*/
}
