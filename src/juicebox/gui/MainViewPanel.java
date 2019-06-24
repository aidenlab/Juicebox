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

package juicebox.gui;

import com.jidesoft.swing.JideButton;
import juicebox.Context;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.MatrixZoomData;
import juicebox.mapcolorui.HeatmapPanel;
import juicebox.mapcolorui.JColorRangePanel;
import juicebox.mapcolorui.ResolutionControl;
import juicebox.mapcolorui.ThumbnailPanel;
import juicebox.track.TrackLabelPanel;
import juicebox.track.TrackPanel;
import juicebox.windowui.*;
import juicebox.windowui.layers.MiniAnnotationsLayerPanel;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.border.LineBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
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
    public static final List<Float> preDefMapColorFractions = new ArrayList<>();
    public static boolean preDefMapColor = false;
    private static JComboBox<Chromosome> chrBox1;
    private static JComboBox<Chromosome> chrBox2;
    private static final JideButton refreshButton = new JideButton();
    private static JComboBox<String> observedNormalizationComboBox, controlNormalizationComboBox;
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
    private static final JPanel hiCPanel = new JPanel(new HiCLayout());
    private static HiCChromosomeFigPanel chromosomePanelX;
    private static HiCChromosomeFigPanel chromosomePanelY;
    private static final JPanel bottomChromosomeFigPanel = new JPanel(new BorderLayout());
    private static final JPanel chrSidePanel = new JPanel(new BorderLayout());
    private static final JPanel chrSidePanel3 = new JPanel(new BorderLayout());
    private static MainMenuBar menuBar;
    private final JToggleButton annotationsPanelToggleButton = new JToggleButton("Show Annotation Panel");
    private final JPanel annotationsPanel = new JPanel(new BorderLayout());
    private final JPanel mainPanel = new JPanel(new BorderLayout());
    private final JPanel bigPanel = new JPanel(new BorderLayout());
    private final JPanel toolbarPanel = new JPanel(new GridBagLayout());
    //private final JPanel bottomPanel = new JPanel();
    private final JPanel chrSelectionPanel = new JPanel(new BorderLayout());
    private final JPanel wrapGapPanel = new JPanel();
    private final JPanel topPanel = new JPanel(new BorderLayout());
    private final JPanel leftPanel = new JPanel(new BorderLayout());
    private final JPanel chrSidePanel2 = new JPanel();
    private final JPanel wrapHeatmapPanel = new JPanel(new BorderLayout());
    private final JPanel rightSidePanel = new JPanel();
    private final JPanel normalizationPanel = new JPanel(new BorderLayout());
    private final JPanel normalizationLabelPanel = new JPanel(new BorderLayout());
    private final JPanel topCenterPanel = new JPanel(new BorderLayout());
    private final JPanel displayOptionPanel = new JPanel(new BorderLayout());
    private final JPanel displayOptionLabelPanel = new JPanel(new BorderLayout());
    private final JPanel chrButtonPanel = new JPanel();
    private final JPanel chrLabelPanel = new JPanel(new BorderLayout());
    private final JLabel chrLabel = new JLabel("Chromosomes");
    private final JLabel normalizationLabel = new JLabel("Normalization");
    private final JLabel displayOptionLabel = new JLabel("Show");
    private MiniAnnotationsLayerPanel miniAnnotationsLayerPanel;
    private boolean tooltipAllowedToUpdate = true;
    private boolean ignoreUpdateThumbnail = false;
    private final JPanel tooltipPanel = new JPanel(new BorderLayout());

    public void setIgnoreUpdateThumbnail(boolean flag) {
        ignoreUpdateThumbnail = flag;
    }

    public JComboBox<Chromosome> getChrBox2() {
        return chrBox2;
    }

    public JComboBox<Chromosome> getChrBox1() {
        return chrBox1;
    }

    public MainMenuBar getMenuBar() {
        return menuBar;
    }

    public void initializeMainView(final SuperAdapter superAdapter, Container contentPane, Dimension screenSize, int taskBarHeight) {

        Dimension bigPanelDim = new Dimension((int) (screenSize.width * .85),
                (int) ((screenSize.height - taskBarHeight) * .9));

        Dimension panelDim = new Dimension((int) (screenSize.width * .75),
                screenSize.height - taskBarHeight);

        Dimension chrBoxDim = new Dimension(95, 70);

        contentPane.setLayout(new BorderLayout());
        contentPane.add(mainPanel, BorderLayout.CENTER);

        toolbarPanel.setBorder(null);
        mainPanel.add(toolbarPanel, BorderLayout.NORTH);

        bigPanel.setPreferredSize(new Dimension(bigPanelDim));
        bigPanel.setMaximumSize(new Dimension(bigPanelDim));
        bigPanel.setMinimumSize(new Dimension(bigPanelDim));

        menuBar = new MainMenuBar(superAdapter);
        contentPane.add(menuBar, BorderLayout.NORTH);

        GridBagConstraints toolbarConstraints = new GridBagConstraints(0, 0, 1, 1,
                0.1, 0, GridBagConstraints.LINE_START, GridBagConstraints.HORIZONTAL,
                new Insets(0, 0, 0, 0), 0, 0);

        // --- Chromosome panel ---

        toolbarPanel.add(chrSelectionPanel, toolbarConstraints);

        chrSelectionPanel.setBorder(LineBorder.createGrayLineBorder());

        chrLabel.setHorizontalAlignment(SwingConstants.CENTER);
        chrLabelPanel.add(chrLabel, BorderLayout.CENTER);
        chrSelectionPanel.add(chrLabelPanel, BorderLayout.PAGE_START);
        chrButtonPanel.setLayout(new BoxLayout(chrButtonPanel, BoxLayout.X_AXIS));

        //---- chrBox1 ----
        chrBox1 = new JComboBox<>(new Chromosome[]{new Chromosome(0, Globals.CHR_ALL, 0)});
        chrBox1.addPopupMenuListener(new BoundsPopupMenuListener<Chromosome>(true, false));
        chrBox1.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                chrBox1ActionPerformed(e);
            }
        });
        chrBox1.setPreferredSize(chrBoxDim);
        chrButtonPanel.add(chrBox1);

        //---- chrBox2 ----
        chrBox2 = new JComboBox<>(new Chromosome[]{new Chromosome(0, Globals.CHR_ALL, 0)});
        chrBox2.addPopupMenuListener(new BoundsPopupMenuListener<Chromosome>(true, false));
        chrBox2.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                chrBox2ActionPerformed(e);
            }
        });
        chrBox2.setPreferredSize(chrBoxDim);
        chrButtonPanel.add(chrBox2);

        //---- refreshButton ----
        refreshButton.setIcon(new ImageIcon(getClass().getResource("/toolbarButtonGraphics/general/Refresh24.gif")));
        refreshButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                superAdapter.safeRefreshButtonActionPerformed();
            }
        });
        refreshButton.setPreferredSize(new Dimension(24, 24));
        chrButtonPanel.add(refreshButton);
        chrSelectionPanel.add(chrButtonPanel, BorderLayout.CENTER);
        chrSelectionPanel.setMinimumSize(new Dimension(200, 70));
        chrSelectionPanel.setPreferredSize(new Dimension(210, 70));

        //======== Display Option Panel ========
        displayOptionLabel.setHorizontalAlignment(SwingConstants.CENTER);
        displayOptionLabelPanel.add(displayOptionLabel, BorderLayout.CENTER);
        displayOptionPanel.add(displayOptionLabelPanel, BorderLayout.PAGE_START);
        JPanel displayOptionButtonPanel = new JPanel();
        displayOptionButtonPanel.setBorder(new EmptyBorder(0, 10, 0, 10));
        displayOptionButtonPanel.setLayout(new GridLayout(1, 0, 20, 0));
        displayOptionComboBox = new JComboBox<>(new MatrixType[]{MatrixType.OBSERVED});
        displayOptionComboBox.setPreferredSize(new Dimension(500, 30));
        displayOptionComboBox.addPopupMenuListener(new BoundsPopupMenuListener<MatrixType>(true, false));
        displayOptionComboBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                superAdapter.safeDisplayOptionComboBoxActionPerformed();
                observedNormalizationComboBox.setEnabled(!isWholeGenome());
                controlNormalizationComboBox.setEnabled(!isWholeGenome());
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

        //======== Normalization Panel ========
        normalizationPanel.setBorder(LineBorder.createGrayLineBorder());
        normalizationLabel.setHorizontalAlignment(SwingConstants.CENTER);
        normalizationLabelPanel.add(normalizationLabel, BorderLayout.CENTER);
        normalizationPanel.add(normalizationLabelPanel, BorderLayout.PAGE_START);

        JPanel normalizationButtonPanel = new JPanel();
        normalizationButtonPanel.setBorder(new EmptyBorder(0, 10, 0, 10));
        normalizationButtonPanel.setLayout(new GridLayout(1, 0, 20, 0));
        observedNormalizationComboBox = new JComboBox<>(new String[]{NormalizationHandler.NONE.getDescription()});
        observedNormalizationComboBox.addPopupMenuListener(new BoundsPopupMenuListener<String>(true, false));
        observedNormalizationComboBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                superAdapter.safeNormalizationComboBoxActionPerformed(e, false);
            }
        });

        controlNormalizationComboBox = new JComboBox<>(new String[]{NormalizationHandler.NONE.getDescription()});
        controlNormalizationComboBox.addPopupMenuListener(new BoundsPopupMenuListener<String>(true, false));
        controlNormalizationComboBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                superAdapter.safeNormalizationComboBoxActionPerformed(e, true);
            }
        });

        normalizationButtonPanel.add(observedNormalizationComboBox);
        normalizationButtonPanel.add(controlNormalizationComboBox);
        normalizationPanel.add(normalizationButtonPanel, BorderLayout.CENTER);
        normalizationPanel.setPreferredSize(new Dimension(180, 70));
        normalizationPanel.setMinimumSize(new Dimension(140, 70));

        toolbarConstraints.gridx = 2;
        toolbarConstraints.weightx = 0.1;
        toolbarPanel.add(normalizationPanel, toolbarConstraints);

        bigPanel.add(hiCPanel, BorderLayout.CENTER);

        wrapGapPanel.setMaximumSize(new Dimension(5, 5));
        wrapGapPanel.setMinimumSize(new Dimension(5, 5));
        wrapGapPanel.setPreferredSize(new Dimension(5, 5));
        wrapGapPanel.setBorder(LineBorder.createBlackLineBorder());
        bigPanel.add(wrapGapPanel, BorderLayout.EAST);

        hiCPanel.add(topPanel, BorderLayout.NORTH);
        trackLabelPanel = new TrackLabelPanel(superAdapter.getHiC());
        hiCPanel.add(trackLabelPanel, HiCLayout.NORTH_WEST);
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

        //---- chromosomeSidePanel ----
        chrSidePanel.setMaximumSize(new Dimension(4000, 50));
        chrSidePanel.setPreferredSize(new Dimension(50, 50));
        chrSidePanel.setMinimumSize(new Dimension(50, 50));
        chrSidePanel.setVisible(true);

        chrSidePanel2.setMaximumSize(new Dimension(50, 50));
        chrSidePanel2.setPreferredSize(new Dimension(50, 50));
        chrSidePanel2.setMinimumSize(new Dimension(50, 50));

        chrSidePanel3.setMaximumSize(new Dimension(50, 4000));
        chrSidePanel3.setPreferredSize(new Dimension(50, 50));
        chrSidePanel3.setMinimumSize(new Dimension(50, 50));
        chrSidePanel3.setVisible(true);

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

        int panelWidth = (int) panelDim.getWidth();
        int panelHeight = (int) panelDim.getHeight();
        System.err.println("Window W: " + panelWidth + " H" + panelHeight);

        wrapHeatmapPanel.setMaximumSize(new Dimension(panelDim));
        wrapHeatmapPanel.setMinimumSize(new Dimension(panelDim));
        wrapHeatmapPanel.setPreferredSize(new Dimension(panelDim));
        wrapHeatmapPanel.setVisible(true);

        heatmapPanel = new HeatmapPanel(superAdapter);
        heatmapPanel.setMaximumSize(new Dimension(panelWidth - 5, panelHeight - 5));
        heatmapPanel.setMinimumSize(new Dimension(panelWidth - 5, panelHeight - 5));
        heatmapPanel.setPreferredSize(new Dimension(panelWidth - 5, panelHeight - 5));

        wrapHeatmapPanel.add(heatmapPanel, BorderLayout.CENTER);
        wrapHeatmapPanel.add(bottomChromosomeFigPanel, BorderLayout.SOUTH);
        wrapHeatmapPanel.add(chromosomePanelY, BorderLayout.EAST);

        hiCPanel.add(wrapHeatmapPanel, BorderLayout.CENTER);

        //======== Resolution Slider Panel ========
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

        //======== Right side panel ========
        int prefRightSideWidth = (int) (screenSize.width * .15);

        rightSidePanel.setLayout(new BoxLayout(rightSidePanel, BoxLayout.Y_AXIS));
        rightSidePanel.setMinimumSize(new Dimension((int) (screenSize.width * .15), screenSize.height));
        rightSidePanel.setPreferredSize(new Dimension(prefRightSideWidth, screenSize.height));
        rightSidePanel.setMaximumSize(new Dimension((int) (screenSize.width * .21), screenSize.height));

        //======== Bird's view mini map ========
        Dimension thumbNailDim = new Dimension(prefRightSideWidth, prefRightSideWidth);
        thumbnailPanel = new ThumbnailPanel(superAdapter);
        thumbnailPanel.setMaximumSize(thumbNailDim);
        thumbnailPanel.setMinimumSize(thumbNailDim);
        thumbnailPanel.setPreferredSize(thumbNailDim);
        // todo eliminate thumbpanel - redundant container?
        rightSidePanel.add(thumbnailPanel, BorderLayout.NORTH);

        //========= mini-annotations panel ======
        int maxMiniAnnotHeight = (screenSize.height - toolbarPanel.getHeight() - taskBarHeight) / 5;
        miniAnnotationsLayerPanel = new MiniAnnotationsLayerPanel(superAdapter, prefRightSideWidth, maxMiniAnnotHeight);

        //========= mouse hover text ======
        int leftoverHeight = screenSize.height - toolbarPanel.getHeight() - taskBarHeight -
                miniAnnotationsLayerPanel.getDynamicHeight() - annotationsPanelToggleButton.getHeight();
        mouseHoverTextPanel = new JEditorPane();
        mouseHoverTextPanel.setEditable(false);
        mouseHoverTextPanel.setContentType("text/html");
        mouseHoverTextPanel.setFont(new Font("sans-serif", Font.PLAIN, 20));
        mouseHoverTextPanel.setBorder(null);
        int mouseTextY = rightSidePanel.getBounds().y + rightSidePanel.getBounds().height;

        Dimension prefTextPanelSize = new Dimension(prefRightSideWidth, leftoverHeight);
        mouseHoverTextPanel.setPreferredSize(prefTextPanelSize);
        tooltipPanel.setPreferredSize(prefTextPanelSize);

        JScrollPane tooltipScroller = new JScrollPane(mouseHoverTextPanel,
                JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
                JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        tooltipScroller.setBorder(null);
        tooltipScroller.setMaximumSize(prefTextPanelSize);

        tooltipPanel.add(tooltipScroller);
        tooltipPanel.setBounds(new Rectangle(new Point(0, mouseTextY), prefTextPanelSize));
        tooltipPanel.setBorder(null);

        rightSidePanel.add(tooltipPanel, BorderLayout.CENTER);

        annotationsPanelToggleButton.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                if (annotationsPanelToggleButton.isSelected()) {
                    annotationsPanelToggleButton.setText("Hide Annotation Panel");
                } else {
                    annotationsPanelToggleButton.setText("Show Annotation Panel");
                }
            }
        });
        annotationsPanelToggleButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (annotationsPanelToggleButton.isSelected()) {
                    superAdapter.setLayersPanelVisible(true);
                    annotationsPanelToggleButton.setText("Hide Annotation Panel");
                } else {
                    superAdapter.setLayersPanelVisible(false);
                    annotationsPanelToggleButton.setText("Show Annotation Panel");
                }
            }
        });
        annotationsPanelToggleButton.setSelected(false);
        annotationsPanel.add(miniAnnotationsLayerPanel, BorderLayout.NORTH);
        miniAnnotationsLayerPanel.setAlignmentX(Component.CENTER_ALIGNMENT);
        annotationsPanel.add(annotationsPanelToggleButton, BorderLayout.SOUTH);
        annotationsPanelToggleButton.setAlignmentX(Component.CENTER_ALIGNMENT);

      rightSidePanel.add(annotationsPanel, BorderLayout.SOUTH);
        annotationsPanel.setAlignmentX(Component.CENTER_ALIGNMENT);

        mainPanel.add(bigPanel, BorderLayout.CENTER);
        mainPanel.add(rightSidePanel, BorderLayout.EAST);

        resetAllColors();
        initialSetToFalse();
    }

    private void initialSetToFalse() {
        JComponent[] comps = new JComponent[]{chrBox1, chrBox2, refreshButton, displayOptionComboBox,
                observedNormalizationComboBox, controlNormalizationComboBox, toolbarPanel, miniAnnotationsLayerPanel, annotationsPanelToggleButton};
        for (JComponent comp : comps) {
            comp.setEnabled(false);
        }
    }

    public void resetAllColors() {
        Color mainBackgroundColor = HiCGlobals.isDarkulaModeEnabled ? Color.darkGray : Color.WHITE;
        JComponent[] components = new JComponent[]{mainPanel, bigPanel, topPanel, trackLabelPanel,
                leftPanel, chrSidePanel, chrSidePanel2, chrSidePanel3, bottomChromosomeFigPanel, wrapHeatmapPanel,
                heatmapPanel, wrapGapPanel, hiCPanel, trackPanelX, trackPanelY, rulerPanelX, rulerPanelY,
                chromosomePanelX, chromosomePanelY, topCenterPanel};

        //annotationsPanelToggleButton
        for (JComponent component : components) {
            component.setBackground(mainBackgroundColor);
        }

        topCenterPanel.setBackground(Color.BLUE);
        normalizationLabelPanel.setBackground(HiCGlobals.backgroundColor);
        chrLabelPanel.setBackground(HiCGlobals.backgroundColor);
        displayOptionLabelPanel.setBackground(HiCGlobals.backgroundColor);

        normalizationPanel.setBackground(HiCGlobals.diffGrayColor);
        displayOptionPanel.setBackground(HiCGlobals.diffGrayColor);
        chrButtonPanel.setBackground(HiCGlobals.diffGrayColor);

        displayOptionPanel.setBorder(LineBorder.createGrayLineBorder());

        heatmapPanel.reset();
    }

    public JPanel getHiCPanel() {
        return hiCPanel;
    }

    public void updateToolTipText(String s) {
        if (tooltipAllowedToUpdate)
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
            MatrixType matrixType = (MatrixType) displayOptionComboBox.getSelectedItem();
            if (MatrixType.isPearsonType(matrixType)) {
                // can't do pearson's genomewide
                displayOptionComboBox.setSelectedIndex(0);
                superAdapter.unsafeDisplayOptionComboBoxActionPerformed();
            }
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
        resolutionSlider.setEnabled(!ChromosomeHandler.isAllByAll(xChrom));
    }

    /**
     * Chromosome "0" is whole genome
     *
     * @param handler for list of chromosomes
     */
    void setChromosomes(ChromosomeHandler handler) {
        heatmapPanel.setChromosomeBoundaries(handler.getChromosomeBoundaries());
        chrBox1.setModel(new DefaultComboBoxModel<>(handler.getChromosomeArray()));
        chrBox2.setModel(new DefaultComboBoxModel<>(handler.getChromosomeArray()));
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
        return ChromosomeHandler.isAllByAll(chr1) || ChromosomeHandler.isAllByAll(chr2);
    }

    private boolean isWholeGenome(HiC hic) {
        Chromosome chr1 = hic.getXContext().getChromosome();
        Chromosome chr2 = hic.getYContext().getChromosome();
        return ChromosomeHandler.isAllByAll(chr1) || ChromosomeHandler.isAllByAll(chr2);
    }

    public void setNormalizationDisplayState(HiC hic) {

        // Test for new dataset ("All"),  or change in chromosome
        if (isWholeGenome()) { // for now only allow observed
            hic.setDisplayOption(MatrixType.OBSERVED);
            displayOptionComboBox.setSelectedIndex(0);
            observedNormalizationComboBox.setSelectedIndex(0);
            controlNormalizationComboBox.setSelectedIndex(0);
        } else if (isInterChromosomal()) {
            if (MatrixType.isOnlyIntrachromosomalType(hic.getDisplayOption())) {
                hic.setDisplayOption(MatrixType.OBSERVED);
                displayOptionComboBox.setSelectedIndex(0);
            }
        }

        observedNormalizationComboBox.setEnabled(!isWholeGenome(hic));
        controlNormalizationComboBox.setEnabled(!isWholeGenome());
        displayOptionComboBox.setEnabled(true);
    }

    public void repaintTrackPanels() {
        trackPanelX.repaint();
        trackPanelY.repaint();
    }

    public void repaintGridRulerPanels() {
        rulerPanelX.repaint();
        rulerPanelY.repaint();
    }

    public String getTrackPanelPrintouts(int x, int y) {
        String trackToolTip = "";
        try {
            String text = trackPanelX.tooltipText(x, y, false);
            if (text != null) trackToolTip += "<span style='color:" + HiCGlobals.topChromosomeColor +
                    "; font-family: arial; font-size: 12pt; '>" + text + "</span>";
            text = trackPanelY.tooltipText(x, y, false);
            if (text != null) trackToolTip += "<span style='color:" + HiCGlobals.leftChromosomeColor +
                    "; font-family: arial; font-size: 12pt; '>" + text + "</span>";
        } catch (Exception ignored) {
        }
        return trackToolTip;
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
            try {
                Image thumbnail = heatmapPanel.getThumbnailImage(zd0, zdControl,
                        thumbnailPanel.getWidth(), thumbnailPanel.getHeight(),
                        hic.getDisplayOption(), hic.getObsNormalizationType(), hic.getControlNormalizationType());
                if (thumbnail != null) {
                    thumbnailPanel.setImage(thumbnail);
                    thumbnailPanel.repaint();
                }
            } catch (Exception ignored) {
                thumbnailPanel.setImage(null);
                thumbnailPanel.repaint();
            }

        } else {
            thumbnailPanel.setImage(null);
        }
    }

    public static void invertAssemblyMatCheck() {
        HiCGlobals.isAssemblyMatCheck = !HiCGlobals.isAssemblyMatCheck;
    }

    private void chrBox1ActionPerformed(ActionEvent e) {
      if (chrBox1.getSelectedIndex() == 0) {
        chrBox2.setSelectedIndex(0);
      } else if (HiCGlobals.isAssemblyMatCheck && chrBox1.getSelectedIndex() == (chrBox1.getItemCount() - 1)) {
        chrBox2.setSelectedIndex(chrBox1.getItemCount() - 1);
      }
    }

    private void chrBox2ActionPerformed(ActionEvent e) {
      if (chrBox2.getSelectedIndex() == 0) {
        chrBox1.setSelectedIndex(0);
      } else if (HiCGlobals.isAssemblyMatCheck && chrBox2.getSelectedIndex() == (chrBox1.getItemCount() - 1)) {
        chrBox1.setSelectedIndex(chrBox1.getItemCount() - 1);
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

    public boolean getShowGridLines() {
        return heatmapPanel == null || heatmapPanel.getShowGridLines();
    }

    public void setShowGridLines(boolean status) {
        if (heatmapPanel != null) {
            heatmapPanel.setShowGridLines(status);
        }
    }

    public String getToolTip() {
        return mouseHoverTextPanel.getText();
    }

    public void setDisplayBox(int indx) {
        displayOptionComboBox.setSelectedIndex(indx);
    }

    public void setNormalizationEnabledForReload() {
        //observedNormalizationComboBox.setEnabled(true);
        observedNormalizationComboBox.setEnabled(!isWholeGenome());
        controlNormalizationComboBox.setEnabled(!isWholeGenome());
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
        annotationsPanelToggleButton.setEnabled(status);
        miniAnnotationsLayerPanel.setEnabled(status);
        menuBar.setEnableForAllElements(status);
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

    public void setEnabledForNormalization(boolean isControl, String[] normalizationOptions, boolean status) {
        if (isControl) {
            if (normalizationOptions != null && normalizationOptions.length == 1) {
                controlNormalizationComboBox.setEnabled(false);
            } else {
                controlNormalizationComboBox.setModel(new DefaultComboBoxModel<>(normalizationOptions));
                controlNormalizationComboBox.setSelectedIndex(0);
                controlNormalizationComboBox.setEnabled(status && !isWholeGenome());
            }
        } else {
            if (normalizationOptions.length == 1) {
                observedNormalizationComboBox.setEnabled(false);
            } else {
                observedNormalizationComboBox.setModel(new DefaultComboBoxModel<>(normalizationOptions));
                observedNormalizationComboBox.setSelectedIndex(0);
                observedNormalizationComboBox.setEnabled(status && !isWholeGenome());
            }
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
            displayOptionComboBox.setModel(new DefaultComboBoxModel<>(options));
            int indx = 0;
            for (int i = 0; i < displayOptionComboBox.getItemCount(); i++) {
                if (originalMatrixType.equals(displayOptionComboBox.getItemAt(i))) {
                    indx = i;
                    break;
                }
            }
            displayOptionComboBox.setSelectedIndex(indx);
        } else {
            displayOptionComboBox.setModel(new DefaultComboBoxModel<>(options));
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

    public boolean isTooltipAllowedToUpdate() {
        return tooltipAllowedToUpdate;
    }

    public void toggleToolTipUpdates(boolean tooltipAllowedToUpdated) {
        this.tooltipAllowedToUpdate = tooltipAllowedToUpdated;
    }

    public JComboBox<String> getObservedNormalizationComboBox() {
        return observedNormalizationComboBox;
    }

    public JComboBox<String> getControlNormalizationComboBox() {
        return controlNormalizationComboBox;
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

    public void setAnnotationsPanelToggleButtonSelected(boolean status) {
        annotationsPanelToggleButton.setSelected(status);
        menuBar.setAnnotationPanelMenuItemSelected(status);
    }

    public void updateMiniAnnotationsLayerPanel(SuperAdapter superAdapter) {
        miniAnnotationsLayerPanel.updateRows(superAdapter);
        rightSidePanel.revalidate();
        rightSidePanel.repaint();
    }

    public boolean unsavedEditsExist() {
        return menuBar.unsavedEditsExist();
    }

    public void addRecentMapMenuEntry(String title, boolean status) {
        menuBar.addRecentMapMenuEntry(title, status);
    }

    public void updatePrevStateNameFromImport(String path) {
        menuBar.updatePrevStateNameFromImport(path);
    }

}
