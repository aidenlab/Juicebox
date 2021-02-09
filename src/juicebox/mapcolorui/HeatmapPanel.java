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

import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.assembly.AssemblyHeatmapHandler;
import juicebox.assembly.AssemblyOperationExecutor;
import juicebox.assembly.AssemblyScaffoldHandler;
import juicebox.data.ChromosomeHandler;
import juicebox.data.MatrixZoomData;
import juicebox.data.basics.Chromosome;
import juicebox.gui.SuperAdapter;
import juicebox.track.feature.AnnotationLayerHandler;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DGuiContainer;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;
import org.broad.igv.renderer.GraphicUtils;
import org.broad.igv.ui.FontManager;
import org.broad.igv.util.Pair;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;


public class HeatmapPanel extends JComponent {
    //public static final int clickDelay1 = (Integer) Toolkit.getDefaultToolkit().getDesktopProperty("awt.multiClickInterval");
    private static final long serialVersionUID = 9000028;
    private final MainWindow mainWindow;
    private final HiC hic;
    private final SuperAdapter superAdapter;
    private final ColorScaleHandler colorScaleHandler = new ColorScaleHandler();
    private final GeneralTileManager tileManager = new GeneralTileManager(colorScaleHandler);
    private final HeatmapMouseHandler mouseHandler;
    private boolean showGridLines = true;
    private final HeatmapClickListener clickListener;
    private long[] chromosomeBoundaries;
    private final BoundingBoxRenderer boundingBoxRenderer = new BoundingBoxRenderer(this);

    public HeatmapPanel(SuperAdapter superAdapter) {
        this.mainWindow = superAdapter.getMainWindow();
        this.superAdapter = superAdapter;
        this.hic = superAdapter.getHiC();
        superAdapter.setPearsonColorScale(colorScaleHandler.getPearsonColorScale());
        mouseHandler = new HeatmapMouseHandler(hic, superAdapter, this);
        clickListener = new HeatmapClickListener(this);
        addMouseMotionListener(mouseHandler);
        addMouseListener(mouseHandler);
        addMouseListener(clickListener);
        addMouseWheelListener(mouseHandler);
    }

    public long[] getChromosomeBoundaries() {
        return this.chromosomeBoundaries;
    }

    public void setChromosomeBoundaries(long[] chromosomeBoundaries) {
        this.chromosomeBoundaries = chromosomeBoundaries;
        boundingBoxRenderer.setChromosomeBoundaries(chromosomeBoundaries);
    }

    public int getMinimumDimension() {
        return Math.min(getWidth(), getHeight());
    }

    @Override
    protected void paintComponent(Graphics g1) {
        Graphics2D g = (Graphics2D) g1;
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        Rectangle clipBounds = g.getClipBounds();
        g.clearRect(clipBounds.x, clipBounds.y, clipBounds.width, clipBounds.height);

        if (HiCGlobals.isDarkulaModeEnabled) {
            g.setColor(Color.darkGray);
            g.fillRect(clipBounds.x, clipBounds.y, clipBounds.width, clipBounds.height);
        }

        // Are we ready to draw?
        final MatrixZoomData zd;
        try {
            zd = hic.getZd();
        } catch (Exception ex) {
            return;
        }

        MatrixZoomData controlZd = null;
        try {
            controlZd = hic.getControlZd();
        } catch (Exception ee) {
            ee.printStackTrace();
        }

        if (hic.getXContext() == null) return;

        // todo pearsons
        if (hic.isInPearsonsMode()) {
            // Possibly force asynchronous computation of pearsons
            if (hic.isPearsonsNotAvailableForFile(false)) {
                JOptionPane.showMessageDialog(this, "Pearson's matrix is not available at this " +
                        "resolution, use 500KB or lower resolution.");
                return;
            }
            if (hic.isInControlPearsonsMode() && hic.isPearsonsNotAvailableForFile(false)) {
                JOptionPane.showMessageDialog(this, "Pearson's matrix is not available at this " +
                        "resolution, use 500KB or lower resolution.");
                return;
            }
        }

        // Same scale used for X & Y (square pixels)
        final double scaleFactor = hic.getScaleFactor();
        final int screenWidth = getBounds().width;
        final int screenHeight = getBounds().height;
        double binOriginX = hic.getXContext().getBinOrigin();
        double bRight = binOriginX + (screenWidth / scaleFactor);
        double binOriginY = hic.getYContext().getBinOrigin();
        double bBottom = binOriginY + (screenHeight / scaleFactor);


        HeatmapRenderer renderer = new HeatmapRenderer(g, colorScaleHandler);
        boolean allTilesNull = tileManager.renderHiCTiles(renderer, binOriginX, binOriginY, bRight, bBottom, zd, controlZd,
                scaleFactor, this.getBounds(), hic, this, superAdapter);

        boolean isWholeGenome = ChromosomeHandler.isWholeGenomeView(hic.getXContext(), hic.getYContext());

        Color color0 = g.getColor();

        if (isWholeGenome) {
            boundingBoxRenderer.drawAllByAllGrid(g, zd, showGridLines, binOriginX, binOriginY, scaleFactor);
        } else {
            boundingBoxRenderer.drawRegularGrid(g, zd, showGridLines, hic.getChromosomeHandler(), binOriginX, binOriginY, scaleFactor);
        }
        CursorRenderer cursorRenderer = new CursorRenderer(this);
        cursorRenderer.drawCursors(g, hic.getCursorPoint(), hic.getDiagonalCursorPoint(),
                binOriginX, binOriginY, scaleFactor, hic.getColorForRuler(), bRight, bBottom);

        g.setColor(color0);

        if (allTilesNull) {
            g.setFont(FontManager.getFont(12));
            GraphicUtils.drawCenteredText("Normalization vectors not available at this resolution.  Try a different normalization.", clipBounds, g);
        } else {
            // Render loops
            int centerX = (int) ((screenWidth / scaleFactor) / 2);
            int centerY = (int) ((screenHeight / scaleFactor) / 2);
            float x1 = (float) (binOriginX * zd.getBinSize());
            float y1 = (float) (binOriginY * zd.getBinSize());
            float x2 = x1 + (float) ((screenWidth / scaleFactor) * zd.getBinSize());
            float y2 = y1 + (float) ((screenHeight / scaleFactor) * zd.getBinSize());
            net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(x1, y1, x2, y2);

            Graphics2D g2 = (Graphics2D) g.create();
            mouseHandler.clearFeaturePairs();

            final boolean activelyEditingAssembly = mouseHandler.getIsActivelyEditingAssembly();
            List<AnnotationLayerHandler> handlers;
            if (activelyEditingAssembly) {
                // Only look at assembly layers if we're in assembly mode
                handlers = superAdapter.getAssemblyLayerHandlers();
            } else {
                handlers = superAdapter.getAllLayers();
            }

            for (AnnotationLayerHandler handler : handlers) {
                List<Feature2D> loops = handler.getNearbyFeatures(zd, zd.getChr1Idx(), zd.getChr2Idx(),
                        centerX, centerY, Feature2DHandler.numberOfLoopsToFind, binOriginX, binOriginY, scaleFactor);
                List<Feature2D> cLoopsReflected = new ArrayList<>();
                for (Feature2D feature2D : loops) {
                    if (zd.getChr1Idx() == zd.getChr2Idx() && !feature2D.isOnDiagonal()) {
                        cLoopsReflected.add(feature2D.reflectionAcrossDiagonal());
                    }
                }

                loops.addAll(cLoopsReflected);
                mouseHandler.addAllFeatures(handler, loops, zd,
                        binOriginX, binOriginY, scaleFactor, activelyEditingAssembly);

                final List<Feature2D> highlightedFeatures = mouseHandler.getHighlightedFeature();
                final boolean showFeatureHighlight = mouseHandler.getShouldShowHighlight();

                FeatureRenderer.render(g2, handler, loops, zd, binOriginX, binOriginY, scaleFactor,
                        highlightedFeatures, showFeatureHighlight, this.getWidth(), this.getHeight());
            }
            mouseHandler.renderMouseAnnotations(g2);
            g2.dispose();
        }
    }

    public Image getThumbnailImage(MatrixZoomData zd0, MatrixZoomData ctrl0, int tw, int th, MatrixType displayOption,
                                   NormalizationType observedNormalizationType, NormalizationType controlNormalizationType) {
        if (MatrixType.isPearsonType(displayOption) && hic.isPearsonsNotAvailableForFile(false)) {
            JOptionPane.showMessageDialog(this, "Pearson's matrix is not available at this resolution");
            return null;
        }

        int maxBinCountX = (int) zd0.getXGridAxis().getBinCount();
        int maxBinCountY = (int) zd0.getYGridAxis().getBinCount();

        int wh = Math.max(maxBinCountX, maxBinCountY); // todo assumption for thumbnail
        //if (wh > 1000) wh=1000; // this can happen with single resolution hic files - breaks thumbnail localization

        BufferedImage image = (BufferedImage) createImage(wh, wh);
        Graphics2D g = image.createGraphics();
        if (HiCGlobals.isDarkulaModeEnabled) {
            g.setColor(Color.darkGray);
            g.fillRect(0, 0, wh, wh);
        }

        HeatmapRenderer renderer = new HeatmapRenderer(g, colorScaleHandler);
        boolean success = renderer.render(0,
                0,
                maxBinCountX,
                maxBinCountY,
                zd0,
                ctrl0,
                displayOption,
                observedNormalizationType,
                controlNormalizationType,
                hic.getExpectedValues(),
                hic.getExpectedControlValues(),
                false);

        if (!success) return null;

        return image.getScaledInstance(tw, th, Image.SCALE_REPLICATE);

    }

    public boolean getShowGridLines() {
        return showGridLines;
    }

    public void setShowGridLines(boolean showGridLines) {
        this.showGridLines = showGridLines;
    }

    public HiC getHiC() {
        return this.hic;
    }

    public MainWindow getMainWindow() {
        return this.mainWindow;
    }

    public SuperAdapter getSuperAdapter() {
        return this.superAdapter;
    }

    public void unsafeSetSelectedChromosomes(Chromosome xChrom, Chromosome yChrom) {
        superAdapter.unsafeSetSelectedChromosomes(xChrom, yChrom);
    }

    public void updateThumbnail() {
        superAdapter.updateThumbnail();
    }

    public void clearTileCache() {
        tileManager.clearTileCache();
    }

    public void launchColorSelectionMenu(Pair<Rectangle, Feature2D> selectedFeaturePair) {
        JColorChooser colorChooser = new JColorChooser(selectedFeaturePair.getSecond().getColor());
        JDialog dialog = JColorChooser.createDialog(new JPanel(null), "feature Color Selection", true, colorChooser,
                null, null);
        dialog.setVisible(true);
        Color c = colorChooser.getColor();
        if (c != null) {
            selectedFeaturePair.getSecond().setColor(c);
        }
    }

    public void enableAssemblyEditing() {
        SuperAdapter.assemblyModeCurrentlyActive = true;
        mouseHandler.setActivelyEditingAssembly(true);
        AssemblyHeatmapHandler.setSuperAdapter(superAdapter);
    }

    public void disableAssemblyEditing() {
        mouseHandler.clearSelectedFeatures();
        superAdapter.getMainViewPanel().toggleToolTipUpdates(true);
        mouseHandler.setActivelyEditingAssembly(false);
        HiCGlobals.splitModeEnabled = false;
        SuperAdapter.assemblyModeCurrentlyActive = false;
    }

    public void removeHighlightedFeature() {
        mouseHandler.setFeatureOptionMenuEnabled(false);
        mouseHandler.eraseHighlightedFeatures();
        superAdapter.repaintTrackPanels();
        repaint();
    }

    public void removeSelection() {
        mouseHandler.clearSelectedFeatures();
        superAdapter.updatePreviousTempSelectedGroups(mouseHandler.getTempSelectedGroup());
        mouseHandler.setTempSelectedGroup(null);
        superAdapter.clearEditsAndUpdateLayers();
        HiCGlobals.splitModeEnabled = false;
        superAdapter.getMainViewPanel().toggleToolTipUpdates(true);
        removeHighlightedFeature();

        Chromosome chrX = superAdapter.getHiC().getXContext().getChromosome();
        Chromosome chrY = superAdapter.getHiC().getYContext().getChromosome();
        superAdapter.getEditLayer().filterTempSelectedGroup(chrX.getIndex(), chrY.getIndex());
        final Feature2D debrisFeature = mouseHandler.getDebrisFeature();
        if (debrisFeature != null) {
            superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(
                    chrX.getIndex(), chrY.getIndex(), debrisFeature);
        }

        mouseHandler.resetCurrentPromptedAssemblyAction();
        mouseHandler.reset();
        repaint();
    }

    public void moveSelectionToEnd() {
        AssemblyScaffoldHandler assemblyHandler = superAdapter.getAssemblyStateTracker().getAssemblyHandler();
        final List<Integer> lastLine = assemblyHandler.getListOfSuperscaffolds().get(assemblyHandler.getListOfSuperscaffolds().size() - 1);
        int lastId = Math.abs(lastLine.get(lastLine.size() - 1)) - 1;
        AssemblyOperationExecutor.moveSelection(superAdapter, getSelectedFeatures(),
                assemblyHandler.getListOfScaffolds().get(lastId).getCurrentFeature2D());
        removeSelection();
    }

    public void reset() {
        colorScaleHandler.reset();
        clearTileCache();
    }

    public void setNewDisplayRange(MatrixType displayOption, double min, double max, String key) {
        colorScaleHandler.setNewDisplayRange(displayOption, min, max, key);
        clearTileCache();
        repaint();
    }

    public HeatmapMouseHandler.PromptedAssemblyAction getCurrentPromptedAssemblyAction() {
        return mouseHandler.getCurrentPromptedAssemblyAction();
    }

    public HeatmapMouseHandler.PromptedAssemblyAction getPromptedAssemblyActionOnClick() {
        return mouseHandler.getPromptedAssemblyActionOnClick();
    }

    public void setPromptedAssemblyActionOnClick(HeatmapMouseHandler.PromptedAssemblyAction promptedAssemblyAction) {
        mouseHandler.setPromptedAssemblyActionOnClick(promptedAssemblyAction);
    }

    public List<Feature2D> getSelectedFeatures() {
        return mouseHandler.getSelectedFeatures();
    }

    public Feature2DGuiContainer getCurrentUpstreamFeature() {
        return mouseHandler.getCurrentUpstreamFeature();
    }

    public Feature2DGuiContainer getCurrentDownstreamFeature() {
        return mouseHandler.getCurrentDownstreamFeature();
    }
}