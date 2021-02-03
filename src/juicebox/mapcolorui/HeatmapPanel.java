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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
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
import juicebox.data.CustomMatrixZoomData;
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


/**
 * @author jrobinso
 * @since Aug 2, 2010
 */
public class HeatmapPanel extends JComponent {
    //public static final int clickDelay1 = (Integer) Toolkit.getDefaultToolkit().getDesktopProperty("awt.multiClickInterval");
    private static final long serialVersionUID = -8017012290342597941L;
    private final MainWindow mainWindow;
    private final HiC hic;
    private final SuperAdapter superAdapter;
    private final HeatmapRenderer renderer = new HeatmapRenderer();
    private final TileManager tileManager = new TileManager(renderer);
    private final HeatmapMouseHandler mouseHandler;
    private boolean showGridLines = true;
    private final HeatmapClickListener clickListener;
    private long[] chromosomeBoundaries;

    public HeatmapPanel(SuperAdapter superAdapter) {
        this.mainWindow = superAdapter.getMainWindow();
        this.superAdapter = superAdapter;
        this.hic = superAdapter.getHiC();
        superAdapter.setPearsonColorScale(renderer.getPearsonColorScale());
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
    }

    public int getMinimumDimension() {
        return Math.min(getWidth(), getHeight());
    }

    @Override
    protected void paintComponent(Graphics g) {
        ((Graphics2D) g).setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

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


        boolean allTilesNull = tileManager.renderHiCTiles(g, binOriginX, binOriginY, bRight, bBottom, zd, controlZd,
                scaleFactor, this.getBounds(), hic, this, superAdapter);

        boolean isWholeGenome = ChromosomeHandler.isAllByAll(hic.getXContext().getChromosome()) &&
                ChromosomeHandler.isAllByAll(hic.getYContext().getChromosome());

        //if (mainWindow.isRefreshTest()) {
        // Draw grid

        if (isWholeGenome) {
            Color color = g.getColor();
            if (HiCGlobals.isDarkulaModeEnabled) {
                g.setColor(Color.LIGHT_GRAY);
            } else {
                g.setColor(Color.DARK_GRAY);
            }

            long maxDimension = chromosomeBoundaries[chromosomeBoundaries.length - 1];

            // Draw grid lines only if option is selected
            if (showGridLines) {
                for (long bound : chromosomeBoundaries) {
                    // vertical lines
                    int xBin = zd.getXGridAxis().getBinNumberForGenomicPosition(bound);
                    int x = (int) ((xBin - binOriginX) * scaleFactor);
                    g.drawLine(x, 0, x, getGridLineHeightLimit(zd, maxDimension));

                    // horizontal lines
                    int yBin = zd.getYGridAxis().getBinNumberForGenomicPosition(bound);
                    int y = (int) ((yBin - binOriginY) * scaleFactor);
                    g.drawLine(0, y, getGridLineWidthLimit(zd, maxDimension), y);
                }
            }

            g.setColor(color);

            //Cover gray background for the empty parts of the matrix:
            if (HiCGlobals.isDarkulaModeEnabled) {
                g.setColor(Color.darkGray);
            } else {
                g.setColor(Color.white);
            }
            g.fillRect(getGridLineHeightLimit(zd, maxDimension), 0, getHeight(), getWidth());
            g.fillRect(0, getGridLineWidthLimit(zd, maxDimension), getHeight(), getWidth());
            g.fillRect(getGridLineHeightLimit(zd, maxDimension), getGridLineWidthLimit(zd, maxDimension), getHeight(), getWidth());

        } else {

            if (showGridLines) {
                Color color = g.getColor();
                if (HiCGlobals.isDarkulaModeEnabled) {
                    g.setColor(Color.LIGHT_GRAY);
                } else {
                    g.setColor(Color.DARK_GRAY);
                }
                if (hic.getChromosomeHandler().isCustomChromosome(zd.getChr1())) {
                    if (zd instanceof CustomMatrixZoomData) {
                        List<Long> xBins = ((CustomMatrixZoomData) zd).getBoundariesOfCustomChromosomeX();
                        //int maxSize = xBins.get(xBins.size() - 1);
                        int maxSize = (int) ((zd.getYGridAxis().getBinCount() - binOriginY) * scaleFactor);
                        for (long xBin : xBins) {
                            int x = (int) ((xBin - binOriginX) * scaleFactor);
                            g.drawLine(x, 0, x, maxSize);
                        }
                    }
                }
                if (hic.getChromosomeHandler().isCustomChromosome(zd.getChr2())) {
                    if (zd instanceof CustomMatrixZoomData) {
                        List<Long> yBins = ((CustomMatrixZoomData) zd).getBoundariesOfCustomChromosomeY();
                        //int maxSize = yBins.get(yBins.size() - 1);
                        int maxSize = (int) ((zd.getXGridAxis().getBinCount() - binOriginX) * scaleFactor);
                        for (long yBin : yBins) {
                            int y = (int) ((yBin - binOriginY) * scaleFactor);
                            g.drawLine(0, y, maxSize, y);
                        }
                    }
                }
                g.setColor(color);
            }
        }

        Point cursorPoint = hic.getCursorPoint();
        if (cursorPoint != null) {
            g.setColor(hic.getColorForRuler());
            g.drawLine(cursorPoint.x, 0, cursorPoint.x, getHeight());
            g.drawLine(0, cursorPoint.y, getWidth(), cursorPoint.y);
        } else {
            Point diagonalCursorPoint = hic.getDiagonalCursorPoint();
            if (diagonalCursorPoint != null) {
                g.setColor(hic.getColorForRuler());
                // quadrant 4 signs in plotting equal to quadrant 1 flipped across x in cartesian plane
                // y = -x + b
                // y + x = b
                int b = diagonalCursorPoint.x + diagonalCursorPoint.y;
                // at x = 0, y = b unless y exceeds height
                int leftEdgeY = Math.min(b, getHeight());
                int leftEdgeX = b - leftEdgeY;
                // at y = 0, x = b unless x exceeds width
                int rightEdgeX = Math.min(b, getWidth());
                int rightEdgeY = b - rightEdgeX;
                g.drawLine(leftEdgeX, leftEdgeY, rightEdgeX, rightEdgeY);

                // now we need to draw the perpendicular
                // line which intersects this at the mouse
                // m = -1, neg reciprocal is 1
                // y2 = x2 + b2
                // y2 - x2 = b2
                int b2 = diagonalCursorPoint.y - diagonalCursorPoint.x;
                // at x2 = 0, y2 = b2 unless y less than 0
                int leftEdgeY2 = Math.max(b2, 0);
                int leftEdgeX2 = leftEdgeY2 - b2;
                // at x2 = width, y2 = width+b2 unless x exceeds height
                int rightEdgeY2 = Math.min(getWidth() + b2, getHeight());
                int rightEdgeX2 = rightEdgeY2 - b2;
                g.drawLine(leftEdgeX2, leftEdgeY2, rightEdgeX2, rightEdgeY2);

                // find a point on the diagonal (binx = biny)
                double binXYOrigin = Math.max(binOriginX, binOriginY);
                // ensure diagonal is in view
                if (binXYOrigin < bRight && binXYOrigin < bBottom) {
                    int xDiag = (int) ((binXYOrigin - binOriginX) * scaleFactor);
                    int yDiag = (int) ((binXYOrigin - binOriginY) * scaleFactor);
                    // see if new point is above the line y2 = x2 + b2
                    // y' less than due to flipped topography
                    int vertDisplacement = yDiag - (xDiag + b2);
                    // displacement takes care of directionality of diagonal
                    // being above/below is the second line we drew
                    int b3 = b2 + (2 * vertDisplacement);
                    // at x2 = 0, y2 = b2 unless y less than 0
                    int leftEdgeY3 = Math.max(b3, 0);
                    int leftEdgeX3 = leftEdgeY3 - b3;
                    // at x2 = width, y2 = width+b2 unless x exceeds height
                    int rightEdgeY3 = Math.min(getWidth() + b3, getHeight());
                    int rightEdgeX3 = rightEdgeY3 - b3;
                    g.drawLine(leftEdgeX3, leftEdgeY3, rightEdgeX3, rightEdgeY3);
                }
            }
        }


        if (allTilesNull) {
            g.setFont(FontManager.getFont(12));
            GraphicUtils.drawCenteredText("Normalization vectors not available at this resolution.  Try a different normalization.", clipBounds, g);
        } else {
            // Render loops
            int centerX = (int) (screenWidth / scaleFactor) / 2;
            int centerY = (int) (screenHeight / scaleFactor) / 2;
            //float x1 = (float) binOriginX * zd.getBinSize();
            //float y1 = (float) binOriginY * zd.getBinSize();
            //float x2 = x1 + (float) (screenWidth / scaleFactor) * zd.getBinSize();
            //float y2 = y1 + (float) (screenHeight / scaleFactor) * zd.getBinSize();
            //net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(x1, y1, x2, y2);

            Graphics2D g2 = (Graphics2D) g.create();
            final boolean activelyEditingAssembly = mouseHandler.getIsActivelyEditingAssembly();
            mouseHandler.clearFeaturePairs(activelyEditingAssembly);

            // Only look at assembly layers if we're in assembly mode
            List<AnnotationLayerHandler> handlers;
            if (activelyEditingAssembly) {
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

                final Feature2D highlightedFeature = mouseHandler.getHighlightedFeature();
                final boolean showFeatureHighlight = mouseHandler.getShouldShowHighlight();

                FeatureRenderer.render(g2, handler, loops, zd, binOriginX, binOriginY, scaleFactor,
                        highlightedFeature, showFeatureHighlight, this.getWidth(), this.getHeight());

            }

            mouseHandler.renderMouseAnnotations(g2);
            g2.dispose();
        }
    }

    private int getGridLineWidthLimit(MatrixZoomData zd, long maxPosition) {
        int w = getWidth();
        if (w < 50 || hic.getScaleFactor() == 0) {
            return 0;
        }
        int xBin = zd.getXGridAxis().getBinNumberForGenomicPosition(maxPosition);
        return (int) (xBin * hic.getScaleFactor());
    }

    private int getGridLineHeightLimit(MatrixZoomData zd, long maxPosition) {
        int h = getHeight();
        if (h < 50 || hic.getScaleFactor() == 0) {
            return 0;
        }
        int yBin = zd.getYGridAxis().getBinNumberForGenomicPosition(maxPosition);
        return (int) (yBin * hic.getScaleFactor());
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
                g, false);

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
        superAdapter.getMainViewPanel().toggleToolTipUpdates(Boolean.TRUE);
        mouseHandler.setActivelyEditingAssembly(false);
        HiCGlobals.splitModeEnabled = false;
        SuperAdapter.assemblyModeCurrentlyActive = false;
    }

    public void removeHighlightedFeature() {
        mouseHandler.setFeatureOptionMenuEnabled(false);
        mouseHandler.eraseHighlightedFeature();
        superAdapter.repaintTrackPanels();
        repaint();
    }

    public void removeSelection() {
        mouseHandler.clearSelectedFeatures();
        superAdapter.updatePreviousTempSelectedGroups(mouseHandler.getTempSelectedGroup());
        mouseHandler.setTempSelectedGroup(null);
        superAdapter.clearEditsAndUpdateLayers();
        HiCGlobals.splitModeEnabled = false;
        superAdapter.getMainViewPanel().toggleToolTipUpdates(Boolean.TRUE);
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
        renderer.reset();
        clearTileCache();
    }

    public void setNewDisplayRange(MatrixType displayOption, double min, double max, String key) {
        renderer.setNewDisplayRange(displayOption, min, max, key);
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