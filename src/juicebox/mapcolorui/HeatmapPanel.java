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

package juicebox.mapcolorui;

import com.jidesoft.swing.JidePopupMenu;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.data.ExpectedValueFunction;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.gui.MainMenuBar;
import juicebox.gui.SuperAdapter;
import juicebox.track.HiCFragmentAxis;
import juicebox.track.HiCGridAxis;
import juicebox.track.feature.Feature2D;
import juicebox.windowui.EditFeatureAttributesDialog;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.renderer.GraphicUtils;
import org.broad.igv.ui.FontManager;
import org.broad.igv.util.ObjectCache;
import org.broad.igv.util.Pair;

import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;

import static java.awt.Toolkit.getDefaultToolkit;


/**
 * @author jrobinso
 * @since Aug 2, 2010
 */
public class HeatmapPanel extends JComponent implements Serializable {

    private static final long serialVersionUID = -8017012290342597941L;

    // used for finding nearby features
    private static final int NUM_NEIGHBORS = 7;
    private final NumberFormat formatter = NumberFormat.getInstance();
    private final MainWindow mainWindow;
    private final HiC hic;
    private final SuperAdapter superAdapter;
    private final int RESIZE_SNAP = 5;
    /**
     * Image tile width in pixels
     */
    private final int imageTileWidth = 500;
    private final ObjectCache<String, ImageTile> tileCache = new ObjectCache<String, ImageTile>(26);
    private final HeatmapRenderer renderer;
    //private final transient List<Pair<Rectangle, Feature2D>> drawnLoopFeatures;
    private transient List<Pair<Rectangle, Feature2D>> customFeaturePairs;
    private Rectangle zoomRectangle;
    private Rectangle annotateRectangle;
    /**
     * Chromosome boundaries in kbases for whole genome view.
     */
    private int[] chromosomeBoundaries;
    private boolean straightEdgeEnabled = false;
    private boolean featureOptionMenuEnabled = false;
    private boolean firstAnnotation;
    private AdjustAnnotation adjustAnnotation = AdjustAnnotation.NONE;
    /**
     * feature highlight related variables
     */
    private boolean showFeatureHighlight = true;
    private Feature2D highlightedFeature = null;
    private Pair<Rectangle, Feature2D> mostRecentRectFeaturePair = null;
    private Pair<Pair<Integer, Integer>, Feature2D> preAdjustLoop = null;
    private boolean changedSize = false;

    /**
     * Initialize heatmap panel
     *
     * @param superAdapter
     */
    public HeatmapPanel(SuperAdapter superAdapter) {
        this.mainWindow = superAdapter.getMainWindow();
        this.superAdapter = superAdapter;
        this.hic = superAdapter.getHiC();
        renderer = new HeatmapRenderer();
        final HeatmapMouseHandler mouseHandler = new HeatmapMouseHandler();
        addMouseMotionListener(mouseHandler);
        addMouseListener(mouseHandler);
        addMouseWheelListener(mouseHandler);
        this.firstAnnotation = true;
    }

    public void setChromosomeBoundaries(int[] chromosomeBoundaries) {
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

        // Are we ready to draw?
        final MatrixZoomData zd;
        try {
            zd = hic.getZd();
        } catch (Exception ex) {
            return;
        }

        if (hic.getXContext() == null) return;

        if (hic.isInPearsonsMode()) {
            // Possibly force asynchronous computation of pearsons
            if (zd.getPearsons(hic.getDataset().getExpectedValues(zd.getZoom(), hic.getNormalizationType())) == null) {
                JOptionPane.showMessageDialog(this, "Pearson's matrix is not available at this resolution, use 500KB or lower resolution.");
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

        // tile numbers
        int tLeft = (int) (binOriginX / imageTileWidth);
        int tRight = (int) Math.ceil(bRight / imageTileWidth);
        int tTop = (int) (binOriginY / imageTileWidth);
        int tBottom = (int) Math.ceil(bBottom / imageTileWidth);

        MatrixType displayOption = hic.getDisplayOption();
        NormalizationType normalizationType = hic.getNormalizationType();

        boolean allTilesNull = true;
        for (int tileRow = tTop; tileRow <= tBottom; tileRow++) {
            for (int tileColumn = tLeft; tileColumn <= tRight; tileColumn++) {

                ImageTile tile;
                try {
                    tile = getImageTile(zd, tileRow, tileColumn, displayOption, normalizationType);
                } catch (Exception e) {
                    return;
                }
                if (tile != null) {
                    allTilesNull = false;

                    int imageWidth = tile.image.getWidth(null);
                    int imageHeight = tile.image.getHeight(null);

                    int xSrc0 = 0;
                    int xSrc1 = imageWidth;
                    int ySrc0 = 0;
                    int ySrc1 = imageHeight;

                    int xDest0 = (int) ((tile.bLeft - binOriginX) * scaleFactor);
                    int xDest1 = (int) ((tile.bLeft + imageWidth - binOriginX) * scaleFactor);
                    int yDest0 = (int) ((tile.bTop - binOriginY) * scaleFactor);
                    int yDest1 = (int) ((tile.bTop + imageHeight - binOriginY) * scaleFactor);

                    // Trim off edges that are out of view -- take care if you attempt to simplify or rearrange this,
                    // its easy to introduce alias and round-off errors due to the int casts.  I suggest leaving it alone.
                    Rectangle bounds = getBounds();
                    final int screenRight = bounds.x + bounds.width;
                    final int screenBottom = bounds.y + bounds.height;
                    if (xDest0 < 0) {
                        int leftExcess = (int) (-xDest0 / scaleFactor);
                        xSrc0 += leftExcess;
                        xDest0 = (int) ((tile.bLeft - binOriginX + leftExcess) * scaleFactor);
                    }
                    if (xDest1 > screenRight) {
                        int rightExcess = (int) ((xDest1 - screenRight) / scaleFactor);
                        xSrc1 -= rightExcess;
                        xDest1 = (int) ((tile.bLeft + imageWidth - binOriginX - rightExcess) * scaleFactor);
                    }
                    if (yDest0 < 0) {
                        int topExcess = (int) (-yDest0 / scaleFactor);
                        ySrc0 += topExcess;
                        yDest0 = (int) ((tile.bTop - binOriginY + topExcess) * scaleFactor);
                    }
                    if (yDest1 > screenBottom) {
                        int bottomExcess = (int) ((yDest1 - screenBottom) / scaleFactor);
                        ySrc1 -= bottomExcess;
                        yDest1 = (int) ((tile.bTop + imageHeight - binOriginY - bottomExcess) * scaleFactor);
                    }


                    //if (mainWindow.isRefreshTest()) {
                    try {
                        if (xDest0 < xDest1 && yDest0 < yDest1 && xSrc0 < xSrc1 && ySrc0 < ySrc1) {
                            // basically ensure that we're not trying to plot empty space
                            // also for some reason we have negative indices sometimes??
                            g.drawImage(tile.image, xDest0, yDest0, xDest1, yDest1, xSrc0, ySrc0, xSrc1, ySrc1, null);
                        }
                    } catch (Exception e) {

                        // handling for svg export
                        try {
                            if (HiCGlobals.printVerboseComments) {
                                System.out.println("svg plotting for\n" + xDest0 + "_" + yDest0 + "_" + xDest1 + "_" +
                                        yDest1 + "_" + xSrc0 + "_" + ySrc0 + "_" + xSrc1 + "_" + ySrc1);
                            }
                            bypassTileAndDirectlyDrawOnGraphics((Graphics2D) g, zd, tileRow, tileColumn,
                                    displayOption, normalizationType,
                                    xDest0, yDest0, xDest1, yDest1, xSrc0, ySrc0, xSrc1, ySrc1);
                            //processedExportRegions.add(newKey);
                        } catch (Exception e2) {
                            System.err.println("SVG export did not work");
                        }
                    }
                    //}


                    //TODO ******** UNCOMMENT *******
                    //Uncomment to draw tile grid (for debugging)
                    //g.drawRect((int) xDest0, (int) yDest0, (int) (xDest1 - xDest0), (int) (yDest1 - yDest0));

                }
            }

            //In case of change to map settings, get map color limits and update slider:
            //TODO: || might not catch all changed at once, if more then one parameter changed...
            if (hic.testZoomChanged() || hic.testDisplayOptionChanged() || hic.testNormalizationTypeChanged()) {
                //In case tender is called as a result of zoom change event, check if
                //We need to update slider with map range:
                String cacheKey = HeatmapRenderer.getColorScaleCacheKey(zd, displayOption);
                renderer.updateColorSliderFromColorScale(superAdapter, displayOption, cacheKey);
            }


            //Uncomment to draw bin grid (for debugging)
//            Graphics2D g2 = (Graphics2D) g.create();
//            g2.setColor(Color.green);
//            g2.setColor(new Color(0, 0, 1.0f, 0.3f));
//            for (int bin = (int) binOriginX; bin <= bRight; bin++) {
//                int pX = (int) ((bin - hic.getXContext().getBinOrigin()) * hic.getScaleFactor());
//                g2.drawLine(pX, 0, pX, getHeight());
//            }
//            for (int bin = (int) binOriginY; bin <= bBottom; bin++) {
//                int pY = (int) ((bin - hic.getYContext().getBinOrigin()) * hic.getScaleFactor());
//                g2.drawLine(0, pY, getWidth(), pY);
//            }
//            g2.dispose();

            boolean isWholeGenome = HiCFileTools.isAllChromosome(hic.getXContext().getChromosome()) &&
                    HiCFileTools.isAllChromosome(hic.getYContext().getChromosome());

            //if (mainWindow.isRefreshTest()) {
            // Draw grid

            if (isWholeGenome) {
                Color color = g.getColor();
                g.setColor(Color.LIGHT_GRAY);

                for (int bound : chromosomeBoundaries) {
                    // vertical lines
                    int xBin = zd.getXGridAxis().getBinNumberForGenomicPosition(bound);
                    int x = (int) ((xBin - binOriginX) * scaleFactor);
                    g.drawLine(x, 0, x, getTickHeight(zd));

                    // horizontal lines
                    int yBin = zd.getYGridAxis().getBinNumberForGenomicPosition(bound);
                    int y = (int) ((yBin - binOriginY) * hic.getScaleFactor());
                    g.drawLine(0, y, getTickWidth(zd), y);
                }

                g.setColor(color);

                //Cover gray background for the empty parts of the matrix:
                g.setColor(Color.white);
                g.fillRect(getTickHeight(zd), 0, getHeight(), getWidth());
                g.fillRect(0, getTickWidth(zd), getHeight(), getWidth());
                g.fillRect(getTickHeight(zd), getTickWidth(zd), getHeight(), getWidth());
            }

            Point cursorPoint = hic.getCursorPoint();
            if (cursorPoint != null) {
                g.setColor(HiCGlobals.RULER_LINE_COLOR);
                g.drawLine(cursorPoint.x, 0, cursorPoint.x, getHeight());
                g.drawLine(0, cursorPoint.y, getWidth(), cursorPoint.y);
            }


            if (allTilesNull) {
                g.setFont(FontManager.getFont(12));
                GraphicUtils.drawCenteredText("Normalization vectors not available at this resolution.  Try a different normalization.", clipBounds, g);

            } else {
                // Render loops
                //drawnLoopFeatures.clear();

                //double x = (screenWidth / scaleFactor)/2.0;//binOriginX;// +(screenWidth / scaleFactor)/2.0;
                //double y = (screenHeight / scaleFactor)/2.0;//binOriginY;// +(screenHeight / scaleFactor)/2.0;

                List<Feature2D> loops = hic.findNearbyFeatures(zd, zd.getChr1Idx(), zd.getChr2Idx(),
                        0, 0, Feature2DHandler.numberOfLoopsToFind);

                List<Feature2D> cLoops = MainMenuBar.customAnnotations.getVisibleLoopList(zd.getChr1Idx(), zd.getChr2Idx());
                List<Feature2D> cLoopsReflected = new ArrayList<Feature2D>();
                for (Feature2D feature2D : cLoops) {
                    if (zd.getChr1Idx() == zd.getChr2Idx() && !feature2D.isOnDiagonal()) {
                        cLoopsReflected.add(feature2D.reflectionAcrossDiagonal());
                    }
                }
                loops.addAll(cLoops);
                loops.addAll(cLoopsReflected);

                customFeaturePairs = Feature2DHandler.featurePairs(cLoops, zd, binOriginX, binOriginY, scaleFactor);
                customFeaturePairs.addAll(Feature2DHandler.featurePairs(cLoopsReflected, zd, binOriginX, binOriginY, scaleFactor));

                Graphics2D g2 = (Graphics2D) g.create();
                //g2.fillOval((int)x, (int)y, 20, 20);

                FeatureRenderer.render(g2, loops, zd, binOriginX, binOriginY, scaleFactor,
                        highlightedFeature, showFeatureHighlight, this.getWidth(), this.getHeight());

                if (zoomRectangle != null) {
                    ((Graphics2D) g).draw(zoomRectangle);
                }

                if (annotateRectangle != null) {
                    ((Graphics2D) g).draw(annotateRectangle);
                }
            }
        }
    }

    private void bypassTileAndDirectlyDrawOnGraphics(Graphics2D g, MatrixZoomData zd, int tileRow, int tileColumn,
                                                     MatrixType displayOption, NormalizationType normalizationType,
                                                     int xDest0, int yDest0, int xDest1, int yDest1, int xSrc0,
                                                     int ySrc0, int xSrc1, int ySrc1) {

        // Image size can be smaller than tile width when zoomed out, or near the edges.

        int maxBinCountX = zd.getXGridAxis().getBinCount();
        int maxBinCountY = zd.getYGridAxis().getBinCount();

        if (maxBinCountX < 0 || maxBinCountY < 0) return;

        int imageWidth = maxBinCountX < imageTileWidth ? maxBinCountX : imageTileWidth;
        int imageHeight = maxBinCountY < imageTileWidth ? maxBinCountY : imageTileWidth;

        final int bx0 = tileColumn * imageTileWidth;
        final int by0 = tileRow * imageTileWidth;

        // set new origins
        g.translate(xDest0, yDest0);

        // scale drawing appropriately
        double widthDest = xDest1 - xDest0;
        double heightDest = yDest1 - yDest0;
        int widthSrc = xSrc1 - xSrc0;
        int heightSrc = ySrc1 - ySrc0;
        double horizontalScaling = widthDest / widthSrc;
        double verticalScaling = heightDest / heightSrc;
        g.scale(horizontalScaling, verticalScaling);

        final int bx0Offset = bx0 + xSrc0;
        final int by0Offset = by0 + ySrc0;

        renderer.render(bx0Offset,
                by0Offset,
                widthSrc,
                heightSrc,
                zd,
                hic.getControlZd(),
                displayOption,
                normalizationType,
                hic.getDataset().getExpectedValues(zd.getZoom(), normalizationType),
                g);

        g.scale(1, 1);
        g.translate(0, 0);
    }

    private int getTickWidth(MatrixZoomData zd) {

        int w = getWidth();
        //int h = getHeight();

        if (w < 50 || hic.getScaleFactor() == 0) {
            return 0;
        }

        int xBin = zd.getXGridAxis().getBinNumberForGenomicPosition(chromosomeBoundaries[chromosomeBoundaries.length - 1]);
        return (int) (xBin * hic.getScaleFactor());
    }

    private int getTickHeight(MatrixZoomData zd) {

        int h = getHeight();
        //int w = getWidth();

        if (h < 50 || hic.getScaleFactor() == 0) {
            return 0;
        }

        int xBin = zd.getXGridAxis().getBinNumberForGenomicPosition(chromosomeBoundaries[chromosomeBoundaries.length - 1]);
        return (int) (xBin * hic.getScaleFactor());
    }

    public Image getThumbnailImage(MatrixZoomData zd0, MatrixZoomData ctrl0, int tw, int th, MatrixType displayOption,
                                   NormalizationType normalizationType) {

        if (displayOption == MatrixType.PEARSON &&
                zd0.getPearsons(hic.getDataset().getExpectedValues(zd0.getZoom(), normalizationType)) == null) {
            JOptionPane.showMessageDialog(this, "Pearson's matrix is not available at this resolution");
            return null;

        }
        int maxBinCountX = zd0.getXGridAxis().getBinCount();
        int maxBinCountY = zd0.getYGridAxis().getBinCount();

        int wh = Math.max(maxBinCountX, maxBinCountY);
        if (wh > 1000) wh=1000; // this can happen with single resolution hic files

        BufferedImage image = (BufferedImage) createImage(wh, wh);
        Graphics2D g = image.createGraphics();
        boolean success = renderer.render(0,
                0,
                maxBinCountX,
                maxBinCountY,
                zd0,
                ctrl0,
                displayOption,
                normalizationType,
                hic.getDataset().getExpectedValues(zd0.getZoom(), normalizationType),
                g);

        if (!success) return null;

        return image.getScaledInstance(tw, th, Image.SCALE_REPLICATE);

    }

    /**
     * Return the specified image tile, scaled by scaleFactor
     *
     * @param zd         Matrix of tile
     * @param tileRow    row index of tile
     * @param tileColumn column index of tile
     * @return image tile
     */
    private ImageTile getImageTile(MatrixZoomData zd, int tileRow, int tileColumn, MatrixType displayOption,
                                   NormalizationType normalizationType) {

        String key = zd.getKey() + "_" + tileRow + "_" + tileColumn + "_ " + displayOption;
        ImageTile tile = tileCache.get(key);

        if (tile == null) {

            // Image size can be smaller than tile width when zoomed out, or near the edges.

            int maxBinCountX = zd.getXGridAxis().getBinCount();
            int maxBinCountY = zd.getYGridAxis().getBinCount();

            if (maxBinCountX < 0 || maxBinCountY < 0) return null;

            int imageWidth = maxBinCountX < imageTileWidth ? maxBinCountX : imageTileWidth;
            int imageHeight = maxBinCountY < imageTileWidth ? maxBinCountY : imageTileWidth;

            BufferedImage image = (BufferedImage) createImage(imageWidth, imageHeight);
            Graphics2D g2D = (Graphics2D) image.getGraphics();

            final int bx0 = tileColumn * imageTileWidth;
            final int by0 = tileRow * imageTileWidth;

            if (!renderer.render(bx0,
                    by0,
                    imageWidth,
                    imageHeight,
                    zd,
                    hic.getControlZd(),
                    displayOption,
                    normalizationType,
                    hic.getDataset().getExpectedValues(zd.getZoom(), normalizationType),
                    g2D)) {
                return null;
            }

            //           if (scaleFactor > 0.999 && scaleFactor < 1.001) {
            tile = new ImageTile(image, bx0, by0);
            tileCache.put(key, tile);
        }
        return tile;
    }

    public void clearTileCache() {
        tileCache.clear();
    }

    private void launchColorSelectionMenu(Pair<Rectangle, Feature2D> selectedFeaturePair) {
        JColorChooser colorChooser = new JColorChooser(selectedFeaturePair.getSecond().getColor());
        JDialog dialog = JColorChooser.createDialog(new JPanel(null), "feature Color Selection", true, colorChooser,
                null, null);
        dialog.setVisible(true);
        Color c = colorChooser.getColor();
        if (c != null) {
            selectedFeaturePair.getSecond().setColor(c);
        }
    }

    private JidePopupMenu getPopupMenu(final int xMousePos, final int yMousePos) {

        JidePopupMenu menu = new JidePopupMenu();

        /* Undo Zoom implementation _UZI
        if(hic.isCanRedoZoomChangeAvailable() || hic.isCanUndoZoomChangeAvailable()) {
            String title = "Undo Zoom";
            final boolean timeToUndoZoom;
            if(hic.isCanRedoZoomChangeAvailable()){
                timeToUndoZoom = false;
                title = "Redo Zoom";
            }
            else {
                timeToUndoZoom = true;
            }
            final JMenuItem mi0UndoOrRedo = new JMenuItem(title);
            mi0UndoOrRedo.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    if(timeToUndoZoom){
                        hic.undoZoomChange();
                    }
                    else {
                        hic.redoZoomChange();
                    }
                }
            });
            menu.add(mi0UndoOrRedo);
        }
        */

        final JCheckBoxMenuItem mi = new JCheckBoxMenuItem("Enable straight edge");
        mi.setSelected(straightEdgeEnabled);
        mi.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (mi.isSelected()) {
                    straightEdgeEnabled = true;
                    setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
                } else {
                    straightEdgeEnabled = false;
                    hic.setCursorPoint(null);
                    setCursor(Cursor.getDefaultCursor());
                    repaint();
                    superAdapter.repaintTrackPanels();
                }

            }
        });
        menu.add(mi);

     /*   final JMenuItem mi2 = new JMenuItem("Goto ...");
        mi2.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String fragmentString = JOptionPane.showInputDialog(HeatmapPanel.this,
                        "Enter fragment or bp range in the form <bp|frag>:x:y");
                if (fragmentString != null) {
                    String[] tokens = Globals.colonPattern.split(fragmentString);
                    HiC.Unit unit = HiC.Unit.FRAG;
                    int idx = 0;
                    if (tokens.length == 3) {
                        if (tokens[idx++].equalsIgnoreCase(HiCFileTools.BP)) {
                            unit = HiC.Unit.BP;
                        }
                    }
                    int x = Integer.parseInt(tokens[idx++].replace(",", ""));
                    int y = (tokens.length > idx) ? Integer.parseInt(tokens[idx].replace(",", "")) : x;

                    if (unit == HiC.Unit.FRAG) {
                        hic.centerFragment(x, y);
                    } else {
                        hic.centerBP(x, y);
                    }


                }
            }
        });*/

        // internally, single sync = what we previously called sync
        final JMenuItem mi3 = new JMenuItem("Broadcast Single Sync");
        mi3.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                hic.broadcastLocation();
            }
        });

        // internally, continuous sync = what we used to call linked
        final JCheckBoxMenuItem mi4 = new JCheckBoxMenuItem("Broadcast Continuous Sync");
        mi4.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                final boolean isLinked = mi4.isSelected();
                if (isLinked) {
                    hic.broadcastLocation();
                }
                hic.setLinkedMode(isLinked);
            }
        });

        final JCheckBoxMenuItem mi5 = new JCheckBoxMenuItem("Freeze hover text");
        mi5.setSelected(!superAdapter.isTooltipAllowedToUpdated());
        mi5.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                superAdapter.toggleToolTipUpdates(!superAdapter.isTooltipAllowedToUpdated());
            }
        });

        final JCheckBoxMenuItem mi6 = new JCheckBoxMenuItem("Copy hover text to clipboard");
        mi6.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                StringSelection stringSelection = new StringSelection(superAdapter.getToolTip());
                Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
                clpbrd.setContents(stringSelection, null);
            }
        });

        final JCheckBoxMenuItem mi7 = new JCheckBoxMenuItem("Copy top position to clipboard");
        mi7.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                StringSelection stringSelection = new StringSelection(hic.getXPosition());
                superAdapter.setPositionChrTop(hic.getXPosition().concat(":").concat(String.valueOf(hic.getXContext().getZoom().getBinSize())));
                superAdapter.setPositionChrLeft(hic.getYPosition().concat(":").concat(String.valueOf(hic.getYContext().getZoom().getBinSize())));
                Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
                clpbrd.setContents(stringSelection, null);
            }
        });

        // TODO - can we remove this second option and just have a copy position to clipboard? Is this used?
        final JCheckBoxMenuItem mi8 = new JCheckBoxMenuItem("Copy left position to clipboard");
        mi8.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                StringSelection stringSelection = new StringSelection(hic.getYPosition());
                superAdapter.setPositionChrTop(hic.getXPosition().concat(":").concat(String.valueOf(hic.getXContext().getZoom().getBinSize())));
                superAdapter.setPositionChrLeft(hic.getYPosition().concat(":").concat(String.valueOf(hic.getYContext().getZoom().getBinSize())));
                Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
                clpbrd.setContents(stringSelection, null);
            }
        });

        final JCheckBoxMenuItem mi85Highlight = new JCheckBoxMenuItem("Highlight");
        mi85Highlight.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                featureOptionMenuEnabled = false;
                showFeatureHighlight = true;
                highlightedFeature = mostRecentRectFeaturePair.getSecond();
                repaint();
            }
        });

        final JCheckBoxMenuItem mi86Toggle = new JCheckBoxMenuItem("Toggle Highlight Visibility");
        mi86Toggle.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                featureOptionMenuEnabled = false;
                showFeatureHighlight = !showFeatureHighlight;
                repaint();
            }
        });

        final JCheckBoxMenuItem mi87Remove = new JCheckBoxMenuItem("Remove Highlight");
        mi87Remove.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                featureOptionMenuEnabled = false;
                highlightedFeature = null;
                repaint();
            }
        });

        final JCheckBoxMenuItem mi9_h = new JCheckBoxMenuItem("Generate Horizontal 1D Track");
        mi9_h.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                hic.generateTrackFromLocation(yMousePos, true);
            }
        });

        final JCheckBoxMenuItem mi9_v = new JCheckBoxMenuItem("Generate Vertical 1D Track");
        mi9_v.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                hic.generateTrackFromLocation(xMousePos, false);
            }
        });


        final JMenuItem mi10_1 = new JMenuItem("Change Color");
        mi10_1.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                featureOptionMenuEnabled = false;
                Pair<Rectangle, Feature2D> featureCopy =
                        new Pair<Rectangle, Feature2D>(mostRecentRectFeaturePair.getFirst(), mostRecentRectFeaturePair.getSecond());
                launchColorSelectionMenu(featureCopy);
            }
        });

        final JMenuItem mi10_2 = new JMenuItem("Change Attributes");
        mi10_2.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                featureOptionMenuEnabled = false;
                new EditFeatureAttributesDialog(mainWindow, mostRecentRectFeaturePair.getSecond(),
                        MainMenuBar.customAnnotations);
            }
        });

        final JMenuItem mi10_3 = new JMenuItem("Delete");
        mi10_3.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                featureOptionMenuEnabled = false;
                Feature2D feature = mostRecentRectFeaturePair.getSecond();
                int chr1Idx = hic.getXContext().getChromosome().getIndex();
                int chr2Idx = hic.getYContext().getChromosome().getIndex();
                MainMenuBar.customAnnotations.removeFromList(chr1Idx, chr2Idx, feature);
            }
        });


        final JMenu configureFeatureMenu = new JMenu("Configure feature");
        configureFeatureMenu.add(mi10_1);
        configureFeatureMenu.add(mi10_2);
        configureFeatureMenu.add(mi10_3);

        if (hic != null) {
            //    menu.add(mi2);
            menu.add(mi3);
            mi4.setSelected(hic.isLinkedMode());
            menu.add(mi4);
            menu.add(mi5);
            menu.add(mi6);
            menu.add(mi7);
            menu.add(mi8);
            if (!hic.getXContext().getChromosome().getName().equals(Globals.CHR_ALL)
                    && MatrixType.isObservedOrControl(hic.getDisplayOption())) {
                menu.addSeparator();
                menu.add(mi9_h);
                menu.add(mi9_v);
            }

            boolean menuSeparatorNotAdded = true;

            if (highlightedFeature != null) {
                menu.addSeparator();
                menuSeparatorNotAdded = false;
                mi86Toggle.setSelected(showFeatureHighlight);
                menu.add(mi86Toggle);
            }

            if (mostRecentRectFeaturePair != null) {//mouseIsOverFeature
                featureOptionMenuEnabled = true;
                if (menuSeparatorNotAdded) {
                    menu.addSeparator();
                }

                if (highlightedFeature != null) {
                    if (mostRecentRectFeaturePair.getSecond() != highlightedFeature) {
                        configureFeatureMenu.add(mi85Highlight);
                        menu.add(mi87Remove);
                    } else {
                        configureFeatureMenu.add(mi87Remove);
                    }
                } else {
                    configureFeatureMenu.add(mi85Highlight);
                }

                menu.add(configureFeatureMenu);
            } else if (highlightedFeature != null) {
                menu.add(mi87Remove);
            }

            //menu.add(mi9);
        }


        return menu;

    }

    private String toolTipText(int x, int y) {
        // Update popup text
        final MatrixZoomData zd;
        HiCGridAxis xGridAxis, yGridAxis;
        try {
            zd = hic.getZd();
            xGridAxis = zd.getXGridAxis();
            yGridAxis = zd.getYGridAxis();
        } catch (Exception e) {
            return "";
        }

        int binX = (int) (hic.getXContext().getBinOrigin() + x / hic.getScaleFactor());
        int binY = (int) (hic.getYContext().getBinOrigin() + y / hic.getScaleFactor());

        int xGenomeStart = xGridAxis.getGenomicStart(binX) + 1; // Conversion from in internal "0" -> 1 base coordinates
        int yGenomeStart = yGridAxis.getGenomicStart(binY) + 1;
        int xGenomeEnd = xGridAxis.getGenomicEnd(binX);
        int yGenomeEnd = yGridAxis.getGenomicEnd(binY);

        if (hic.isWholeGenome()) {

            Chromosome xChrom = null;
            Chromosome yChrom = null;
            for (int i = 0; i < chromosomeBoundaries.length; i++) {
                if (xChrom == null && chromosomeBoundaries[i] > xGenomeStart) {
                    xChrom = hic.getChromosomes().get(i + 1);
                    break;
                }
            }
            for (int i = 0; i < chromosomeBoundaries.length; i++) {
                if (yChrom == null && chromosomeBoundaries[i] > yGenomeStart) {
                    yChrom = hic.getChromosomes().get(i + 1);
                    break;
                }
            }
            if (xChrom != null && yChrom != null) {

                int leftBoundaryX = xChrom.getIndex() == 1 ? 0 : chromosomeBoundaries[xChrom.getIndex() - 2];
                int leftBoundaryY = yChrom.getIndex() == 1 ? 0 : chromosomeBoundaries[yChrom.getIndex() - 2];

                int xChromPos = (xGenomeStart - leftBoundaryX) * 1000;
                int yChromPos = (yGenomeStart - leftBoundaryY) * 1000;

                String txt = "";
                txt += "<html><span style='color:" + HiCGlobals.topChromosomeColor + "; font-family: arial; font-size: 12pt;'>";
                txt += xChrom.getName();
                txt += ":";
                txt += String.valueOf(xChromPos);
                txt += "</span><br><span style='color:" + HiCGlobals.leftChromosomeColor + "; font-family: arial; font-size: 12pt;'>";
                txt += yChrom.getName();
                txt += ":";
                txt += String.valueOf(yChromPos);
                txt += "</span></html>";

                if (xChrom.getName().toLowerCase().contains("chr")) {
                    hic.setXPosition(xChrom.getName() + ":" + String.valueOf(xChromPos));
                } else {
                    hic.setXPosition("chr" + xChrom.getName() + ":" + String.valueOf(xChromPos));
                }
                if (yChrom.getName().toLowerCase().contains("chr")) {
                    hic.setYPosition(yChrom.getName() + ":" + String.valueOf(yChromPos));
                } else {
                    hic.setYPosition("chr" + yChrom.getName() + ":" + String.valueOf(yChromPos));
                }
                return txt;
            }

        } else {

            //Update Position in hic. Used for clipboard copy:
            if (hic.getXContext().getChromosome().getName().toLowerCase().contains("chr")) {
                hic.setXPosition(hic.getXContext().getChromosome().getName() + ":" + formatter.format(xGenomeStart) + "-" + formatter.format(xGenomeEnd));
            } else {
                hic.setXPosition("chr" + hic.getXContext().getChromosome().getName() + ":" + formatter.format(xGenomeStart) + "-" + formatter.format(xGenomeEnd));
            }
            if (hic.getYContext().getChromosome().getName().toLowerCase().contains("chr")) {
                hic.setYPosition(hic.getYContext().getChromosome().getName() + ":" + formatter.format(yGenomeStart) + "-" + formatter.format(yGenomeEnd));
            } else {
                hic.setYPosition("chr" + hic.getYContext().getChromosome().getName() + ":" + formatter.format(yGenomeStart) + "-" + formatter.format(yGenomeEnd));
            }

            //int binX = (int) ((mainWindow.xContext.getOrigin() + e.getX() * mainWindow.xContext.getScale()) / getBinWidth());
            //int binY = (int) ((mainWindow.yContext.getOrigin() + e.getY() * mainWindow.yContext.getScale()) / getBinWidth());
            StringBuilder txt = new StringBuilder();

            txt.append("<html><span style='color:" + HiCGlobals.topChromosomeColor + "; font-family: arial; font-size: 12pt; '>");
            txt.append(hic.getXContext().getChromosome().getName());
            txt.append(":");
            txt.append(formatter.format(xGenomeStart));
            txt.append("-");
            txt.append(formatter.format(xGenomeEnd));

            if (xGridAxis instanceof HiCFragmentAxis) {
                String fragNumbers;
                int binSize = zd.getZoom().getBinSize();
                if (binSize == 1) {
                    fragNumbers = formatter.format(binX);
                } else {
                    int leftFragment = binX * binSize;
                    int rightFragment = ((binX + 1) * binSize) - 1;
                    fragNumbers = formatter.format(leftFragment) + "-" + formatter.format(rightFragment);
                }
                txt.append("  (");
                txt.append(fragNumbers);
                txt.append("  len=");
                txt.append(formatter.format(xGenomeEnd - xGenomeStart));
                txt.append(")");
            }

            txt.append("</span><br><span style='color:" + HiCGlobals.leftChromosomeColor + "; font-family: arial; font-size: 12pt; '>");
            txt.append(hic.getYContext().getChromosome().getName());
            txt.append(":");
            txt.append(formatter.format(yGenomeStart));
            txt.append("-");
            txt.append(formatter.format(yGenomeEnd));

            if (yGridAxis instanceof HiCFragmentAxis) {
                String fragNumbers;
                int binSize = zd.getZoom().getBinSize();
                if (binSize == 1) {
                    fragNumbers = formatter.format(binY);
                } else {
                    int leftFragment = binY * binSize;
                    int rightFragment = ((binY + 1) * binSize) - 1;
                    fragNumbers = formatter.format(leftFragment) + "-" + formatter.format(rightFragment);
                }
                txt.append("  (");
                txt.append(fragNumbers);
                txt.append("  len=");
                txt.append(formatter.format(yGenomeEnd - yGenomeStart));
                txt.append(")");
            }
            txt.append("</span><span style='font-family: arial; font-size: 12pt;'>");

            if (hic.isInPearsonsMode()) {
                float value = zd.getPearsonValue(binX, binY, hic.getNormalizationType());
                if (!Float.isNaN(value)) {

                    txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append("value = ");
                    txt.append(value);
                    txt.append("</span>");

                }
            } else {
                float value = hic.getNormalizedObservedValue(binX, binY);
                if (!Float.isNaN(value)) {
                    txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append("observed value = ");
                    txt.append(getFloatString(value));
                    txt.append("</span>");
                }

                int c1 = hic.getXContext().getChromosome().getIndex();
                int c2 = hic.getYContext().getChromosome().getIndex();
                double ev = 0;
                if (c1 == c2) {
                    ExpectedValueFunction df = hic.getExpectedValues();
                    if (df != null) {
                        int distance = Math.abs(binX - binY);
                        ev = df.getExpectedValue(c1, distance);
                    }
                } else {
                    ev = zd.getAverageCount();
                }

                String evString = ev < 0.001 || Double.isNaN(ev) ? String.valueOf(ev) : formatter.format(ev);
                txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
                txt.append("expected value = ");
                txt.append(evString);
                txt.append("</span>");
                if (ev > 0 && !Float.isNaN(value)) {
                    txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append("O/E            = ");
                    txt.append(formatter.format(value / ev));
                    txt.append("</span>");
                } else {
                    txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append("O/E            = NaN");
                    txt.append("</span>");
                }

                MatrixZoomData controlZD = hic.getControlZd();
                if (controlZD != null) {
                    float controlValue = controlZD.getObservedValue(binX, binY, hic.getNormalizationType());
                    txt.append("<br><br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append("control value = ");
                    txt.append(getFloatString(controlValue));
                    txt.append("</span>");

                    double obsValue = (value / zd.getAverageCount());
                    txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append("observed/average = ");
                    txt.append(getFloatString((float) obsValue));
                    txt.append("</span>");

                    double ctlValue = (float) (controlValue / controlZD.getAverageCount());
                    txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append("control/average = ");
                    txt.append(getFloatString((float) ctlValue));
                    txt.append("</span>");

                    if (value > 0 && controlValue > 0) {
                        double ratio = obsValue / ctlValue;
                        txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
                        txt.append("O'/C' = ");
                        txt.append(getFloatString((float) ratio));
                        txt.append("</span>");
                    }
                }
            }

            Point currMouse = new Point(x, y);
            double minDistance = Double.POSITIVE_INFINITY;
            //mouseIsOverFeature = false;
            mostRecentRectFeaturePair = null;

            List<Pair<Rectangle, Feature2D>> neighbors = hic.findNearbyFeaturePairs(zd, zd.getChr1Idx(), zd.getChr2Idx(), x, y, NUM_NEIGHBORS);
            neighbors.addAll(customFeaturePairs);

            for (Pair<Rectangle, Feature2D> loop : neighbors) {
                if (loop.getFirst().contains(x, y)) {
                    // TODO - why is this code duplicated in this file?
                    txt.append("<br><br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append(loop.getSecond().tooltipText());
                    txt.append("</span>");

                    double distance = currMouse.distance(loop.getFirst().getX(), loop.getFirst().getY());
                    if (distance < minDistance) {
                        minDistance = distance;
                        mostRecentRectFeaturePair = loop;
                    }
                    //mouseIsOverFeature = true;
                }
            }
            txt.append("</html>");
            return txt.toString();
        }

        return null;
    }

    private String getFloatString(float value) {
        String valueString;
        if (Float.isNaN(value)) {
            valueString = "NaN";
        } else if (value < 0.001) {
            valueString = String.valueOf(value);
        } else {
            valueString = formatter.format(value);
        }
        return valueString;
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

    private enum AdjustAnnotation {LEFT, RIGHT, NONE}

    private enum DragMode {ZOOM, ANNOTATE, RESIZE, PAN, SELECT, NONE}

//    @Override
//    public String getToolTipText(MouseEvent e) {
//        return toolTipText(e.getX(), e.getY());
//
//    }

    static class ImageTile {
        final int bLeft;
        final int bTop;
        final Image image;

        ImageTile(Image image, int bLeft, int py0) {
            this.bLeft = bLeft;
            this.bTop = py0;
            this.image = image;
        }
    }

    class HeatmapMouseHandler extends MouseAdapter {

        DragMode dragMode = DragMode.NONE;
        private Point lastMousePoint;

        @Override
        public void mouseEntered(MouseEvent e) {
            if (straightEdgeEnabled) {
                setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
            } else {
                setCursor(Cursor.getDefaultCursor());
            }
        }

        @Override
        public void mouseExited(MouseEvent e) {
            hic.setCursorPoint(null);
            if (straightEdgeEnabled) {
                superAdapter.repaintTrackPanels();
            }
        }

        @Override
        public void mousePressed(final MouseEvent e) {
            featureOptionMenuEnabled = false;

            if (hic.isWholeGenome()) {
                if (e.isPopupTrigger()) {
                    getPopupMenu(e.getX(), e.getY()).show(HeatmapPanel.this, e.getX(), e.getY());
                }
                return;
            }
            // Priority is right click
            if (e.isPopupTrigger()) {
                getPopupMenu(e.getX(), e.getY()).show(HeatmapPanel.this, e.getX(), e.getY());
                // Alt down for zoom
            } else if (e.isAltDown()) {
                dragMode = DragMode.ZOOM;
                // Shift down for custom annotations
            } else if (e.isShiftDown()) {
                boolean showWarning = false;

                if (superAdapter.unsavedEditsExist() && firstAnnotation && showWarning) {
                    firstAnnotation = false;
                    String text = "There are unsaved hand annotations from your previous session! \n" +
                            "Go to 'Annotations > Hand Annotations > Load Last' to restore.";
                    System.err.println(text);
                    JOptionPane.showMessageDialog(superAdapter.getMainWindow(), text);
                }

                dragMode = DragMode.ANNOTATE;
                MainMenuBar.customAnnotationHandler.updateSelectionPoint(e.getX(), e.getY());
                MainMenuBar.customAnnotationHandler.doPeak();

                setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
                // Corners for resize annotation
            } else if (adjustAnnotation != AdjustAnnotation.NONE) {
                dragMode = DragMode.RESIZE;
                Feature2D loop = mostRecentRectFeaturePair.getSecond();
                // Resizing upper left corner, keep end points stationary
                if (adjustAnnotation == AdjustAnnotation.LEFT) {
                    MainMenuBar.customAnnotationHandler.setStationaryEnd(loop.getEnd1(), loop.getEnd2());
                    // Resizing lower right corner, keep start points stationary
                } else {
                    MainMenuBar.customAnnotationHandler.setStationaryStart(loop.getStart1(), loop.getStart2());
                }


                try {
                    HiCGridAxis xAxis = hic.getZd().getXGridAxis();
                    HiCGridAxis yAxis = hic.getZd().getYGridAxis();
                    final double scaleFactor = hic.getScaleFactor();
                    double binOriginX = hic.getXContext().getBinOrigin();
                    double binOriginY = hic.getYContext().getBinOrigin();
                    loop.doTest();//TODO meh - please comment why test?
                    annotateRectangle = Feature2DHandler.rectangleFromFeature(xAxis, yAxis, loop, binOriginX, binOriginY, scaleFactor);
                    int chr1Idx = hic.getXContext().getChromosome().getIndex();
                    int chr2Idx = hic.getYContext().getChromosome().getIndex();
                    preAdjustLoop = new Pair<Pair<Integer, Integer>, Feature2D>(new Pair<Integer, Integer>(chr1Idx, chr2Idx), loop);

                } catch (Exception ex) {
                    ex.printStackTrace();
                }

            } else {
                dragMode = DragMode.PAN;
                setCursor(MainWindow.fistCursor);
            }
            lastMousePoint = e.getPoint();
        }


        @Override
        public void mouseReleased(final MouseEvent e) {
            if (e.isPopupTrigger()) {
                getPopupMenu(e.getX(), e.getY()).show(HeatmapPanel.this, e.getX(), e.getY());
                dragMode = DragMode.NONE;
                lastMousePoint = null;
                zoomRectangle = null;
                annotateRectangle = null;
                setCursor(straightEdgeEnabled ? Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR) : Cursor.getDefaultCursor());
            } else if ((dragMode == DragMode.ZOOM || dragMode == DragMode.SELECT) && zoomRectangle != null) {
                Runnable runnable = new Runnable() {
                    @Override
                    public void run() {
                        unsafeDragging();
                    }
                };
                mainWindow.executeLongRunningTask(runnable, "Mouse Drag");
            } else if (dragMode == DragMode.ANNOTATE) {
                // New annotation is added (not single click) and new feature from custom annotation
                MainMenuBar.customAnnotationHandler.addFeature(hic, MainMenuBar.customAnnotations);
                restoreDefaultVariables();
            } else if (dragMode == DragMode.RESIZE) {
                // New annotation is added (not single click) and new feature from custom annotation
                int idx1 = preAdjustLoop.getFirst().getFirst();
                int idx2 = preAdjustLoop.getFirst().getSecond();
                Feature2D loop = preAdjustLoop.getSecond();
                // Add a new loop if it was resized (prevents deletion on single click)
                if (MainMenuBar.customAnnotations.hasLoop(idx1, idx2, loop) && changedSize == true) {
                    MainMenuBar.customAnnotations.removeFromList(idx1, idx2, loop);

//                    // Snap to nearest neighbor, if close enough
//                    MatrixZoomData zd = null;
//                    try {
//                        zd = hic.getZd();
//                    } catch (Exception exception) {
//                        exception.printStackTrace();
//                    }
//                    List<Pair<Rectangle, Feature2D>> neighbors = hic.findNearbyFeaturePairs(zd, zd.getChr1Idx(), zd.getChr2Idx(), e.getX(), e.getY(), NUM_NEIGHBORS);
//                    // Look for left neighbors
//                    if (adjustAnnotation == AdjustAnnotation.LEFT) {
//                        for (Pair<Rectangle, Feature2D> neighbor : neighbors){
//                            double neighborEdge = neighbor.getFirst().getX() + neighbor.getFirst().getWidth();
//
//                        }
//                    // Look for right neighbors
//                    } else {
//
//                    }
                    MainMenuBar.customAnnotationHandler.addFeature(hic, MainMenuBar.customAnnotations);
                    MainMenuBar.customAnnotationHandler.setLastItem(idx1, idx2, loop);
                }
                restoreDefaultVariables();
            } else {
                setCursor(straightEdgeEnabled ? Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR) : Cursor.getDefaultCursor());
            }
        }

        private void restoreDefaultVariables() {
            dragMode = DragMode.NONE;
            adjustAnnotation = AdjustAnnotation.NONE;
            annotateRectangle = null;
            lastMousePoint = null;
            zoomRectangle = null;
            preAdjustLoop = null;
            hic.setCursorPoint(null);
            changedSize = false;
            setCursor(Cursor.getDefaultCursor());
            repaint();
            superAdapter.repaintTrackPanels();
        }

        private void unsafeDragging() {
            final double scaleFactor1 = hic.getScaleFactor();
            double binX = hic.getXContext().getBinOrigin() + (zoomRectangle.x / scaleFactor1);
            double binY = hic.getYContext().getBinOrigin() + (zoomRectangle.y / scaleFactor1);
            double wBins = (int) (zoomRectangle.width / scaleFactor1);
            double hBins = (int) (zoomRectangle.height / scaleFactor1);

            try {
                final MatrixZoomData currentZD = hic.getZd();
                int xBP0 = currentZD.getXGridAxis().getGenomicStart(binX);

                int yBP0 = currentZD.getYGridAxis().getGenomicEnd(binY);

                double newXBinSize = wBins * currentZD.getBinSize() / getWidth();
                double newYBinSize = hBins * currentZD.getBinSize() / getHeight();
                double newBinSize = Math.max(newXBinSize, newYBinSize);

                hic.zoomToDrawnBox(xBP0, yBP0, newBinSize);
            } catch (Exception e) {
                e.printStackTrace();
            }

            dragMode = DragMode.NONE;
            lastMousePoint = null;
            zoomRectangle = null;
            setCursor(straightEdgeEnabled ? Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR) : Cursor.getDefaultCursor());
        }


        @Override
        final public void mouseDragged(final MouseEvent e) {

            Rectangle lastRectangle, damageRect;
            int x, y;
            double x_d, y_d;

            try {
                hic.getZd();
            } catch (Exception ex) {
                return;
            }

            if (hic.isWholeGenome()) {
                return;
            }

            if (lastMousePoint == null) {
                lastMousePoint = e.getPoint();
                return;
            }

            int deltaX = e.getX() - lastMousePoint.x;
            int deltaY = e.getY() - lastMousePoint.y;
            double deltaX_d = e.getX() - lastMousePoint.x;
            double deltaY_d = e.getY() - lastMousePoint.y;

            switch (dragMode) {
                case ZOOM:
                    lastRectangle = zoomRectangle;

                    if (deltaX == 0 || deltaY == 0) {
                        return;
                    }

                    // Constrain aspect ratio of zoom rectangle to that of panel
                    double aspectRatio = (double) getWidth() / getHeight();
                    if (deltaX * aspectRatio > deltaY) {
                        deltaY = (int) (deltaX / aspectRatio);
                    } else {
                        deltaX = (int) (deltaY * aspectRatio);
                    }

                    x = deltaX > 0 ? lastMousePoint.x : lastMousePoint.x + deltaX;
                    y = deltaY > 0 ? lastMousePoint.y : lastMousePoint.y + deltaY;
                    zoomRectangle = new Rectangle(x, y, Math.abs(deltaX), Math.abs(deltaY));

                    damageRect = lastRectangle == null ? zoomRectangle : zoomRectangle.union(lastRectangle);
                    damageRect.x--;
                    damageRect.y--;
                    damageRect.width += 2;
                    damageRect.height += 2;
                    paintImmediately(damageRect);

                    break;
                case ANNOTATE:
                    lastRectangle = annotateRectangle;

                    if (deltaX_d == 0 || deltaY_d == 0) {
                        return;
                    }

                    x = deltaX > 0 ? lastMousePoint.x : lastMousePoint.x + deltaX;
                    y = deltaY > 0 ? lastMousePoint.y : lastMousePoint.y + deltaY;
                    annotateRectangle = new Rectangle(x, y, Math.abs(deltaX), Math.abs(deltaY));

                    damageRect = lastRectangle == null ? annotateRectangle : annotateRectangle.union(lastRectangle);
                    MainMenuBar.customAnnotationHandler.updateSelectionRegion(damageRect);
                    damageRect.x--;
                    damageRect.y--;
                    damageRect.width += 2;
                    damageRect.height += 2;
                    paintImmediately(damageRect);
                    break;
                case RESIZE:
                    if (deltaX_d == 0 || deltaY_d == 0) {
                        return;
                    }

                    lastRectangle = annotateRectangle;
                    double rectX;
                    double rectY;

                    // Resizing upper left corner
                    if (adjustAnnotation == AdjustAnnotation.LEFT) {
                        rectX = annotateRectangle.getX() + annotateRectangle.getWidth();
                        rectY = annotateRectangle.getY() + annotateRectangle.getHeight();
                        // Resizing lower right corner
                    } else {
                        rectX = annotateRectangle.getX();
                        rectY = annotateRectangle.getY();
                    }
                    deltaX_d = e.getX() - rectX;
                    deltaY_d = e.getY() - rectY;

                    x_d = deltaX_d > 0 ? rectX : rectX + deltaX_d;
                    y_d = deltaY_d > 0 ? rectY : rectY + deltaY_d;

                    annotateRectangle = new Rectangle((int) x_d, (int) y_d, (int) Math.abs(deltaX_d), (int) Math.abs(deltaY_d));
                    damageRect = lastRectangle == null ? annotateRectangle : annotateRectangle.union(lastRectangle);
                    damageRect.width += 1;
                    damageRect.height += 1;
                    paintImmediately(damageRect);
                    MainMenuBar.customAnnotationHandler.updateSelectionRegion(damageRect);
                    changedSize = true;
                    break;
                default:
                    lastMousePoint = e.getPoint();    // Always save the last Point

                    double deltaXBins = -deltaX / hic.getScaleFactor();
                    double deltaYBins = -deltaY / hic.getScaleFactor();
                    hic.moveBy(deltaXBins, deltaYBins);

            }

        }

        private void unsafeMouseClickSubActionA(final MouseEvent eF) {
            double binX = hic.getXContext().getBinOrigin() + (eF.getX() / hic.getScaleFactor());
            double binY = hic.getYContext().getBinOrigin() + (eF.getY() / hic.getScaleFactor());


            Chromosome xChrom = null;
            Chromosome yChrom = null;

            try {
                int xGenome = hic.getZd().getXGridAxis().getGenomicMid(binX);
                int yGenome = hic.getZd().getYGridAxis().getGenomicMid(binY);
                for (int i = 0; i < chromosomeBoundaries.length; i++) {
                    if (xChrom == null && chromosomeBoundaries[i] > xGenome) {
                        xChrom = hic.getChromosomes().get(i + 1);
                    }
                    if (yChrom == null && chromosomeBoundaries[i] > yGenome) {
                        yChrom = hic.getChromosomes().get(i + 1);
                    }
                }
            } catch (Exception ex) {
                // do nothing, leave chromosomes null
            }
            if (xChrom != null && yChrom != null) {

                final Chromosome xC = xChrom;
                final Chromosome yC = yChrom;
                superAdapter.unsafeSetSelectedChromosomes(xC, yC);
            }

            //Only if zoom is changed All->Chr:
            superAdapter.updateThumbnail();
        }

        private void unsafeMouseClickSubActionB(double centerBinX, double centerBinY, HiCZoom newZoom) {
            try {
                final int xGenome = hic.getZd().getXGridAxis().getGenomicMid(centerBinX);
                final int yGenome = hic.getZd().getYGridAxis().getGenomicMid(centerBinY);

                hic.unsafeActuallySetZoomAndLocation("", "", newZoom, xGenome, yGenome, -1, false,
                        HiC.ZoomCallType.STANDARD, true);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        private void safeMouseClicked(final MouseEvent eF) {

            if (!eF.isPopupTrigger() && eF.getButton() == MouseEvent.BUTTON1 && !eF.isControlDown()) {

                try {
                    hic.getZd();
                } catch (Exception e) {
                    return;
                }

                if (hic.isWholeGenome()) {
                    //avoid double click...
                    if (eF.getClickCount() == 1) {

                        Runnable runnable = new Runnable() {
                            public void run() {
                                unsafeMouseClickSubActionA(eF);
                            }
                        };
                        mainWindow.executeLongRunningTask(runnable, "Mouse Click Set Chr");

                    }
                } else if (eF.getClickCount() == 2) {

                    // Double click,  zoom and center on click location
                    try {
                        final HiCZoom currentZoom = hic.getZd().getZoom();
                        final HiCZoom nextPotentialZoom = hic.getDataset().getNextZoom(currentZoom, !eF.isAltDown());
                        final HiCZoom newZoom = hic.isResolutionLocked() ||
                                hic.isPearsonEdgeCaseEncountered(nextPotentialZoom) ? currentZoom : nextPotentialZoom;

                        // If newZoom == currentZoom adjust scale factor (no change in resolution)
                        final double centerBinX = hic.getXContext().getBinOrigin() + (eF.getX() / hic.getScaleFactor());
                        final double centerBinY = hic.getYContext().getBinOrigin() + (eF.getY() / hic.getScaleFactor());

                        if (newZoom.equals(currentZoom)) {
                            double mult = eF.isAltDown() ? 0.5 : 2.0;
                            double newScaleFactor = Math.max(1.0, hic.getScaleFactor() * mult);
                            hic.setScaleFactor(newScaleFactor);
                            hic.getXContext().setBinOrigin(Math.max(0, (int) (centerBinX - (getWidth() / (2 * newScaleFactor)))));
                            hic.getYContext().setBinOrigin(Math.max(0, (int) (centerBinY - (getHeight() / (2 * newScaleFactor)))));
                            mainWindow.repaint();
                        } else {
                            Runnable runnable = new Runnable() {
                                public void run() {
                                    unsafeMouseClickSubActionB(centerBinX, centerBinY, newZoom);
                                }
                            };
                            mainWindow.executeLongRunningTask(runnable, "Mouse Click Zoom");
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        }

        @Override
        public void mouseClicked(MouseEvent e) {

            if (hic == null) return;
            safeMouseClicked(e);
        }

        @Override
        public void mouseMoved(MouseEvent e) {
            try {
                hic.getZd();
            } catch (Exception ex) {
                return;
            }
            if (hic.getXContext() != null) {
                adjustAnnotation = AdjustAnnotation.NONE;

                // Update tool tip text
                if (!featureOptionMenuEnabled) {
                    superAdapter.updateToolTipText(toolTipText(e.getX(), e.getY()));
                }
                // Set check if hovering over feature corner
                if (mostRecentRectFeaturePair != null) {

                    Rectangle loop = mostRecentRectFeaturePair.getFirst();
                    Point mousePoint = e.getPoint();
                    int minDist = 20;
                    // Mouse near top left corner
                    if ((Math.abs(loop.getMinX() - mousePoint.getX()) <= minDist &&
                            Math.abs(loop.getMinY() - mousePoint.getY()) <= minDist)) {
                        adjustAnnotation = AdjustAnnotation.LEFT;
                        setCursor(Cursor.getPredefinedCursor(Cursor.NW_RESIZE_CURSOR));
                        // Mouse near bottom right corner
                    } else if (Math.abs(loop.getMaxX() - mousePoint.getX()) <= minDist &&
                            Math.abs(loop.getMaxY() - mousePoint.getY()) <= minDist) {
                        adjustAnnotation = AdjustAnnotation.RIGHT;
                        setCursor(Cursor.getPredefinedCursor(Cursor.SE_RESIZE_CURSOR));
                    } else {
                        setCursor(Cursor.getDefaultCursor());
                    }
                } else {
                    setCursor(Cursor.getDefaultCursor());
                }

                if (straightEdgeEnabled || e.isShiftDown()) {
                    synchronized (this) {
                        hic.setCursorPoint(e.getPoint());
                        superAdapter.repaintTrackPanels();
                    }
                } else if (adjustAnnotation == AdjustAnnotation.NONE) {
                    hic.setCursorPoint(null);
                    setCursor(Cursor.getDefaultCursor());
                }
                repaint();
            }
        }

        @Override
        public void mouseWheelMoved(MouseWheelEvent e) {
            int scroll = e.getWheelRotation();
            hic.moveBy(scroll, scroll);
            superAdapter.updateToolTipText(toolTipText(e.getX(), e.getY()));
        }


    }
}
