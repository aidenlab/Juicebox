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

import com.jidesoft.swing.JidePopupMenu;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.data.ExpectedValueFunction;
import juicebox.data.MatrixZoomData;
import juicebox.track.HiCFragmentAxis;
import juicebox.track.HiCGridAxis;
import juicebox.track.feature.Feature2D;
import juicebox.windowui.EditFeatureAttributesDialog;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.MatrixType;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.renderer.GraphicUtils;
import org.broad.igv.ui.FontManager;
import org.broad.igv.util.ObjectCache;
import org.broad.igv.util.Pair;

import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
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
    private final NumberFormat formatter = NumberFormat.getInstance();
    private final MainWindow mainWindow;
    private final HiC hic;
    /**
     * Image tile width in pixels
     */
    private final int imageTileWidth = 500;
    private final ObjectCache<String, ImageTile> tileCache = new ObjectCache<String, ImageTile>(26);
    private final transient List<Pair<Rectangle, Feature2D>> drawnLoopFeatures;
    private final HeatmapRenderer renderer;
    private Rectangle zoomRectangle;
    private Rectangle annotateRectangle;
    /**
     * Chromosome boundaries in kbases for whole genome view.
     */
    private int[] chromosomeBoundaries;
    private boolean straightEdgeEnabled = false;
    private boolean featureOptionMenuEnabled = false;

    /**
     * feature highlight related variables
     */
    private boolean showFeatureHighlight = true;
    private boolean mouseIsOverFeature = false;
    private Pair<Rectangle, Feature2D> highlightedFeature = null;
    private Pair<Rectangle, Feature2D> mostRecentRectFeaturePair = null;

    /**
     * @param mainWindow
     * @param hic
     */
    public HeatmapPanel(MainWindow mainWindow, HiC hic) {
        this.mainWindow = mainWindow;
        this.hic = hic;
        renderer = new HeatmapRenderer(mainWindow, hic);
        final HeatmapMouseHandler mouseHandler = new HeatmapMouseHandler();
        addMouseListener(mouseHandler);
        addMouseMotionListener(mouseHandler);
        drawnLoopFeatures = new ArrayList<Pair<Rectangle, Feature2D>>();
        //setToolTipText(""); // Turns tooltip on
    }

    public void reset() {
        renderer.reset();
        clearTileCache();
    }

    public void setObservedRange(double min, double max) {
        renderer.setObservedRange(min, max);
        clearTileCache();
        repaint();
    }

    public void setOEMax(double max) {
        renderer.setOEMax(max);
        clearTileCache();
        repaint();

    }

    public void setPreDefRange(double min, double max) {
        renderer.setPreDefRange(min, max);
        clearTileCache();
        repaint();

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
        final MatrixZoomData zd = hic.getZd();
        if (zd == null || hic.getXContext() == null) return;

        if (hic.getDisplayOption() == MatrixType.PEARSON) {
            // Possibly force asynchronous computation of pearsons
            if (zd.getPearsons(hic.getDataset().getExpectedValues(zd.getZoom(), hic.getNormalizationType())) == null) {
                JOptionPane.showMessageDialog(this, "Pearson's matrix is not available at this resolution");
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

        boolean allTilesNull = true;
        for (int tileRow = tTop; tileRow <= tBottom; tileRow++) {
            for (int tileColumn = tLeft; tileColumn <= tRight; tileColumn++) {

                ImageTile tile = getImageTile(zd, tileRow, tileColumn, displayOption);
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
                        g.drawImage(tile.image, xDest0, yDest0, xDest1, yDest1, xSrc0, ySrc0, xSrc1, ySrc1, null);
                    //}
                    //TODO ******** UNCOMMENT *******

                    // Uncomment to draw tile grid (for debugging)
                    //g.drawRect((int) xDest0, (int) yDest0, (int) (xDest1 - xDest0), (int) (yDest1 - yDest0));

                }
            }

            //In case of change to map settings, get map color limits and update slider:
            //TODO: || might not catch all changed at once, if more then one parameter changed...
            if (hic.testZoomChanged() || hic.testDisplayOptionChanged() || hic.testNormalizationTypeChanged()) {
                //In case tender is called as a result of zoom change event, check if
                //We need to update slider with map range:
                renderer.updateColorSliderFromColorScale(zd, displayOption);
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

            boolean isWholeGenome = (hic.getXContext().getChromosome().getName().equals("All") &&
                    hic.getYContext().getChromosome().getName().equals("All"));

            //if (mainWindow.isRefreshTest()) {
                // Draw grid
                if (isWholeGenome) {
                    Color color = g.getColor();
                    g.setColor(Color.LIGHT_GRAY);

                    List<Chromosome> chromosomes = hic.getChromosomes();
                    // Index 0 is whole genome
                    int xGenomeCoord = 0;
                    int x = 0;
                    for (int i = 1; i < chromosomes.size(); i++) {
                        Chromosome c = chromosomes.get(i);
                        xGenomeCoord += (c.getLength() / 1000);
                        int xBin = zd.getXGridAxis().getBinNumberForGenomicPosition(xGenomeCoord);
                        x = (int) (xBin * scaleFactor);
                        g.drawLine(x, 0, x, getTickHeight(g));
                    }


                    int yGenomeCoord = 0;
                    int y = 0;
                    for (int i = 1; i < chromosomes.size(); i++) {
                        Chromosome c = chromosomes.get(i);
                        yGenomeCoord += (c.getLength() / 1000);
                        int yBin = zd.getYGridAxis().getBinNumberForGenomicPosition(yGenomeCoord);
                        y = (int) (yBin * hic.getScaleFactor());
                        g.drawLine(0, y, getTickWidth(g), y);
                    }

                    g.setColor(color);

                    //Cover gray background for the empty parts of the matrix:
                    g.setColor(Color.white);
                    g.fillRect(getTickHeight(g), 0, getHeight(), getWidth());
                    g.fillRect(0, getTickWidth(g), getHeight(), getWidth());
                    g.fillRect(getTickHeight(g), getTickWidth(g), getHeight(), getWidth());
                }

                Point cursorPoint = hic.getCursorPoint();
                if (cursorPoint != null) {
                    g.setColor(MainWindow.RULER_LINE_COLOR);
                    g.drawLine(cursorPoint.x, 0, cursorPoint.x, getHeight());
                    g.drawLine(0, cursorPoint.y, getWidth(), cursorPoint.y);
                }


                if (allTilesNull) {
                    g.setFont(FontManager.getFont(12));
                    GraphicUtils.drawCenteredText("Normalization vectors not available at this resolution.  Try a different normalization.", clipBounds, g);

                } else {
                    // Render loops
                    drawnLoopFeatures.clear();

                    List<Feature2D> loops = hic.getVisibleLoopList(zd.getChr1Idx(), zd.getChr2Idx());
                    // customLoops is array with zero or more loops
                    List<Feature2D> customLoops = MainWindow.customAnnotations.getVisibleLoopList(zd.getChr1Idx(), zd.getChr2Idx());
                    if (loops == null) {
                        loops = customLoops;
                    } else {
                        loops.addAll(customLoops);
                    }

                    FeatureRenderer.render((Graphics2D) g.create(), loops, zd, binOriginX, binOriginY, scaleFactor, drawnLoopFeatures,
                            highlightedFeature, showFeatureHighlight, this.getWidth(), this.getHeight());

                    if (zoomRectangle != null) {
                        ((Graphics2D) g).draw(zoomRectangle);
                    }

                    if (annotateRectangle != null) {
                        ((Graphics2D) g).draw(annotateRectangle);
                    }

                }
            //}
            //UNCOMMENT TO OUTLINE "selected" BIN TODO - is this safe to delete?
            //        if(hic.getSelectedBin() != null) {
            //            int pX = (int) ((hic.getSelectedBin().x - hic.xContext.getBinOrigin()) * hic.xContext.getScaleFactor());
            //            int pY = (int) ((hic.getSelectedBin().y - hic.yContext.getBinOrigin()) * hic.yContext.getScaleFactor());
            //            int w = (int) hic.xContext.getScaleFactor() - 1;
            //            int h = (int) hic.yContext.getScaleFactor() - 1;
            //            g.setColor(Color.green);
            //            g.drawRect(pX, pY, w, h);
            //        }

        }
    }

    private int getTickWidth(Graphics g) {

        int w = getWidth();
        int h = getHeight();

        if (w < 50 || hic.getScaleFactor() == 0) {
            return 0;
        }

        List<Chromosome> chromosomes = hic.getChromosomes();
        // Index 0 is whole genome
        int genomeCoord = 0;
        for (int i = 1; i < chromosomes.size(); i++) {

            Chromosome c = chromosomes.get(i);
            genomeCoord += (c.getLength() / 1000);
        }

        int xBin = hic.getZd().getXGridAxis().getBinNumberForGenomicPosition(genomeCoord);
        return (int) (xBin * hic.getScaleFactor());
    }

    private int getTickHeight(Graphics g) {

        int w = getHeight();
        int h = getWidth();

        if (w < 50 || hic.getScaleFactor() == 0) {
            return 0;
        }

        List<Chromosome> chromosomes = hic.getChromosomes();
        // Index 0 is whole genome
        int genomeCoord = 0;
        for (int i = 1; i < chromosomes.size(); i++) {

            Chromosome c = chromosomes.get(i);
            genomeCoord += (c.getLength() / 1000);
        }

        int xBin = hic.getZd().getXGridAxis().getBinNumberForGenomicPosition(genomeCoord);
        return (int) (xBin * hic.getScaleFactor());
    }


    public Image getThumbnailImage(MatrixZoomData zd0, MatrixZoomData ctrl0, int tw, int th, MatrixType displayOption) {

        if (hic.getDisplayOption() == MatrixType.PEARSON &&
                zd0.getPearsons(hic.getDataset().getExpectedValues(zd0.getZoom(), hic.getNormalizationType())) == null) {
            JOptionPane.showMessageDialog(this, "Pearson's matrix is not available at this resolution");
            return null;

        }
        int maxBinCountX = zd0.getXGridAxis().getBinCount();
        int maxBinCountY = zd0.getYGridAxis().getBinCount();

        int wh = Math.max(maxBinCountX, maxBinCountY);

        BufferedImage image = (BufferedImage) createImage(wh, wh);
        Graphics2D g = image.createGraphics();
        boolean success = renderer.render(0,
                0,
                maxBinCountX,
                maxBinCountY,
                zd0,
                ctrl0,
                displayOption,
                hic.getNormalizationType(),
                hic.getDataset().getExpectedValues(zd0.getZoom(), hic.getNormalizationType()),
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
    private ImageTile getImageTile(MatrixZoomData zd, int tileRow, int tileColumn, MatrixType displayOption) {

        String key = zd.getKey() + "_" + tileRow + "_" + tileColumn + "_ " + displayOption;
        ImageTile tile = tileCache.get(key);

        if (tile == null) {

            // Image size can be smaller than tile width when zoomed out, or near the edges.
            int maxBinCountX = hic.getZd().getXGridAxis().getBinCount();
            int maxBinCountY = hic.getZd().getYGridAxis().getBinCount();

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
                    hic.getZd(),
                    hic.getControlZd(),
                    displayOption,
                    hic.getNormalizationType(),
                    hic.getDataset().getExpectedValues(hic.getZd().getZoom(), hic.getNormalizationType()),
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
        if(c != null) {
            selectedFeaturePair.getSecond().setColor(c);
        }
    }

    private JidePopupMenu getPopupMenu() {

        JidePopupMenu menu = new JidePopupMenu();

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
                    mainWindow.repaintTrackPanels();
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
                        if (tokens[idx++].toLowerCase().equals("bp")) {
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
                hic.broadcastState();
            }
        });

        // internally, continuous sync = what we used to call linked
        final JCheckBoxMenuItem mi4 = new JCheckBoxMenuItem("Broadcast Continuous Sync");
        mi4.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                final boolean isLinked = mi4.isSelected();
                if (isLinked) {
                    hic.broadcastState();
                }
                hic.setLinkedMode(isLinked);
            }
        });

        final JCheckBoxMenuItem mi5 = new JCheckBoxMenuItem("Freeze hover text");
        mi5.setSelected(!mainWindow.isTooltipAllowedToUpdated());
        mi5.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                mainWindow.toggleToolTipUpdates(!mainWindow.isTooltipAllowedToUpdated());
            }
        });

        final JCheckBoxMenuItem mi6 = new JCheckBoxMenuItem("Copy hover text to clipboard");
        mi6.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                StringSelection stringSelection = new StringSelection(mainWindow.getToolTip());
                Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
                clpbrd.setContents(stringSelection, null);
            }
        });

        final JCheckBoxMenuItem mi7 = new JCheckBoxMenuItem("Copy top position to clipboard");
        mi7.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                StringSelection stringSelection = new StringSelection(hic.getXPosition());
                mainWindow.setPositionChrTop(hic.getXPosition().concat(":").concat(String.valueOf(hic.getXContext().getZoom().getBinSize())));
                mainWindow.setPositionChrLeft(hic.getYPosition().concat(":").concat(String.valueOf(hic.getYContext().getZoom().getBinSize())));
                Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
                clpbrd.setContents(stringSelection, null);
            }
        });

        final JCheckBoxMenuItem mi8 = new JCheckBoxMenuItem("Copy left position to clipboard");
        mi8.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                StringSelection stringSelection = new StringSelection(hic.getYPosition());
                mainWindow.setPositionChrTop(hic.getXPosition().concat(":").concat(String.valueOf(hic.getXContext().getZoom().getBinSize())));
                mainWindow.setPositionChrLeft(hic.getYPosition().concat(":").concat(String.valueOf(hic.getYContext().getZoom().getBinSize())));
                Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
                clpbrd.setContents(stringSelection, null);
            }
        });

        final JCheckBoxMenuItem mi85 = new JCheckBoxMenuItem("Highlight feature");
        mi85.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                featureOptionMenuEnabled = false;
                showFeatureHighlight = true;
                highlightedFeature = mostRecentRectFeaturePair;
                repaint();
            }
        });

        final JCheckBoxMenuItem mi86 = new JCheckBoxMenuItem("Toggle feature Highlight");
        mi86.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                featureOptionMenuEnabled = false;
                showFeatureHighlight = !showFeatureHighlight;
                repaint();
            }
        });

        final JCheckBoxMenuItem mi87 = new JCheckBoxMenuItem("Remove Highlight");
        mi87.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                featureOptionMenuEnabled = false;
                highlightedFeature = null;
                repaint();
            }
        });

        final JCheckBoxMenuItem mi9 = new JCheckBoxMenuItem("Generate 1D Tracks");
        mi9.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

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
                new EditFeatureAttributesDialog(mostRecentRectFeaturePair.getSecond(),
                        MainWindow.customAnnotations);
            }
        });

        final JMenu configureFeatureMenu = new JMenu("Configure feature");
        configureFeatureMenu.add(mi10_1);
        configureFeatureMenu.add(mi10_2);

        if (hic != null) {
            //    menu.add(mi2);
            menu.add(mi3);
            mi4.setSelected(hic.isLinkedMode());
            menu.add(mi4);
            menu.add(mi5);
            menu.add(mi6);
            menu.add(mi7);
            menu.add(mi8);

            if (mostRecentRectFeaturePair != null) {//mouseIsOverFeature
                featureOptionMenuEnabled = true;
                menu.add(mi85);
                menu.add(configureFeatureMenu);
            }
            if (highlightedFeature != null) {
                menu.add(mi86);
                menu.add(mi87);
            }
            //menu.add(mi9);
        }


        return menu;

    }

    private String toolTipText(int x, int y) {
        // Update popup text
        final MatrixZoomData zd = hic.getZd();
        if (zd == null) return "";
        HiCGridAxis xGridAxis = zd.getXGridAxis();
        HiCGridAxis yGridAxis = zd.getYGridAxis();


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
                }
                if (yChrom == null && chromosomeBoundaries[i] > yGenomeStart) {
                    yChrom = hic.getChromosomes().get(i + 1);
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

            if (hic.getDisplayOption() == MatrixType.PEARSON) {
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
            mouseIsOverFeature = false;
            mostRecentRectFeaturePair = null;
            for (Pair<Rectangle, Feature2D> loop : drawnLoopFeatures) {
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
                    mouseIsOverFeature = true;

                }
            }

            txt.append("</html>");
            return txt.toString();
        }

        return null;
    }


    // TODO - not generating tracks at present, just some relevant code pieces - MSS
    private String trackGenerator(int x, int y) {
        // Update popup text
        final MatrixZoomData zd = hic.getZd();
        if (zd == null) return "";
        HiCGridAxis xGridAxis = zd.getXGridAxis();
        HiCGridAxis yGridAxis = zd.getYGridAxis();

        String topColor = "0000FF";
        String leftColor = "009900";

        int binX = (int) (hic.getXContext().getBinOrigin() + x / hic.getScaleFactor());
        int binY = (int) (hic.getYContext().getBinOrigin() + y / hic.getScaleFactor());

        int xGenomeStart = xGridAxis.getGenomicStart(binX) + 1; // Conversion from in internal "0" -> 1 base coordinates
        int yGenomeStart = yGridAxis.getGenomicStart(binY) + 1;
        int xGenomeEnd = xGridAxis.getGenomicEnd(binX);
        int yGenomeEnd = yGridAxis.getGenomicEnd(binY);

        if (!hic.isWholeGenome()) {

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

            txt.append("<html><span style='color:#" + topColor + "; font-family: arial; font-size: 12pt; '>");
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

            txt.append("</span><br><span style='color:#" + leftColor + "; font-family: arial; font-size: 12pt; '>");
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

            if (hic.getDisplayOption() == MatrixType.PEARSON) {
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

            for (Pair<Rectangle, Feature2D> loop : drawnLoopFeatures) {
                if (loop.getFirst().contains(x, y)) {
                    txt.append("<br><br><span style='font-family: arial; font-size: 12pt;'>");
                    txt.append(loop.getSecond().tooltipText());
                    txt.append("</span>");

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

    enum DragMode {NONE, PAN, ZOOM, SELECT, ANNOTATE}


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
                mainWindow.repaintTrackPanels();
            }
        }

        @Override
        public void mousePressed(final MouseEvent e) {
            featureOptionMenuEnabled = false;

            if (hic.isWholeGenome()) {
                if (e.isPopupTrigger()) {
                    getPopupMenu().show(HeatmapPanel.this, e.getX(), e.getY());
                }
                return;
            }
            // Priority is right click
            if (e.isPopupTrigger()) {
                getPopupMenu().show(HeatmapPanel.this, e.getX(), e.getY());
            // Alt down for zoom
            } else if (e.isAltDown()) {
                dragMode = DragMode.ZOOM;
            } else if (e.isShiftDown()){
                dragMode = DragMode.ANNOTATE;
                MainWindow.customAnnotationHandler.updateSelectionPoint(e.getX(), e.getY());
                MainWindow.customAnnotationHandler.doPeak();

                setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));

            } else {
                dragMode = DragMode.PAN;
                setCursor(MainWindow.fistCursor);
            }
            lastMousePoint = e.getPoint();

        }


        @Override
        public void mouseReleased(final MouseEvent e) {
            if (e.isPopupTrigger()) {
                getPopupMenu().show(HeatmapPanel.this, e.getX(), e.getY());
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
                MainWindow.customAnnotationHandler.addFeature(MainWindow.customAnnotations);
                dragMode = DragMode.NONE;
                annotateRectangle = null;
                lastMousePoint = null;
                zoomRectangle = null;


                hic.setCursorPoint(null);
                setCursor(Cursor.getDefaultCursor());
                repaint();
                mainWindow.repaintTrackPanels();




                //hic.setCursorPoint(null);
                //setCursor(straightEdgeEnabled ? Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR) : Cursor.getDefaultCursor());

                //
                //straightEdgeEnabled = false;

                //hic.setCursorPoint(null);
                //setCursor(Cursor.getDefaultCursor());
                //repaint();
                //mainWindow.repaintTrackPanels();
                //

            } else {
                    setCursor(straightEdgeEnabled ? Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR) : Cursor.getDefaultCursor());
            }
        }

        private void unsafeDragging() {
            final double scaleFactor1 = hic.getScaleFactor();
            double binX = hic.getXContext().getBinOrigin() + (zoomRectangle.x / scaleFactor1);
            double binY = hic.getYContext().getBinOrigin() + (zoomRectangle.y / scaleFactor1);
            double wBins = (int) (zoomRectangle.width / scaleFactor1);
            double hBins = (int) (zoomRectangle.height / scaleFactor1);

            final MatrixZoomData currentZD = hic.getZd();
            int xBP0 = currentZD.getXGridAxis().getGenomicStart(binX);

            int yBP0 = currentZD.getYGridAxis().getGenomicEnd(binY);

            double newXBinSize = wBins * currentZD.getBinSize() / getWidth();
            double newYBinSize = hBins * currentZD.getBinSize() / getHeight();
            double newBinSize = Math.max(newXBinSize, newYBinSize);

            hic.zoomTo(xBP0, yBP0, newBinSize);

            dragMode = DragMode.NONE;
            lastMousePoint = null;
            zoomRectangle = null;
            setCursor(straightEdgeEnabled ? Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR) : Cursor.getDefaultCursor());
        }


        @Override
        final public void mouseDragged(final MouseEvent e) {

            Rectangle lastRectangle, damageRect;
            int x, y;

            if (hic.getZd() == null || hic.isWholeGenome()) {
                return;
            }

            if (lastMousePoint == null) {
                lastMousePoint = e.getPoint();
                return;
            }

            int deltaX = e.getX() - lastMousePoint.x;
            int deltaY = e.getY() - lastMousePoint.y;
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

                    if (deltaX == 0 || deltaY == 0) {
                        return;
                    }

                    x = deltaX > 0 ? lastMousePoint.x : lastMousePoint.x + deltaX;
                    y = deltaY > 0 ? lastMousePoint.y : lastMousePoint.y + deltaY;
                    annotateRectangle = new Rectangle(x, y, Math.abs(deltaX), Math.abs(deltaY));

                    damageRect = lastRectangle == null ? annotateRectangle : annotateRectangle.union(lastRectangle);
                    damageRect.x--;
                    damageRect.y--;
                    damageRect.width += 2;
                    damageRect.height += 2;
                    paintImmediately(damageRect);
                    MainWindow.customAnnotationHandler.updateSelectionRegion(damageRect);
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

            int xGenome = hic.getZd().getXGridAxis().getGenomicMid(binX);
            int yGenome = hic.getZd().getYGridAxis().getGenomicMid(binY);

            Chromosome xChrom = null;
            Chromosome yChrom = null;
            for (int i = 0; i < chromosomeBoundaries.length; i++) {
                if (xChrom == null && chromosomeBoundaries[i] > xGenome) {
                    xChrom = hic.getChromosomes().get(i + 1);
                }
                if (yChrom == null && chromosomeBoundaries[i] > yGenome) {
                    yChrom = hic.getChromosomes().get(i + 1);
                }
            }
            if (xChrom != null && yChrom != null) {

                final Chromosome xC = xChrom;
                final Chromosome yC = yChrom;
                mainWindow.unsafeSetSelectedChromosomes(xC, yC);
            }

            //Only if zoom is changed All->Chr:
            mainWindow.updateThumbnail();
        }

        private void unsafeMouseClickSubActionB(double centerBinX, double centerBinY, HiCZoom newZoom) {
            final int xGenome = hic.getZd().getXGridAxis().getGenomicMid(centerBinX);
            final int yGenome = hic.getZd().getYGridAxis().getGenomicMid(centerBinY);

            hic.setZoom(newZoom, xGenome, yGenome);
            mainWindow.updateZoom(newZoom);
        }

        private void safeMouseClicked(final MouseEvent eF) {

            if (!eF.isPopupTrigger() && eF.getButton() == MouseEvent.BUTTON1 && !eF.isControlDown()) {

                if (hic.getZd() == null) return;
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
                    final HiCZoom currentZoom = hic.getZd().getZoom();
                    final HiCZoom newZoom = mainWindow.isResolutionLocked() ? currentZoom :
                            hic.getDataset().getNextZoom(currentZoom, !eF.isAltDown());

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
            if (hic.getXContext() != null && hic.getZd() != null) {
                if (!featureOptionMenuEnabled) {
                    mainWindow.updateToolTipText(toolTipText(e.getX(), e.getY()));
                }

                if (straightEdgeEnabled || e.isShiftDown()) {
                    synchronized (this) {
                        hic.setCursorPoint(e.getPoint());
                        repaint();
                        mainWindow.repaintTrackPanels();
                    }
                } else {
                    hic.setCursorPoint(null);
                    setCursor(Cursor.getDefaultCursor());
                    repaint();
                }
            }
        }
    }
}
