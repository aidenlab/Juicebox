/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2014 Broad Institute, Aiden Lab
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
import juicebox.MainWindow;
import juicebox.data.ExpectedValueFunction;
import juicebox.data.MatrixZoomData;
import juicebox.track.Feature2D;
import juicebox.track.HiCFragmentAxis;
import juicebox.track.HiCGridAxis;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.MatrixType;
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
    /**
     * Chromosome boundaries in kbases for whole genome view.
     */
    private int[] chromosomeBoundaries;
    private boolean straightEdgeEnabled = false;

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

                    g.drawImage(tile.image, xDest0, yDest0, xDest1, yDest1, xSrc0, ySrc0, xSrc1, ySrc1, null);

                    // Uncomment to draw tile grid (for debugging)
                    // g.drawRect((int) xDest0, (int) yDest0, (int) (xDest1 - xDest0), (int) (yDest1 - yDest0));

                }
            }

            //In case of change to map settings, get map color limits and update slider:
            //TBD: || might not catch all changed at once, if more then one parameter changed...
            if (hic.testZoomChanged() || hic.testDisplayOptionChanged() || hic.testNormalizationTypeChanged()) {
                //In case tender is called as a result of zoom change event, check if
                //We need to update slider with map range:
                renderer.updateColorSliderFromColorScale(zd, displayOption);
            }


            // Uncomment to draw bin grid (for debugging)
//            Graphics2D g2 = (Graphics2D) g.create();
//            g2.setColor(Color.green);
//            g2.setColor(new Color(0, 0, 1.0f, 0.3f));
//            for (int bin = (int) binOriginX; bin <= bRight; bin++) {
//                int pX = (int) ((bin - hic.getXContext().getBinOrigin()) * hic.getXContext().getScaleFactor());
//                g2.drawLine(pX, 0, pX, getHeight());
//            }
//            for (int bin = (int) binOriginY; bin <= bBottom; bin++) {
//                int pY = (int) ((bin - hic.getYContext().getBinOrigin()) * hic.getYContext().getScaleFactor());
//                g2.drawLine(0, pY, getWidth(), pY);
//            }
//            g2.dispose();

            boolean isWholeGenome = (hic.getXContext().getChromosome().getName().equals("All") &&
                    hic.getYContext().getChromosome().getName().equals("All"));


            // Draw grid
            if (isWholeGenome) {
                Color color = g.getColor();
                g.setColor(Color.lightGray);

                List<Chromosome> chromosomes = hic.getChromosomes();
                // Index 0 is whole genome
                int xGenomeCoord = 0;
                for (int i = 1; i < chromosomes.size(); i++) {
                    Chromosome c = chromosomes.get(i);
                    xGenomeCoord += (c.getLength() / 1000);
                    int xBin = zd.getXGridAxis().getBinNumberForGenomicPosition(xGenomeCoord);
                    int x = (int) (xBin * scaleFactor);
                    g.drawLine(x, 0, x, getHeight());
                }
                int yGenomeCoord = 0;
                for (int i = 1; i < chromosomes.size(); i++) {
                    Chromosome c = chromosomes.get(i);
                    yGenomeCoord += (c.getLength() / 1000);
                    int yBin = zd.getYGridAxis().getBinNumberForGenomicPosition(yGenomeCoord);
                    int y = (int) (yBin * hic.getScaleFactor());
                    g.drawLine(0, y, getWidth(), y);
                }

                g.setColor(color);
            }

            Point cursorPoint = hic.getCursorPoint();
            if (cursorPoint != null) {
                g.setColor(MainWindow.RULER_LINE_COLOR);
                g.drawLine(cursorPoint.x, 0, cursorPoint.x, getHeight());
                g.drawLine(0, cursorPoint.y, getWidth(), cursorPoint.y);
            }

        }
        if (allTilesNull) {
            g.setFont(FontManager.getFont(12));
            GraphicUtils.drawCenteredText("Normalization vectors not available at this resolution.  Try a different normalization.", clipBounds, g);

        } else {
            // Render loops
            drawnLoopFeatures.clear();

            List<Feature2D> loops = hic.getVisibleLoopList(zd.getChr1Idx(), zd.getChr2Idx());
            Graphics2D loopGraphics = (Graphics2D) g.create();
            if (loops != null && loops.size() > 0) {

                // Note: we're assuming feature.chr1 == zd.chr1, and that chr1 is on x-axis
                HiCGridAxis xAxis = zd.getXGridAxis();
                HiCGridAxis yAxis = zd.getYGridAxis();
                boolean sameChr = zd.getChr1Idx() == zd.getChr2Idx();

                for (Feature2D feature : loops) {

                    loopGraphics.setColor(feature.getColor());

                    int binStart1 = xAxis.getBinNumberForGenomicPosition(feature.getStart1());
                    int binEnd1 = xAxis.getBinNumberForGenomicPosition(feature.getEnd1());
                    int binStart2 = yAxis.getBinNumberForGenomicPosition(feature.getStart2());
                    int binEnd2 = yAxis.getBinNumberForGenomicPosition(feature.getEnd2());

                    int x = (int) ((binStart1 - binOriginX) * scaleFactor);
                    int y = (int) ((binStart2 - binOriginY) * scaleFactor);
                    int w = (int) Math.max(1, scaleFactor * (binEnd1 - binStart1));
                    int h = (int) Math.max(1, scaleFactor * (binEnd2 - binStart2));
                    loopGraphics.drawRect(x, y, w, h);
                    //loopGraphics.drawLine(x,y,x,y+w);
                    //loopGraphics.drawLine(x,y+w,x+h,y+w);
                    //System.out.println(binStart1 + "-" + binEnd1);
                    if (w > 5) {
                        // Thick line if there is room.
                        loopGraphics.drawRect(x + 1, y + 1, w - 2, h - 2);
                        //   loopGraphics.drawLine(x+1,y+1,x+1,y+w-1);
                        //   loopGraphics.drawLine(x+1,y+w-1,x+h-1,y+w-1);
                    }
                    drawnLoopFeatures.add(new Pair<Rectangle, Feature2D>(new Rectangle(x - 1, y - 1, w + 2, h + 2), feature));

                    if (sameChr && !(binStart1 == binStart2 && binEnd1 == binEnd2)) {
                        x = (int) ((binStart2 - binOriginX) * scaleFactor);
                        y = (int) ((binStart1 - binOriginY) * scaleFactor);
                        w = (int) Math.max(1, scaleFactor * (binEnd2 - binStart2));
                        h = (int) Math.max(1, scaleFactor * (binEnd1 - binStart1));
                        loopGraphics.drawRect(x, y, w, h);
                        if (w > 5) {
                            loopGraphics.drawRect(x, y, w, h);
                        }
                        drawnLoopFeatures.add(new Pair<Rectangle, Feature2D>(new Rectangle(x - 1, y - 1, w + 2, h + 2), feature));
                    }

                }

                loopGraphics.dispose();
            }

            if (zoomRectangle != null) {
                ((Graphics2D) g).draw(zoomRectangle);
            }

        }
        //UNCOMMENT TO OUTLINE "selected" BIN
//        if(hic.getSelectedBin() != null) {
//            int pX = (int) ((hic.getSelectedBin().x - hic.xContext.getBinOrigin()) * hic.xContext.getScaleFactor());
//            int pY = (int) ((hic.getSelectedBin().y - hic.yContext.getBinOrigin()) * hic.yContext.getScaleFactor());
//            int w = (int) hic.xContext.getScaleFactor() - 1;
//            int h = (int) hic.yContext.getScaleFactor() - 1;
//            g.setColor(Color.green);
//            g.drawRect(pX, pY, w, h);
//        }
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

    JidePopupMenu getPopupMenu() {

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

        final JMenuItem mi2 = new JMenuItem("Goto ...");
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
        });


        final JMenuItem mi3 = new JMenuItem("Sync");
        mi3.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                hic.broadcastState();
            }
        });


        final JCheckBoxMenuItem mi4 = new JCheckBoxMenuItem("Linked");
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

        final JCheckBoxMenuItem mi4_25 = new JCheckBoxMenuItem("Freeze hover text");
        mi4_25.setSelected(!mainWindow.isTooltipAllowedToUpdated());
        mi4_25.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                mainWindow.toggleToolTipUpdates(!mainWindow.isTooltipAllowedToUpdated());
            }
        });

        final JCheckBoxMenuItem mi4_5 = new JCheckBoxMenuItem("Copy hover text to clipboard");
        mi4_5.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                StringSelection stringSelection = new StringSelection(mainWindow.getToolTip());
                Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
                clpbrd.setContents(stringSelection, null);
            }
        });

        final JCheckBoxMenuItem mi5 = new JCheckBoxMenuItem("Copy top position to clipboard");
        mi5.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                StringSelection stringSelection = new StringSelection(hic.getXPosition());
                mainWindow.setPositionChrTop(hic.getXPosition().concat(":").concat(String.valueOf(hic.getXContext().getZoom().getBinSize())));
                mainWindow.setPositionChrLeft(hic.getYPosition().concat(":").concat(String.valueOf(hic.getYContext().getZoom().getBinSize())));
                Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
                clpbrd.setContents(stringSelection, null);
            }
        });

        final JCheckBoxMenuItem mi6 = new JCheckBoxMenuItem("Copy left position to clipboard");
        mi6.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                StringSelection stringSelection = new StringSelection(hic.getYPosition());
                mainWindow.setPositionChrTop(hic.getXPosition().concat(":").concat(String.valueOf(hic.getXContext().getZoom().getBinSize())));
                mainWindow.setPositionChrLeft(hic.getYPosition().concat(":").concat(String.valueOf(hic.getYContext().getZoom().getBinSize())));
                Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
                clpbrd.setContents(stringSelection, null);
            }
        });

        if (hic != null) {
            menu.add(mi2);
            menu.add(mi3);
            mi4.setSelected(hic.isLinkedMode());
            menu.add(mi4);
            menu.add(mi4_25);
            menu.add(mi4_5);
            menu.add(mi5);
            menu.add(mi6);
        }


        return menu;

    }

    private String toolTipText(int x, int y) {
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
                    txt += "<html><span style='color:#" + topColor + "; font-family: arial; font-size: 12pt;'>";
                    txt += xChrom.getName();
                    txt += ":";
                    txt += String.valueOf(xChromPos);
                    txt += "</span><br><span style='color:#" + leftColor + "; font-family: arial; font-size: 12pt;'>";
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

    enum DragMode {NONE, PAN, ZOOM, SELECT}


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

            if (hic.isWholeGenome()) {
                return;
            }

            if (e.isPopupTrigger()) {
                getPopupMenu().show(HeatmapPanel.this, e.getX(), e.getY());
            } else if (e.isAltDown()) {
                dragMode = DragMode.ZOOM;
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
                setCursor(straightEdgeEnabled ? Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR) : Cursor.getDefaultCursor());

            } else if ((dragMode == DragMode.ZOOM || dragMode == DragMode.SELECT) && zoomRectangle != null) {

                Runnable runnable = new Runnable() {
                    @Override
                    public void run() {
                        unsafeDragging();
                    }
                };
                mainWindow.executeLongRunningTask(runnable, "Mouse Drag");
            }
        }

        private void unsafeDragging(){
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
                    Rectangle lastRectangle = zoomRectangle;

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


                    int x = deltaX > 0 ? lastMousePoint.x : lastMousePoint.x + deltaX;
                    int y = deltaY > 0 ? lastMousePoint.y : lastMousePoint.y + deltaY;
                    zoomRectangle = new Rectangle(x, y, Math.abs(deltaX), Math.abs(deltaY));

                    Rectangle damageRect = lastRectangle == null ? zoomRectangle : zoomRectangle.union(lastRectangle);
                    damageRect.x--;
                    damageRect.y--;
                    damageRect.width += 2;
                    damageRect.height += 2;
                    paintImmediately(damageRect);

                    break;
                default:

                    // int dx = (int) (deltaX * hic.xContext.getScale());
                    // int dy = (int) (deltaY * hic.yContext.getScale());
                    lastMousePoint = e.getPoint();    // Always save the last Point

                    double deltaXBins = -deltaX / hic.getScaleFactor();
                    double deltaYBins = -deltaY / hic.getScaleFactor();
                    hic.moveBy(deltaXBins, deltaYBins);

            }

        }

        private void unsafeMouseClicked(final MouseEvent eF) {

            if (!eF.isPopupTrigger() && eF.getButton() == MouseEvent.BUTTON1 && !eF.isControlDown()) {

                if (hic.isWholeGenome()) {
                    //avoid double click...
                    if (eF.getClickCount() == 1) {


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

                    } else {
                        return;
                    }

                } else if (eF.getClickCount() == 1) {

                    // Double click,  zoom and center on click location
                    final HiCZoom currentZoom = hic.getZd().getZoom();
                    final HiCZoom newZoom = mainWindow.isResolutionLocked() ? currentZoom :
                            hic.getDataset().getNextZoom(currentZoom, !eF.isAltDown());

                    // If newZoom == currentZoom adjust scale factor (no change in resolution)
                    double centerBinX = hic.getXContext().getBinOrigin() + (eF.getX() / hic.getScaleFactor());
                    double centerBinY = hic.getYContext().getBinOrigin() + (eF.getY() / hic.getScaleFactor());

                    if (newZoom.equals(currentZoom)) {
                        double mult = eF.isAltDown() ? 0.5 : 2.0;
                        double newScaleFactor = Math.max(1.0, hic.getScaleFactor() * mult);
                        hic.setScaleFactor(newScaleFactor);
                        hic.getXContext().setBinOrigin(Math.max(0, (int) (centerBinX - (getWidth() / (2 * newScaleFactor)))));
                        hic.getYContext().setBinOrigin(Math.max(0, (int) (centerBinY - (getHeight() / (2 * newScaleFactor)))));
                        mainWindow.repaint();
                    } else {

                        final int xGenome = hic.getZd().getXGridAxis().getGenomicMid(centerBinX);
                        final int yGenome = hic.getZd().getYGridAxis().getGenomicMid(centerBinY);

                        hic.setZoom(newZoom, xGenome, yGenome);
                        mainWindow.updateZoom(newZoom);

                    }

                } else {

//                    if (hic.getXContext() == null) return;
//
//                    int binX = (int) (hic.getXContext().getBinOrigin() + eF.getX() / hic.getScaleFactor());
//                    int binY = (int) (hic.getYContext().getBinOrigin() + eF.getY() / hic.getScaleFactor());
//
//                    hic.setSelectedBin(new Point(binX, binY));
//                    repaint();
                    return;
                }
            }
        }

        @Override
        public void mouseClicked(MouseEvent e) {

            if (hic == null) return;

            final MouseEvent eF = e;
            Runnable runnable = new Runnable() {
                public void run() {
                    unsafeMouseClicked(eF);
                }
            };
            mainWindow.executeLongRunningTask(runnable, "Mouse Click");
        }

        @Override
        public void mouseMoved(MouseEvent e) {
            if (hic.getXContext() != null && hic.getZd() != null) {
                mainWindow.updateToolTipText(toolTipText(e.getX(), e.getY()));


                if (straightEdgeEnabled) {
                    synchronized (this) {
                        hic.setCursorPoint(e.getPoint());

//                        // Main panel
//                        Rectangle damageRectX = new Rectangle();
//                        damageRectX.x = (lastCursorPoint != null ? Math.min(lastCursorPoint.x, e.getX()) : e.getX()) - 1;
//                        damageRectX.y = 0;
//                        damageRectX.width = lastCursorPoint == null ? 2 : Math.abs(e.getX() - lastCursorPoint.x) + 2;
//                        damageRectX.height = getHeight();
//                        paintImmediately(damageRectX);
//
//                        Rectangle damageRectY = new Rectangle();
//                        damageRectY.x = 0;
//                        damageRectY.y = (lastCursorPoint != null ? Math.min(lastCursorPoint.y, e.getY()) : e.getY()) - 1;
//                        damageRectY.width = getWidth();
//                        damageRectY.height = lastCursorPoint == null ? 2 : Math.abs(e.getY() - lastCursorPoint.y) + 2;
//                        paintImmediately(damageRectY);
//
//                        // Track panels
//                        damageRectX.height = mainWindow.trackPanelX.getHeight();
//                        mainWindow.trackPanelX.paintImmediately(damageRectX);
//
//                        damageRectY.width = mainWindow.trackPanelY.getWidth();
//                        mainWindow.trackPanelY.paintImmediately(damageRectY);

                        repaint();
                        mainWindow.repaintTrackPanels();
                    }
                }
            }
        }
    }
}
