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

package juicebox.mapcolorui;

import com.jidesoft.swing.JidePopupMenu;
import juicebox.HiC;
import juicebox.HiCGlobals;
import juicebox.MainWindow;
import juicebox.assembly.AssemblyHeatmapHandler;
import juicebox.assembly.AssemblyOperationExecutor;
import juicebox.assembly.AssemblyScaffoldHandler;
import juicebox.assembly.Scaffold;
import juicebox.data.*;
import juicebox.gui.SuperAdapter;
import juicebox.track.HiCFragmentAxis;
import juicebox.track.HiCGridAxis;
import juicebox.track.feature.AnnotationLayer;
import juicebox.track.feature.AnnotationLayerHandler;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DGuiContainer;
import juicebox.windowui.EditFeatureAttributesDialog;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;
import oracle.net.jdbc.nl.UninitializedObjectException;
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
import java.util.Collections;
import java.util.List;

import static java.awt.Toolkit.getDefaultToolkit;


/**
 * @author jrobinso
 * @since Aug 2, 2010
 */
public class HeatmapPanel extends JComponent implements Serializable {

  //    public static final int clickDelay1 = (Integer) Toolkit.getDefaultToolkit().getDesktopProperty("awt.multiClickInterval");
  public static final int clickDelay = 500;
  private static final long serialVersionUID = -8017012290342597941L;
  // used for finding nearby features
  private static final int NUM_NEIGHBORS = 7;
  /**
   * Image tile width in pixels
   */
  private static final int imageTileWidth = 500;
  private final int RESIZE_SNAP = 5;
  private final NumberFormat formatter = NumberFormat.getInstance();
  private final MainWindow mainWindow;
  private final HiC hic;
  private final SuperAdapter superAdapter;
  private final ObjectCache<String, ImageTile> tileCache = new ObjectCache<>(26);
  private final HeatmapRenderer renderer;
  //private final transient List<Pair<Rectangle, Feature2D>> drawnLoopFeatures;
  private final transient List<Feature2DGuiContainer> allFeaturePairs = new ArrayList<>();
  private final transient List<Feature2DGuiContainer> allMainFeaturePairs = new ArrayList<>();
  private final transient List<Feature2DGuiContainer> allEditFeaturePairs = new ArrayList<>();
  private int debrisFeatureSize = RESIZE_SNAP;
  private Rectangle zoomRectangle;
  private Rectangle annotateRectangle;
  /**
   * Chromosome boundaries in kbases for whole genome view.
   */
  private int[] chromosomeBoundaries;
  private boolean straightEdgeEnabled = false, diagonalEdgeEnabled = false;
  private boolean featureOptionMenuEnabled = false;
  private boolean firstAnnotation;
  private AdjustAnnotation adjustAnnotation = AdjustAnnotation.NONE;
  /**
   * feature highlight related variables
   */
  private boolean showFeatureHighlight = true;
  private Feature2D highlightedFeature = null;
  private Feature2D debrisFeature = null;
  private Feature2D tempSelectedGroup = null;
  private List<Feature2D> selectedFeatures = null;
  private List<Feature2D> lastSelectedFeatures = null;
  private Feature2DGuiContainer currentFeature = null;
  private Pair<Pair<Integer, Integer>, Feature2D> preAdjustLoop = null;
  private boolean changedSize = false;
  private Feature2DGuiContainer currentUpstreamFeature = null;
  private Feature2DGuiContainer currentDownstreamFeature = null;
  /**
   * Heatmap grids variables
   */
  private boolean showGridLines = true;
  /**
   * Heatmap mouse variables
   */
  private Robot heatmapMouseBot;
  private boolean activelyEditingAssembly = false;
  private PromptedAssemblyAction currentPromptedAssemblyAction = PromptedAssemblyAction.NONE;
  private PromptedAssemblyAction promptedAssemblyActionOnClick = PromptedAssemblyAction.NONE;

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
    superAdapter.setPearsonColorScale(renderer.getPearsonColorScale());
    final HeatmapMouseHandler mouseHandler = new HeatmapMouseHandler();
    final HeatmapClickListener clickListener = new HeatmapClickListener(this);
    addMouseMotionListener(mouseHandler);
    addMouseListener(mouseHandler);
    addMouseListener(clickListener);
    addMouseWheelListener(mouseHandler);
    this.firstAnnotation = true;
    try {
      heatmapMouseBot = new Robot();
    } catch (AWTException ignored) {
    }
  }

  public int[] getChromosomeBoundaries() {
    return this.chromosomeBoundaries;
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
    } catch (Exception ignored) {
      ignored.printStackTrace();
    }

    if (hic.getXContext() == null) return;

    // todo pearsons
    if (hic.isInPearsonsMode()) {
      // Possibly force asynchronous computation of pearsons
      if (hic.isPearsonsNotAvailable(false)) {
        JOptionPane.showMessageDialog(this, "Pearson's matrix is not available at this " +
            "resolution, use 500KB or lower resolution.");
        return;
      }
      if (hic.isInControlPearsonsMode() && hic.isPearsonsNotAvailable(false)) {
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

    // tile numbers
    int tLeft = (int) (binOriginX / imageTileWidth);
    int tRight = (int) Math.ceil(bRight / imageTileWidth);
    int tTop = (int) (binOriginY / imageTileWidth);
    int tBottom = (int) Math.ceil(bBottom / imageTileWidth);

    //System.out.println("binX "+binOriginX+" "+bRight+" binY "+binOriginY+" "+bBottom);
    //System.out.println("tileX "+tLeft+" "+tRight+" tileY "+tTop+" "+tBottom);

    MatrixType displayOption = hic.getDisplayOption();
      NormalizationType observedNormalizationType = hic.getObsNormalizationType();
      NormalizationType controlNormalizationType = hic.getControlNormalizationType();

    boolean allTilesNull = true;
    for (int tileRow = tTop; tileRow <= tBottom; tileRow++) {
      for (int tileColumn = tLeft; tileColumn <= tRight; tileColumn++) {

        ImageTile tile;
        try {
            tile = getImageTile(zd, controlZd, tileRow, tileColumn, displayOption, observedNormalizationType, controlNormalizationType);
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
                      displayOption, observedNormalizationType, controlNormalizationType,
                  xDest0, yDest0, xDest1, yDest1, xSrc0, ySrc0, xSrc1, ySrc1);
              //processedExportRegions.add(newKey);
            } catch (Exception e2) {
              System.err.println("SVG export did not work");
            }
          }
          //}


          //TODO ******** UNCOMMENT *******
          //Uncomment to draw tile grid (for debugging)
          if (HiCGlobals.displayTiles) {
            g.drawRect(xDest0, yDest0, (xDest1 - xDest0), (yDest1 - yDest0));
          }

        }
      }

      //In case of change to map settings, get map color limits and update slider:
      //TODO: || might not catch all changed at once, if more then one parameter changed...
      if (hic.testZoomChanged() || hic.testDisplayOptionChanged() || hic.testNormalizationTypeChanged()) {
        //In case tender is called as a result of zoom change event, check if
        //We need to update slider with map range:
          String cacheKey = HeatmapRenderer.getColorScaleCacheKey(zd, displayOption, observedNormalizationType, controlNormalizationType);
        renderer.updateColorSliderFromColorScale(superAdapter, displayOption, cacheKey);
        //debrisFeatureSize = (int) (debrisFeatureSize * scaleFactor);
      }


      //Uncomment to draw bin grid (for debugging)
//            Graphics2D g2 = (Graphics2D) g.create();
//            g2.setAssociatedFeatureColor(Color.green);
//            g2.setAssociatedFeatureColor(new Color(0, 0, 1.0f, 0.3f));
//            for (int bin = (int) binOriginX; bin <= bRight; bin++) {
//                int pX = (int) ((bin - hic.getXContext().getBinOrigin()) * hic.getScaleFactor());
//                g2.drawLine(pX, 0, pX, getHeight());
//            }
//            for (int bin = (int) binOriginY; bin <= bBottom; bin++) {
//                int pY = (int) ((bin - hic.getYContext().getBinOrigin()) * hic.getScaleFactor());
//                g2.drawLine(0, pY, getWidth(), pY);
//            }
//            g2.dispose();

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

        int maxDimension = chromosomeBoundaries[chromosomeBoundaries.length - 1];

        // Draw grid lines only if option is selected
        if (showGridLines) {
          for (int bound : chromosomeBoundaries) {
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
              List<Integer> xBins = ((CustomMatrixZoomData) zd).getBoundariesOfCustomChromosomeX();
              //int maxSize = xBins.get(xBins.size() - 1);
              int maxSize = (int) ((zd.getYGridAxis().getBinCount() - binOriginY) * scaleFactor);
              for (int xBin : xBins) {
                int x = (int) ((xBin - binOriginX) * scaleFactor);
                g.drawLine(x, 0, x, maxSize);
              }
            }
          }
          if (hic.getChromosomeHandler().isCustomChromosome(zd.getChr2())) {
            if (zd instanceof CustomMatrixZoomData) {
              List<Integer> yBins = ((CustomMatrixZoomData) zd).getBoundariesOfCustomChromosomeY();
              //int maxSize = yBins.get(yBins.size() - 1);
              int maxSize = (int) ((zd.getXGridAxis().getBinCount() - binOriginX) * scaleFactor);
              for (int yBin : yBins) {
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
        float x1 = (float) binOriginX * zd.getBinSize();
        float y1 = (float) binOriginY * zd.getBinSize();
        float x2 = x1 + (float) (screenWidth / scaleFactor) * zd.getBinSize();
        float y2 = y1 + (float) (screenHeight / scaleFactor) * zd.getBinSize();
        net.sf.jsi.Rectangle currentWindow = new net.sf.jsi.Rectangle(x1, y1, x2, y2);

        Graphics2D g2 = (Graphics2D) g.create();
        allFeaturePairs.clear();
        if (activelyEditingAssembly) {
          allMainFeaturePairs.clear();
          allEditFeaturePairs.clear();
        }

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
          allFeaturePairs.addAll(handler.convertToFeaturePairs(handler, loops, zd, binOriginX, binOriginY, scaleFactor));

          if (activelyEditingAssembly) {
            if (handler == superAdapter.getMainLayer()) {
              allMainFeaturePairs.addAll(superAdapter.getMainLayer().convertToFeaturePairs(handler, loops, zd, binOriginX, binOriginY, scaleFactor));
            } else if (handler == superAdapter.getEditLayer() && selectedFeatures != null && !selectedFeatures.isEmpty()) {
              allEditFeaturePairs.addAll(superAdapter.getEditLayer().convertToFeaturePairs(handler, loops, zd, binOriginX, binOriginY, scaleFactor));
            }
          }

          FeatureRenderer.render(g2, handler, loops, zd, binOriginX, binOriginY, scaleFactor,
              highlightedFeature, showFeatureHighlight, this.getWidth(), this.getHeight());

        }
        g2.dispose();

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
                                                   MatrixType displayOption, NormalizationType observedNormalizationType,
                                                   NormalizationType controlNormalizationType,
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
            observedNormalizationType,
            controlNormalizationType,
        hic.getExpectedValues(),
        hic.getExpectedControlValues(),
        g, false);

    g.scale(1, 1);
    g.translate(0, 0);
  }

  private int getGridLineWidthLimit(MatrixZoomData zd, int maxPosition) {
    int w = getWidth();
    if (w < 50 || hic.getScaleFactor() == 0) {
      return 0;
    }
    int xBin = zd.getXGridAxis().getBinNumberForGenomicPosition(maxPosition);
    return (int) (xBin * hic.getScaleFactor());
  }

  private int getGridLineHeightLimit(MatrixZoomData zd, int maxPosition) {
    int h = getHeight();
    if (h < 50 || hic.getScaleFactor() == 0) {
      return 0;
    }
    int yBin = zd.getYGridAxis().getBinNumberForGenomicPosition(maxPosition);
    return (int) (yBin * hic.getScaleFactor());
  }

  public Image getThumbnailImage(MatrixZoomData zd0, MatrixZoomData ctrl0, int tw, int th, MatrixType displayOption,
                                 NormalizationType observedNormalizationType, NormalizationType controlNormalizationType) {
    if (MatrixType.isPearsonType(displayOption) && hic.isPearsonsNotAvailable(false)) {
      JOptionPane.showMessageDialog(this, "Pearson's matrix is not available at this resolution");
      return null;
    }

    int maxBinCountX = zd0.getXGridAxis().getBinCount();
    int maxBinCountY = zd0.getYGridAxis().getBinCount();

    int wh = Math.max(maxBinCountX, maxBinCountY);
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


  /**
   * Return the specified image tile, scaled by scaleFactor
   *
   * @param zd         Matrix of tile
   * @param tileRow    row index of tile
   * @param tileColumn column index of tile
   * @return image tile
   */
  private ImageTile getImageTile(MatrixZoomData zd, MatrixZoomData controlZd, int tileRow, int tileColumn, MatrixType displayOption,
                                 NormalizationType obsNormalizationType, NormalizationType ctrlNormalizationType) {

    String key = zd.getTileKey(tileRow, tileColumn, displayOption);
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
      if (HiCGlobals.isDarkulaModeEnabled) {
        g2D.setColor(Color.darkGray);
        g2D.fillRect(0, 0, imageWidth, imageHeight);
      }

      final int bx0 = tileColumn * imageTileWidth;
      final int by0 = tileRow * imageTileWidth;

      //System.out.println("tx "+tileColumn+" ty "+tileRow+" bx "+bx0+" by "+by0);

      if (!renderer.render(bx0,
          by0,
          imageWidth,
          imageHeight,
          zd,
          controlZd,
          displayOption,
              obsNormalizationType,
              ctrlNormalizationType,
          hic.getExpectedValues(),
          hic.getExpectedControlValues(),
          g2D, true)) {
        return null;
      }

      //           if (scaleFactor > 0.999 && scaleFactor < 1.001) {
      tile = new ImageTile(image, bx0, by0);
      tileCache.put(key, tile);
    }
    return tile;
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

    if (SuperAdapter.assemblyModeCurrentlyActive) {
      getAssemblyPopupMenu(xMousePos, yMousePos, menu);
      menu.addSeparator();
    }


    final JMenuItem miUndoZoom = new JMenuItem("Undo Zoom");
    miUndoZoom.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        hic.setCursorPoint(new Point(xMousePos, yMousePos));
        hic.undoZoomAction();
      }
    });
    miUndoZoom.setEnabled(hic.getZoomActionTracker().validateUndoZoom());
    menu.add(miUndoZoom);

    final JMenuItem miRedoZoom = new JMenuItem("Redo Zoom");
    miRedoZoom.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        hic.setCursorPoint(new Point(xMousePos, yMousePos));
        hic.redoZoomAction();
      }
    });
    miRedoZoom.setEnabled(hic.getZoomActionTracker().validateRedoZoom());
    menu.add(miRedoZoom);

    // add Jump to Diagonal menu items
    addJumpToDiagonalMenuItems(menu, xMousePos, yMousePos);

    final JCheckBoxMenuItem mi = new JCheckBoxMenuItem("Enable straight edge");
    mi.setSelected(straightEdgeEnabled);
    mi.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (mi.isSelected()) {
          straightEdgeEnabled = true;
          diagonalEdgeEnabled = false;
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

    final JCheckBoxMenuItem miv2 = new JCheckBoxMenuItem("Enable diagonal edge");
    miv2.setSelected(diagonalEdgeEnabled);
    miv2.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (miv2.isSelected()) {
          straightEdgeEnabled = false;
          diagonalEdgeEnabled = true;
          setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
        } else {
          diagonalEdgeEnabled = false;
          hic.setDiagonalCursorPoint(null);
          setCursor(Cursor.getDefaultCursor());
          repaint();
          superAdapter.repaintTrackPanels();
        }

      }
    });
    menu.add(miv2);

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
          HiCGlobals.wasLinkedBeforeMousePress = false;
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

    final JMenuItem mi6 = new JMenuItem("Copy hover text to clipboard");
    mi6.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        StringSelection stringSelection = new StringSelection(superAdapter.getToolTip());
        Clipboard clpbrd = getDefaultToolkit().getSystemClipboard();
        clpbrd.setContents(stringSelection, null);
      }
    });

    final JMenuItem mi7 = new JMenuItem("Copy top position to clipboard");
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
    final JMenuItem mi8 = new JMenuItem("Copy left position to clipboard");
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
        highlightedFeature = currentFeature.getFeature2D();
        addHighlightedFeature(highlightedFeature);

      }
    });

    final JCheckBoxMenuItem mi86Toggle = new JCheckBoxMenuItem("Toggle Highlight Visibility");
    mi86Toggle.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        featureOptionMenuEnabled = false;
        showFeatureHighlight = !showFeatureHighlight;
        hic.setShowFeatureHighlight(showFeatureHighlight);
        repaint();
      }
    });

    final JMenuItem mi87Remove = new JMenuItem("Remove Highlight");
    mi87Remove.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        removeHighlightedFeature();
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
            new Pair<>(currentFeature.getRectangle(), currentFeature.getFeature2D());
        launchColorSelectionMenu(featureCopy);
      }
    });

    final JMenuItem mi10_2 = new JMenuItem("Change Attributes");
    mi10_2.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        featureOptionMenuEnabled = false;
        new EditFeatureAttributesDialog(mainWindow, currentFeature.getFeature2D(),
            superAdapter.getActiveLayerHandler().getAnnotationLayer());
      }
    });

    final JMenuItem mi10_3 = new JMenuItem("Delete");
    mi10_3.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        featureOptionMenuEnabled = false;
        Feature2D feature = currentFeature.getFeature2D();
        int chr1Idx = hic.getXContext().getChromosome().getIndex();
        int chr2Idx = hic.getYContext().getChromosome().getIndex();
        try {
          superAdapter.getActiveLayerHandler().removeFromList(hic.getZd(), chr1Idx, chr2Idx, 0, 0,
              Feature2DHandler.numberOfLoopsToFind, hic.getXContext().getBinOrigin(),
              hic.getYContext().getBinOrigin(), hic.getScaleFactor(), feature);
        } catch (Exception ee) {
          System.err.println("Could not remove custom annotation");
        }
        superAdapter.refresh();
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
      if (!ChromosomeHandler.isAllByAll(hic.getXContext().getChromosome())
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

      if (currentFeature != null) {//mouseIsOverFeature
        featureOptionMenuEnabled = true;
        if (menuSeparatorNotAdded) {
          menu.addSeparator();
        }

        if (highlightedFeature != null) {
          if (currentFeature.getFeature2D() != highlightedFeature) {
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

  public void enableAssemblyEditing() {
    SuperAdapter.assemblyModeCurrentlyActive = true;
    activelyEditingAssembly = true;
    AssemblyHeatmapHandler.setSuperAdapter(superAdapter);
  }

  public void disableAssemblyEditing() {
    updateSelectedFeatures(false);
    if (selectedFeatures != null) {
      selectedFeatures.clear();
    }
    superAdapter.getMainViewPanel().toggleToolTipUpdates(Boolean.TRUE);
    activelyEditingAssembly = false;
    HiCGlobals.splitModeEnabled = false;
    SuperAdapter.assemblyModeCurrentlyActive = false;
  }

  private void addHighlightedFeature(Feature2D feature2D) {
    highlightedFeature = feature2D;
    featureOptionMenuEnabled = false;
    showFeatureHighlight = true;
    hic.setShowFeatureHighlight(showFeatureHighlight);
    hic.setHighlightedFeature(highlightedFeature);
    superAdapter.repaintTrackPanels();
    repaint();
  }

  private void removeHighlightedFeature() {
    featureOptionMenuEnabled = false;
    highlightedFeature = null;
    hic.setHighlightedFeature(highlightedFeature);
    superAdapter.repaintTrackPanels();
    repaint();
  }

  private JidePopupMenu getAssemblyPopupMenu(final int xMousePos, final int yMousePos, JidePopupMenu menu) {


//        final JMenuItem miRepeatSelection = new JCheckBoxMenuItem("Repeat selection");
//        miRepeatSelection.addActionListener(new ActionListener() {
//            @Override
//            public void actionPerformed(ActionEvent e) {
//                selectedFeatures=lastSelectedFeatures;
//                updateSelectedFeatures(true);
//
//                Chromosome chrX = superAdapter.getHiC().getXContext().getChromosome();
//                Chromosome chrY = superAdapter.getHiC().getYContext().getChromosome();
//                superAdapter.getEditLayer().filterTempSelectedGroup(chrX.getIndex(), chrY.getIndex());
//                repaint();
//
//                if (superAdapter.getMainLayer().getLayerVisibility()) {
//                    tempSelectedGroup = superAdapter.getEditLayer().addTempSelectedGroup(selectedFeatures, hic);
//                    addHighlightedFeature(tempSelectedGroup);
//                }
//
////                superAdapter.getMainViewPanel().toggleToolTipUpdates(Boolean.TRUE);
////                superAdapter.updateMainViewPanelToolTipText(toolTipText(e.getX(), e.getY()));
////                superAdapter.getMainViewPanel().toggleToolTipUpdates(selectedFeatures.isEmpty());
//
////                currentPromptedAssemblyAction = PromptedAssemblyAction.NONE;
//
////                restoreDefaultVariables();
//
//            }
//        });
//        //System.out.println("I am in repeatselection and "+lastSelectedFeatures.size());
//        miRepeatSelection.setEnabled(lastSelectedFeatures!=null && !lastSelectedFeatures.isEmpty());
//        menu.add(miRepeatSelection);

    final JMenuItem miMoveToDebris = new JMenuItem("Move to debris");
    miMoveToDebris.setEnabled(selectedFeatures != null && !selectedFeatures.isEmpty());
    miMoveToDebris.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        moveSelectionToEnd();
      }
    });
    menu.add(miMoveToDebris);

    final JMenuItem groupItems = new JMenuItem("Remove chr boundaries");
    groupItems.setEnabled(selectedFeatures != null && selectedFeatures.size() > 1);
    groupItems.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        AssemblyOperationExecutor.multiMerge(superAdapter, selectedFeatures);

        // Cleanup
        removeSelection();
      }
    });
    menu.add(groupItems);

    final JMenuItem splitItems = new JMenuItem("Add chr boundaries");
    splitItems.setEnabled(selectedFeatures != null && !selectedFeatures.isEmpty());
    splitItems.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        AssemblyOperationExecutor.multiSplit(superAdapter, selectedFeatures);

        // Cleanup
        removeSelection();
      }
    });
    menu.add(splitItems);

    final JMenuItem miUndo = new JMenuItem("Undo");
    miUndo.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        superAdapter.getAssemblyStateTracker().undo();
        removeSelection();
        superAdapter.refresh();
      }
    });
    miUndo.setEnabled(superAdapter.getAssemblyStateTracker().checkUndo());
    menu.add(miUndo);


    final JMenuItem miRedo = new JMenuItem("Redo");
    miRedo.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        superAdapter.getAssemblyStateTracker().redo();
        removeSelection();
        superAdapter.refresh();
      }
    });
    miRedo.setEnabled(superAdapter.getAssemblyStateTracker().checkRedo());
    menu.add(miRedo);

    return menu;
  }

  private void addJumpToDiagonalMenuItems(JidePopupMenu menu, int xMousePos, int yMousePos) {

    final double preJumpBinOriginX = hic.getXContext().getBinOrigin();
    final double preJumpBinOriginY = hic.getYContext().getBinOrigin();

    // xMousePos and yMousePos coordinates are relative to the heatmap panel and not the screen
    final int clickedBinX = (int) (preJumpBinOriginX + xMousePos / hic.getScaleFactor());
    final int clickedBinY = (int) (preJumpBinOriginY + yMousePos / hic.getScaleFactor());

    // these coordinates are relative to the screen and not the heatmap panel
    final int defaultPointerDestinationX = (int) (getLocationOnScreen().getX() + xMousePos);
    final int defaultPointerDestinationY = (int) (getLocationOnScreen().getY() + yMousePos);

    // get maximum number of bins on the X and Y axes
    Matrix matrix = hic.getMatrix();
    MatrixZoomData matrixZoomData = matrix.getZoomData(hic.getZoom());
    final int binCountX = matrixZoomData.getXGridAxis().getBinCount();
    final int binCountY = matrixZoomData.getYGridAxis().getBinCount();

    if (clickedBinX > clickedBinY) {

      final JMenuItem jumpToDiagonalLeft = new JMenuItem('\u25C0' + "  Jump To Diagonal");
      jumpToDiagonalLeft.setSelected(straightEdgeEnabled);
      jumpToDiagonalLeft.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
          double postJumpBinOriginX = preJumpBinOriginX - (clickedBinX - clickedBinY);
          hic.moveBy(clickedBinY - clickedBinX, 0);
          if (postJumpBinOriginX < 0) {
            heatmapMouseBot.mouseMove((int) (defaultPointerDestinationX + postJumpBinOriginX * hic.getScaleFactor()), defaultPointerDestinationY);
            return;
          }
          heatmapMouseBot.mouseMove(defaultPointerDestinationX, defaultPointerDestinationY);
        }
      });
      menu.add(jumpToDiagonalLeft);

      final JMenuItem jumpToDiagonalDown = new JMenuItem('\u25BC' + "  Jump To Diagonal");
      jumpToDiagonalDown.setSelected(straightEdgeEnabled);
      jumpToDiagonalDown.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
          double postJumpBinOriginY = preJumpBinOriginY + (clickedBinX - clickedBinY);
          hic.moveBy(0, clickedBinX - clickedBinY);
          if (postJumpBinOriginY + getHeight() / hic.getScaleFactor() > binCountY) {
            heatmapMouseBot.mouseMove(defaultPointerDestinationX, (int) (defaultPointerDestinationY + (postJumpBinOriginY + getHeight() / hic.getScaleFactor() - binCountY)));
            return;
          }
          heatmapMouseBot.mouseMove(defaultPointerDestinationX, defaultPointerDestinationY);
        }
      });
      menu.add(jumpToDiagonalDown);

    } else if (clickedBinX < clickedBinY) {

      final JMenuItem jumpToDiagonalUp = new JMenuItem('\u25B2' + "  Jump To Diagonal");
      jumpToDiagonalUp.setSelected(straightEdgeEnabled);
      jumpToDiagonalUp.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
          double postJumpBinOriginY = preJumpBinOriginY - (clickedBinY - clickedBinX);
          hic.moveBy(0, clickedBinX - clickedBinY);
          if (postJumpBinOriginY < 0) {
            heatmapMouseBot.mouseMove(defaultPointerDestinationX, (int) (defaultPointerDestinationY + postJumpBinOriginY * hic.getScaleFactor()));
            return;
          }
          heatmapMouseBot.mouseMove(defaultPointerDestinationX, defaultPointerDestinationY);
        }
      });
      menu.add(jumpToDiagonalUp);

      final JMenuItem jumpToDiagonalRight = new JMenuItem('\u25B6' + "  Jump To Diagonal");
      jumpToDiagonalRight.setSelected(straightEdgeEnabled);
      jumpToDiagonalRight.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
          double postJumpBinOriginX = preJumpBinOriginX + (clickedBinY - clickedBinX);
          hic.moveBy(clickedBinY - clickedBinX, 0);
          if (postJumpBinOriginX + getWidth() / hic.getScaleFactor() > binCountX) {
            heatmapMouseBot.mouseMove((int) (defaultPointerDestinationX + (postJumpBinOriginX + getWidth() / hic.getScaleFactor() - binCountX)), defaultPointerDestinationY);
            return;
          }
          heatmapMouseBot.mouseMove(defaultPointerDestinationX, defaultPointerDestinationY);
        }
      });
      menu.add(jumpToDiagonalRight);
    }
  }

  public void removeSelection() {
    updateSelectedFeatures(false);
    if (selectedFeatures != null) {
      selectedFeatures.clear();
    }
    superAdapter.updatePreviousTempSelectedGroups(tempSelectedGroup);
    tempSelectedGroup = null;
    superAdapter.getEditLayer().clearAnnotations();
    if (superAdapter.getActiveLayerHandler() != superAdapter.getMainLayer()) {
      superAdapter.setActiveLayerHandler(superAdapter.getMainLayer());
      superAdapter.getLayersPanel().updateBothLayersPanels(superAdapter);
    }
    HiCGlobals.splitModeEnabled = false;
    superAdapter.getMainViewPanel().toggleToolTipUpdates(Boolean.TRUE);
    removeHighlightedFeature();

    Chromosome chrX = superAdapter.getHiC().getXContext().getChromosome();
    Chromosome chrY = superAdapter.getHiC().getYContext().getChromosome();
    superAdapter.getEditLayer().filterTempSelectedGroup(chrX.getIndex(), chrY.getIndex());
    if (debrisFeature != null) {
      superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(chrX.getIndex(), chrY.getIndex(), debrisFeature);
    }
    currentPromptedAssemblyAction = PromptedAssemblyAction.NONE;
    debrisFeatureSize = RESIZE_SNAP;
    repaint();
  }


  private void executeSplitMenuAction() {

    AssemblyOperationExecutor.splitContig(selectedFeatures.get(0), debrisFeature, superAdapter, hic, true);

    HiCGlobals.splitModeEnabled = false;
    Chromosome chrX = superAdapter.getHiC().getXContext().getChromosome();
    Chromosome chrY = superAdapter.getHiC().getYContext().getChromosome();
    superAdapter.getEditLayer().filterTempSelectedGroup(chrX.getIndex(), chrY.getIndex());
    superAdapter.getEditLayer().clearAnnotations();
    superAdapter.setActiveLayerHandler(superAdapter.getMainLayer());
    debrisFeature = null;
    //moveDebrisToEnd();
    removeSelection();
    debrisFeatureSize = RESIZE_SNAP;
  }

  void moveSelectionToEnd() {
    AssemblyScaffoldHandler assemblyHandler = superAdapter.getAssemblyStateTracker().getAssemblyHandler();
    final List<Integer> lastLine = assemblyHandler.getListOfSuperscaffolds().get(assemblyHandler.getListOfSuperscaffolds().size() - 1);
    int lastId = Math.abs(lastLine.get(lastLine.size() - 1)) - 1;
    AssemblyOperationExecutor.moveSelection(superAdapter, selectedFeatures, assemblyHandler.getListOfScaffolds().get(lastId).getCurrentFeature2D());
    removeSelection();
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
          xChrom = hic.getChromosomeHandler().getChromosomeFromIndex(i + 1);
          break;
        }
      }
      for (int i = 0; i < chromosomeBoundaries.length; i++) {
        if (yChrom == null && chromosomeBoundaries[i] > yGenomeStart) {
          yChrom = hic.getChromosomeHandler().getChromosomeFromIndex(i + 1);
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
          hic.setXPosition(xChrom.getName() + ":" + xChromPos);
        } else {
          hic.setXPosition("chr" + xChrom.getName() + ":" + xChromPos);
        }
        if (yChrom.getName().toLowerCase().contains("chr")) {
          hic.setYPosition(yChrom.getName() + ":" + yChromPos);
        } else {
          hic.setYPosition("chr" + yChrom.getName() + ":" + yChromPos);
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
      txt.append(formatter.format(Math.round((xGenomeStart - 1) * HiCGlobals.hicMapScale + 1)));
      txt.append("-");
      txt.append(formatter.format(Math.round(xGenomeEnd) * HiCGlobals.hicMapScale));

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
      txt.append(formatter.format(Math.round((yGenomeStart - 1) * HiCGlobals.hicMapScale + 1)));
      txt.append("-");
      txt.append(formatter.format(Math.round(yGenomeEnd * HiCGlobals.hicMapScale)));

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
          float value = zd.getPearsonValue(binX, binY, hic.getObsNormalizationType());
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
          txt.append("observed value (O) = ");
          txt.append(getFloatString(value));
          txt.append("</span>");
        }

        int c1 = hic.getXContext().getChromosome().getIndex();
        int c2 = hic.getYContext().getChromosome().getIndex();

        double ev = getExpectedValue(c1, c2, binX, binY, zd, hic.getExpectedValues());
        String evString = ev < 0.001 || Double.isNaN(ev) ? String.valueOf(ev) : formatter.format(ev);
        txt.append("<br><span style='font-family: arial; font-size: 12pt;'>expected value (E) = ").append(evString).append("</span>");
        if (ev > 0 && !Float.isNaN(value)) {
          txt.append("<br><span style='font-family: arial; font-size: 12pt;'>O/E            = ");
          txt.append(formatter.format(value / ev)).append("</span>");
        } else {
          txt.append("<br><span style='font-family: arial; font-size: 12pt;'>O/E            = NaN</span>");
        }

        MatrixZoomData controlZD = hic.getControlZd();
        if (controlZD != null) {
            float controlValue = hic.getNormalizedControlValue(binX, binY);
          txt.append("<br><br><span style='font-family: arial; font-size: 12pt;'>");
          txt.append("control value (C) = ");
          txt.append(getFloatString(controlValue));
          txt.append("</span>");

          double evCtrl = getExpectedValue(c1, c2, binX, binY, controlZD, hic.getExpectedControlValues());
          String evStringCtrl = evCtrl < 0.001 || Double.isNaN(evCtrl) ? String.valueOf(evCtrl) : formatter.format(evCtrl);
          txt.append("<br><span style='font-family: arial; font-size: 12pt;'>expected control value (EC) = ").append(evStringCtrl).append("</span>");
          if (evCtrl > 0 && !Float.isNaN(controlValue)) {
            txt.append("<br><span style='font-family: arial; font-size: 12pt;'>C/EC            = ");
            txt.append(formatter.format(controlValue / evCtrl)).append("</span>");
          } else {
            txt.append("<br><span style='font-family: arial; font-size: 12pt;'>C/EC            = NaN</span>");
          }

          double obsAvg = zd.getAverageCount();
          double obsValue = (value / obsAvg);
          txt.append("<br><br><span style='font-family: arial; font-size: 12pt;'>");
          txt.append("average observed value (AVG) = ").append(getFloatString((float) obsAvg));
          txt.append("<br>O' = O/AVG = ").append(getFloatString((float) obsValue));
          txt.append("</span>");

          double ctrlAvg = controlZD.getAverageCount();
          double ctlValue = (float) (controlValue / ctrlAvg);
          txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
          txt.append("average control value (AVGC) = ").append(getFloatString((float) ctrlAvg));
          txt.append("<br>C' = C/AVGC = ").append(getFloatString((float) ctlValue));
          txt.append("</span>");

          if (value > 0 && controlValue > 0) {
            double ratio = obsValue / ctlValue;
            txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
            txt.append("O'/C' = ").append(getFloatString((float) ratio));
            txt.append("</span>");

            double diff = (obsValue - ctlValue) * (obsAvg / 2. + ctrlAvg / 2.);
            txt.append("<br><span style='font-family: arial; font-size: 12pt;'>");
            txt.append("(O'-C')*(AVG/2 + AVGC/2) = ");
            txt.append(getFloatString((float) diff));
            txt.append("</span>");
          }
        }

        txt.append(superAdapter.getTrackPanelPrintouts(x, y));
      }

      Point currMouse = new Point(x, y);
      double minDistance = Double.POSITIVE_INFINITY;
      //mouseIsOverFeature = false;
      currentFeature = null;
      if (activelyEditingAssembly) {
        // current feature is populated only from all main feature pairs, contains does not work
        for (Feature2DGuiContainer loop : allMainFeaturePairs) {
          if (loop.getRectangle().contains(x, y)) {
            currentFeature = loop;
          }
        }

        if (selectedFeatures != null && !selectedFeatures.isEmpty()) {
          Collections.sort(selectedFeatures);
          for (Feature2D feature2D : selectedFeatures) {
            txt.append("<br><br><span style='font-family: arial; font-size: 12pt;'>");
            txt.append(feature2D.tooltipText());
            txt.append("</span>");
          }
        } else {
          for (Feature2DGuiContainer loop : allFeaturePairs) {
            if (loop.getRectangle().contains(x, y)) {
              // TODO - why is this code duplicated in this file?
              txt.append("<br><br><span style='font-family: arial; font-size: 12pt;'>");
              txt.append(loop.getFeature2D().tooltipText());
              txt.append("</span>");
            }
          }
        }
      } else {
        int numLayers = superAdapter.getAllLayers().size();
        int globalPriority = numLayers;
        for (Feature2DGuiContainer loop : allFeaturePairs) {
          if (loop.getRectangle().contains(x, y)) {
            // TODO - why is this code duplicated in this file?
            txt.append("<br><br><span style='font-family: arial; font-size: 12pt;'>");
            txt.append(loop.getFeature2D().tooltipText());
            txt.append("</span>");
            int layerNum = superAdapter.getAllLayers().indexOf(loop.getAnnotationLayerHandler());
            int loopPriority = numLayers - layerNum;
            double distance = currMouse.distance(loop.getRectangle().getX(), loop.getRectangle().getY());
            if (distance < minDistance && loopPriority <= globalPriority) {
              minDistance = distance;
              currentFeature = loop;
              globalPriority = loopPriority;
            }
            //mouseIsOverFeature = true;
          }
        }

      }

      txt.append("<br>");
      txt.append("</html>");
      return txt.toString();
    }

    return null;
  }

  private double getExpectedValue(int c1, int c2, int binX, int binY, MatrixZoomData zd,
                                  ExpectedValueFunction df) {
    double ev = 0;
    if (c1 == c2) {
      if (df != null) {
        int distance = Math.abs(binX - binY);
        ev = df.getExpectedValue(c1, distance);
      }
    } else {
      ev = zd.getAverageCount();
    }
    return ev;
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

  private void setProperCursor() {
    if (straightEdgeEnabled || diagonalEdgeEnabled) {
      setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
    } else {
      setCursor(Cursor.getDefaultCursor());
    }
  }

  private void updateSelectedFeatures(boolean status) {
    if (selectedFeatures != null) {
      for (Feature2D feature2D : selectedFeatures) {
        feature2D.setSetIsSelectedColorUpdate(status);
      }
    }
  }

  //  public Feature2D generateDebrisFeature(int xMousePos, int yMousePos) {
//        final double scaleFactor = hic.getScaleFactor();
//        double binOriginX = hic.getXContext().getBinOrigin();
//        double binOriginY = hic.getYContext().getBinOrigin();
//        Rectangle annotateRectangle = new Rectangle(xMousePos, (int) (yMousePos + (binOriginX - binOriginY) * scaleFactor), RESIZE_SNAP, RESIZE_SNAP);
//        superAdapter.getEditLayer().updateSelectionRegion(annotateRectangle);
//        return superAdapter.getEditLayer().generateFeature(hic);
//    }

  public void toggleActivelyEditingAssembly() {
    this.activelyEditingAssembly = !this.activelyEditingAssembly;
  }

  //private enum AdjustAnnotation {LEFT, RIGHT, NONE}

  public PromptedAssemblyAction getCurrentPromptedAssemblyAction() {
    return this.currentPromptedAssemblyAction;
  }

  public PromptedAssemblyAction getPromptedAssemblyActionOnClick() {
    return this.promptedAssemblyActionOnClick;
  }

  public void setPromptedAssemblyActionOnClick(PromptedAssemblyAction promptedAssemblyAction) {
    this.promptedAssemblyActionOnClick = promptedAssemblyAction;
  }

  private void setDebrisFeauture(Feature2D debrisFeature) {
    this.debrisFeature = debrisFeature;
  }

  public List<Feature2D> getSelectedFeatures() {
    return this.selectedFeatures;
  }

  public Feature2DGuiContainer getCurrentUpstreamFeature() {
    return this.currentUpstreamFeature;
  }

  public Feature2DGuiContainer getCurrentDownstreamFeature() {
    return this.currentDownstreamFeature;
  }

  private enum AdjustAnnotation {LEFT, RIGHT, NONE}

//    @Override
//    public String getToolTipText(MouseEvent e) {
//        return toolTipText(e.getX(), e.getY());
//
//    }

  private enum DragMode {ZOOM, ANNOTATE, RESIZE, PAN, SELECT, NONE}

  public enum PromptedAssemblyAction {REGROUP, PASTE, INVERT, CUT, ADJUST, NONE, PASTETOP, PASTEBOTTOM}

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

  private Feature2DGuiContainer getMouseHoverSuperscaffold(int x, int y) {
    final Point mousePoint = calculateSelectionPoint(x, y);

    if (activelyEditingAssembly) {
      for (Feature2DGuiContainer loop : allFeaturePairs) {
        if (loop.getFeature2D().getFeatureType() == Feature2D.FeatureType.SUPERSCAFFOLD) {
          if (new Rectangle(
                  loop.getFeature2D().getStart1(), loop.getFeature2D().getStart2(),
                  loop.getFeature2D().getWidth1(), loop.getFeature2D().getWidth2()).contains(mousePoint)) {
            return loop;
          }
        }
      }
    }

    return null;
  }

  private Point calculateSelectionPoint(int unscaledX, int unscaledY) {
    final MatrixZoomData zd;
    try {
      zd = hic.getZd();
    } catch (Exception err) {
      return null;
    }

    final HiCGridAxis xAxis = zd.getXGridAxis();
    final HiCGridAxis yAxis = zd.getYGridAxis();
    final double binOriginX = hic.getXContext().getBinOrigin();
    final double binOriginY = hic.getYContext().getBinOrigin();
    final double scale = hic.getScaleFactor();

    float x = (float) (((unscaledX / scale) + binOriginX) * xAxis.getBinSize());
    float y = (float) (((unscaledY / scale) + binOriginY) * yAxis.getBinSize());
    return new Point((int) x, (int) y);
  }

  class HeatmapMouseHandler extends MouseAdapter {

    private static final int clickLong = 400;
    DragMode dragMode = DragMode.NONE;
    double startTime, endTime, holdTime;
    private Point lastMousePoint;
    private Point lastPressedMousePoint;

    @Override
    public void mouseEntered(MouseEvent e) {
      setProperCursor();
    }

    @Override
    public void mouseExited(MouseEvent e) {
      hic.setCursorPoint(null);
      if (straightEdgeEnabled || diagonalEdgeEnabled) {
        superAdapter.repaintTrackPanels();
      }
    }

    @Override
    public void mousePressed(final MouseEvent e) {
      startTime = System.nanoTime();
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
      } else {

        // turn off continuous sync for dragging
        if (hic.isLinkedMode()) {
          HiCGlobals.wasLinkedBeforeMousePress = true;
          hic.setLinkedMode(false);
        }

        // Alt down for zoom
        if (e.isAltDown()) {
          dragMode = DragMode.ZOOM;
          // Shift down for custom annotations
        } else if (e.isShiftDown() && (activelyEditingAssembly || superAdapter.getActiveLayerHandler().getAnnotationLayerType() != AnnotationLayer.LayerType.SCAFFOLD)) {

          if (!activelyEditingAssembly) {
            boolean showWarning = false;

            if (superAdapter.unsavedEditsExist() && firstAnnotation && showWarning) {
              firstAnnotation = false;
              String text = "There are unsaved hand annotations from your previous session! \n" +
                  "Go to 'Annotations > Hand Annotations > Load Last' to restore.";
              System.err.println(text);
              JOptionPane.showMessageDialog(superAdapter.getMainWindow(), text);
            }

            //superAdapter.getActiveLayerHandler().updateSelectionPoint(e.getX(), e.getY());
            superAdapter.getActiveLayerHandler().doPeak();
          }

          dragMode = DragMode.ANNOTATE;
          //superAdapter.getActiveLayer().updateSelectionPoint(e.getX(), e.getY());
          superAdapter.getActiveLayerHandler().doPeak();
          setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
          // Corners for resize annotation

          try {
            List<Feature2D> newSelectedFeatures = superAdapter.getMainLayer().getSelectedFeatures(hic, e.getX(), e.getY());
            if (!selectedFeatures.get(0).equals(newSelectedFeatures.get(0))) {

              HiCGlobals.splitModeEnabled = false;
              superAdapter.setActiveLayerHandler(superAdapter.getMainLayer());
              superAdapter.getLayersPanel().updateBothLayersPanels(superAdapter);
              superAdapter.getEditLayer().clearAnnotations();
            }
            if (selectedFeatures.size() == 1 && selectedFeatures.get(0).equals(newSelectedFeatures.get(0))) {
              HiCGlobals.splitModeEnabled = true;
            }
          } catch (Exception ignored) {
          }
        } else if (adjustAnnotation != AdjustAnnotation.NONE) {
          dragMode = DragMode.RESIZE;
          Feature2D loop;
          if (activelyEditingAssembly && currentPromptedAssemblyAction == PromptedAssemblyAction.ADJUST) {
            loop = superAdapter.getEditLayer().getFeatureHandler().getFeatureList().get(1, 1).get(0);
          } else {
            loop = currentFeature.getFeature2D();
          }
          // Resizing upper left corner, keep end points stationary
          if (adjustAnnotation == AdjustAnnotation.LEFT) {
            superAdapter.getActiveLayerHandler().setStationaryEnd(loop.getEnd1(), loop.getEnd2());
            // Resizing lower right corner, keep start points stationary
          } else {
            superAdapter.getActiveLayerHandler().setStationaryStart(loop.getStart1(), loop.getStart2());
          }


          try {
            HiCGridAxis xAxis = hic.getZd().getXGridAxis();
            HiCGridAxis yAxis = hic.getZd().getYGridAxis();
            final double scaleFactor = hic.getScaleFactor();
            double binOriginX = hic.getXContext().getBinOrigin();
            double binOriginY = hic.getYContext().getBinOrigin();
            loop.doTest();//TODO meh - please comment why test?
            // hic.getFeature2DHandler()
            annotateRectangle = superAdapter.getActiveLayerHandler().getFeatureHandler().getRectangleFromFeature(
                xAxis, yAxis, loop, binOriginX, binOriginY, scaleFactor);
            int chr1Idx = hic.getXContext().getChromosome().getIndex();
            int chr2Idx = hic.getYContext().getChromosome().getIndex();
            preAdjustLoop = new Pair<>(new Pair<>(chr1Idx, chr2Idx), loop);

          } catch (Exception ex) {
            ex.printStackTrace();
          }

        } else if (!e.isShiftDown() && currentPromptedAssemblyAction == PromptedAssemblyAction.CUT) {
          Feature2D debrisFeature = generateDebrisFeature(e, debrisFeatureSize);
          setDebrisFeauture(debrisFeature);
          int chr1Idx = hic.getXContext().getChromosome().getIndex();
          int chr2Idx = hic.getYContext().getChromosome().getIndex();
          if (debrisFeature != null) {
            superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(chr1Idx, chr2Idx, debrisFeature);
          }
          superAdapter.getEditLayer().getAnnotationLayer().add(chr1Idx, chr2Idx, debrisFeature);
          HiCGlobals.splitModeEnabled = true;
          superAdapter.setActiveLayerHandler(superAdapter.getEditLayer());
          restoreDefaultVariables();
          repaint();
        } else {
          dragMode = DragMode.PAN;
          setCursor(MainWindow.fistCursor);
        }
        lastMousePoint = e.getPoint();
        lastPressedMousePoint = e.getPoint();
      }
    }


    @Override
    public void mouseReleased(final MouseEvent e) {
      endTime = System.nanoTime();
      if (e.isPopupTrigger()) {
        getPopupMenu(e.getX(), e.getY()).show(HeatmapPanel.this, e.getX(), e.getY());
        dragMode = DragMode.NONE;
        lastMousePoint = null;
        zoomRectangle = null;
        annotateRectangle = null;
        setProperCursor();
        // After popup, priority is assembly mode, highlighting those features.

      } /*else if (HiCGlobals.splitModeEnabled && activelyEditingAssembly) {
                if (dragMode == DragMode.ANNOTATE) {
                    Feature2D feature2D = superAdapter.getActiveLayerHandler().generateFeature(hic); //TODO can modify split to wait for user to accept split
                    if (feature2D == null) {
                        int x = (int) lastMousePoint.getX();
                        int y = (int) lastMousePoint.getY();
                        Rectangle annotateRectangle = new Rectangle(x, y, 1, 1);
                        superAdapter.getActiveLayerHandler().updateSelectionRegion(annotateRectangle);
                        feature2D = superAdapter.getActiveLayerHandler().generateFeature(hic); //TODO can modify split to wait for user to accept split
                    }
                    AnnotationLayerHandler editLayerHandler = superAdapter.getEditLayer();
                    debrisFeature = feature2D;
//                    editLayerHandler.getAnnotationLayer().add(Hic.);
                    int chr1Idx = hic.getXContext().getChromosomeFromName().getIndex();
                    int chr2Idx = hic.getYContext().getChromosomeFromName().getIndex();
//                    executeSplitMenuAction(selectedFeatures.get(0),debrisFeature);
                    editLayerHandler.getAnnotationLayer().add(chr1Idx, chr2Idx, debrisFeature);
                    restoreDefaultVariables();

                }

            }*/ else {
        // turn on continuous sync after dragging
        if (HiCGlobals.wasLinkedBeforeMousePress) {
          HiCGlobals.wasLinkedBeforeMousePress = false;
          hic.setLinkedMode(true);

          if (lastPressedMousePoint != null) {
            double deltaX = e.getX() - lastPressedMousePoint.getX();
            double deltaY = e.getY() - lastPressedMousePoint.getY();
            if (Math.abs(deltaX) > 0 && Math.abs(deltaY) > 0) {
              hic.broadcastLocation();
            }
          } else hic.broadcastLocation();
        }

        if (activelyEditingAssembly && HiCGlobals.splitModeEnabled && currentPromptedAssemblyAction == PromptedAssemblyAction.CUT) {
          // disable long click: it seems that no one is using it anyway. But let's keep it commented around for now..
//                    holdTime = (endTime - startTime) / Math.pow(10, 6);
          //Short click: execute split, long click: expert mode leave annotation be for editing purposes
//                    if (holdTime <= clickDelay) {
          debrisFeature = generateDebrisFeature(e, debrisFeatureSize);
          executeSplitMenuAction();
//                    }
          currentPromptedAssemblyAction = PromptedAssemblyAction.NONE;
        }
        if (activelyEditingAssembly && (dragMode == DragMode.ANNOTATE || currentPromptedAssemblyAction == PromptedAssemblyAction.ADJUST)) {
          // New annotation is added (not single click) and new feature from custom annotation

          updateSelectedFeatures(false);
          List<Feature2D> newSelectedFeatures = superAdapter.getMainLayer().getSelectedFeatures(hic, e.getX(), e.getY());

          // selects superscaffold
          if ((newSelectedFeatures == null || newSelectedFeatures.size() == 0) && (selectedFeatures == null || selectedFeatures.size() == 0)) {
            Feature2DGuiContainer newSelectedSuperscaffold = getMouseHoverSuperscaffold(e.getX(), e.getY());

            if (newSelectedSuperscaffold != null) {
              final List<Integer> curScaffolds = superAdapter.getAssemblyStateTracker().getAssemblyHandler().getListOfSuperscaffolds().get(
                      Integer.parseInt(newSelectedSuperscaffold.getFeature2D().getAttribute("Superscaffold #")) - 1);

              newSelectedFeatures.clear();
              for (int scaffold : curScaffolds) {
                Feature2D curScaffold = superAdapter.getAssemblyStateTracker().getAssemblyHandler().getListOfScaffolds().get(Math.abs(scaffold) - 1).getCurrentFeature2D();
                newSelectedFeatures.add(curScaffold);
              }
            }
          }

          Collections.sort(newSelectedFeatures);

          // Damage rectangle is not precise, adjust boundaries...
          try {
            if (currentPromptedAssemblyAction == PromptedAssemblyAction.ADJUST && selectedFeatures != null) {
              if (adjustAnnotation == AdjustAnnotation.LEFT) {
                while (!selectedFeatures.contains(newSelectedFeatures.get(newSelectedFeatures.size() - 1)) && !newSelectedFeatures.isEmpty()) {
                  newSelectedFeatures.remove(newSelectedFeatures.size() - 1);
                }
              } else {
                while (!selectedFeatures.contains(newSelectedFeatures.get(0)) && !newSelectedFeatures.isEmpty()) {
                  newSelectedFeatures.remove(0);
                }
              }
            }
          } catch (Exception e1) {
            removeSelection();
          }

          if (HiCGlobals.translationInProgress) {
            translationInProgressMouseReleased(newSelectedFeatures);
          } else {
            if (selectedFeatures != null && selectedFeatures.equals(newSelectedFeatures) && currentPromptedAssemblyAction != PromptedAssemblyAction.ADJUST) {
              removeSelection();
            } else {
              selectedFeatures = newSelectedFeatures;
            }
          }
          updateSelectedFeatures(true);

          Chromosome chrX = superAdapter.getHiC().getXContext().getChromosome();
          Chromosome chrY = superAdapter.getHiC().getYContext().getChromosome();
          superAdapter.getEditLayer().filterTempSelectedGroup(chrX.getIndex(), chrY.getIndex());
          repaint();

          if (selectedFeatures != null && !selectedFeatures.isEmpty()) {
            if (superAdapter.getMainLayer().getLayerVisibility()) {
              tempSelectedGroup = superAdapter.getEditLayer().addTempSelectedGroup(selectedFeatures, hic);
              addHighlightedFeature(tempSelectedGroup);
            }
          } else {
            removeHighlightedFeature();
          }

          //getAssemblyPopupMenu(e.getX(), e.getY()).show(HeatmapPanel.this, e.getX(), e.getY());

          superAdapter.getMainViewPanel().toggleToolTipUpdates(Boolean.TRUE);
          superAdapter.updateMainViewPanelToolTipText(toolTipText(e.getX(), e.getY()));
          superAdapter.getMainViewPanel().toggleToolTipUpdates(selectedFeatures.isEmpty());

          currentPromptedAssemblyAction = PromptedAssemblyAction.NONE;

          restoreDefaultVariables();
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
          superAdapter.getActiveLayerHandler().addFeature(hic);
          restoreDefaultVariables();
        } else if (dragMode == DragMode.RESIZE) {
          // New annotation is added (not single click) and new feature from custom annotation
          int idx1 = preAdjustLoop.getFirst().getFirst();
          int idx2 = preAdjustLoop.getFirst().getSecond();

          Feature2D secondLoop = preAdjustLoop.getSecond();
          // Add a new loop if it was resized (prevents deletion on single click)

          try {
            final double scaleFactor = hic.getScaleFactor();
            final int screenWidth = HeatmapPanel.this.getBounds().width;
            final int screenHeight = HeatmapPanel.this.getBounds().height;
            int centerX = (int) (screenWidth / scaleFactor) / 2;
            int centerY = (int) (screenHeight / scaleFactor) / 2;

            if (superAdapter.getActiveLayerHandler().hasLoop(hic.getZd(), idx1, idx2, centerX, centerY,
                Feature2DHandler.numberOfLoopsToFind, hic.getXContext().getBinOrigin(),
                hic.getYContext().getBinOrigin(), hic.getScaleFactor(), secondLoop) && changedSize) {
              Feature2D oldFeature2D = secondLoop.deepCopy();

              superAdapter.getActiveLayerHandler().removeFromList(hic.getZd(), idx1, idx2, centerX, centerY,
                  Feature2DHandler.numberOfLoopsToFind, hic.getXContext().getBinOrigin(),
                  hic.getYContext().getBinOrigin(), hic.getScaleFactor(), secondLoop);

              Feature2D tempFeature2D = superAdapter.getActiveLayerHandler().addFeature(hic);
              superAdapter.getActiveLayerHandler().setLastItem(idx1, idx2, secondLoop);
              for (String newKey : oldFeature2D.getAttributeKeys()) {
                tempFeature2D.setAttribute(newKey, oldFeature2D.getAttribute(newKey));
              }

              //remove preadjust loop from list
              if (activelyEditingAssembly && HiCGlobals.splitModeEnabled) {
                debrisFeature = tempFeature2D;
              }
            }
          } catch (Exception ee) {
            System.err.println("Unable to remove pre-resized loop");
          }
          restoreDefaultVariables();
        } else {
          setProperCursor();
        }
      }
    }

    // works for only one selected contig
    private void translationInProgressMouseReleased(List<Feature2D> newSelectedFeatures) {
      if (selectedFeatures != null && !selectedFeatures.isEmpty()) {
        Feature2D featureDestination = newSelectedFeatures.get(0);
        AssemblyOperationExecutor.moveSelection(superAdapter, selectedFeatures, featureDestination);
        repaint();
      }

      if (selectedFeatures != null && newSelectedFeatures != null) {
        selectedFeatures.addAll(newSelectedFeatures);
      }
      HiCGlobals.translationInProgress = Boolean.FALSE;
      removeSelection(); //TODO fix this so that highlight moves with translated selection

    }

    Feature2D generateDebrisFeature(final MouseEvent eF, int debrisFeatureSize) {
      final double scaleFactor = hic.getScaleFactor();
      double binOriginX = hic.getXContext().getBinOrigin();
      double binOriginY = hic.getYContext().getBinOrigin();
      Point mousePoint = eF.getPoint();
      double x = mousePoint.getX();
      double y = mousePoint.getY();
      int rightCorner = (int) Math.max(x, y + (binOriginY - binOriginX) * scaleFactor);
      Rectangle annotateRectangle = new Rectangle(rightCorner - debrisFeatureSize, (int) (rightCorner - debrisFeatureSize - (binOriginY - binOriginX) * scaleFactor), debrisFeatureSize, debrisFeatureSize);
      superAdapter.getEditLayer().updateSelectionRegion(annotateRectangle);
      debrisFeature = superAdapter.getEditLayer().generateFeature(hic);
      return debrisFeature;
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
      setProperCursor();
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
          superAdapter.getActiveLayerHandler().updateSelectionRegion(damageRect);
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
          superAdapter.getActiveLayerHandler().updateSelectionRegion(damageRect);
          changedSize = true;
          break;
        default:
          lastMousePoint = e.getPoint();    // Always save the last Point

          double deltaXBins = -deltaX / hic.getScaleFactor();
          double deltaYBins = -deltaY / hic.getScaleFactor();
          hic.moveBy(deltaXBins, deltaYBins);
      }
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
        currentPromptedAssemblyAction = PromptedAssemblyAction.NONE;
        // Update tool tip text
        if (!featureOptionMenuEnabled) {
          superAdapter.updateMainViewPanelToolTipText(toolTipText(e.getX(), e.getY()));
        }
        // Set check if hovering over feature corner

        // Following was commented out since it was causing flickering of the cursor on windows machines, don't know if was necessary
//        setCursor(Cursor.getDefaultCursor());
        int minDist = 20;
        if (currentFeature != null) {

          boolean resizeable = (currentFeature.getAnnotationLayerHandler().getAnnotationLayerType() != AnnotationLayer.LayerType.SCAFFOLD) && (currentFeature.getAnnotationLayerHandler().getAnnotationLayerType() != AnnotationLayer.LayerType.SUPERSCAFFOLD);
//                    if (activelyEditingAssembly) {
//                        resizeable = (resizeable && HiCGlobals.splitModeEnabled);
//                    }
          if (resizeable) {
            Rectangle loop = currentFeature.getRectangle();
            Point mousePoint = e.getPoint();
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
            }
          }

        }
        if (activelyEditingAssembly && !allMainFeaturePairs.isEmpty() && !e.isShiftDown()) {

          final double scaleFactor = hic.getScaleFactor();
          double binOriginX = hic.getXContext().getBinOrigin();
          double binOriginY = hic.getYContext().getBinOrigin();

          Point mousePoint = e.getPoint();
          double x = mousePoint.getX();
          double y = mousePoint.getY();

          // this is a good place to handle inserts to top and bottom as it should be done even if individual
          // features at the beginning of the assembly are not visible
          // find the x and y in relation to the displayed screen
          int topLeftCornerX = (int) ((0 - binOriginX) * scaleFactor);
          int topLeftCornerY = (int) ((0 - binOriginY) * scaleFactor);

          List<Scaffold> listOfScaffolds =
              superAdapter.getAssemblyStateTracker().getAssemblyHandler().getListOfAggregateScaffolds();
          int
              lastGenomicBin = 0;
          try {
            lastGenomicBin = listOfScaffolds.get(listOfScaffolds.size() - 1).getCurrentFeature2D().getEnd2() /
                hic.getZd().getBinSize();
          } catch (UninitializedObjectException e1) {
            e1.printStackTrace();
          }
          int bottomRightCornerX = (int) ((lastGenomicBin - binOriginX) * scaleFactor);
          int bottomRightCornerY = (int) ((lastGenomicBin - binOriginY) * scaleFactor);

          if (selectedFeatures != null && !selectedFeatures.isEmpty()) {
            if (mousePoint.getX() - topLeftCornerX >= 0 &&
                mousePoint.getX() - topLeftCornerX <= minDist &&
                mousePoint.getY() - topLeftCornerY >= 0 &&
                mousePoint.getY() - topLeftCornerY <= minDist) {
              setCursor(MainWindow.pasteNWCursor);
              currentPromptedAssemblyAction = PromptedAssemblyAction.PASTETOP;
            }
            if (bottomRightCornerX - mousePoint.getX() >= 0 &&
                bottomRightCornerX - mousePoint.getX() <= minDist &&
                bottomRightCornerY - mousePoint.getY() >= 0 &&
                bottomRightCornerY - mousePoint.getY() <= minDist) {
              setCursor(MainWindow.pasteSECursor);
              currentPromptedAssemblyAction = PromptedAssemblyAction.PASTEBOTTOM;
            }
          }

          currentUpstreamFeature = null;
          currentDownstreamFeature = null;

          for (Feature2DGuiContainer asmFragment : allMainFeaturePairs) {
            if (asmFragment.getRectangle().contains(x, x + (binOriginX - binOriginY) * scaleFactor)) {
              currentUpstreamFeature = asmFragment;
            }
            if (asmFragment.getRectangle().contains(y + (binOriginY - binOriginX) * scaleFactor, y)) {
              currentDownstreamFeature = asmFragment;
            }
          }

          if (currentUpstreamFeature != null && currentDownstreamFeature != null) {
            if (currentUpstreamFeature.getFeature2D().getStart1() > currentDownstreamFeature.getFeature2D().getStart1()) {
              Feature2DGuiContainer temp = currentUpstreamFeature;
              currentUpstreamFeature = currentDownstreamFeature;
              currentDownstreamFeature = temp;
            }

            // inserting from highlight: keeping for future development
//            if (selectedFeatures != null && !selectedFeatures.isEmpty()) {
//
//              // upstream feature is the same
//              if (currentUpstreamFeature.getFeature2D().getStart1() >= selectedFeatures.get(0).getStart1() &&
//                  currentUpstreamFeature.getFeature2D().getEnd1() <=
//                      selectedFeatures.get(selectedFeatures.size() - 1).getEnd1()) {
//
//                int topYright = currentUpstreamFeature.getRectangle().y;
//                int bottomYright =
//                    currentUpstreamFeature.getRectangle().y + (int) currentUpstreamFeature.getRectangle().getHeight();
//                int leftXbottom = currentUpstreamFeature.getRectangle().x;
//                int rightXbottom =
//                    currentUpstreamFeature.getRectangle().x + (int) currentUpstreamFeature.getRectangle().getWidth();
//
//                if (mousePoint.getY() >= topYright && mousePoint.getY() <= bottomYright) {
//
//                  if ((mousePoint.getX() >= currentDownstreamFeature.getRectangle().getMinX() &&
//                      mousePoint.getX() <= currentDownstreamFeature.getRectangle().getMinX() + minDist)) {
//
//                    // if the start doesn't match the end of the previous one, there's a gap, do not insert
//                    if (currentDownstreamFeature.getFeature2D().getStart1() ==
//                        allMainFeaturePairs.get(idxUp - 1).getFeature2D().getEnd1()) {
//                      setCursor(MainWindow.pasteSWCursor);
//                      currentUpstreamFeature = allMainFeaturePairs.get(idxUp - 1);
//                      currentPromptedAssemblyAction = PromptedAssemblyAction.PASTE;
//                    }
//                  }
//                }
//
//                if (mousePoint.getX() >= leftXbottom && mousePoint.getX() <= rightXbottom) {
//                  // -y axis
//                  if ((mousePoint.getY() >= currentDownstreamFeature.getRectangle().getMinY() &&
//                      mousePoint.getY() <= currentDownstreamFeature.getRectangle().getMinY() + minDist)) {
//                    if (currentDownstreamFeature.getFeature2D().getStart1() ==
//                        allMainFeaturePairs.get(idxDown - 1).getFeature2D().getEnd1()) {
//                      setCursor(MainWindow.pasteNECursor);
//                      currentUpstreamFeature = allMainFeaturePairs.get(idxDown - 1);
//                      currentPromptedAssemblyAction = PromptedAssemblyAction.PASTE;
//                    }
//                  }
//                }
//              }
//
//              // downstream feature the same
//              if ((currentDownstreamFeature.getFeature2D().getStart1() >= selectedFeatures.get(0).getStart1() &&
//                  currentDownstreamFeature.getFeature2D().getEnd1() <=
//                      selectedFeatures.get(selectedFeatures.size() - 1).getEnd1())
//                  ) {
//                int topYleft = currentDownstreamFeature.getRectangle().y;
//                int bottomYleft =
//                    currentDownstreamFeature.getRectangle().y +
//                        (int) currentDownstreamFeature.getRectangle().getHeight();
//                int leftXtop = currentDownstreamFeature.getRectangle().x;
//                int rightXtop =
//                    currentDownstreamFeature.getRectangle().x +
//                        (int) currentDownstreamFeature.getRectangle().getWidth();
//
//                // y axis
//                if (mousePoint.getX() >= leftXtop && mousePoint.getX() <= rightXtop) {
//                  if ((mousePoint.getY() >= currentUpstreamFeature.getRectangle().getMaxY() - minDist &&
//                      mousePoint.getY() <= currentUpstreamFeature.getRectangle().getMaxY())) {
//                    setCursor(MainWindow.pasteSWCursor);
//                    currentPromptedAssemblyAction = PromptedAssemblyAction.PASTE;
//                  }
//                }
//                // -x axis
//                else if (mousePoint.getY() >= topYleft && mousePoint.getY() <= bottomYleft) {
//                  if ((mousePoint.getX() >= currentUpstreamFeature.getRectangle().getMaxX() - minDist &&
//                      mousePoint.getX() <= (currentUpstreamFeature.getRectangle().getMaxX()))) {
//                    setCursor(MainWindow.pasteNECursor);
//                    currentPromptedAssemblyAction = PromptedAssemblyAction.PASTE;
//                  }
//                }
//              }
//            }

            if (!HiCGlobals.splitModeEnabled && (currentUpstreamFeature.getFeature2D().getEnd1() == currentDownstreamFeature.getFeature2D().getStart1()) || (currentDownstreamFeature == null && currentUpstreamFeature == null)) {

              if ((mousePoint.getX() - currentUpstreamFeature.getRectangle().getMaxX() >= 0) &&
                  (mousePoint.getX() - currentUpstreamFeature.getRectangle().getMaxX() <= minDist) &&
                  (currentUpstreamFeature.getRectangle().getMaxY() - mousePoint.getY() >= 0) &&
                  (currentUpstreamFeature.getRectangle().getMaxY() - mousePoint.getY() <= minDist)) {
                if (selectedFeatures == null || selectedFeatures.isEmpty()) {
                  setCursor(MainWindow.groupSWCursor);
                  currentPromptedAssemblyAction = PromptedAssemblyAction.REGROUP;
                } else if (!(currentUpstreamFeature.getFeature2D().getEnd1() >= selectedFeatures.get(0).getStart1() &&
                    currentUpstreamFeature.getFeature2D().getEnd1() <=
                        selectedFeatures.get(selectedFeatures.size() - 1).getEnd1())) {
                  setCursor(MainWindow.pasteSWCursor);
                  currentPromptedAssemblyAction = PromptedAssemblyAction.PASTE;
                }
              } else if ((currentUpstreamFeature.getRectangle().getMaxX() - mousePoint.getX() >= 0) &&
                  (currentUpstreamFeature.getRectangle().getMaxX() - mousePoint.getX() <= minDist) &&
                  (mousePoint.getY() - currentUpstreamFeature.getRectangle().getMaxY() >= 0) &&
                  (mousePoint.getY() - currentUpstreamFeature.getRectangle().getMaxY() <= minDist)) {
                if (selectedFeatures == null || selectedFeatures.isEmpty()) {
                  setCursor(MainWindow.groupNECursor);
                  currentPromptedAssemblyAction = PromptedAssemblyAction.REGROUP;
                } else if (!(currentUpstreamFeature.getFeature2D().getEnd1() >= selectedFeatures.get(0).getStart1() &&
                    currentUpstreamFeature.getFeature2D().getEnd1() <=
                        selectedFeatures.get(selectedFeatures.size() - 1).getEnd1())) {
                  setCursor(MainWindow.pasteNECursor);
                  currentPromptedAssemblyAction = PromptedAssemblyAction.PASTE;
                }
              }
            }
          }

          if (!HiCGlobals.splitModeEnabled && selectedFeatures!=null && !selectedFeatures.isEmpty()){

            for (Feature2DGuiContainer asmFragment : allEditFeaturePairs) {
              if (asmFragment.getFeature2D().equals(tempSelectedGroup) && !asmFragment.getFeature2D().equals(debrisFeature)) {
                if (Math.abs(asmFragment.getRectangle().getMaxX() - mousePoint.getX()) < minDist &&
                    Math.abs(asmFragment.getRectangle().getMinY() - mousePoint.getY()) < minDist) {
                  setCursor(MainWindow.invertSWCursor);
                  if (debrisFeature != null) {
                    int chr1Idx = hic.getXContext().getChromosome().getIndex();
                    int chr2Idx = hic.getYContext().getChromosome().getIndex();
                    superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(chr1Idx, chr2Idx, debrisFeature);
                  }
                  currentPromptedAssemblyAction = PromptedAssemblyAction.INVERT;
                } else if (Math.abs(asmFragment.getRectangle().getMinX() - mousePoint.getX()) < minDist &&
                    Math.abs(asmFragment.getRectangle().getMaxY() - mousePoint.getY()) < minDist) {
                  setCursor(MainWindow.invertNECursor);
                  if (debrisFeature != null) {
                    int chr1Idx = hic.getXContext().getChromosome().getIndex();
                    int chr2Idx = hic.getYContext().getChromosome().getIndex();
                    superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(chr1Idx, chr2Idx, debrisFeature);
                  }
                  currentPromptedAssemblyAction = PromptedAssemblyAction.INVERT;
                } else if (selectedFeatures.size() == 1 && Math.abs(x - (y + (binOriginY - binOriginX) * scaleFactor)) < minDist &&
                    Math.abs(y - (x + (binOriginX - binOriginY) * scaleFactor)) < minDist &&
                    x - asmFragment.getRectangle().getMinX() > debrisFeatureSize + RESIZE_SNAP + scaleFactor &&
                    asmFragment.getRectangle().getMaxX() - x > RESIZE_SNAP + scaleFactor &&
                    y - asmFragment.getRectangle().getMinY() > debrisFeatureSize + RESIZE_SNAP + scaleFactor &&
                    asmFragment.getRectangle().getMaxY() - y > RESIZE_SNAP + scaleFactor) {
                  setCursor(MainWindow.scissorCursor);
                  currentPromptedAssemblyAction = PromptedAssemblyAction.CUT;

                  int chr1Idx = hic.getXContext().getChromosome().getIndex();
                  int chr2Idx = hic.getYContext().getChromosome().getIndex();
                  if (debrisFeature != null) {
                    superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(chr1Idx, chr2Idx, debrisFeature);
                  }
                  generateDebrisFeature(e, debrisFeatureSize);
                  superAdapter.getEditLayer().getAnnotationLayer().add(chr1Idx, chr2Idx, debrisFeature);
                } else if (Math.abs(x - asmFragment.getRectangle().getMinX()) <= RESIZE_SNAP &&
                    Math.abs(y - asmFragment.getRectangle().getMinY()) <= RESIZE_SNAP &&
                    y + x < asmFragment.getRectangle().getMaxX() + asmFragment.getRectangle().getMinY()) {
                  setCursor(Cursor.getPredefinedCursor(Cursor.NW_RESIZE_CURSOR));
                  currentPromptedAssemblyAction = PromptedAssemblyAction.ADJUST;
                  adjustAnnotation = AdjustAnnotation.LEFT;
                } else if (Math.abs(asmFragment.getRectangle().getMaxX() - x) <= RESIZE_SNAP &&
                    Math.abs(asmFragment.getRectangle().getMaxY() - y) <= RESIZE_SNAP &&
                    y + x > asmFragment.getRectangle().getMaxX() + asmFragment.getRectangle().getMinY()) {
                  setCursor(Cursor.getPredefinedCursor(Cursor.SE_RESIZE_CURSOR));
                  currentPromptedAssemblyAction = PromptedAssemblyAction.ADJUST;
                  adjustAnnotation = AdjustAnnotation.RIGHT;
                } else if (debrisFeature != null) {
                  int chr1Idx = hic.getXContext().getChromosome().getIndex();
                  int chr2Idx = hic.getYContext().getChromosome().getIndex();
                  superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(chr1Idx, chr2Idx, debrisFeature);
                }
              }
            }
          }
        }

        if (hic.isWholeGenome()) {
          synchronized (this) {
            hic.setGWCursorPoint(e.getPoint());
            superAdapter.repaintGridRulerPanels();
          }
        } else {
          hic.setGWCursorPoint(null);
        }

        if (straightEdgeEnabled || e.isShiftDown()) {
          synchronized (this) {
            hic.setCursorPoint(e.getPoint());
            superAdapter.repaintTrackPanels();
          }
        } else if (diagonalEdgeEnabled) {
          synchronized (this) {
            hic.setDiagonalCursorPoint(e.getPoint());
            superAdapter.repaintTrackPanels();
          }
        } else if (adjustAnnotation == AdjustAnnotation.NONE && currentPromptedAssemblyAction == PromptedAssemblyAction.NONE) {
          hic.setCursorPoint(null);
          setCursor(Cursor.getDefaultCursor());
        }
        repaint();
      }

    }

    @Override
    public void mouseWheelMoved(MouseWheelEvent e) {
      try {
        if (currentPromptedAssemblyAction == PromptedAssemblyAction.CUT) {

          final double scaleFactor = hic.getScaleFactor();
          double binOriginX = hic.getXContext().getBinOrigin();
          double binOriginY = hic.getYContext().getBinOrigin();
          Point mousePoint = e.getPoint();
          double x = mousePoint.getX();
          double y = mousePoint.getY();
          int rightCorner = (int) Math.max(x, y);

          debrisFeatureSize = debrisFeatureSize - e.getUnitsToScroll();
          if (rightCorner - debrisFeatureSize < currentFeature.getRectangle().getMinX() + RESIZE_SNAP) {
            debrisFeatureSize = rightCorner - (int) currentFeature.getRectangle().getMinX() - RESIZE_SNAP - 1;
          }
          if (debrisFeatureSize <= scaleFactor) {
            debrisFeatureSize = (int) Math.max(scaleFactor, 1);
          }

          int chr1Idx = hic.getXContext().getChromosome().getIndex();
          int chr2Idx = hic.getYContext().getChromosome().getIndex();
          if (debrisFeature != null) {
            superAdapter.getEditLayer().getAnnotationLayer().getFeatureHandler().getFeatureList().checkAndRemoveFeature(chr1Idx, chr2Idx, debrisFeature);
          }
          generateDebrisFeature(e, debrisFeatureSize);
          superAdapter.getEditLayer().getAnnotationLayer().add(chr1Idx, chr2Idx, debrisFeature);
          repaint();
          return;
        }
        int scroll = 0;

        if (System.getProperty("os.name").toLowerCase().indexOf("win") >= 0) {
          double precScroll = e.getPreciseWheelRotation();

          if (precScroll >= 0) {
            scroll = (int) Math.ceil(precScroll);
          } else {
            scroll = (int) Math.floor(precScroll);
          }
        } else {
          scroll = e.getWheelRotation();
        }

        hic.moveBy(scroll, scroll);
        superAdapter.updateMainViewPanelToolTipText(toolTipText(e.getX(), e.getY()));
      } catch (Exception e2) {
        repaint();
      }
    }
  }
}