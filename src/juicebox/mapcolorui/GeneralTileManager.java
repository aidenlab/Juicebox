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
import juicebox.data.MatrixZoomData;
import juicebox.gui.SuperAdapter;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;

import javax.swing.*;
import java.awt.*;

public class GeneralTileManager {
    private static final int imageTileWidth = 500;

    private final HiCMapTileManager mapTileManager;

    public GeneralTileManager(ColorScaleHandler colorScaleHandler) {
        mapTileManager = new HiCMapTileManager(colorScaleHandler);
    }

    public boolean renderHiCTiles(HeatmapRenderer renderer, double binOriginX, double binOriginY, double bRight, double bBottom,
                                  MatrixZoomData zd, MatrixZoomData controlZd,
                                  double scaleFactor, Rectangle bounds, HiC hic, JComponent parent, SuperAdapter superAdapter) {

        boolean allTilesNull = true;
        MatrixType displayOption = hic.getDisplayOption();
        NormalizationType observedNormalizationType = hic.getObsNormalizationType();
        NormalizationType controlNormalizationType = hic.getControlNormalizationType();

        // tile numbers
        int tLeft = (int) (binOriginX / imageTileWidth);
        int tRight = (int) Math.ceil(bRight / imageTileWidth);
        int tTop = (int) (binOriginY / imageTileWidth);
        int tBottom = (int) Math.ceil(bBottom / imageTileWidth);

        for (int tileRow = tTop; tileRow <= tBottom; tileRow++) {
            for (int tileColumn = tLeft; tileColumn <= tRight; tileColumn++) {

                ImageTile tile = null;
                try {
                    tile = mapTileManager.getImageTile(zd, controlZd, tileRow, tileColumn, displayOption,
                            observedNormalizationType, controlNormalizationType, hic, parent);
                } catch (Exception e) {
                    System.err.println(e.getMessage());

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


                    try {
                        if (xDest0 < xDest1 && yDest0 < yDest1 && xSrc0 < xSrc1 && ySrc0 < ySrc1) {
                            // basically ensure that we're not trying to plot empty space
                            // also for some reason we have negative indices sometimes??
                            renderer.drawImage(tile.image, xDest0, yDest0, xDest1, yDest1, xSrc0, ySrc0, xSrc1, ySrc1);
                        }
                    } catch (Exception e) {

                        // handling for svg export
                        try {
                            if (HiCGlobals.printVerboseComments) {
                                System.out.println("svg plotting for\n" + xDest0 + "_" + yDest0 + "_" + xDest1 + "_" +
                                        yDest1 + "_" + xSrc0 + "_" + ySrc0 + "_" + xSrc1 + "_" + ySrc1);
                            }
                            bypassTileAndDirectlyDrawOnGraphics(renderer, zd, tileRow, tileColumn,
                                    displayOption, observedNormalizationType, controlNormalizationType,
                                    xDest0, yDest0, xDest1, yDest1, xSrc0, ySrc0, xSrc1, ySrc1, hic);
                            //processedExportRegions.add(newKey);
                        } catch (Exception e2) {
                            System.err.println("SVG export did not work");
                        }
                    }

                    if (HiCGlobals.displayTiles) {
                        renderer.drawRect(xDest0, yDest0, (xDest1 - xDest0), (yDest1 - yDest0));
                    }
                }
            }
        }

        //In case of change to map settings, get map color limits and update slider:
        //TODO: || might not catch all changed at once, if more then one parameter changed...
        if (hic.testZoomChanged() || hic.testDisplayOptionChanged() || hic.testNormalizationTypeChanged()) {
            //In case render is called as a result of zoom change event, check if
            //We need to update slider with map range:
            String cacheKey = HeatmapRenderer.getColorScaleCacheKey(zd, displayOption, observedNormalizationType, controlNormalizationType);
            mapTileManager.updateColorSliderFromColorScale(superAdapter, displayOption, cacheKey);
            //debrisFeatureSize = (int) (debrisFeatureSize * scaleFactor);
        }

        return allTilesNull;
    }



    private void bypassTileAndDirectlyDrawOnGraphics(HeatmapRenderer renderer, MatrixZoomData zd, int tileRow, int tileColumn,
                                                     MatrixType displayOption, NormalizationType observedNormalizationType,
                                                     NormalizationType controlNormalizationType,
                                                     int xDest0, int yDest0, int xDest1, int yDest1, int xSrc0,
                                                     int ySrc0, int xSrc1, int ySrc1, HiC hic) {

        // Image size can be smaller than tile width when zoomed out, or near the edges.

        long maxBinCountX = zd.getXGridAxis().getBinCount();
        long maxBinCountY = zd.getYGridAxis().getBinCount();

        if (maxBinCountX < 0 || maxBinCountY < 0) return;

        final int bx0 = tileColumn * imageTileWidth;
        final int by0 = tileRow * imageTileWidth;

        // set new origins
        renderer.translate(xDest0, yDest0);

        // scale drawing appropriately
        double widthDest = xDest1 - xDest0;
        double heightDest = yDest1 - yDest0;
        int widthSrc = xSrc1 - xSrc0;
        int heightSrc = ySrc1 - ySrc0;
        double horizontalScaling = widthDest / widthSrc;
        double verticalScaling = heightDest / heightSrc;
        renderer.scale(horizontalScaling, verticalScaling);

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
                false);

        renderer.scale(1, 1);
        renderer.translate(0, 0);
    }

    public void clearTileCache() {
        mapTileManager.clearTileCache();
    }

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
}
