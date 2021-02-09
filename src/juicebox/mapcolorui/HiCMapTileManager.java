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
import juicebox.data.ExpectedValueFunction;
import juicebox.data.MatrixZoomData;
import juicebox.gui.SuperAdapter;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;
import org.broad.igv.util.ObjectCache;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class HiCMapTileManager {
    private static final int imageTileWidth = 500;
    private final ObjectCache<String, GeneralTileManager.ImageTile> tileCache = new ObjectCache<>(30);
    private final ColorScaleHandler colorScaleHandler;

    public HiCMapTileManager(ColorScaleHandler colorScaleHandler) {
        this.colorScaleHandler = colorScaleHandler;
    }

    public void clearTileCache() {
        tileCache.clear();
    }

    public GeneralTileManager.ImageTile getImageTile(MatrixZoomData zd, MatrixZoomData controlZd, int tileRow, int tileColumn, MatrixType displayOption,
                                                     NormalizationType obsNormalizationType, NormalizationType ctrlNormalizationType,
                                                     HiC hic, JComponent parent) {

        String key = zd.getTileKey(tileRow, tileColumn, displayOption);
        GeneralTileManager.ImageTile tile = tileCache.get(key);

        if (tile == null) {

            // Image size can be smaller than tile width when zoomed out, or near the edges.

            long maxBinCountX = zd.getXGridAxis().getBinCount();
            long maxBinCountY = zd.getYGridAxis().getBinCount();

            if (maxBinCountX < 0 || maxBinCountY < 0) return null;

            int imageWidth = maxBinCountX < imageTileWidth ? (int) maxBinCountX : imageTileWidth;
            int imageHeight = maxBinCountY < imageTileWidth ? (int) maxBinCountY : imageTileWidth;
            final int bx0 = tileColumn * imageTileWidth;
            final int by0 = tileRow * imageTileWidth;

            Image image = renderDataWithCPU(parent, bx0, by0, imageWidth, imageHeight,
                    zd, controlZd, displayOption, obsNormalizationType, ctrlNormalizationType,
                    hic.getExpectedValues(), hic.getExpectedControlValues());

            // if (scaleFactor > 0.999 && scaleFactor < 1.001) {
            tile = new GeneralTileManager.ImageTile(image, bx0, by0);
            tileCache.put(key, tile);
        }
        return tile;
    }

    private BufferedImage renderDataWithCPU(JComponent parent, int bx0, int by0, int imageWidth, int imageHeight,
                                            MatrixZoomData zd, MatrixZoomData controlZd, MatrixType displayOption,
                                            NormalizationType obsNormalizationType, NormalizationType ctrlNormalizationType,
                                            ExpectedValueFunction expectedValues, ExpectedValueFunction expectedControlValues) {
        BufferedImage image = (BufferedImage) parent.createImage(imageWidth, imageHeight);
        Graphics2D g2D = (Graphics2D) image.getGraphics();
        if (HiCGlobals.isDarkulaModeEnabled) {
            g2D.setColor(Color.darkGray);
            g2D.fillRect(0, 0, imageWidth, imageHeight);
        }

        HeatmapRenderer renderer = new HeatmapRenderer(g2D, colorScaleHandler);
        if (!renderer.render(bx0, by0, imageWidth, imageHeight,
                zd, controlZd, displayOption,
                obsNormalizationType, ctrlNormalizationType,
                expectedValues, expectedControlValues, true)) {
            return null;
        }
        return image;
    }

    public void updateColorSliderFromColorScale(SuperAdapter superAdapter, MatrixType displayOption, String cacheKey) {
        colorScaleHandler.updateColorSliderFromColorScale(superAdapter, displayOption, cacheKey);
    }
}
