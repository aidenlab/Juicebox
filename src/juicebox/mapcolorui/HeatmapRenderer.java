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

import juicebox.HiCGlobals;
import juicebox.data.Block;
import juicebox.data.ContactRecord;
import juicebox.data.ExpectedValueFunction;
import juicebox.data.MatrixZoomData;
import juicebox.gui.SuperAdapter;
import juicebox.matrix.BasicMatrix;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;
import org.broad.igv.renderer.ColorScale;

import java.awt.*;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author jrobinso
 * @since Aug 11, 2010
 */
public class HeatmapRenderer {

    private final ColorScaleHandler colorScaleHandler = new ColorScaleHandler();
    private static final int PIXEL_WIDTH = 1, PIXEL_HEIGHT = 1;
    public static float PSEUDOCOUNT = 1f;

    public static String getColorScaleCacheKey(MatrixZoomData zd, MatrixType displayOption, NormalizationType obsNorm, NormalizationType ctrlNorm) {
        return zd.getColorScaleKey(displayOption, obsNorm, ctrlNorm);
    }

    public boolean render(int originX, int originY, int width, int height,
                          final MatrixZoomData zd, final MatrixZoomData controlZD,
                          final MatrixType displayOption,
                          final NormalizationType observedNormalizationType, final NormalizationType controlNormalizationType,
                          final ExpectedValueFunction df, final ExpectedValueFunction controlDF,
                          Graphics2D g, boolean isImportant) {
        g.setRenderingHint(RenderingHints.KEY_COLOR_RENDERING, RenderingHints.VALUE_COLOR_RENDER_SPEED);
        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);

        int chr1 = zd.getChr1Idx();
        int chr2 = zd.getChr2Idx();
        int x = originX;
        int y = originY;

        boolean isWholeGenome = chr1 == 0 && chr2 == 0;
        boolean sameChr = (chr1 == chr2);

        if (sameChr) {
            // Data is transposable, transpose if necessary.  Convention is to use lower diagonal
            if (x > y) {
                //noinspection SuspiciousNameCombination
                x = originY;
                y = originX;
                int tmp = width;
                width = height;
                height = tmp;
            }
        }

        int maxX = x + width - 1;
        int maxY = y + height - 1;

        String key = zd.getColorScaleKey(displayOption, observedNormalizationType, controlNormalizationType);
        String controlKey = zd.getColorScaleKey(displayOption, observedNormalizationType, controlNormalizationType);

        float pseudocountObs = PSEUDOCOUNT;
        float pseudocountCtrl = PSEUDOCOUNT;

        if (displayOption == MatrixType.NORM2) {
            renderNorm2(g, zd, isWholeGenome, observedNormalizationType, key, displayOption,
                    originX, originY, width, height);
        } else if (displayOption == MatrixType.NORM2CTRL) {
            renderNorm2(g, controlZD, isWholeGenome, controlNormalizationType, controlKey, displayOption,
                    originX, originY, width, height);
        } else if (displayOption == MatrixType.NORM2OBSVSCTRL) {
            if (controlDF == null) {
                System.err.println("Control DF is NULL");
                return false;
            }
            renderNorm2VS(g, zd, controlZD, isWholeGenome, observedNormalizationType,
                    controlNormalizationType, key, displayOption,
                    originX, originY, width, height);
        } else if (displayOption == MatrixType.PEARSON) {
            renderPearsons(g, zd, df, key, originX, originY, width, height);
        } else if (displayOption == MatrixType.PEARSONCTRL) {
            if (controlDF == null) {
                System.err.println("Control DF is NULL");
                return false;
            }
            renderPearsons(g, controlZD, controlDF, controlKey, originX, originY, width, height);
        } else if (displayOption == MatrixType.PEARSONVS) {

            if (controlDF == null) {
                System.err.println("Control DF is NULL");
                return false;
            }
            renderPearsonsVS(g, zd, controlZD, df, controlDF, key, originX, originY, width, height);
        } else if (displayOption == MatrixType.CONTROL) {
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (controlZD == null || ctrlBlocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

            renderSimpleMap(g, ctrlBlocks, cs, width, height, sameChr, originX, originY);
        } else if (displayOption == MatrixType.LOGC) {
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (controlZD == null || ctrlBlocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

            renderSimpleLogMap(g, ctrlBlocks, cs, width, height, sameChr, originX, originY);
        } else if (displayOption == MatrixType.OECTRLV2 || displayOption == MatrixType.OECTRL) {
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (controlZD == null || ctrlBlocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

            renderObservedOverExpectedMap(g, chr1, ctrlBlocks, controlDF, controlZD,
                    cs, sameChr, originX, originY, width, height, 0);
        } else if (displayOption == MatrixType.OECTRLP1V2 || displayOption == MatrixType.OECTRLP1) {
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (controlZD == null || ctrlBlocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

            renderObservedOverExpectedMap(g, chr1, ctrlBlocks, controlDF, controlZD,
                    cs, sameChr, originX, originY, width, height, pseudocountCtrl);
        } else if (displayOption == MatrixType.LOGCEO) {
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (controlZD == null || ctrlBlocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

            renderLogObservedBaseExpectedMap(g, chr1, ctrlBlocks, controlDF, controlZD, cs,
                    sameChr, originX, originY, width, height);
        } else if (displayOption == MatrixType.VS) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null && ctrlBlocks == null) return false;
            if (blocks.isEmpty() && ctrlBlocks.isEmpty()) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, ctrlBlocks, 1f);

            renderSimpleVSMap(g, blocks, ctrlBlocks, zd, controlZD, originX, originY, width, height, cs, sameChr);
        } else if (displayOption == MatrixType.LOGVS) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null && ctrlBlocks == null) return false;
            if (blocks.isEmpty() && ctrlBlocks.isEmpty()) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, ctrlBlocks, 1f);

            renderSimpleLogVSMap(g, blocks, ctrlBlocks, zd, controlZD, originX, originY, width, height, cs, sameChr);
        } else if (displayOption == MatrixType.OEVSV2 || displayOption == MatrixType.OEVS) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null && ctrlBlocks == null) return false;
            if (blocks.isEmpty() && ctrlBlocks.isEmpty()) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, ctrlBlocks, 1f);

            renderObservedOverExpectedVSMap(g, chr1, blocks, ctrlBlocks, df, controlDF,
                    zd, controlZD, cs, sameChr, originX, originY, width, height, 0, 0);
        } else if (displayOption == MatrixType.OEVSP1V2 || displayOption == MatrixType.OEVSP1) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null && ctrlBlocks == null) return false;
            if (blocks.isEmpty() && ctrlBlocks.isEmpty()) return false;

            ColorScale cs;
            if (blocks.isEmpty()) {
                cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, ctrlBlocks, 1f);
            } else {
                cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);
            }

            renderObservedOverExpectedVSMap(g, chr1, blocks, ctrlBlocks, df, controlDF,
                    zd, controlZD, cs, sameChr, originX, originY, width, height, pseudocountObs, pseudocountCtrl);
        } else if (displayOption == MatrixType.LOGEOVS) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null && ctrlBlocks == null) return false;
            if (blocks.isEmpty() && ctrlBlocks.isEmpty()) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, ctrlBlocks, 1f);

            renderLogObsOverExpVSMap(g, chr1, blocks, ctrlBlocks, df, controlDF,
                    zd, controlZD, cs, sameChr, originX, originY, width, height);

        } else if (displayOption == MatrixType.OCMEVS) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null && ctrlBlocks == null) return false;
            if (blocks.isEmpty() && ctrlBlocks.isEmpty()) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, ctrlBlocks, 1f);

            renderLogObsMinusExpVSMap(g, chr1, blocks, ctrlBlocks, df, controlDF,
                    zd, controlZD, cs, sameChr, originX, originY, width, height);

        } else if (displayOption == MatrixType.EXPECTED) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderExpectedMap(g, zd, df, sameChr, cs, originX, originY, width, height, chr1);
        } else if (displayOption == MatrixType.OEV2 || displayOption == MatrixType.OE) {

            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderObservedOverExpectedMap(g, chr1, blocks, df, zd,
                    cs, sameChr, originX, originY, width, height, 0);

        } else if (displayOption == MatrixType.OEP1V2 || displayOption == MatrixType.OEP1) {

            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;

            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderObservedOverExpectedMap(g, chr1, blocks, df, zd,
                    cs, sameChr, originX, originY, width, height, pseudocountObs);
        } else if (displayOption == MatrixType.LOGEO) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderLogObservedBaseExpectedMap(g, chr1, blocks, df, zd,
                    cs, sameChr, originX, originY, width, height);
        } else if (displayOption == MatrixType.EXPLOGEO) {

            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;

            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderNewBaseEMap(g, chr1, blocks, df, zd,
                    cs, sameChr, originX, originY, width, height);

        } else if (displayOption == MatrixType.EXPLOGCEO) {
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (ctrlBlocks == null || controlZD == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, ctrlBlocks, 1f);
            renderNewBaseEMap(g, chr1, ctrlBlocks, controlDF, controlZD,
                    cs, sameChr, originX, originY, width, height);

        } else if (displayOption == MatrixType.OERATIOV2 || displayOption == MatrixType.OERATIO) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null) return false;
            if (controlZD == null) return false;
            if (sameChr && (df == null || controlDF == null)) return false;

            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderOERatioMap(g, blocks, ctrlBlocks, zd, controlZD, df, controlDF, originX, originY,
                    width, height, 0, 0, cs, sameChr, controlNormalizationType, chr1);

        } else if (displayOption == MatrixType.OERATIOP1V2 || displayOption == MatrixType.OERATIOP1) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null) return false;
            if (controlZD == null) return false;
            if (sameChr && (df == null || controlDF == null)) return false;

            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderOERatioMap(g, blocks, ctrlBlocks, zd, controlZD, df, controlDF, originX, originY,
                    width, height, pseudocountObs, pseudocountCtrl, cs, sameChr, controlNormalizationType, chr1);

        } else if (displayOption == MatrixType.LOGEORATIOV2 || displayOption == MatrixType.LOGEORATIO) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            if (sameChr && (df == null || controlDF == null)) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderLogRatioWithExpMap(g, blocks, ctrlBlocks, zd, controlZD, chr1, df, controlDF,
                    originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.OERATIOMINUS) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            if (sameChr && (df == null || controlDF == null)) return false;

            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderOERatioMinus(g, blocks, ctrlBlocks, zd, controlZD, df, controlDF, chr1,
                    0, 0, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);

        } else if (displayOption == MatrixType.OERATIOMINUSP1) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            if (sameChr && (df == null || controlDF == null)) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderOERatioMinus(g, blocks, ctrlBlocks, zd, controlZD, df, controlDF, chr1,
                    pseudocountObs, pseudocountCtrl, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.RATIOV2 || displayOption == MatrixType.RATIO) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderRatioWithAvgMap(g, blocks, ctrlBlocks, zd, controlZD,
                    0, 0, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.RATIOP1V2 || displayOption == MatrixType.RATIOP1) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderRatioWithAvgMap(g, blocks, ctrlBlocks, zd, controlZD,
                    pseudocountObs, pseudocountCtrl, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.LOGRATIOV2 || displayOption == MatrixType.LOGRATIO) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderLogRatioWithAvgMap(g, blocks, ctrlBlocks, zd, controlZD,
                    originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.RATIO0V2 || displayOption == MatrixType.RATIO0) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            if (sameChr && (df == null || controlDF == null)) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderRatioWithExpMap(g, blocks, ctrlBlocks, zd, controlZD, chr1, df, controlDF,
                    0, 0, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.RATIO0P1V2 || displayOption == MatrixType.RATIO0P1) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            if (sameChr && (df == null || controlDF == null)) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderRatioWithExpMap(g, blocks, ctrlBlocks, zd, controlZD, chr1, df, controlDF,
                    pseudocountObs, pseudocountCtrl, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.DIFF) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;

            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderDiffMap(g, blocks, ctrlBlocks, zd, controlZD, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.LOG) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;

            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderSimpleLogMap(g, blocks, cs, width, height, sameChr, originX, originY);
        } else if (displayOption == MatrixType.OBSERVED) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderSimpleMap(g, blocks, cs, width, height, sameChr, originX, originY);

        } else {
            System.err.println("Invalid display option: " + displayOption);
            return false;
        }
        return true;
    }

    private void renderLogRatioWithExpMap(Graphics2D g, List<Block> blocks, List<Block> ctrlBlocks,
                                          MatrixZoomData zd, MatrixZoomData controlZD, int chr1,
                                          ExpectedValueFunction df, ExpectedValueFunction controlDF,
                                          int originX, int originY, int width, int height,
                                          ColorScale cs, boolean sameChr, NormalizationType controlNormalizationType) {
        Map<String, Block> controlBlocks = new HashMap<>();
        for (Block b : ctrlBlocks) {
            controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
        }


        float averageCount = (float) zd.getAverageCount();
        float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

        if (sameChr) {

            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {

                    Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

                    for (ContactRecord rec : recs) {
                        ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                        if (ctrlRecord != null) {
                            float num = rec.getCounts();
                            float den = ctrlRecord.getCounts();

                            int binX = rec.getBinX();
                            int binY = rec.getBinY();
                            int dist = Math.abs(binX - binY);

                            float obsExpected = (float) df.getExpectedValue(chr1, dist);
                            float ctrlExpected = (float) controlDF.getExpectedValue(chr1, dist);

                            if (ratioPainting2(g, cs, num, den, obsExpected, ctrlExpected)) continue;

                            intraPainting(g, originX, originY, width, height, binX, binY);
                        }
                    }
                }
            }
        } else {
            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {

                    Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

                    float obsExpected = (averageCount > 0 ? averageCount : 1);
                    float ctrlExpected = (ctrlAverageCount > 0 ? ctrlAverageCount : 1);

                    for (ContactRecord rec : recs) {
                        ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                        if (ctrlRecord != null) {
                            float num = rec.getCounts();
                            float den = ctrlRecord.getCounts();

                            int binX = rec.getBinX();
                            int binY = rec.getBinY();

                            if (ratioPainting2(g, cs, num, den, obsExpected, ctrlExpected)) continue;

                            aboveDiagonalPainting(g, originX, originY, width, height, binX, binY);
                        }
                    }
                }
            }
        }
    }

    private boolean ratioPainting2(Graphics2D g, ColorScale cs, float num, float den, float obsExpected, float ctrlExpected) {
        float score = (float) ((Math.log(num + 1) / Math.log(obsExpected + 1)) / (Math.log(den + 1) / Math.log(ctrlExpected + 1)));
        if (Float.isNaN(score) || Float.isInfinite(score)) return true;

        Color color = cs.getColor(score);
        g.setColor(color);
        return false;
    }

    private void renderLogRatioWithAvgMap(Graphics2D g, List<Block> blocks, List<Block> ctrlBlocks,
                                          MatrixZoomData zd, MatrixZoomData controlZD,
                                          int originX, int originY, int width, int height,
                                          ColorScale cs, boolean sameChr, NormalizationType controlNormalizationType) {
        float averageCount = (float) zd.getAverageCount();
        float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

        Map<String, Block> controlBlocks = new HashMap<>();
        for (Block b : ctrlBlocks) {
            controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
        }

        for (Block b : blocks) {
            Collection<ContactRecord> recs = b.getContactRecords();

            Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

            if (recs != null) {
                for (ContactRecord rec : recs) {
                    ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                    if (ctrlRecord != null) {
                        float num = (float) Math.log(rec.getCounts() / averageCount + 1);
                        float den = (float) Math.log(ctrlRecord.getCounts() / ctrlAverageCount + 1);
                        ratioPainting(g, originX, originY, width, height, cs, sameChr, rec, num, den);
                    }
                }
            }
        }
    }

    private void renderDiffMap(Graphics2D g, List<Block> blocks, List<Block> ctrlBlocks,
                               MatrixZoomData zd, MatrixZoomData controlZD,
                               int originX, int originY, int width, int height, ColorScale cs,
                               boolean sameChr, NormalizationType controlNormalizationType) {
        float averageCount = (float) zd.getAverageCount();
        float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();
        float averageAcrossMapAndControl = (averageCount / 2f + ctrlAverageCount / 2f);

        Map<String, Block> controlBlocks = new HashMap<>();
        for (Block b : ctrlBlocks) {
            controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
        }

        for (Block b : blocks) {
            Collection<ContactRecord> recs = b.getContactRecords();

            Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

            if (recs != null) {
                for (ContactRecord rec : recs) {
                    ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                    if (ctrlRecord != null) {
                        float num = rec.getCounts() / averageCount;
                        float den = ctrlRecord.getCounts() / ctrlAverageCount;
                        float score = (num - den) * averageAcrossMapAndControl;
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                        intraPainting2(g, originX, originY, width, height, cs, sameChr, rec, score);
                    }
                }
            }
        }
    }

    private void renderRatioWithExpMap(Graphics2D g, List<Block> blocks, List<Block> ctrlBlocks,
                                       MatrixZoomData zd, MatrixZoomData controlZD,
                                       int chr1, ExpectedValueFunction df, ExpectedValueFunction controlDF,
                                       float pseudocountObs, float pseudocountCtrl,
                                       int originX, int originY, int width, int height,
                                       ColorScale cs, boolean sameChr, NormalizationType controlNormalizationType) {

        Map<String, Block> controlBlocks = new HashMap<>();
        for (Block b : ctrlBlocks) {
            controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
        }

        for (Block b : blocks) {
            Collection<ContactRecord> recs = b.getContactRecords();

            Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

            if (recs != null) {
                for (ContactRecord rec : recs) {
                    ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                    if (ctrlRecord != null) {
                        float num = (float) ((rec.getCounts() + pseudocountObs) / (df.getExpectedValue(chr1, 0) + pseudocountObs));
                        float den = (float) ((ctrlRecord.getCounts() + pseudocountCtrl) / (controlDF.getExpectedValue(chr1, 0) + pseudocountCtrl));
                        ratioPainting(g, originX, originY, width, height, cs, sameChr, rec, num, den);
                    }
                }
            }
        }
    }

    private void ratioPainting(Graphics2D g, int originX, int originY, int width, int height, ColorScale cs, boolean sameChr, ContactRecord rec, float num, float den) {
        float score = num / den;
        if (Float.isNaN(score) || Float.isInfinite(score)) return;

        intraPainting2(g, originX, originY, width, height, cs, sameChr, rec, score);
    }

    private void intraPainting2(Graphics2D g, int originX, int originY, int width, int height, ColorScale cs, boolean sameChr, ContactRecord rec, float score) {
        Color color = cs.getColor(score);
        g.setColor(color);

        int binX = rec.getBinX();
        int binY = rec.getBinY();
        aboveDiagonalPainting(g, originX, originY, width, height, binX, binY);

        belowDiagonalPainting(g, width, height, sameChr, originX, originY, binX, binY);
    }

    private void renderRatioWithAvgMap(Graphics2D g, List<Block> blocks, List<Block> ctrlBlocks,
                                       MatrixZoomData zd, MatrixZoomData controlZD,
                                       float pseudocountObs, float pseudocountCtrl,
                                       int originX, int originY, int width, int height,
                                       ColorScale cs, boolean sameChr, NormalizationType controlNormalizationType) {
        float averageCount = (float) zd.getAverageCount();
        float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

        Map<String, Block> controlBlocks = new HashMap<>();
        for (Block b : ctrlBlocks) {
            controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
        }

        for (Block b : blocks) {
            Collection<ContactRecord> recs = b.getContactRecords();

            Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

            if (recs != null) {
                for (ContactRecord rec : recs) {
                    ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                    if (ctrlRecord != null) {
                        float num = (rec.getCounts() + pseudocountObs) / (averageCount + pseudocountObs);
                        float den = (ctrlRecord.getCounts() + pseudocountCtrl) / (ctrlAverageCount + pseudocountCtrl);
                        ratioPainting(g, originX, originY, width, height, cs, sameChr, rec, num, den);
                    }
                }
            }
        }
    }

    private void renderOERatioMinus(Graphics2D g, List<Block> blocks, List<Block> ctrlBlocks,
                                    MatrixZoomData zd, MatrixZoomData controlZD,
                                    ExpectedValueFunction df, ExpectedValueFunction controlDF, int chr1,
                                    float pseudocountObs, float pseudocountCtrl, int originX, int originY,
                                    int width, int height, ColorScale cs, boolean sameChr,
                                    NormalizationType controlNormalizationType) {
        Map<String, Block> controlBlocks = new HashMap<>();
        for (Block b : ctrlBlocks) {
            controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
        }

        float averageCount = (float) zd.getAverageCount();
        float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

        if (sameChr) {
            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {

                    Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

                    for (ContactRecord rec : recs) {
                        ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                        if (ctrlRecord != null) {
                            float num = rec.getCounts() + pseudocountObs;
                            float den = ctrlRecord.getCounts() + pseudocountCtrl;

                            int binX = rec.getBinX();
                            int binY = rec.getBinY();
                            int dist = Math.abs(binX - binY);

                            float obsExpected = (float) df.getExpectedValue(chr1, dist) + pseudocountObs;
                            float ctrlExpected = (float) controlDF.getExpectedValue(chr1, dist) + pseudocountCtrl;

                            float score = (num / obsExpected) - (den / ctrlExpected);
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            Color color = cs.getColor(score);
                            g.setColor(color);

                            intraPainting(g, originX, originY, width, height, binX, binY);
                        }
                    }
                }
            }
        } else {
            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {

                    Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

                    float obsExpected = (averageCount > 0 ? averageCount : 1);
                    float ctrlExpected = (ctrlAverageCount > 0 ? ctrlAverageCount : 1);
                    obsExpected += pseudocountObs;
                    ctrlExpected += pseudocountCtrl;

                    for (ContactRecord rec : recs) {
                        ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                        if (ctrlRecord != null) {
                            float num = rec.getCounts() + pseudocountObs;
                            float den = ctrlRecord.getCounts() + pseudocountCtrl;

                            int binX = rec.getBinX();
                            int binY = rec.getBinY();

                            float score = (num / obsExpected) - (den / ctrlExpected);
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            Color color = cs.getColor(score);
                            g.setColor(color);

                            aboveDiagonalPainting(g, originX, originY, width, height, binX, binY);
                        }
                    }
                }
            }
        }
    }

    private Map<String, ContactRecord> linkRecords(MatrixZoomData zd, NormalizationType controlNormalizationType, Map<String, Block> controlBlocks, Block b) {
        Map<String, ContactRecord> controlRecords = new HashMap<>();
        Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
        if (cb != null) {
            for (ContactRecord ctrlRec : cb.getContactRecords()) {
                controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
            }
        }
        return controlRecords;
    }

    private void renderOERatioMap(Graphics2D g, List<Block> blocks, List<Block> ctrlBlocks,
                                  MatrixZoomData zd, MatrixZoomData controlZD,
                                  ExpectedValueFunction df, ExpectedValueFunction controlDF,
                                  int originX, int originY, int width, int height,
                                  float pseudocountObs, float pseudocountCtrl, ColorScale cs, boolean sameChr,
                                  NormalizationType controlNormalizationType, int chr1) {
        Map<String, Block> controlBlocks = new HashMap<>();
        for (Block b : ctrlBlocks) {
            controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
        }

        if (sameChr) {
            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {

                    Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

                    for (ContactRecord rec : recs) {
                        ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                        if (ctrlRecord != null) {
                            float num = rec.getCounts() + pseudocountObs;
                            float den = ctrlRecord.getCounts() + pseudocountCtrl;

                            int binX = rec.getBinX();
                            int binY = rec.getBinY();
                            int dist = Math.abs(binX - binY);

                            float obsExpected = (float) df.getExpectedValue(chr1, dist) + pseudocountObs;
                            float ctrlExpected = (float) controlDF.getExpectedValue(chr1, dist) + pseudocountCtrl;

                            float score = (num / obsExpected) / (den / ctrlExpected);
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            Color color = cs.getColor(score);
                            g.setColor(color);

                            intraPainting(g, originX, originY, width, height, binX, binY);
                        }
                    }
                }
            }
        } else {
            float averageCount = (float) zd.getAverageCount();
            float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();
            float obsExpected = (averageCount > 0 ? averageCount : 1);
            float ctrlExpected = (ctrlAverageCount > 0 ? ctrlAverageCount : 1);
            obsExpected += pseudocountObs;
            ctrlExpected += pseudocountCtrl;

            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {

                    Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

                    for (ContactRecord rec : recs) {
                        ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                        if (ctrlRecord != null) {
                            float num = rec.getCounts() + pseudocountObs;
                            float den = ctrlRecord.getCounts() + pseudocountCtrl;

                            int binX = rec.getBinX();
                            int binY = rec.getBinY();

                            float score = (num / obsExpected) / (den / ctrlExpected);
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            Color color = cs.getColor(score);
                            g.setColor(color);

                            aboveDiagonalPainting(g, originX, originY, width, height, binX, binY);
                        }
                    }
                }
            }
        }
    }

    private void renderNewBaseEMap(Graphics2D g, int chr1, List<Block> blocks, ExpectedValueFunction df, MatrixZoomData zd,
                                   ColorScale cs, boolean sameChr, int originX, int originY, int width, int height) {
        if (sameChr) {
            if (df != null) {
                for (Block b : blocks) {
                    Collection<ContactRecord> recs = b.getContactRecords();
                    if (recs != null) {
                        for (ContactRecord rec : recs) {
                            int binX = rec.getBinX();
                            int binY = rec.getBinY();
                            int dist = Math.abs(binX - binY);
                            float expected = (float) df.getExpectedValue(chr1, dist);

                            float score = (float) Math.exp((Math.log(rec.getCounts() + 1) / Math.log(expected + 1)));
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            Color color = cs.getColor(score);
                            g.setColor(color);

                            intraPainting(g, originX, originY, width, height, binX, binY);
                        }
                    }
                }
            }
        } else {
            float averageCount = (float) zd.getAverageCount();
            float expected = (averageCount > 0 ? averageCount : 1);
            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {

                    for (ContactRecord rec : recs) {
                        float score = (float) Math.exp((Math.log(rec.getCounts() + 1) / Math.log(expected + 1)));
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                        Color color = cs.getColor(score);
                        g.setColor(color);

                        interPainting(g, originX, originY, width, height, rec);
                    }
                }
            }
        }
    }

    private void renderExpectedMap(Graphics2D g, MatrixZoomData zd, ExpectedValueFunction df,
                                   boolean sameChr, ColorScale cs, int originX, int originY,
                                   int width, int height, int chr1) {
        if (sameChr) {
            if (df != null) {
                for (int px = 0; px <= width; px++) {
                    for (int py = 0; py <= height; py++) {
                        int binX = px + originX;
                        int binY = py + originY;

                        int dist = Math.abs(binX - binY);
                        float expected = (float) df.getExpectedValue(chr1, dist);
                        Color color = cs.getColor(expected);
                        g.setColor(color);
                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
                    }
                }
            }
        } else {
            float averageCount = (float) zd.getAverageCount();
            float expected = (averageCount > 0 ? averageCount : 1);
            Color color = cs.getColor(expected);
            g.setColor(color);
            for (int px = 0; px <= width; px++) {
                for (int py = 0; py <= height; py++) {
                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
                }
            }
        }
    }

    private void renderLogObsMinusExpVSMap(Graphics2D g, int chr1, List<Block> blocks, List<Block> ctrlBlocks,
                                           ExpectedValueFunction df, ExpectedValueFunction controlDF,
                                           MatrixZoomData zd, MatrixZoomData controlZD, ColorScale cs,
                                           boolean sameChr, int originX, int originY, int width, int height) {
        if (zd != null && df != null) {
            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {

                        float score = rec.getCounts();
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                        int binX = rec.getBinX();
                        int binY = rec.getBinY();
                        int px = binX - originX;
                        int py = binY - originY;

                        if (px > -1 && py > -1 && px <= width && py <= height) {
                            int dist = Math.abs(binX - binY);
                            float expected = (float) df.getExpectedValue(chr1, dist);
                            score = rec.getCounts() - expected;
                            Color color = cs.getColor(score);
                            g.setColor(color);
                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
                        }
                    }
                }
            }
        }
        if (sameChr && controlZD != null && controlDF != null) {
            for (Block b : ctrlBlocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {

                        float score = rec.getCounts();
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                        int binX = rec.getBinX();
                        int binY = rec.getBinY();

                        if (binX != binY) {
                            int dist = Math.abs(binX - binY);
                            float expected = (float) controlDF.getExpectedValue(chr1, dist);
                            score = rec.getCounts() - expected;

                            Color color = cs.getColor(score);
                            g.setColor(color);
                            aboveDiagonalPainting(g, originX, originY, width, height, binY, binX);
                        }
                    }
                }
            }
        }
    }

    private void renderLogObsOverExpVSMap(Graphics2D g, int chr1, List<Block> blocks, List<Block> ctrlBlocks,
                                          ExpectedValueFunction df, ExpectedValueFunction controlDF,
                                          MatrixZoomData zd, MatrixZoomData controlZD, ColorScale cs,
                                          boolean sameChr, int originX, int originY, int width, int height) {
        if (zd != null && df != null) {
            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {
                        int binX = rec.getBinX();
                        int binY = rec.getBinY();
                        int px = binX - originX;
                        int py = binY - originY;

                        if (px > -1 && py > -1 && px <= width && py <= height) {
                            int dist = Math.abs(binX - binY);
                            float expected = (float) df.getExpectedValue(chr1, dist);
                            float score = (float) (Math.log(rec.getCounts() + 1) / Math.log(expected + 1));
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;
                            Color color = cs.getColor(score);
                            g.setColor(color);
                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
                        }
                    }
                }
            }
        }
        if (sameChr && controlZD != null && controlDF != null) {
            for (Block b : ctrlBlocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {
                        int binX = rec.getBinX();
                        int binY = rec.getBinY();

                        if (binX != binY) {
                            int dist = Math.abs(binX - binY);
                            float expected = (float) controlDF.getExpectedValue(chr1, dist);
                            float score = (float) (Math.log(rec.getCounts() + 1) / Math.log(expected + 1));
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            Color color = cs.getColor(score);
                            g.setColor(color);
                            aboveDiagonalPainting(g, originX, originY, width, height, binY, binX);
                        }
                    }
                }
            }
        }
    }

    private void renderSimpleLogVSMap(Graphics2D g, List<Block> blocks, List<Block> ctrlBlocks,
                                      MatrixZoomData zd, MatrixZoomData controlZD,
                                      int originX, int originY, int width, int height, ColorScale cs, boolean sameChr) {

        float averageCount = (float) zd.getAverageCount();
        float ctrlAverageCount = (float) controlZD.getAverageCount();
        float averageAcrossMapAndControl = (averageCount + ctrlAverageCount) / 2;

        if (blocks != null) {
            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {

                        float score = (float) Math.log(averageAcrossMapAndControl * (rec.getCounts() / averageCount) + 1);
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                        int binX = rec.getBinX();
                        int binY = rec.getBinY();
                        int px = binX - originX;
                        int py = binY - originY;

                        Color color = cs.getColor(score);
                        g.setColor(color);

                        if (px > -1 && py > -1 && px <= width && py <= height) {
                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
                        }
                    }
                }
            }
        }
        if (sameChr && ctrlBlocks != null) {
            for (Block b : ctrlBlocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {

                        float score = (float) Math.log(averageAcrossMapAndControl * (rec.getCounts() / ctrlAverageCount) + 1);
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                        int binX = rec.getBinX();
                        int binY = rec.getBinY();

                        belowDiagonalPainting2(g, originX, originY, width, height, cs, score, binX, binY);
                    }
                }
            }
        }
    }

    private void belowDiagonalPainting2(Graphics2D g, int originX, int originY, int width, int height, ColorScale cs, float score, int binX, int binY) {
        if (binX != binY) {
            Color color = cs.getColor(score);
            g.setColor(color);
            int px = (binY - originX);
            int py = (binX - originY);
            if (px > -1 && py > -1 && px <= width && py <= height) {
                g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
            }
        }
    }

    private void renderSimpleVSMap(Graphics2D g, List<Block> blocks, List<Block> ctrlBlocks,
                                   MatrixZoomData zd, MatrixZoomData controlZD,
                                   int originX, int originY, int width, int height, ColorScale cs, boolean sameChr) {
        float averageCount = (float) zd.getAverageCount();
        float ctrlAverageCount = (float) controlZD.getAverageCount();
        float averageAcrossMapAndControl = (averageCount + ctrlAverageCount) / 2;

        if (blocks != null) {
            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {

                        float score = rec.getCounts();
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;
                        score = (score / averageCount) * averageAcrossMapAndControl;

                        int binX = rec.getBinX();
                        int binY = rec.getBinY();
                        int px = binX - originX;
                        int py = binY - originY;

                        Color color = cs.getColor(score);
                        g.setColor(color);

                        if (px > -1 && py > -1 && px <= width && py <= height) {
                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
                        }
                    }
                }
            }
        }
        if (sameChr && ctrlBlocks != null) {
            for (Block b : ctrlBlocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {

                        float score = rec.getCounts();
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;
                        score = (score / ctrlAverageCount) * averageAcrossMapAndControl;

                        int binX = rec.getBinX();
                        int binY = rec.getBinY();

                        belowDiagonalPainting2(g, originX, originY, width, height, cs, score, binX, binY);
                    }
                }
            }
        }
    }

    private void renderLogObservedBaseExpectedMap(Graphics2D g, int chrom, List<Block> blocks, ExpectedValueFunction df,
                                                  MatrixZoomData zd, ColorScale cs, boolean sameChr,
                                                  int originX, int originY, int width, int height) {
        if (sameChr) {
            if (df != null) {
                for (Block b : blocks) {
                    Collection<ContactRecord> recs = b.getContactRecords();
                    if (recs != null) {
                        for (ContactRecord rec : recs) {
                            int binX = rec.getBinX();
                            int binY = rec.getBinY();
                            int dist = Math.abs(binX - binY);
                            float expected = (float) df.getExpectedValue(chrom, dist);

                            float score = (float) (Math.log(rec.getCounts() + 1) / Math.log(expected + 1));
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            Color color = cs.getColor(score);
                            g.setColor(color);

                            intraPainting(g, originX, originY, width, height, binX, binY);
                        }
                    }
                }
            }
        } else {
            float averageCount = (float) zd.getAverageCount();
            float expected = (averageCount > 0 ? averageCount : 1);

            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {
                        float score = (float) (Math.log(rec.getCounts() + 1) / Math.log(expected + 1));
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                        Color color = cs.getColor(score);
                        g.setColor(color);

                        interPainting(g, originX, originY, width, height, rec);
                    }
                }
            }
        }
    }

    private void renderObservedOverExpectedVSMap(Graphics2D g, int chrom, List<Block> blocks, List<Block> ctrlBlocks,
                                                 ExpectedValueFunction df, ExpectedValueFunction controlDF,
                                                 MatrixZoomData zd, MatrixZoomData controlZD, ColorScale cs,
                                                 boolean sameChr, int originX, int originY, int width, int height,
                                                 float pseudocountObs, float pseudocountCtrl) {
        if (zd != null && blocks != null && df != null) {
            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {

                        float score = rec.getCounts();
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                        int binX = rec.getBinX();
                        int binY = rec.getBinY();
                        int px = binX - originX;
                        int py = binY - originY;

                        if (px > -1 && py > -1 && px <= width && py <= height) {
                            int dist = Math.abs(binX - binY);
                            float expected = (float) df.getExpectedValue(chrom, dist);
                            score = (rec.getCounts() + pseudocountObs) / (expected + pseudocountObs);
                            Color color = cs.getColor(score);
                            g.setColor(color);
                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
                        }
                    }
                }
            }
        }
        if (sameChr && controlZD != null && ctrlBlocks != null && controlDF != null) {
            for (Block b : ctrlBlocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {

                        float score = rec.getCounts();
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                        int binX = rec.getBinX();
                        int binY = rec.getBinY();

                        if (binX != binY) {
                            int dist = Math.abs(binX - binY);
                            float expected = (float) controlDF.getExpectedValue(chrom, dist);
                            score = (rec.getCounts() + pseudocountCtrl) / (expected + pseudocountCtrl);

                            Color color = cs.getColor(score);
                            g.setColor(color);
                            aboveDiagonalPainting(g, originX, originY, width, height, binY, binX);
                        }
                    }
                }
            }
        }
    }

    private void renderObservedOverExpectedMap(Graphics2D g, int chrom, List<Block> blocks, ExpectedValueFunction df,
                                               MatrixZoomData zd, ColorScale cs, boolean sameChr,
                                               int originX, int originY, int width, int height, float pseudocount) {
        if (sameChr) {
            if (df != null) {
                for (Block b : blocks) {
                    Collection<ContactRecord> recs = b.getContactRecords();
                    if (recs != null) {
                        for (ContactRecord rec : recs) {
                            int binX = rec.getBinX();
                            int binY = rec.getBinY();
                            int dist = Math.abs(binX - binY);
                            float expected = (float) df.getExpectedValue(chrom, dist);

                            float score = (rec.getCounts() + pseudocount) / (expected + pseudocount);
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            Color color = cs.getColor(score);
                            g.setColor(color);

                            intraPainting(g, originX, originY, width, height, binX, binY);
                        }
                    }
                }
            }
        } else {
            float averageCount = (float) zd.getAverageCount();
            float expected = (averageCount > 0 ? averageCount : 1);

            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {
                        float score = (rec.getCounts() + pseudocount) / (expected + pseudocount);
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                        Color color = cs.getColor(score);
                        g.setColor(color);

                        interPainting(g, originX, originY, width, height, rec);
                    }
                }
            }
        }
    }

    private void interPainting(Graphics2D g, int originX, int originY, int width, int height, ContactRecord rec) {
        int binX = rec.getBinX();
        int binY = rec.getBinY();
        aboveDiagonalPainting(g, originX, originY, width, height, binX, binY);
    }

    private void intraPainting(Graphics2D g, int originX, int originY, int width, int height, int binX, int binY) {
        aboveDiagonalPainting(g, originX, originY, width, height, binX, binY);
        int px;
        int py;

        if (binX != binY) {
            px = binY - originX;
            py = binX - originY;
            if (px > -1 && py > -1 && px <= width && py <= height) {
                g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
            }
        }
    }

    private void aboveDiagonalPainting(Graphics2D g, int originX, int originY, int width, int height, int binX, int binY) {
        int px = binX - originX;
        int py = binY - originY;
        if (px > -1 && py > -1 && px <= width && py <= height) {
            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
        }
    }

    private void renderSimpleMap(Graphics2D g, List<Block> blocks, ColorScale cs,
                                 int width, int height, boolean sameChr, int originX, int originY) {
        for (Block b : blocks) {
            Collection<ContactRecord> recs = b.getContactRecords();
            if (recs != null) {
                for (ContactRecord rec : recs) {
                    float score = rec.getCounts();
                    simpleColoring(g, cs, width, height, sameChr, originX, originY, rec, score);
                }
            }
        }
    }

    private void renderSimpleLogMap(Graphics2D g, List<Block> blocks, ColorScale cs,
                                    int width, int height, boolean sameChr, int originX, int originY) {
        for (Block b : blocks) {
            Collection<ContactRecord> recs = b.getContactRecords();
            if (recs != null) {
                for (ContactRecord rec : recs) {
                    float score = (float) Math.log(1 + rec.getCounts());
                    simpleColoring(g, cs, width, height, sameChr, originX, originY, rec, score);
                }
            }
        }
    }

    private void simpleColoring(Graphics2D g, ColorScale cs, int width, int height, boolean sameChr, int originX, int originY, ContactRecord rec, float score) {
        if (Float.isNaN(score) || Float.isInfinite(score)) return;

        int binX = rec.getBinX();
        int binY = rec.getBinY();
        int px = binX - originX;
        int py = binY - originY;

        Color color = cs.getColor(score);
        g.setColor(color);

        if (px > -1 && py > -1 && px <= width && py <= height) {
            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
        }

        belowDiagonalPainting(g, width, height, sameChr, originX, originY, binX, binY);
    }

    private void belowDiagonalPainting(Graphics2D g, int width, int height, boolean sameChr, int originX, int originY, int binX, int binY) {
        int px;
        int py;
        if (sameChr && binX != binY) {
            px = binY - originX;
            py = binX - originY;
            if (px > -1 && py > -1 && px <= width && py <= height) {
                g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
            }
        }
    }

    private void renderPearsonsVS(Graphics2D g, MatrixZoomData zd, MatrixZoomData controlZD,
                                  ExpectedValueFunction df, ExpectedValueFunction controlDF,
                                  String key, int originX, int originY,
                                  int width, int height) {
        BasicMatrix bm1 = zd.getPearsons(df);
        BasicMatrix bm2 = controlZD.getPearsons(controlDF);
        PearsonColorScale pearsonColorScale = colorScaleHandler.getPearsonColorScale();
        if (pearsonColorScale.doesNotContainKey(key)) {
            float min = Math.min(bm1.getLowerValue(), bm2.getLowerValue());
            float max = Math.max(bm1.getUpperValue(), bm2.getUpperValue());
            pearsonColorScale.setMinMax(key, min, max);
        }
        renderDenseMatrix(bm1, bm2, originX, originY, width, height, pearsonColorScale, key, g, null);
    }

    private void renderPearsons(Graphics2D g, MatrixZoomData zd, ExpectedValueFunction df,
                                String key, int originX, int originY, int width, int height) {
        BasicMatrix bm = zd.getPearsons(df);
        PearsonColorScale pearsonColorScale = colorScaleHandler.getPearsonColorScale();
        if (pearsonColorScale.doesNotContainKey(key)) {
            pearsonColorScale.setMinMax(key, bm.getLowerValue(), bm.getUpperValue());
        }
        renderDenseMatrix(bm, null, originX, originY, width, height, pearsonColorScale, key, g, null);
    }

    private void renderNorm2VS(Graphics2D g, MatrixZoomData zd, MatrixZoomData controlZD,
                               boolean isWholeGenome, NormalizationType observedNormalizationType,
                               NormalizationType controlNormalizationType, String key,
                               MatrixType displayOption, int originX, int originY, int width, int height) {
        BasicMatrix bm1 = zd.getNormSquared(observedNormalizationType);
        BasicMatrix bm2 = controlZD.getNormSquared(controlNormalizationType);

        double percentile = isWholeGenome ? 99 : 95;
        float max = colorScaleHandler.computePercentile(bm1, bm2, percentile);

        ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, null, max);

        renderDenseMatrix(bm1, bm2, originX, originY, width, height, null, key, g, cs);
    }

    private void renderNorm2(Graphics2D g, MatrixZoomData zd, boolean isWholeGenome,
                             NormalizationType normType, String key, MatrixType displayOption,
                             int originX, int originY, int width, int height) {
        BasicMatrix bm = zd.getNormSquared(normType);
        double percentile = isWholeGenome ? 99 : 95;
        float max = colorScaleHandler.computePercentile(bm, percentile);
        ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, null, max);
        renderDenseMatrix(bm, null, originX, originY, width, height, null, key, g, cs);
    }

    private List<Block> getTheBlocks(MatrixZoomData zd, int x, int y, int maxX, int maxY, NormalizationType normType, boolean isImportant) {
        List<Block> blocks = null;
        if (zd != null) {
            try {
                blocks = zd.getNormalizedBlocksOverlapping(x, y, maxX, maxY, normType, isImportant, false);
            } catch (Exception ee) {
                if (HiCGlobals.printVerboseComments) ee.printStackTrace();
            }
        }
        return blocks;
    }


    /**
     * Render a dense matrix. Used for Pearsons correlation.  The bitmap is drawn at 1 data point
     * per pixel, scaling happens elsewhere.
     *
     * @param bm1        Matrix to render
     * @param bm2        Matrix to render
     * @param originX    origin in pixels
     * @param originY    origin in pixels
     * @param colorScale color scale to apply
     * @param key        id for view
     * @param g          graphics to render matrix into
     */
    private void renderDenseMatrix(BasicMatrix bm1, BasicMatrix bm2, int originX, int originY, int width, int height,
                                   PearsonColorScale colorScale, String key, Graphics2D g, ColorScale cs) {
        int endX = Math.min(originX + width, bm1.getColumnDimension());
        int endY = Math.min(originY + height, bm1.getRowDimension());

        // TODO -- need to check bounds before drawing
        for (int row = originY; row < endY; row++) {
            for (int col = originX; col < endX; col++) {

                float score = bm1.getEntry(row, col);
                Color color = colorScaleHandler.getDenseMatrixColor(key, score, colorScale, cs);
                int px = col - originX;
                int py = row - originY;
                g.setColor(color);

                g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
                // Assuming same chromosome
                if (col != row) {
                    if (bm2 != null) {
                        float controlScore = bm2.getEntry(row, col);
                        Color controlColor = colorScaleHandler.getDenseMatrixColor(key, controlScore, colorScale, cs);
                        px = row - originX;
                        py = col - originY;
                        g.setColor(controlColor);
                    } else {
                        px = row - originX;
                        py = col - originY;
                    }
                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
                }
            }
        }
    }

    public void reset() {
        colorScaleHandler.reset();
    }

    public PearsonColorScale getPearsonColorScale() {
        return colorScaleHandler.getPearsonColorScale();
    }

    public void setNewDisplayRange(MatrixType displayOption, double min, double max, String key) {
        colorScaleHandler.setNewDisplayRange(displayOption, min, max, key);
    }

    public void updateColorSliderFromColorScale(SuperAdapter superAdapter, MatrixType displayOption, String cacheKey) {
        colorScaleHandler.updateColorSliderFromColorScale(superAdapter, displayOption, cacheKey);
    }
}
