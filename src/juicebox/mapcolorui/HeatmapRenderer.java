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
import java.util.List;
import java.util.*;

/**
 * @author jrobinso
 * @since Aug 11, 2010
 */
public class HeatmapRenderer {

    public static float PSEUDO_COUNT = 1f;
    protected static final int PIXEL_WIDTH = 1, PIXEL_HEIGHT = 1;
    private final ColorScaleHandler colorScaleHandler;
    private final Graphics2D g;

    public HeatmapRenderer(Graphics2D g, ColorScaleHandler colorScaleHandler) {
        this.g = g;
        this.colorScaleHandler = colorScaleHandler;
    }

    public static String getColorScaleCacheKey(MatrixZoomData zd, MatrixType displayOption, NormalizationType obsNorm, NormalizationType ctrlNorm) {
        return zd.getColorScaleKey(displayOption, obsNorm, ctrlNorm);
    }

    @SuppressWarnings("SuspiciousNameCombination")
    public boolean render(int originX, int originY, int width, int height,
                          final MatrixZoomData zd, final MatrixZoomData controlZD,
                          final MatrixType displayOption,
                          final NormalizationType observedNormalizationType, final NormalizationType controlNormalizationType,
                          final ExpectedValueFunction df, final ExpectedValueFunction controlDF,
                          boolean isImportant) {
        if (g != null) {
            g.setRenderingHint(RenderingHints.KEY_COLOR_RENDERING, RenderingHints.VALUE_COLOR_RENDER_SPEED);
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);
        }

        int chr1 = zd.getChr1Idx();
        int chr2 = zd.getChr2Idx();
        int x = originX;
        int y = originY;

        boolean isWholeGenome = chr1 == 0 && chr2 == 0;
        boolean sameChr = (chr1 == chr2);

        if (sameChr) {
            // transpose if necessary; convention is to use upper diagonal
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

        float pseudoCountObs = PSEUDO_COUNT;
        float pseudoCountCtrl = PSEUDO_COUNT;

        if (displayOption == MatrixType.NORM2) {
            renderNorm2(zd, isWholeGenome, observedNormalizationType, key, displayOption,
                    originX, originY, width, height);
        } else if (displayOption == MatrixType.NORM2CTRL) {
            renderNorm2(controlZD, isWholeGenome, controlNormalizationType, controlKey, displayOption,
                    originX, originY, width, height);
        } else if (displayOption == MatrixType.NORM2OBSVSCTRL) {
            if (controlDF == null) {
                System.err.println("Control DF is NULL");
                return false;
            }
            renderNorm2VS(zd, controlZD, isWholeGenome, observedNormalizationType,
                    controlNormalizationType, key, displayOption,
                    originX, originY, width, height);
        } else if (displayOption == MatrixType.PEARSON) {
            renderPearson(zd, df, key, originX, originY, width, height);
        } else if (displayOption == MatrixType.PEARSONCTRL) {
            if (controlDF == null) {
                System.err.println("Control DF is NULL");
                return false;
            }
            renderPearson(controlZD, controlDF, controlKey, originX, originY, width, height);
        } else if (displayOption == MatrixType.PEARSONVS) {

            if (controlDF == null) {
                System.err.println("Control DF is NULL");
                return false;
            }
            renderPearsonVS(zd, controlZD, df, controlDF, key, originX, originY, width, height);
        } else if (displayOption == MatrixType.CONTROL) {
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (controlZD == null || ctrlBlocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

            renderSimpleMap(ctrlBlocks, cs, width, height, sameChr, originX, originY);
        } else if (displayOption == MatrixType.LOGC) {
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (controlZD == null || ctrlBlocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

            renderSimpleLogMap(ctrlBlocks, cs, width, height, sameChr, originX, originY);
        } else if (displayOption == MatrixType.OECTRLV2 || displayOption == MatrixType.OECTRL) {
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (controlZD == null || ctrlBlocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

            renderObservedOverExpectedMap(chr1, ctrlBlocks, controlDF, controlZD,
                    cs, sameChr, originX, originY, width, height, 0);
        } else if (displayOption == MatrixType.OECTRLP1V2 || displayOption == MatrixType.OECTRLP1) {
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (controlZD == null || ctrlBlocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

            renderObservedOverExpectedMap(chr1, ctrlBlocks, controlDF, controlZD,
                    cs, sameChr, originX, originY, width, height, pseudoCountCtrl);
        } else if (displayOption == MatrixType.LOGCEO) {
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (controlZD == null || ctrlBlocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

            renderLogObservedBaseExpectedMap(chr1, ctrlBlocks, controlDF, controlZD, cs,
                    sameChr, originX, originY, width, height);
        } else if (displayOption == MatrixType.LOGEOVS || displayOption == MatrixType.OCMEVS ||
                displayOption == MatrixType.VS || displayOption == MatrixType.LOGVS ||
                displayOption == MatrixType.OEVSV2 || displayOption == MatrixType.OEVS ||
                displayOption == MatrixType.OEVSP1V2 || displayOption == MatrixType.OEVSP1 ||
                displayOption == MatrixType.OERATIOV2 || displayOption == MatrixType.OERATIO ||
                displayOption == MatrixType.OERATIOP1V2 || displayOption == MatrixType.OERATIOP1) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null) return false;
            if (blocks.isEmpty() && ctrlBlocks.isEmpty()) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, ctrlBlocks, 1f);

            if (displayOption == MatrixType.LOGEOVS) {
                renderLogObsOverExpVSMap(chr1, blocks, ctrlBlocks, df, controlDF,
                        zd, controlZD, cs, sameChr, originX, originY, width, height);
            } else if (displayOption == MatrixType.OCMEVS) { //
                renderLogObsMinusExpVSMap(chr1, blocks, ctrlBlocks, df, controlDF,
                        zd, controlZD, cs, sameChr, originX, originY, width, height);
            } else if (displayOption == MatrixType.VS) {
                renderSimpleVSMap(blocks, ctrlBlocks, zd, controlZD, originX, originY, width, height, cs, sameChr);
            } else if (displayOption == MatrixType.LOGVS) {
                renderSimpleLogVSMap(blocks, ctrlBlocks, zd, controlZD, originX, originY, width, height, cs, sameChr);
            } else if (displayOption == MatrixType.OEVSP1V2 || displayOption == MatrixType.OEVSP1) {
                renderObservedOverExpectedVSMap(chr1, blocks, ctrlBlocks, df, controlDF,
                        zd, controlZD, cs, sameChr, originX, originY, width, height, pseudoCountObs, pseudoCountCtrl);
            } else if (displayOption == MatrixType.OEVSV2 || displayOption == MatrixType.OEVS) {
                renderObservedOverExpectedVSMap(chr1, blocks, ctrlBlocks, df, controlDF,
                        zd, controlZD, cs, sameChr, originX, originY, width, height, 0, 0);
            } else if (displayOption == MatrixType.OERATIOV2 || displayOption == MatrixType.OERATIO) {
                if (controlZD == null) return false;
                if (sameChr && (df == null || controlDF == null)) return false;
                renderOERatioMap(blocks, ctrlBlocks, zd, controlZD, df, controlDF, originX, originY,
                        width, height, 0, 0, cs, sameChr, controlNormalizationType, chr1);
            } else if (displayOption == MatrixType.OERATIOP1V2 || displayOption == MatrixType.OERATIOP1) {
                if (controlZD == null) return false;
                if (sameChr && (df == null || controlDF == null)) return false;
                renderOERatioMap(blocks, ctrlBlocks, zd, controlZD, df, controlDF, originX, originY,
                        width, height, pseudoCountObs, pseudoCountCtrl, cs, sameChr, controlNormalizationType, chr1);
            }

        } else if (displayOption == MatrixType.EXPECTED) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderExpectedMap(zd, df, sameChr, cs, originX, originY, width, height, chr1);
        } else if (displayOption == MatrixType.OEV2 || displayOption == MatrixType.OE) {

            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderObservedOverExpectedMap(chr1, blocks, df, zd,
                    cs, sameChr, originX, originY, width, height, 0);

        } else if (displayOption == MatrixType.OEP1V2 || displayOption == MatrixType.OEP1) {

            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;

            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderObservedOverExpectedMap(chr1, blocks, df, zd,
                    cs, sameChr, originX, originY, width, height, pseudoCountObs);
        } else if (displayOption == MatrixType.LOGEO) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderLogObservedBaseExpectedMap(chr1, blocks, df, zd,
                    cs, sameChr, originX, originY, width, height);
        } else if (displayOption == MatrixType.EXPLOGEO) {

            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;

            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderNewBaseEMap(chr1, blocks, df, zd,
                    cs, sameChr, originX, originY, width, height);

        } else if (displayOption == MatrixType.EXPLOGCEO) {
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (ctrlBlocks == null || controlZD == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, ctrlBlocks, 1f);
            renderNewBaseEMap(chr1, ctrlBlocks, controlDF, controlZD,
                    cs, sameChr, originX, originY, width, height);

        } else if (displayOption == MatrixType.LOGEORATIOV2 || displayOption == MatrixType.LOGEORATIO) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            if (sameChr && (df == null || controlDF == null)) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderLogRatioWithExpMap(blocks, ctrlBlocks, zd, controlZD, chr1, df, controlDF,
                    originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.OERATIOMINUS) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            if (sameChr && (df == null || controlDF == null)) return false;

            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderOERatioMinus(blocks, ctrlBlocks, zd, controlZD, df, controlDF, chr1,
                    0, 0, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);

        } else if (displayOption == MatrixType.OERATIOMINUSP1) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            if (sameChr && (df == null || controlDF == null)) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderOERatioMinus(blocks, ctrlBlocks, zd, controlZD, df, controlDF, chr1,
                    pseudoCountObs, pseudoCountCtrl, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.RATIOV2 || displayOption == MatrixType.RATIO) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderRatioWithAvgMap(blocks, ctrlBlocks, zd, controlZD,
                    0, 0, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.RATIOP1V2 || displayOption == MatrixType.RATIOP1) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderRatioWithAvgMap(blocks, ctrlBlocks, zd, controlZD,
                    pseudoCountObs, pseudoCountCtrl, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.LOGRATIOV2 || displayOption == MatrixType.LOGRATIO) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderLogRatioWithAvgMap(blocks, ctrlBlocks, zd, controlZD,
                    originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.RATIO0V2 || displayOption == MatrixType.RATIO0) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            if (sameChr && (df == null || controlDF == null)) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderRatioWithExpMap(blocks, ctrlBlocks, zd, controlZD, chr1, df, controlDF,
                    0, 0, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.RATIO0P1V2 || displayOption == MatrixType.RATIO0P1) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;
            if (sameChr && (df == null || controlDF == null)) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderRatioWithExpMap(blocks, ctrlBlocks, zd, controlZD, chr1, df, controlDF,
                    pseudoCountObs, pseudoCountCtrl, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.DIFF) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
            if (blocks == null || ctrlBlocks == null || controlZD == null) return false;

            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderDiffMap(blocks, ctrlBlocks, zd, controlZD, originX, originY, width, height,
                    cs, sameChr, controlNormalizationType);
        } else if (displayOption == MatrixType.LOG) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;

            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderSimpleLogMap(blocks, cs, width, height, sameChr, originX, originY);
        } else if (displayOption == MatrixType.OBSERVED) {
            List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
            if (blocks == null) return false;
            ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

            renderSimpleMap(blocks, cs, width, height, sameChr, originX, originY);

        } else {
            System.err.println("Invalid display option: " + displayOption);
            return false;
        }
        return true;
    }

    private void renderLogRatioWithExpMap(List<Block> blocks, List<Block> ctrlBlocks,
                                          MatrixZoomData zd, MatrixZoomData controlZD, int chr1,
                                          ExpectedValueFunction df, ExpectedValueFunction controlDF,
                                          int originX, int originY, int width, int height,
                                          ColorScale cs, boolean sameChr, NormalizationType controlNormalizationType) {
        Map<String, Block> controlBlocks = convertBlockListToMap(ctrlBlocks, controlZD);

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

                            float obsExpected = getExpectedValue(df, chr1, rec);
                            float ctrlExpected = getExpectedValue(controlDF, chr1, rec);

                            if (logPainting(cs, num, den, obsExpected, ctrlExpected)) continue;

                            intraPainting(originX, originY, width, height, rec);
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

                            if (logPainting(cs, num, den, obsExpected, ctrlExpected)) continue;

                            aboveDiagonalPainting(originX, originY, width, height, rec);
                        }
                    }
                }
            }
        }
    }

    private Map<String, Block> convertBlockListToMap(List<Block> ctrlBlocks, MatrixZoomData controlZD) {
        Map<String, Block> controlBlocks = new HashMap<>();
        for (Block b : ctrlBlocks) {
            controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
        }
        return controlBlocks;
    }

    private void renderLogRatioWithAvgMap(List<Block> blocks, List<Block> ctrlBlocks,
                                          MatrixZoomData zd, MatrixZoomData controlZD,
                                          int originX, int originY, int width, int height,
                                          ColorScale cs, boolean sameChr, NormalizationType controlNormalizationType) {
        float averageCount = (float) zd.getAverageCount();
        float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

        Map<String, Block> controlBlocks = convertBlockListToMap(ctrlBlocks, controlZD);

        for (Block b : blocks) {
            Collection<ContactRecord> recs = b.getContactRecords();

            Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

            if (recs != null) {
                for (ContactRecord rec : recs) {
                    ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                    if (ctrlRecord != null) {
                        float num = (float) Math.log(rec.getCounts() / averageCount + 1);
                        float den = (float) Math.log(ctrlRecord.getCounts() / ctrlAverageCount + 1);
                        ratioPainting(originX, originY, width, height, cs, sameChr, rec, num, den);
                    }
                }
            }
        }
    }

    private void renderDiffMap(List<Block> blocks, List<Block> ctrlBlocks,
                               MatrixZoomData zd, MatrixZoomData controlZD,
                               int originX, int originY, int width, int height, ColorScale cs,
                               boolean sameChr, NormalizationType controlNormalizationType) {
        float averageCount = (float) zd.getAverageCount();
        float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();
        float averageAcrossMapAndControl = (averageCount / 2f + ctrlAverageCount / 2f);

        Map<String, Block> controlBlocks = convertBlockListToMap(ctrlBlocks, controlZD);

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
                        setColor(cs.getColor(score));
                        intraPainting2(originX, originY, width, height, sameChr, rec);
                    }
                }
            }
        }
    }

    private void renderRatioWithExpMap(List<Block> blocks, List<Block> ctrlBlocks,
                                       MatrixZoomData zd, MatrixZoomData controlZD,
                                       int chr1, ExpectedValueFunction df, ExpectedValueFunction controlDF,
                                       float pseudoCountObs, float pseudoCountCtrl,
                                       int originX, int originY, int width, int height,
                                       ColorScale cs, boolean sameChr, NormalizationType controlNormalizationType) {

        Map<String, Block> controlBlocks = convertBlockListToMap(ctrlBlocks, controlZD);

        for (Block b : blocks) {
            Collection<ContactRecord> recs = b.getContactRecords();

            Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

            if (recs != null) {
                for (ContactRecord rec : recs) {
                    ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                    if (ctrlRecord != null) {
                        float num = ((rec.getCounts() + pseudoCountObs) / (getExpectedValue(df, chr1, 0, 0) + pseudoCountObs));
                        float den = ((ctrlRecord.getCounts() + pseudoCountCtrl) / (getExpectedValue(controlDF, chr1, 0, 0) + pseudoCountCtrl));
                        ratioPainting(originX, originY, width, height, cs, sameChr, rec, num, den);
                    }
                }
            }
        }
    }

    private void renderRatioWithAvgMap(List<Block> blocks, List<Block> ctrlBlocks,
                                       MatrixZoomData zd, MatrixZoomData controlZD,
                                       float pseudoCountObs, float pseudoCountCtrl,
                                       int originX, int originY, int width, int height,
                                       ColorScale cs, boolean sameChr, NormalizationType controlNormalizationType) {
        float averageCount = (float) zd.getAverageCount();
        float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

        Map<String, Block> controlBlocks = convertBlockListToMap(ctrlBlocks, controlZD);

        for (Block b : blocks) {
            Collection<ContactRecord> recs = b.getContactRecords();

            Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

            if (recs != null) {
                for (ContactRecord rec : recs) {
                    ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                    if (ctrlRecord != null) {
                        float num = (rec.getCounts() + pseudoCountObs) / (averageCount + pseudoCountObs);
                        float den = (ctrlRecord.getCounts() + pseudoCountCtrl) / (ctrlAverageCount + pseudoCountCtrl);
                        ratioPainting(originX, originY, width, height, cs, sameChr, rec, num, den);
                    }
                }
            }
        }
    }

    private void renderOERatioMinus(List<Block> blocks, List<Block> ctrlBlocks,
                                    MatrixZoomData zd, MatrixZoomData controlZD,
                                    ExpectedValueFunction df, ExpectedValueFunction controlDF, int chr1,
                                    float pseudoCountObs, float pseudoCountCtrl, int originX, int originY,
                                    int width, int height, ColorScale cs, boolean sameChr,
                                    NormalizationType controlNormalizationType) {
        Map<String, Block> controlBlocks = convertBlockListToMap(ctrlBlocks, controlZD);

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
                            float num = rec.getCounts() + pseudoCountObs;
                            float den = ctrlRecord.getCounts() + pseudoCountCtrl;

                            float obsExpected = getExpectedValue(df, chr1, rec) + pseudoCountObs;
                            float ctrlExpected = getExpectedValue(controlDF, chr1, rec) + pseudoCountCtrl;

                            float score = (num / obsExpected) - (den / ctrlExpected);
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            setColor(cs.getColor(score));

                            intraPainting(originX, originY, width, height, rec);
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
                    obsExpected += pseudoCountObs;
                    ctrlExpected += pseudoCountCtrl;

                    for (ContactRecord rec : recs) {
                        ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                        if (ctrlRecord != null) {
                            float num = rec.getCounts() + pseudoCountObs;
                            float den = ctrlRecord.getCounts() + pseudoCountCtrl;

                            float score = (num / obsExpected) - (den / ctrlExpected);
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            setColor(cs.getColor(score));

                            aboveDiagonalPainting(originX, originY, width, height, rec);
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

    private void renderOERatioMap(List<Block> blocks, List<Block> ctrlBlocks,
                                  MatrixZoomData zd, MatrixZoomData controlZD,
                                  ExpectedValueFunction df, ExpectedValueFunction controlDF,
                                  int originX, int originY, int width, int height,
                                  float pseudoCountObs, float pseudoCountCtrl, ColorScale cs, boolean sameChr,
                                  NormalizationType controlNormalizationType, int chr1) {
        Map<String, Block> controlBlocks = convertBlockListToMap(ctrlBlocks, controlZD);

        if (sameChr) {
            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {

                    Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

                    for (ContactRecord rec : recs) {
                        ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                        if (ctrlRecord != null) {
                            float num = rec.getCounts() + pseudoCountObs;
                            float den = ctrlRecord.getCounts() + pseudoCountCtrl;

                            float obsExpected = getExpectedValue(df, chr1, rec) + pseudoCountObs;
                            float ctrlExpected = getExpectedValue(controlDF, chr1, rec) + pseudoCountCtrl;

                            float score = (num / obsExpected) / (den / ctrlExpected);
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            setColor(cs.getColor(score));

                            intraPainting(originX, originY, width, height, rec);
                        }
                    }
                }
            }
        } else {
            float averageCount = (float) zd.getAverageCount();
            float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();
            float obsExpected = (averageCount > 0 ? averageCount : 1);
            float ctrlExpected = (ctrlAverageCount > 0 ? ctrlAverageCount : 1);
            obsExpected += pseudoCountObs;
            ctrlExpected += pseudoCountCtrl;

            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {

                    Map<String, ContactRecord> controlRecords = linkRecords(zd, controlNormalizationType, controlBlocks, b);

                    for (ContactRecord rec : recs) {
                        ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                        if (ctrlRecord != null) {
                            float num = rec.getCounts() + pseudoCountObs;
                            float den = ctrlRecord.getCounts() + pseudoCountCtrl;

                            float score = (num / obsExpected) / (den / ctrlExpected);
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            setColor(cs.getColor(score));

                            aboveDiagonalPainting(originX, originY, width, height, rec);
                        }
                    }
                }
            }
        }
    }

    private void renderNewBaseEMap(int chr1, List<Block> blocks, ExpectedValueFunction df, MatrixZoomData zd,
                                   ColorScale cs, boolean sameChr, int originX, int originY, int width, int height) {
        if (sameChr) {
            if (df != null) {
                for (Block b : blocks) {
                    Collection<ContactRecord> recs = b.getContactRecords();
                    if (recs != null) {
                        for (ContactRecord rec : recs) {
                            float expected = getExpectedValue(df, chr1, rec);

                            float score = (float) Math.exp((Math.log(rec.getCounts() + 1) / Math.log(expected + 1)));
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            setColor(cs.getColor(score));

                            intraPainting(originX, originY, width, height, rec);
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

                        setColor(cs.getColor(score));

                        interPainting(originX, originY, width, height, rec);
                    }
                }
            }
        }
    }

    private void renderExpectedMap(MatrixZoomData zd, ExpectedValueFunction df,
                                   boolean sameChr, ColorScale cs, int originX, int originY,
                                   int width, int height, int chr1) {
        if (sameChr) {
            if (df != null) {
                for (int px = 0; px <= width; px++) {
                    for (int py = 0; py <= height; py++) {
                        int binX = px + originX;
                        int binY = py + originY;
                        float expected = getExpectedValue(df, chr1, binX, binY);
                        setColor(cs.getColor(expected));
                        directPixelPainting(px, py);
                    }
                }
            }
        } else {
            float averageCount = (float) zd.getAverageCount();
            float expected = (averageCount > 0 ? averageCount : 1);
            setColor(cs.getColor(expected));
            for (int px = 0; px <= width; px++) {
                for (int py = 0; py <= height; py++) {
                    directPixelPainting(px, py);
                }
            }
        }
    }

    private void renderLogObsMinusExpVSMap(int chr1, List<Block> blocks, List<Block> ctrlBlocks,
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

                        float expected = getExpectedValue(df, chr1, rec);
                        score = rec.getCounts() - expected;
                        setColor(cs.getColor(score));

                        aboveDiagonalPainting(originX, originY, width, height, rec);
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
                            float expected = getExpectedValue(controlDF, chr1, rec);
                            score = rec.getCounts() - expected;

                            setColor(cs.getColor(score));
                            belowDiagonalPainting(originX, originY, width, height, rec);
                        }
                    }
                }
            }
        }
    }

    private void renderLogObsOverExpVSMap(int chr1, List<Block> blocks, List<Block> ctrlBlocks,
                                          ExpectedValueFunction df, ExpectedValueFunction controlDF,
                                          MatrixZoomData zd, MatrixZoomData controlZD, ColorScale cs,
                                          boolean sameChr, int originX, int originY, int width, int height) {
        if (zd != null && df != null) {
            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {
                        float expected = getExpectedValue(df, chr1, rec);
                        float score = (float) (Math.log(rec.getCounts() + 1) / Math.log(expected + 1));
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;
                        setColor(cs.getColor(score));

                        aboveDiagonalPainting(originX, originY, width, height, rec);
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
                            float expected = getExpectedValue(controlDF, chr1, rec);
                            float score = (float) (Math.log(rec.getCounts() + 1) / Math.log(expected + 1));
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            setColor(cs.getColor(score));
                            belowDiagonalPainting(originX, originY, width, height, rec);
                        }
                    }
                }
            }
        }
    }

    private void renderSimpleLogVSMap(List<Block> blocks, List<Block> ctrlBlocks,
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

                        setColor(cs.getColor(score));

                        aboveDiagonalPainting(originX, originY, width, height, rec);
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

                        setColor(cs.getColor(score));

                        belowDiagonalPainting(originX, originY, width, height, rec);
                    }
                }
            }
        }
    }


    private void renderSimpleVSMap(List<Block> blocks, List<Block> ctrlBlocks,
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

                        setColor(cs.getColor(score));

                        aboveDiagonalPainting(originX, originY, width, height, rec);
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

                        setColor(cs.getColor(score));
                        belowDiagonalPainting(originX, originY, width, height, rec);
                    }
                }
            }
        }
    }

    private void renderLogObservedBaseExpectedMap(int chromosome, List<Block> blocks, ExpectedValueFunction df,
                                                  MatrixZoomData zd, ColorScale cs, boolean sameChr,
                                                  int originX, int originY, int width, int height) {
        if (sameChr) {
            if (df != null) {
                for (Block b : blocks) {
                    Collection<ContactRecord> recs = b.getContactRecords();
                    if (recs != null) {
                        for (ContactRecord rec : recs) {
                            float expected = getExpectedValue(df, chromosome, rec);

                            float score = (float) (Math.log(rec.getCounts() + 1) / Math.log(expected + 1));
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            setColor(cs.getColor(score));

                            intraPainting(originX, originY, width, height, rec);
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

                        setColor(cs.getColor(score));

                        interPainting(originX, originY, width, height, rec);
                    }
                }
            }
        }
    }

    private void renderObservedOverExpectedVSMap(int chromosome, List<Block> blocks, List<Block> ctrlBlocks,
                                                 ExpectedValueFunction df, ExpectedValueFunction controlDF,
                                                 MatrixZoomData zd, MatrixZoomData controlZD, ColorScale cs,
                                                 boolean sameChr, int originX, int originY, int width, int height,
                                                 float pseudoCountObs, float pseudoCountCtrl) {
        if (zd != null && blocks != null && df != null) {
            for (Block b : blocks) {
                Collection<ContactRecord> recs = b.getContactRecords();
                if (recs != null) {
                    for (ContactRecord rec : recs) {

                        float score = rec.getCounts();
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                        float expected = getExpectedValue(df, chromosome, rec);
                        score = (rec.getCounts() + pseudoCountObs) / (expected + pseudoCountObs);

                        setColor(cs.getColor(score));
                        aboveDiagonalPainting(originX, originY, width, height, rec);
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
                            float expected = getExpectedValue(controlDF, chromosome, rec);
                            score = (rec.getCounts() + pseudoCountCtrl) / (expected + pseudoCountCtrl);

                            setColor(cs.getColor(score));
                            belowDiagonalPainting(originX, originY, width, height, rec);
                        }
                    }
                }
            }
        }
    }

    private float getExpectedValue(ExpectedValueFunction df, int chromosome, ContactRecord record) {
        return getExpectedValue(df, chromosome, record.getBinX(), record.getBinY());
    }

    private float getExpectedValue(ExpectedValueFunction df, int chromosome, int binX, int binY) {
        int dist = Math.abs(binX - binY);
        return (float) df.getExpectedValue(chromosome, dist);
    }

    private void renderObservedOverExpectedMap(int chromosome, List<Block> blocks, ExpectedValueFunction df,
                                               MatrixZoomData zd, ColorScale cs, boolean sameChr,
                                               int originX, int originY, int width, int height, float pseudoCount) {
        if (sameChr) {
            if (df != null) {
                for (Block b : blocks) {
                    Collection<ContactRecord> recs = b.getContactRecords();
                    if (recs != null) {
                        for (ContactRecord rec : recs) {
                            float expected = getExpectedValue(df, chromosome, rec);

                            float score = (rec.getCounts() + pseudoCount) / (expected + pseudoCount);
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            setColor(cs.getColor(score));

                            intraPainting(originX, originY, width, height, rec);
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
                        float score = (rec.getCounts() + pseudoCount) / (expected + pseudoCount);
                        if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                        setColor(cs.getColor(score));

                        interPainting(originX, originY, width, height, rec);
                    }
                }
            }
        }
    }


    private void renderSimpleMap(List<Block> blocks, ColorScale cs,
                                 int width, int height, boolean sameChr, int originX, int originY) {
        for (Block b : blocks) {
            Collection<ContactRecord> recs = b.getContactRecords();
            if (recs != null) {
                for (ContactRecord rec : recs) {
                    float score = rec.getCounts();
                    simplePainting(cs, width, height, sameChr, originX, originY, rec, score);
                }
            }
        }
    }

    private void renderSimpleLogMap(List<Block> blocks, ColorScale cs,
                                    int width, int height, boolean sameChr, int originX, int originY) {
        for (Block b : blocks) {
            Collection<ContactRecord> recs = b.getContactRecords();
            if (recs != null) {
                for (ContactRecord rec : recs) {
                    float score = (float) Math.log(1 + rec.getCounts());
                    simplePainting(cs, width, height, sameChr, originX, originY, rec, score);
                }
            }
        }
    }

    private void renderPearsonVS(MatrixZoomData zd, MatrixZoomData controlZD,
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
        renderDenseMatrix(bm1, bm2, originX, originY, width, height, pearsonColorScale, key, null);
    }

    private void renderPearson(MatrixZoomData zd, ExpectedValueFunction df,
                               String key, int originX, int originY, int width, int height) {
        BasicMatrix bm = zd.getPearsons(df);
        PearsonColorScale pearsonColorScale = colorScaleHandler.getPearsonColorScale();
        if (pearsonColorScale.doesNotContainKey(key)) {
            pearsonColorScale.setMinMax(key, bm.getLowerValue(), bm.getUpperValue());
        }
        renderDenseMatrix(bm, null, originX, originY, width, height, pearsonColorScale, key, null);
    }

    private void renderNorm2VS(MatrixZoomData zd, MatrixZoomData controlZD,
                               boolean isWholeGenome, NormalizationType observedNormalizationType,
                               NormalizationType controlNormalizationType, String key,
                               MatrixType displayOption, int originX, int originY, int width, int height) {
        BasicMatrix bm1 = zd.getNormSquared(observedNormalizationType);
        BasicMatrix bm2 = controlZD.getNormSquared(controlNormalizationType);

        double percentile = isWholeGenome ? 99 : 95;
        float max = colorScaleHandler.computePercentile(bm1, bm2, percentile);

        ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, null, max);

        renderDenseMatrix(bm1, bm2, originX, originY, width, height, null, key, cs);
    }

    private void renderNorm2(MatrixZoomData zd, boolean isWholeGenome,
                             NormalizationType normType, String key, MatrixType displayOption,
                             int originX, int originY, int width, int height) {
        BasicMatrix bm = zd.getNormSquared(normType);
        double percentile = isWholeGenome ? 99 : 95;
        float max = colorScaleHandler.computePercentile(bm, percentile);
        ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, null, max);
        renderDenseMatrix(bm, null, originX, originY, width, height, null, key, cs);
    }

    private List<Block> getTheBlocks(MatrixZoomData zd, int x, int y, int maxX, int maxY, NormalizationType normType, boolean isImportant) {
        if (zd != null) {
            try {
                return zd.getNormalizedBlocksOverlapping(x, y, maxX, maxY, normType, isImportant, false);
            } catch (Exception ee) {
                if (HiCGlobals.printVerboseComments) ee.printStackTrace();
            }
        }
        return new ArrayList<>();
    }


    /**
     * Render a dense matrix. Used for Pearson correlation.  The bitmap is drawn at 1 data point
     * per pixel, scaling happens elsewhere.
     *
     * @param bm1        Matrix to render
     * @param bm2        Matrix to render
     * @param originX    origin in pixels
     * @param originY    origin in pixels
     * @param colorScale color scale to apply
     * @param key        id for view
     */
    private void renderDenseMatrix(BasicMatrix bm1, BasicMatrix bm2, int originX, int originY, int width, int height,
                                   PearsonColorScale colorScale, String key, ColorScale cs) {
        int endX = Math.min(originX + width, bm1.getColumnDimension());
        int endY = Math.min(originY + height, bm1.getRowDimension());

        // TODO -- need to check bounds before drawing
        for (int row = originY; row < endY; row++) {
            for (int col = originX; col < endX; col++) {

                float score = bm1.getEntry(row, col);
                Color color = colorScaleHandler.getDenseMatrixColor(key, score, colorScale, cs);
                setColor(color);

                directDensePainting(originX, originY, col, row);
                // Assuming same chromosome
                if (col != row) {
                    if (bm2 != null) {
                        float controlScore = bm2.getEntry(row, col);
                        Color controlColor = colorScaleHandler.getDenseMatrixColor(key, controlScore, colorScale, cs);
                        setColor(controlColor);
                    }
                    directDensePainting(originX, originY, row, col);
                }
            }
        }
    }

    public void updateColorSliderFromColorScale(SuperAdapter superAdapter, MatrixType displayOption, String cacheKey) {
        colorScaleHandler.updateColorSliderFromColorScale(superAdapter, displayOption, cacheKey);
    }

    private void interPainting(int originX, int originY, int width, int height, ContactRecord rec) {
        aboveDiagonalPainting(originX, originY, width, height, rec);
    }

    private void simplePainting(ColorScale cs, int width, int height, boolean sameChr, int originX, int originY, ContactRecord rec, float score) {
        if (Float.isNaN(score) || Float.isInfinite(score)) return;
        setColor(cs.getColor(score));

        aboveDiagonalPainting(originX, originY, width, height, rec);
        if (sameChr) belowDiagonalPainting(originX, originY, width, height, rec);
    }

    private boolean logPainting(ColorScale cs, float num, float den, float obsExpected, float ctrlExpected) {
        float score = (float) ((Math.log(num + 1) / Math.log(obsExpected + 1)) / (Math.log(den + 1) / Math.log(ctrlExpected + 1)));
        if (Float.isNaN(score) || Float.isInfinite(score)) return true;
        setColor(cs.getColor(score));
        return false;
    }

    private void ratioPainting(int originX, int originY, int width, int height, ColorScale cs, boolean sameChr, ContactRecord rec, float num, float den) {
        float score = num / den;
        if (Float.isNaN(score) || Float.isInfinite(score)) return;
        setColor(cs.getColor(score));
        intraPainting2(originX, originY, width, height, sameChr, rec);
    }

    private void intraPainting2(int originX, int originY, int width, int height, boolean sameChr, ContactRecord rec) {
        aboveDiagonalPainting(originX, originY, width, height, rec);
        if (sameChr) belowDiagonalPainting(originX, originY, width, height, rec);
    }

    private void intraPainting(int originX, int originY, int width, int height, ContactRecord rec) {
        aboveDiagonalPainting(originX, originY, width, height, rec);
        belowDiagonalPainting(originX, originY, width, height, rec);
    }

    @SuppressWarnings("SuspiciousNameCombination")
    private void belowDiagonalPainting(int originX, int originY, int width, int height, ContactRecord rec) {
        int binX = rec.getBinX();
        int binY = rec.getBinY();
        if (binX != binY) {
            actualDiagonalPainting(originX, originY, width, height, binY, binX);
        }
    }

    //justPainting(originX, originY, width, height, rec);
    private void aboveDiagonalPainting(int originX, int originY, int width, int height, ContactRecord rec) {
        actualDiagonalPainting(originX, originY, width, height, rec.getBinX(), rec.getBinY());
    }

    private void actualDiagonalPainting(int originX, int originY, int width, int height, int binX, int binY) {
        int px = binX - originX;
        int py = binY - originY;
        if (px > -1 && py > -1 && px <= width && py <= height) {
            directPixelPainting(px, py);
        }
    }

    private void directDensePainting(int originX, int originY, int binX, int binY) {
        int px = binX - originX;
        int py = binY - originY;
        directPixelPainting(px, py);
    }

    protected void setColor(Color color) {
        g.setColor(color);
    }

    protected void directPixelPainting(int px, int py) {
        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_HEIGHT);
    }

    public void translate(int x, int y) {
        g.translate(x, y);
    }

    public void scale(double sx, double sy) {
        g.scale(sx, sy);
    }

    public void drawImage(Image image, int xDest0, int yDest0, int xDest1, int yDest1, int xSrc0, int ySrc0, int xSrc1, int ySrc1) {
        g.drawImage(image, xDest0, yDest0, xDest1, yDest1, xSrc0, ySrc0, xSrc1, ySrc1, null);
    }

    public void drawRect(int x, int y, int width, int height) {
        g.drawRect(x, y, width, height);
    }
}