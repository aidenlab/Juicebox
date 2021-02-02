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
    private static final int PIXEL_WIDTH = 1;
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

        switch (displayOption) {
            case NORM2: {
                renderNorm2(g, zd, isWholeGenome, observedNormalizationType, key, displayOption,
                        originX, originY, width, height);
                break;
            }
            case NORM2CTRL: {
                renderNorm2(g, controlZD, isWholeGenome, controlNormalizationType, controlKey, displayOption,
                        originX, originY, width, height);
                break;
            }
            case NORM2OBSVSCTRL: {
                if (controlDF == null) {
                    System.err.println("Control DF is NULL");
                    return false;
                }
                renderNorm2VS(g, zd, controlZD, isWholeGenome, observedNormalizationType,
                        controlNormalizationType, key, controlKey, displayOption,
                        originX, originY, width, height);
                break;
            }
            case PEARSON: {
                renderPearsons(g, zd, df, key, originX, originY, width, height);
                break;
            }
            case PEARSONCTRL: {
                if (controlDF == null) {
                    System.err.println("Control DF is NULL");
                    return false;
                }
                renderPearsons(g, controlZD, controlDF, controlKey, originX, originY, width, height);
                break;
            }
            case PEARSONVS: {

                if (controlDF == null) {
                    System.err.println("Control DF is NULL");
                    return false;
                }
                renderPearsonsVS(g, zd, controlZD, df, controlDF, key, controlKey, originX, originY, width, height);
                break;
            }
            case CONTROL: {
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (controlZD == null || ctrlBlocks == null) return false;
                ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

                renderSimpleMap(g, ctrlBlocks, cs, width, height, sameChr, originX, originY);
                break;
            }
            case LOGC: {
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (controlZD == null || ctrlBlocks == null) return false;
                ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

                renderSimpleLogMap(g, ctrlBlocks, cs, width, height, sameChr, originX, originY);
                break;
            }
            case OECTRLV2:
            case OECTRL: {
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (controlZD == null || ctrlBlocks == null) return false;
                ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

                renderObservedOverExpectedMap(g, chr1, ctrlBlocks, controlDF, controlZD,
                        cs, sameChr, originX, originY, width, height, 0);
                break;
            }
            case OECTRLP1V2:
            case OECTRLP1: {
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (controlZD == null || ctrlBlocks == null) return false;
                ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

                renderObservedOverExpectedMap(g, chr1, ctrlBlocks, controlDF, controlZD,
                        cs, sameChr, originX, originY, width, height, pseudocountCtrl);
                break;
            }
            case LOGCEO: {
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (controlZD == null || ctrlBlocks == null) return false;
                ColorScale cs = colorScaleHandler.getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

                renderLogObservedBaseExpectedMap(g, chr1, ctrlBlocks, controlDF, controlZD, cs,
                        sameChr, originX, originY, width, height);
                break;
            }
            case VS: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null && ctrlBlocks == null) return false;
                if (blocks.isEmpty() && ctrlBlocks.isEmpty()) return false;
                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, ctrlBlocks, 1f);

                float averageCount = (float) zd.getAverageCount();
                float ctrlAverageCount = (float) controlZD.getAverageCount();
                float averageAcrossMapAndControl = (averageCount + ctrlAverageCount) / 2;

                if (zd != null && blocks != null) {
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
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }
                            }
                        }
                    }
                }
                if (sameChr && controlZD != null && ctrlBlocks != null) {
                    for (Block b : ctrlBlocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {
                            for (ContactRecord rec : recs) {

                                float score = rec.getCounts();
                                if (Float.isNaN(score) || Float.isInfinite(score)) continue;
                                score = (score / ctrlAverageCount) * averageAcrossMapAndControl;

                                int binX = rec.getBinX();
                                int binY = rec.getBinY();

                                if (binX != binY) {
                                    Color color = cs.getColor(score);
                                    g.setColor(color);
                                    int px = (binY - originX);
                                    int py = (binX - originY);
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case LOGVS: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null && ctrlBlocks == null) return false;
                if (blocks.isEmpty() && ctrlBlocks.isEmpty()) return false;
                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, ctrlBlocks, 1f);

                float averageCount = (float) zd.getAverageCount();
                float ctrlAverageCount = (float) controlZD.getAverageCount();
                float averageAcrossMapAndControl = (averageCount + ctrlAverageCount) / 2;

                if (zd != null && blocks != null) {
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
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }
                            }
                        }
                    }
                }
                if (sameChr && controlZD != null && ctrlBlocks != null) {
                    for (Block b : ctrlBlocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {
                            for (ContactRecord rec : recs) {

                                float score = (float) Math.log(averageAcrossMapAndControl * (rec.getCounts() / ctrlAverageCount) + 1);
                                if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                int binX = rec.getBinX();
                                int binY = rec.getBinY();

                                if (binX != binY) {
                                    Color color = cs.getColor(score);
                                    g.setColor(color);
                                    int px = (binY - originX);
                                    int py = (binX - originY);
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case OEVSV2:
            case OEVS: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null && ctrlBlocks == null) return false;
                if (blocks.isEmpty() && ctrlBlocks.isEmpty()) return false;
                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, ctrlBlocks, 1f);

                renderObservedOverExpectedVSMap(g, chr1, blocks, ctrlBlocks, df, controlDF,
                        zd, controlZD, cs, sameChr, originX, originY, width, height, 0, 0);
                break;
            }
            case OEVSP1V2:
            case OEVSP1: {
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
                break;
            }
            case LOGEOVS: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null && ctrlBlocks == null) return false;
                if (blocks.isEmpty() && ctrlBlocks.isEmpty()) return false;
                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, ctrlBlocks, 1f);

                if (zd != null && blocks != null && df != null) {
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
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
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
                                int binX = rec.getBinX();
                                int binY = rec.getBinY();

                                if (binX != binY) {
                                    int dist = Math.abs(binX - binY);
                                    float expected = (float) controlDF.getExpectedValue(chr1, dist);
                                    float score = (float) (Math.log(rec.getCounts() + 1) / Math.log(expected + 1));
                                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);
                                    int px = binY - originX;
                                    int py = binX - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case OCMEVS: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null && ctrlBlocks == null) return false;
                if (blocks.isEmpty() && ctrlBlocks.isEmpty()) return false;
                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, ctrlBlocks, 1f);

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
                                    float expected = (float) df.getExpectedValue(chr1, dist);
                                    score = rec.getCounts() - expected;
                                    Color color = cs.getColor(score);
                                    g.setColor(color);
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
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
                                    float expected = (float) controlDF.getExpectedValue(chr1, dist);
                                    score = rec.getCounts() - expected;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);
                                    int px = binY - originX;
                                    int py = binX - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case EXPECTED: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                if (blocks == null) return false;
                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

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
                                g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
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
                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                        }
                    }
                }
                break;
            }
            case OEV2:
            case OE: {

                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                if (blocks == null || zd == null) return false;

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);
                float averageCount = (float) zd.getAverageCount();

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

                                    float score = rec.getCounts() / expected;
                                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);

                                    int px = binX - originX;
                                    int py = binY - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }

                                    if (binX != binY) {
                                        px = binY - originX;
                                        py = binX - originY;
                                        if (px > -1 && py > -1 && px <= width && py <= height) {
                                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (Block b : blocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {
                            float expected = (averageCount > 0 ? averageCount : 1);
                            for (ContactRecord rec : recs) {
                                float score = rec.getCounts() / expected;
                                if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                Color color = cs.getColor(score);
                                g.setColor(color);

                                int binX = rec.getBinX();
                                int binY = rec.getBinY();
                                int px = binX - originX;
                                int py = binY - originY;
                                if (px > -1 && py > -1 && px <= width && py <= height) {
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }
                            }
                        }
                    }
                }

                break;
            }
            case OEP1V2:
            case OEP1: {

                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                if (blocks == null || zd == null) return false;

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);
                float averageCount = (float) zd.getAverageCount();

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

                                    float score = (rec.getCounts() + pseudocountObs) / (expected + pseudocountObs);
                                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);

                                    int px = binX - originX;
                                    int py = binY - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }

                                    if (binX != binY) {
                                        px = binY - originX;
                                        py = binX - originY;
                                        if (px > -1 && py > -1 && px <= width && py <= height) {
                                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (Block b : blocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {
                            float expected = (averageCount > 0 ? averageCount : 1);
                            for (ContactRecord rec : recs) {
                                float score = (rec.getCounts() + pseudocountObs) / (expected + pseudocountObs);
                                if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                Color color = cs.getColor(score);
                                g.setColor(color);

                                int binX = rec.getBinX();
                                int binY = rec.getBinY();
                                int px = binX - originX;
                                int py = binY - originY;
                                if (px > -1 && py > -1 && px <= width && py <= height) {
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }
                            }
                        }
                    }
                }

                break;
            }
            case LOGEO: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                if (blocks == null || zd == null) return false;
                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                renderLogObservedBaseExpectedMap(g, chr1, blocks, df, zd,
                        cs, sameChr, originX, originY, width, height);
                break;
            }
            case EXPLOGEO: {

                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                if (blocks == null || zd == null) return false;

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);
                float averageCount = (float) zd.getAverageCount();

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

                                    int px = binX - originX;
                                    int py = binY - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }

                                    if (binX != binY) {
                                        px = binY - originX;
                                        py = binX - originY;
                                        if (px > -1 && py > -1 && px <= width && py <= height) {
                                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (Block b : blocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {
                            float expected = (averageCount > 0 ? averageCount : 1);
                            for (ContactRecord rec : recs) {
                                float score = (float) Math.exp((Math.log(rec.getCounts() + 1) / Math.log(expected + 1)));
                                if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                Color color = cs.getColor(score);
                                g.setColor(color);

                                int binX = rec.getBinX();
                                int binY = rec.getBinY();
                                int px = binX - originX;
                                int py = binY - originY;
                                if (px > -1 && py > -1 && px <= width && py <= height) {
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }
                            }
                        }
                    }
                }

                break;
            }
            case OERATIOV2:
            case OERATIO: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null || ctrlBlocks == null || controlZD == null || zd == null) return false;

                Map<String, Block> controlBlocks = new HashMap<>();
                for (Block b : ctrlBlocks) {
                    controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
                }

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                float averageCount = (float) zd.getAverageCount();
                float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

                if (sameChr) {
                    if (df == null || controlDF == null) return false;

                    for (Block b : blocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {

                            Map<String, ContactRecord> controlRecords = new HashMap<>();
                            Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                            if (cb != null) {
                                for (ContactRecord ctrlRec : cb.getContactRecords()) {
                                    controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                                }
                            }

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

                                    float score = (num / obsExpected) / (den / ctrlExpected);
                                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);

                                    int px = binX - originX;
                                    int py = binY - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }

                                    if (binX != binY) {
                                        px = binY - originX;
                                        py = binX - originY;
                                        if (px > -1 && py > -1 && px <= width && py <= height) {
                                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (Block b : blocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {

                            Map<String, ContactRecord> controlRecords = new HashMap<>();
                            Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                            if (cb != null) {
                                for (ContactRecord ctrlRec : cb.getContactRecords()) {
                                    controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                                }
                            }

                            float obsExpected = (averageCount > 0 ? averageCount : 1);
                            float ctrlExpected = (ctrlAverageCount > 0 ? ctrlAverageCount : 1);

                            for (ContactRecord rec : recs) {
                                ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                                if (ctrlRecord != null) {
                                    float num = rec.getCounts();
                                    float den = ctrlRecord.getCounts();

                                    int binX = rec.getBinX();
                                    int binY = rec.getBinY();

                                    float score = (num / obsExpected) / (den / ctrlExpected);
                                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);

                                    int px = binX - originX;
                                    int py = binY - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case OERATIOP1V2:
            case OERATIOP1: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null || ctrlBlocks == null || controlZD == null || zd == null) return false;

                Map<String, Block> controlBlocks = new HashMap<>();
                for (Block b : ctrlBlocks) {
                    controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
                }

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                float averageCount = (float) zd.getAverageCount();
                float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

                if (sameChr) {
                    if (df == null || controlDF == null) return false;

                    for (Block b : blocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {

                            Map<String, ContactRecord> controlRecords = new HashMap<>();
                            Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                            if (cb != null) {
                                for (ContactRecord ctrlRec : cb.getContactRecords()) {
                                    controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                                }
                            }

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

                                    float score = ((num + pseudocountObs) / (obsExpected + pseudocountObs)) / ((den + pseudocountCtrl) / (ctrlExpected + pseudocountCtrl));
                                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);

                                    int px = binX - originX;
                                    int py = binY - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }

                                    if (binX != binY) {
                                        px = binY - originX;
                                        py = binX - originY;
                                        if (px > -1 && py > -1 && px <= width && py <= height) {
                                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (Block b : blocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {

                            Map<String, ContactRecord> controlRecords = new HashMap<>();
                            Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                            if (cb != null) {
                                for (ContactRecord ctrlRec : cb.getContactRecords()) {
                                    controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                                }
                            }

                            float obsExpected = (averageCount > 0 ? averageCount : 1);
                            float ctrlExpected = (ctrlAverageCount > 0 ? ctrlAverageCount : 1);

                            for (ContactRecord rec : recs) {
                                ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                                if (ctrlRecord != null) {
                                    float num = rec.getCounts();
                                    float den = ctrlRecord.getCounts();

                                    int binX = rec.getBinX();
                                    int binY = rec.getBinY();

                                    float score = ((num + pseudocountObs) / (obsExpected + pseudocountObs)) / ((den + pseudocountCtrl) / (ctrlExpected + pseudocountCtrl));
                                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);

                                    int px = binX - originX;
                                    int py = binY - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case LOGEORATIOV2:
            case LOGEORATIO: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null || ctrlBlocks == null || controlZD == null || zd == null) return false;

                Map<String, Block> controlBlocks = new HashMap<>();
                for (Block b : ctrlBlocks) {
                    controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
                }

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                float averageCount = (float) zd.getAverageCount();
                float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

                if (sameChr) {
                    if (df == null || controlDF == null) return false;

                    for (Block b : blocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {

                            Map<String, ContactRecord> controlRecords = new HashMap<>();
                            Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                            if (cb != null) {
                                for (ContactRecord ctrlRec : cb.getContactRecords()) {
                                    controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                                }
                            }

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

                                    float score = (float) ((Math.log(num + 1) / Math.log(obsExpected + 1)) / (Math.log(den + 1) / Math.log(ctrlExpected + 1)));
                                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);

                                    int px = binX - originX;
                                    int py = binY - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }

                                    if (binX != binY) {
                                        px = binY - originX;
                                        py = binX - originY;
                                        if (px > -1 && py > -1 && px <= width && py <= height) {
                                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (Block b : blocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {

                            Map<String, ContactRecord> controlRecords = new HashMap<>();
                            Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                            if (cb != null) {
                                for (ContactRecord ctrlRec : cb.getContactRecords()) {
                                    controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                                }
                            }

                            float obsExpected = (averageCount > 0 ? averageCount : 1);
                            float ctrlExpected = (ctrlAverageCount > 0 ? ctrlAverageCount : 1);

                            for (ContactRecord rec : recs) {
                                ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                                if (ctrlRecord != null) {
                                    float num = rec.getCounts();
                                    float den = ctrlRecord.getCounts();

                                    int binX = rec.getBinX();
                                    int binY = rec.getBinY();

                                    float score = (float) ((Math.log(num + 1) / Math.log(obsExpected + 1)) / (Math.log(den + 1) / Math.log(ctrlExpected + 1)));
                                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);

                                    int px = binX - originX;
                                    int py = binY - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case OERATIOMINUS: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null || ctrlBlocks == null || controlZD == null || zd == null) return false;

                Map<String, Block> controlBlocks = new HashMap<>();
                for (Block b : ctrlBlocks) {
                    controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
                }

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                float averageCount = (float) zd.getAverageCount();
                float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

                if (sameChr) {
                    if (df == null || controlDF == null) return false;

                    for (Block b : blocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {

                            Map<String, ContactRecord> controlRecords = new HashMap<>();
                            Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                            if (cb != null) {
                                for (ContactRecord ctrlRec : cb.getContactRecords()) {
                                    controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                                }
                            }

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

                                    float score = (num / obsExpected) - (den / ctrlExpected);
                                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);

                                    int px = binX - originX;
                                    int py = binY - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }

                                    if (binX != binY) {
                                        px = binY - originX;
                                        py = binX - originY;
                                        if (px > -1 && py > -1 && px <= width && py <= height) {
                                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (Block b : blocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {

                            Map<String, ContactRecord> controlRecords = new HashMap<>();
                            Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                            if (cb != null) {
                                for (ContactRecord ctrlRec : cb.getContactRecords()) {
                                    controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                                }
                            }

                            float obsExpected = (averageCount > 0 ? averageCount : 1);
                            float ctrlExpected = (ctrlAverageCount > 0 ? ctrlAverageCount : 1);

                            for (ContactRecord rec : recs) {
                                ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                                if (ctrlRecord != null) {
                                    float num = rec.getCounts();
                                    float den = ctrlRecord.getCounts();

                                    int binX = rec.getBinX();
                                    int binY = rec.getBinY();

                                    float score = (num / obsExpected) - (den / ctrlExpected);
                                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);

                                    int px = binX - originX;
                                    int py = binY - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;

            }
            case OERATIOMINUSP1: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null || ctrlBlocks == null || controlZD == null || zd == null) return false;

                Map<String, Block> controlBlocks = new HashMap<>();
                for (Block b : ctrlBlocks) {
                    controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
                }

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                float averageCount = (float) zd.getAverageCount();
                float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

                if (sameChr) {
                    if (df == null || controlDF == null) return false;

                    for (Block b : blocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {

                            Map<String, ContactRecord> controlRecords = new HashMap<>();
                            Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                            if (cb != null) {
                                for (ContactRecord ctrlRec : cb.getContactRecords()) {
                                    controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                                }
                            }

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

                                    float score = ((num + pseudocountObs) / (obsExpected + pseudocountObs)) - ((den + pseudocountCtrl) / (ctrlExpected + pseudocountCtrl));
                                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);

                                    int px = binX - originX;
                                    int py = binY - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }

                                    if (binX != binY) {
                                        px = binY - originX;
                                        py = binX - originY;
                                        if (px > -1 && py > -1 && px <= width && py <= height) {
                                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (Block b : blocks) {
                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {

                            Map<String, ContactRecord> controlRecords = new HashMap<>();
                            Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                            if (cb != null) {
                                for (ContactRecord ctrlRec : cb.getContactRecords()) {
                                    controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                                }
                            }

                            float obsExpected = (averageCount > 0 ? averageCount : 1);
                            float ctrlExpected = (ctrlAverageCount > 0 ? ctrlAverageCount : 1);

                            for (ContactRecord rec : recs) {
                                ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                                if (ctrlRecord != null) {
                                    float num = rec.getCounts();
                                    float den = ctrlRecord.getCounts();

                                    int binX = rec.getBinX();
                                    int binY = rec.getBinY();

                                    float score = ((num + pseudocountObs) / (obsExpected + pseudocountObs)) - ((den + pseudocountCtrl) / (ctrlExpected + pseudocountCtrl));
                                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                    Color color = cs.getColor(score);
                                    g.setColor(color);

                                    int px = binX - originX;
                                    int py = binY - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;

            }
            case RATIOV2:
            case RATIO: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null || ctrlBlocks == null || zd == null || controlZD == null) return false;

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                float averageCount = (float) zd.getAverageCount();
                float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

                Map<String, Block> controlBlocks = new HashMap<>();
                for (Block b : ctrlBlocks) {
                    controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
                }

                for (Block b : blocks) {
                    Collection<ContactRecord> recs = b.getContactRecords();

                    Map<String, ContactRecord> controlRecords = new HashMap<>();
                    Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                    if (cb != null) {
                        for (ContactRecord ctrlRec : cb.getContactRecords()) {
                            controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                        }
                    }

                    if (recs != null) {
                        for (ContactRecord rec : recs) {
                            ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                            if (ctrlRecord != null) {
                                float num = rec.getCounts() / averageCount;
                                float den = ctrlRecord.getCounts() / ctrlAverageCount;
                                float score = num / den;
                                if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                Color color = cs.getColor(score);
                                g.setColor(color);

                                int binX = rec.getBinX();
                                int binY = rec.getBinY();
                                int px = binX - originX;
                                int py = binY - originY;
                                if (px > -1 && py > -1 && px <= width && py <= height) {
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }

                                if (sameChr && binX != binY) {
                                    px = binY - originX;
                                    py = binX - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case RATIOP1V2:
            case RATIOP1: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null || ctrlBlocks == null || zd == null || controlZD == null) return false;

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                float averageCount = (float) zd.getAverageCount();
                float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

                Map<String, Block> controlBlocks = new HashMap<>();
                for (Block b : ctrlBlocks) {
                    controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
                }

                for (Block b : blocks) {
                    Collection<ContactRecord> recs = b.getContactRecords();

                    Map<String, ContactRecord> controlRecords = new HashMap<>();
                    Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                    if (cb != null) {
                        for (ContactRecord ctrlRec : cb.getContactRecords()) {
                            controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                        }
                    }

                    if (recs != null) {
                        for (ContactRecord rec : recs) {
                            ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                            if (ctrlRecord != null) {
                                float num = (rec.getCounts() + pseudocountObs) / (averageCount + pseudocountObs);
                                float den = (ctrlRecord.getCounts() + pseudocountCtrl) / (ctrlAverageCount + pseudocountCtrl);
                                float score = num / den;
                                if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                Color color = cs.getColor(score);
                                g.setColor(color);

                                int binX = rec.getBinX();
                                int binY = rec.getBinY();
                                int px = binX - originX;
                                int py = binY - originY;
                                if (px > -1 && py > -1 && px <= width && py <= height) {
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }

                                if (sameChr && binX != binY) {
                                    px = binY - originX;
                                    py = binX - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case LOGRATIOV2:
            case LOGRATIO: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null || ctrlBlocks == null || zd == null || controlZD == null) return false;

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                float averageCount = (float) zd.getAverageCount();
                float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();

                Map<String, Block> controlBlocks = new HashMap<>();
                for (Block b : ctrlBlocks) {
                    controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
                }

                for (Block b : blocks) {
                    Collection<ContactRecord> recs = b.getContactRecords();

                    Map<String, ContactRecord> controlRecords = new HashMap<>();
                    Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                    if (cb != null) {
                        for (ContactRecord ctrlRec : cb.getContactRecords()) {
                            controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                        }
                    }

                    if (recs != null) {
                        for (ContactRecord rec : recs) {
                            ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                            if (ctrlRecord != null) {
                                float num = (float) Math.log(rec.getCounts() / averageCount + 1);
                                float den = (float) Math.log(ctrlRecord.getCounts() / ctrlAverageCount + 1);
                                float score = num / den;
                                if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                Color color = cs.getColor(score);
                                g.setColor(color);

                                int binX = rec.getBinX();
                                int binY = rec.getBinY();
                                int px = binX - originX;
                                int py = binY - originY;
                                if (px > -1 && py > -1 && px <= width && py <= height) {
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }

                                if (sameChr && binX != binY) {
                                    px = binY - originX;
                                    py = binX - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case RATIO0V2:
            case RATIO0: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null || ctrlBlocks == null || zd == null || controlZD == null || df == null || controlDF == null)
                    return false;

                Map<String, Block> controlBlocks = new HashMap<>();
                for (Block b : ctrlBlocks) {
                    controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
                }

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                for (Block b : blocks) {
                    Collection<ContactRecord> recs = b.getContactRecords();

                    Map<String, ContactRecord> controlRecords = new HashMap<>();
                    Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                    if (cb != null) {
                        for (ContactRecord ctrlRec : cb.getContactRecords()) {
                            controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                        }
                    }

                    if (recs != null) {
                        for (ContactRecord rec : recs) {
                            ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                            if (ctrlRecord != null) {
                                float num = (float) (rec.getCounts() / df.getExpectedValue(chr1, 0));
                                float den = (float) (ctrlRecord.getCounts() / controlDF.getExpectedValue(chr1, 0));
                                float score = num / den;
                                if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                Color color = cs.getColor(score);
                                g.setColor(color);

                                int binX = rec.getBinX();
                                int binY = rec.getBinY();
                                int px = binX - originX;
                                int py = binY - originY;
                                if (px > -1 && py > -1 && px <= width && py <= height) {
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }

                                if (sameChr && binX != binY) {
                                    px = binY - originX;
                                    py = binX - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case RATIO0P1V2:
            case RATIO0P1: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null || ctrlBlocks == null || zd == null || controlZD == null || df == null || controlDF == null)
                    return false;

                Map<String, Block> controlBlocks = new HashMap<>();
                for (Block b : ctrlBlocks) {
                    controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
                }

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                for (Block b : blocks) {
                    Collection<ContactRecord> recs = b.getContactRecords();

                    Map<String, ContactRecord> controlRecords = new HashMap<>();
                    Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                    if (cb != null) {
                        for (ContactRecord ctrlRec : cb.getContactRecords()) {
                            controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                        }
                    }

                    if (recs != null) {
                        for (ContactRecord rec : recs) {
                            ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                            if (ctrlRecord != null) {
                                float num = (float) ((rec.getCounts() + pseudocountObs) / (df.getExpectedValue(chr1, 0) + pseudocountObs));
                                float den = (float) ((ctrlRecord.getCounts() + pseudocountCtrl) / (controlDF.getExpectedValue(chr1, 0) + pseudocountCtrl));
                                float score = num / den;
                                if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                Color color = cs.getColor(score);
                                g.setColor(color);

                                int binX = rec.getBinX();
                                int binY = rec.getBinY();
                                int px = binX - originX;
                                int py = binY - originY;
                                if (px > -1 && py > -1 && px <= width && py <= height) {
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }

                                if (sameChr && binX != binY) {
                                    px = binY - originX;
                                    py = binX - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case DIFF: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                List<Block> ctrlBlocks = getTheBlocks(controlZD, x, y, maxX, maxY, controlNormalizationType, isImportant);
                if (blocks == null || ctrlBlocks == null || controlZD == null || zd == null) return false;

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                float averageCount = (float) zd.getAverageCount();
                float ctrlAverageCount = controlZD == null ? 1 : (float) controlZD.getAverageCount();
                float averageAcrossMapAndControl = (averageCount / 2f + ctrlAverageCount / 2f);

                Map<String, Block> controlBlocks = new HashMap<>();
                for (Block b : ctrlBlocks) {
                    controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
                }

                for (Block b : blocks) {
                    Collection<ContactRecord> recs = b.getContactRecords();

                    Map<String, ContactRecord> controlRecords = new HashMap<>();
                    Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                    if (cb != null) {
                        for (ContactRecord ctrlRec : cb.getContactRecords()) {
                            controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                        }
                    }

                    if (recs != null) {
                        for (ContactRecord rec : recs) {
                            ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                            if (ctrlRecord != null) {
                                float num = rec.getCounts() / averageCount;
                                float den = ctrlRecord.getCounts() / ctrlAverageCount;
                                float score = (num - den) * averageAcrossMapAndControl;
                                if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                                Color color = cs.getColor(score);
                                g.setColor(color);

                                int binX = rec.getBinX();
                                int binY = rec.getBinY();
                                int px = binX - originX;
                                int py = binY - originY;
                                if (px > -1 && py > -1 && px <= width && py <= height) {
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }

                                if (sameChr && binX != binY) {
                                    px = binY - originX;
                                    py = binX - originY;
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case LOG: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                if (blocks == null) return false;

                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                for (Block b : blocks) {
                    Collection<ContactRecord> recs = b.getContactRecords();
                    if (recs != null) {
                        for (ContactRecord rec : recs) {
                            float score = (float) Math.log(1 + rec.getCounts());
                            if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                            Color color = cs.getColor(score);
                            g.setColor(color);

                            int binX = rec.getBinX();
                            int binY = rec.getBinY();
                            int px = binX - originX;
                            int py = binY - originY;
                            if (px > -1 && py > -1 && px <= width && py <= height) {
                                g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                            }

                            if (sameChr && binX != binY) {
                                px = binY - originX;
                                py = binX - originY;
                                if (px > -1 && py > -1 && px <= width && py <= height) {
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }
                            }
                        }
                    }
                }
                break;
            }
            case OBSERVED:
            default: {
                List<Block> blocks = getTheBlocks(zd, x, y, maxX, maxY, observedNormalizationType, isImportant);
                if (blocks == null) return false;
                ColorScale cs = colorScaleHandler.getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                renderSimpleMap(g, blocks, cs, width, height, sameChr, originX, originY);

                break;
            }
        }
        return true;
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

                            int px = binX - originX;
                            int py = binY - originY;
                            if (px > -1 && py > -1 && px <= width && py <= height) {
                                g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                            }

                            if (binX != binY) {
                                px = binY - originX;
                                py = binX - originY;
                                if (px > -1 && py > -1 && px <= width && py <= height) {
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }
                            }
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

                        int binX = rec.getBinX();
                        int binY = rec.getBinY();
                        int px = binX - originX;
                        int py = binY - originY;
                        if (px > -1 && py > -1 && px <= width && py <= height) {
                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                        }
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
                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
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
                            int px = binY - originX;
                            int py = binX - originY;
                            if (px > -1 && py > -1 && px <= width && py <= height) {
                                g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                            }
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

                            int px = binX - originX;
                            int py = binY - originY;
                            if (px > -1 && py > -1 && px <= width && py <= height) {
                                g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                            }

                            if (binX != binY) {
                                px = binY - originX;
                                py = binX - originY;
                                if (px > -1 && py > -1 && px <= width && py <= height) {
                                    g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                                }
                            }
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

                        int binX = rec.getBinX();
                        int binY = rec.getBinY();
                        int px = binX - originX;
                        int py = binY - originY;
                        if (px > -1 && py > -1 && px <= width && py <= height) {
                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                        }
                    }
                }
            }
        }
    }

    private void renderSimpleMap(Graphics2D g, List<Block> blocks, ColorScale cs,
                                 int width, int height, boolean sameChr, int originX, int originY) {
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

                    Color color = cs.getColor(score);
                    g.setColor(color);

                    if (px > -1 && py > -1 && px <= width && py <= height) {
                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                    }

                    if (sameChr && binX != binY) {
                        px = binY - originX;
                        py = binX - originY;
                        if (px > -1 && py > -1 && px <= width && py <= height) {
                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                        }
                    }
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
                    if (Float.isNaN(score) || Float.isInfinite(score)) continue;

                    int binX = rec.getBinX();
                    int binY = rec.getBinY();
                    int px = binX - originX;
                    int py = binY - originY;

                    Color color = cs.getColor(score);
                    g.setColor(color);

                    if (px > -1 && py > -1 && px <= width && py <= height) {
                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                    }

                    if (sameChr && binX != binY) {
                        px = binY - originX;
                        py = binX - originY;
                        if (px > -1 && py > -1 && px <= width && py <= height) {
                            g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                        }
                    }
                }
            }
        }
    }

    private void renderPearsonsVS(Graphics2D g, MatrixZoomData zd, MatrixZoomData controlZD,
                                  ExpectedValueFunction df, ExpectedValueFunction controlDF,
                                  String key, String controlKey, int originX, int originY,
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
                               NormalizationType controlNormalizationType, String key, String controlKey,
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

                //noinspection SuspiciousNameCombination
                g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                // Assuming same chromosome
                if (col != row) {
                    if (bm2 != null) {
                        float controlScore = bm2.getEntry(row, col);
                        Color controlColor = colorScaleHandler.getDenseMatrixColor(key, controlScore, colorScale, cs);
                        px = row - originX;
                        py = col - originY;
                        g.setColor(controlColor);
                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                    } else {
                        px = row - originX;
                        py = col - originY;
                        g.fillRect(px, py, PIXEL_WIDTH, PIXEL_WIDTH);
                    }
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
