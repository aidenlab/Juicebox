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

import juicebox.HiCGlobals;
import juicebox.data.Block;
import juicebox.data.ContactRecord;
import juicebox.data.ExpectedValueFunction;
import juicebox.data.MatrixZoomData;
import juicebox.gui.MainViewPanel;
import juicebox.gui.SuperAdapter;
import juicebox.matrix.BasicMatrix;
import juicebox.windowui.MatrixType;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.stat.StatUtils;
import org.broad.igv.renderer.ColorScale;
import org.broad.igv.renderer.ContinuousColorScale;
import org.broad.igv.util.collections.DoubleArrayList;

import java.awt.*;
import java.util.List;
import java.util.*;

/**
 * @author jrobinso
 * @since Aug 11, 2010
 */
class HeatmapRenderer {

    private final PearsonColorScale pearsonColorScale;
    private final Map<String, ContinuousColorScale> observedColorScaleMap = new HashMap<>();
    private final Map<String, OEColorScale> ratioColorScaleMap = new HashMap<>();
    private final PreDefColorScale preDefColorScale;

    public HeatmapRenderer() {

        pearsonColorScale = new PearsonColorScale();

        preDefColorScale = new PreDefColorScale("Template",
                new Color[]{
                        new Color(18, 129, 242),
                        new Color(113, 153, 89),
                        new Color(117, 170, 101),
                        new Color(149, 190, 113),
                        new Color(178, 214, 117),
                        new Color(202, 226, 149),
                        new Color(222, 238, 161),
                        new Color(242, 238, 161),
                        new Color(238, 222, 153),
                        new Color(242, 206, 133),
                        new Color(234, 182, 129),
                        new Color(218, 157, 121),
                        new Color(194, 141, 125),
                        new Color(214, 157, 145),
                        new Color(226, 174, 165),
                        new Color(222, 186, 182),
                        new Color(238, 198, 210),
                        new Color(255, 206, 226),
                        new Color(250, 218, 234),
                        new Color(255, 222, 230),
                        new Color(255, 230, 242),
                        new Color(255, 242, 255),
                        new Color(255, 0, 0)
                },
                // elevation
                new int[]{
                        -1,
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        100
                }
        );
    }

    public static String getColorScaleCacheKey(MatrixZoomData zd, MatrixType displayOption, NormalizationType obsNorm, NormalizationType ctrlNorm) {
        return zd.getColorScaleKey(displayOption, obsNorm, ctrlNorm);
    }

    public boolean render(int originX,
                          int originY,
                          int width,
                          int height,
                          final MatrixZoomData zd,
                          final MatrixZoomData controlZD,
                          final MatrixType displayOption,
                          final NormalizationType observedNormalizationType,
                          final NormalizationType controlNormalizationType,
                          final ExpectedValueFunction df,
                          final ExpectedValueFunction controlDF,
                          Graphics2D g,
                          boolean isImportant) {


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

        switch (displayOption) {
            case NORM2: {
                BasicMatrix bm = zd.getNormSquared(observedNormalizationType);
                double percentile = isWholeGenome ? 99 : 95;
                float max = computePercentile(bm, percentile);

                ColorScale cs = getColorScale(key, displayOption, isWholeGenome, null, max);

                renderDenseMatrix(bm, null, originX, originY, width, height, null, key, g, cs);

                break;
            }
            case NORM2CTRL: {
                BasicMatrix bm = controlZD.getNormSquared(controlNormalizationType);
                double percentile = isWholeGenome ? 99 : 95;
                float max = computePercentile(bm, percentile);

                ColorScale cs = getColorScale(controlKey, displayOption, isWholeGenome, null, max);

                renderDenseMatrix(bm, null, originX, originY, width, height, null, key, g, cs);

                break;
            }
            case NORM2OBSVSCTRL: {
                if (controlDF == null) {
                    System.err.println("Control DF is NULL");
                    return false;
                }

                BasicMatrix bm1 = zd.getNormSquared(observedNormalizationType);
                BasicMatrix bm2 = controlZD.getNormSquared(controlNormalizationType);

                double percentile = isWholeGenome ? 99 : 95;
                float max = computePercentile(bm1, percentile) + computePercentile(bm2, percentile);

                ColorScale cs = getColorScale(key, displayOption, isWholeGenome, null, max);

                renderDenseMatrix(bm1, bm2, originX, originY, width, height, null, key, g, cs);
                break;
            }
            case PEARSON: {
                BasicMatrix bm = zd.getPearsons(df);

                if (pearsonColorScale.doesNotContainKey(key)) {
                    pearsonColorScale.setMinMax(key, bm.getLowerValue(), bm.getUpperValue());
                }

                renderDenseMatrix(bm, null, originX, originY, width, height, pearsonColorScale, key, g, null);

                break;
            }
            case PEARSONCTRL: {
                if (controlDF == null) {
                    System.err.println("Control DF is NULL");
                    return false;
                }

                BasicMatrix bm = controlZD.getPearsons(controlDF);

                if (pearsonColorScale.doesNotContainKey(controlKey)) {
                    pearsonColorScale.setMinMax(key, bm.getLowerValue(), bm.getUpperValue());
                }
                renderDenseMatrix(bm, null, originX, originY, width, height, pearsonColorScale, key, g, null);

                break;
            }
            case PEARSONVS: {

                if (controlDF == null) {
                    System.err.println("Control DF is NULL");
                    return false;
                }

                BasicMatrix bm1 = zd.getPearsons(df);
                BasicMatrix bm2 = controlZD.getPearsons(controlDF);

                if (pearsonColorScale.doesNotContainKey(key)) {
                    float min = Math.min(bm1.getLowerValue(), bm2.getLowerValue());
                    float max = Math.max(bm1.getUpperValue(), bm2.getUpperValue());
                    pearsonColorScale.setMinMax(key, min, max);
                }

                renderDenseMatrix(bm1, bm2, originX, originY, width, height, pearsonColorScale, key, g, null);
                break;
            }
            default:
                // Iterate through blocks overlapping visible region
                if (HiCGlobals.printVerboseComments && isImportant)
                    System.out.println("the sys x " + x + " " + maxX + " y " + y + " " + maxY);

                List<Block> blocks = null;
                try {
                    if (zd != null)
                        blocks = zd.getNormalizedBlocksOverlapping(x, y, maxX, maxY, observedNormalizationType, isImportant, false);
                } catch (Exception ignored) {
                    System.err.println("problems with MZD");
                    if (HiCGlobals.printVerboseComments) ignored.printStackTrace();
                }

                List<Block> ctrlBlocks = null;
                try {
                    if (controlZD != null)
                        ctrlBlocks = controlZD.getNormalizedBlocksOverlapping(x, y, maxX, maxY, controlNormalizationType, isImportant, false);
                } catch (Exception ignored) {
                    if (HiCGlobals.printVerboseComments) ignored.printStackTrace();
                }

                if (blocks == null && ctrlBlocks == null) {
                    System.err.println("Both ZoomData objects are null");
                    return false;
                }

                if (displayOption == MatrixType.CONTROL || displayOption == MatrixType.OECTRL) {
                    if (controlZD != null && ctrlBlocks != null) {

                        ColorScale cs = getColorScale(controlKey, displayOption, isWholeGenome, ctrlBlocks, 1f);

                        for (Block b : ctrlBlocks) {

                            Collection<ContactRecord> recs = b.getContactRecords();
                            if (recs != null) {
                                for (ContactRecord rec : recs) {

                                    double score = rec.getCounts();
                                    if (Double.isNaN(score)) continue;

                                    int binX = rec.getBinX();
                                    int binY = rec.getBinY();
                                    int px = binX - originX;
                                    int py = binY - originY;

                                    if (displayOption == MatrixType.OECTRL) {
                                        if (controlDF != null) {
                                            int dist = Math.abs(binX - binY);
                                            double expected = controlDF.getExpectedValue(chr1, dist);
                                            score = rec.getCounts() / expected;
                                        } else {
                                            continue;
                                        }
                                    }

                                    Color color = cs.getColor((float) score);
                                    g.setColor(color);

                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, HiCGlobals.BIN_PIXEL_WIDTH, HiCGlobals.BIN_PIXEL_WIDTH);
                                    }

                                    if (sameChr && (rec.getBinX() != rec.getBinY())) {
                                        px = (binY - originX);
                                        py = (binX - originY);
                                        if (px > -1 && py > -1 && px <= width && py <= height) {
                                            g.fillRect(px, py, HiCGlobals.BIN_PIXEL_WIDTH, HiCGlobals.BIN_PIXEL_WIDTH);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        return false;
                    }
                } else if (displayOption == MatrixType.VS || displayOption == MatrixType.OEVS || displayOption == MatrixType.OCMEVS) {

                    List<Block> comboBlocks = new ArrayList<>();

                    if (blocks != null) comboBlocks.addAll(blocks);
                    if (ctrlBlocks != null) comboBlocks.addAll(ctrlBlocks);
                    if (comboBlocks.isEmpty()) return false;

                    ColorScale cs = getColorScale(key, displayOption, isWholeGenome, comboBlocks, 1f);

                    double averageCount = zd.getAverageCount();
                    double ctrlAverageCount = controlZD == null ? 1 : controlZD.getAverageCount();
                    double averageAcrossMapAndControl = (averageCount + ctrlAverageCount) / 2;


                    if (zd != null && blocks != null) {
                        for (Block b : blocks) {

                            Collection<ContactRecord> recs = b.getContactRecords();
                            if (recs != null) {
                                for (ContactRecord rec : recs) {

                                    double score = rec.getCounts() / averageCount;
                                    score = score * averageAcrossMapAndControl;
                                    if (Double.isNaN(score)) continue;

                                    int binX = rec.getBinX();
                                    int binY = rec.getBinY();

                                    int px = binX - originX;
                                    int py = binY - originY;

                                    if (displayOption == MatrixType.OEVS || displayOption == MatrixType.OCMEVS) {
                                        if (df != null) {
                                            int dist = Math.abs(binX - binY);
                                            double expected = df.getExpectedValue(chr1, dist);
                                            if (displayOption == MatrixType.OEVS) {
                                                score = rec.getCounts() / expected;
                                            } else {
                                                score = rec.getCounts() - expected;
                                            }
                                        } else {
                                            continue;
                                        }
                                    }
                                    Color color = cs.getColor((float) score);
                                    g.setColor(color);

                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, HiCGlobals.BIN_PIXEL_WIDTH, HiCGlobals.BIN_PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                    if (controlZD != null && ctrlBlocks != null) {
                        for (Block b : ctrlBlocks) {
                            Collection<ContactRecord> recs = b.getContactRecords();
                            if (recs != null) {
                                for (ContactRecord rec : recs) {

                                    double score = rec.getCounts() / ctrlAverageCount;
                                    score = score * averageAcrossMapAndControl;
                                    if (Double.isNaN(score)) continue;

                                    int binX = rec.getBinX();
                                    int binY = rec.getBinY();

                                    if (displayOption == MatrixType.OEVS || displayOption == MatrixType.OCMEVS) {
                                        if (controlDF != null) {
                                            //score = getObservedComparedToExpected()
                                            int dist = Math.abs(binX - binY);
                                            double expected = controlDF.getExpectedValue(chr1, dist);

                                            if (displayOption == MatrixType.OEVS) {
                                                score = rec.getCounts() / expected;
                                            } else {
                                                score = rec.getCounts() - expected;
                                            }
                                        } else {
                                            continue;
                                        }
                                    }
                                    Color color = cs.getColor((float) score);
                                    g.setColor(color);

                                    if (sameChr && (rec.getBinX() != rec.getBinY())) {
                                        int px = (binY - originX);
                                        int py = (binX - originY);
                                        if (px > -1 && py > -1 && px <= width && py <= height) {
                                            g.fillRect(px, py, HiCGlobals.BIN_PIXEL_WIDTH, HiCGlobals.BIN_PIXEL_WIDTH);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {

                    boolean hasControl = controlZD != null && ctrlBlocks != null && MatrixType.isSimpleControlType(displayOption);
                    Map<String, Block> controlBlocks = new HashMap<>();
                    if (hasControl) {
                        for (Block b : ctrlBlocks) {
                            controlBlocks.put(controlZD.getNormLessBlockKey(b), b);
                        }
                    }

                    ColorScale cs = getColorScale(key, displayOption, isWholeGenome, blocks, 1f);

                    double averageCount = zd.getAverageCount();
                    double ctrlAverageCount = controlZD == null ? 1 : controlZD.getAverageCount();
                    double averageAcrossMapAndControl = (averageCount / 2. + ctrlAverageCount / 2.);

                    for (Block b : blocks) {

                        Collection<ContactRecord> recs = b.getContactRecords();
                        if (recs != null) {

                            Map<String, ContactRecord> controlRecords = new HashMap<>();
                            if (hasControl) {
                                Block cb = controlBlocks.get(zd.getNormLessBlockKey(b));
                                if (cb != null) {
                                    for (ContactRecord ctrlRec : cb.getContactRecords()) {
                                        controlRecords.put(ctrlRec.getKey(controlNormalizationType), ctrlRec);
                                    }
                                }
                            }

                            for (ContactRecord rec : recs) {
                                double score = Double.NaN;
                                if (displayOption == MatrixType.OE || displayOption == MatrixType.EXPECTED) {
                                    double expected = 0;
                                    // todo
                                    // if(zd instanceof CustomMatrixZoomData){
                                    //    expected = ((CustomMatrixZoomData)zd).getExpected(rec.getBinX(), rec.getBinY(), df);
                                    //} else
                                    if (chr1 == chr2) {
                                        if (df != null) {
                                            int binX = rec.getBinX();
                                            int binY = rec.getBinY();
                                            int dist = Math.abs(binX - binY);
                                            expected = df.getExpectedValue(chr1, dist);
                                        }
                                    } else {
                                        expected = (averageCount > 0 ? averageCount : 1);
                                    }

                                    if (displayOption == MatrixType.OE) {
                                        score = rec.getCounts() / expected;
                                    } else {
                                        score = expected;
                                    }
                                } else if (displayOption == MatrixType.RATIO && hasControl) {
                                    ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                                    if (ctrlRecord != null && ctrlRecord.getCounts() > 0) {
                                        double num = rec.getCounts() / averageCount;
                                        double den = ctrlRecord.getCounts() / ctrlAverageCount;
                                        //score = rec.getCounts() / ctrlRecord.getCounts();
                                        // System.err.println(ctrlAverageCount + " " + averageCount);
                                        score = num / den;
                                    }
                                } else if (displayOption == MatrixType.DIFF && hasControl) {
                                    ContactRecord ctrlRecord = controlRecords.get(rec.getKey(controlNormalizationType));
                                    if (ctrlRecord != null && ctrlRecord.getCounts() > 0) {
                                        double num = rec.getCounts() / averageCount;
                                        double den = ctrlRecord.getCounts() / ctrlAverageCount;
                                        score = (num - den) * averageAcrossMapAndControl;
                                    }
                                } else {
                                    score = rec.getCounts();
                                }
                                if (Double.isNaN(score)) continue;

                                Color color = cs.getColor((float) score);
                                g.setColor(color);

                                int px = rec.getBinX() - originX;
                                int py = rec.getBinY() - originY;
                                if (px > -1 && py > -1 && px <= width && py <= height) {
                                    g.fillRect(px, py, HiCGlobals.BIN_PIXEL_WIDTH, HiCGlobals.BIN_PIXEL_WIDTH);
                                }

                                if (sameChr && (rec.getBinX() != rec.getBinY())) {
                                    px = (rec.getBinY() - originX);
                                    py = (rec.getBinX() - originY);
                                    if (px > -1 && py > -1 && px <= width && py <= height) {
                                        g.fillRect(px, py, HiCGlobals.BIN_PIXEL_WIDTH, HiCGlobals.BIN_PIXEL_WIDTH);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
        }
        return true;
    }


    private ColorScale getColorScale(String key, MatrixType displayOption, boolean wholeGenome, List<Block> blocks, float givenMax) {

        if (displayOption == MatrixType.RATIO || displayOption == MatrixType.OE
                || displayOption == MatrixType.OECTRL || displayOption == MatrixType.OEVS
                || displayOption == MatrixType.DIFF) {

            OEColorScale oeColorScale = ratioColorScaleMap.get(key);
            if (oeColorScale == null) {
                oeColorScale = new OEColorScale(displayOption);
                ratioColorScaleMap.put(key, oeColorScale);
            }
            return oeColorScale;
        } else if (MatrixType.isSimpleType(displayOption)) {

            if (MainViewPanel.preDefMapColor) {
                return preDefColorScale;
            } else {
                //todo: why is the key flicking between resolutions when rendering a switch from "whole genome" to chromosome view?
                ContinuousColorScale observedColorScale = observedColorScaleMap.get(key);
                if (observedColorScale == null) {
                    double percentile = wholeGenome ? 99 : 95;
                    float max = givenMax;
                    if (blocks != null) {
                        max = computePercentile(blocks, percentile);
                    }

                    //observedColorScale = new ContinuousColorScale(0, max, Color.white, Color.red);
                    if (HiCGlobals.isDarkulaModeEnabled) {
                        observedColorScale = new ContinuousColorScale(0, max, Color.black, HiCGlobals.HIC_MAP_COLOR);
                    } else {
                        observedColorScale = new ContinuousColorScale(0, max, Color.white, HiCGlobals.HIC_MAP_COLOR);
                    }
                    observedColorScaleMap.put(key, observedColorScale);
                    //mainWindow.updateColorSlider(0, 2 * max, max);
                }
                return observedColorScale;
            }
        } else {
            return null;
        }
    }

    public void updateColorSliderFromColorScale(SuperAdapter superAdapter, MatrixType displayOption, String key) {

        if (MatrixType.isSimpleType(displayOption)) {

            ContinuousColorScale observedColorScale = observedColorScaleMap.get(key);
            if ((observedColorScale != null)) {
                superAdapter.updateColorSlider(0, observedColorScale.getMinimum(), observedColorScale.getMaximum(), observedColorScale.getMaximum() * 2);
            }
            if (MainViewPanel.preDefMapColor) {
                updatePreDefColors();
                superAdapter.updateColorSlider(0, PreDefColorScale.getMinimum(), PreDefColorScale.getMaximum(), PreDefColorScale.getMaximum() * 2);
            }
        } else if (MatrixType.isComparisonType(displayOption)) {
            OEColorScale oeColorScale = ratioColorScaleMap.get(key);

            if (oeColorScale == null) {
                oeColorScale = new OEColorScale(displayOption);
                ratioColorScaleMap.put(key, oeColorScale);
            }
            superAdapter.updateRatioColorSlider((int) oeColorScale.getMax(), oeColorScale.getThreshold());
        }
    }

    private float computePercentile(List<Block> blocks, double p) {


        DoubleArrayList dal = new DoubleArrayList(10000);
        if (blocks != null) {
            for (Block b : blocks) {
                for (ContactRecord rec : b.getContactRecords()) {
                    // Filter diagonal
                    if (Math.abs(rec.getBinX() - rec.getBinY()) > 1) {
                        float val = rec.getCounts();  // view with average multiplied

                        dal.add(val);
                    }
                }
            }
        }
        return dal.size() == 0 ? 1 : (float) StatUtils.percentile(dal.toArray(), p);
    }

    private float computePercentile(BasicMatrix bm, double p) {
        DoubleArrayList dal = new DoubleArrayList(10000);

        for (int i = 0; i < bm.getRowDimension(); i++) {
            for (int j = i + 1; j < bm.getColumnDimension(); j++) {
                dal.add(bm.getEntry(i, j));
            }
        }

        return dal.size() == 0 ? 1 : (float) StatUtils.percentile(dal.toArray(), p);
    }


    /**
     * Render a dense matrix. Used for Pearsons correlation.  The bitmap is drawn at 1 data point
     * per pixel, scaling happens elsewhere.
     * @param bm1         Matrix to render
     * @param bm2         Matrix to render
     * @param originX    origin in pixels
     * @param originY    origin in pixels
     * @param colorScale color scale to apply
     * @param key
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
                Color color = getDenseMatrixColor(key, score, colorScale, cs);
                int px = col - originX;
                int py = row - originY;
                g.setColor(color);

                //noinspection SuspiciousNameCombination
                g.fillRect(px, py, HiCGlobals.BIN_PIXEL_WIDTH, HiCGlobals.BIN_PIXEL_WIDTH);
                // Assuming same chromosome
                if (col != row) {
                    if (bm2 != null) {
                        float controlScore = bm2.getEntry(row, col);
                        Color controlColor = getDenseMatrixColor(key, controlScore, colorScale, cs);
                        px = row - originX;
                        py = col - originY;
                        g.setColor(controlColor);
                        g.fillRect(px, py, HiCGlobals.BIN_PIXEL_WIDTH, HiCGlobals.BIN_PIXEL_WIDTH);
                    } else {
                        px = row - originX;
                        py = col - originY;
                        g.fillRect(px, py, HiCGlobals.BIN_PIXEL_WIDTH, HiCGlobals.BIN_PIXEL_WIDTH);
                    }
                }
            }
        }
    }

    private Color getDenseMatrixColor(String key, float score, PearsonColorScale pearsonColorScale, ColorScale genericColorScale) {
        Color color;
        if (Float.isNaN(score)) {
            color = Color.gray;
        } else {
            if (pearsonColorScale != null) {
                color = score == 0 ? Color.black : pearsonColorScale.getColor(key, score);
            } else {
                color = genericColorScale.getColor(score);
            }
        }
        return color;
    }

    public void reset() {
        observedColorScaleMap.clear();
        ratioColorScaleMap.clear();
    }

    private void updatePreDefColors() {
        int arrSize = MainViewPanel.preDefMapColorGradient.size();

        //ImmutableSortedSet<Integer> set = ContiguousSet.create(Range.closed(0, arrSize), DiscreteDomain.integers());
        //Integer[] arrTmp = new Integer[arrSize];//set.toArray(new Integer[arrSize]);
        final int[] arrScores = new int[arrSize];

        for (int idx = 0; idx < arrSize; idx++) {
            arrScores[idx] = idx;
        }

        preDefColorScale.updateColors(MainViewPanel.preDefMapColorGradient.toArray(new Color[arrSize]), arrScores);
    }

    public void setNewDisplayRange(MatrixType displayOption, double min, double max, String key) {

        if (MatrixType.isComparisonType(displayOption)) {

            OEColorScale oeColorScale = ratioColorScaleMap.get(key);
            if (oeColorScale == null) {
                oeColorScale = new OEColorScale(displayOption);
                ratioColorScaleMap.put(key, oeColorScale);
            }
            oeColorScale.setThreshold(max);

        } else if (MainViewPanel.preDefMapColor) {

            preDefColorScale.setPreDefRange(min, max);

        } else if (MatrixType.isSimpleType(displayOption)) {

            ContinuousColorScale observedColorScale = observedColorScaleMap.get(key);
            if (observedColorScale == null) {
                if (HiCGlobals.isDarkulaModeEnabled) {
                    observedColorScale = new ContinuousColorScale(min, max, Color.black, HiCGlobals.HIC_MAP_COLOR);
                } else {
                    observedColorScale = new ContinuousColorScale(min, max, Color.white, HiCGlobals.HIC_MAP_COLOR);
                }
                observedColorScaleMap.put(key, observedColorScale);
            }
            observedColorScale.setNegEnd(min);
            observedColorScale.setPosEnd(max);
        }
    }

    public PearsonColorScale getPearsonColorScale() {
        return pearsonColorScale;
    }
}
