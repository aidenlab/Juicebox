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
import juicebox.gui.SuperAdapter;
import juicebox.matrix.BasicMatrix;
import juicebox.windowui.MatrixType;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.broad.igv.renderer.ColorScale;
import org.broad.igv.renderer.ContinuousColorScale;

import java.awt.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ColorScaleHandler {
    private final PearsonColorScale pearsonColorScale = new PearsonColorScale();
    private final Map<String, ContinuousColorScale> observedColorScaleMap = new HashMap<>();
    private final Map<String, OEColorScale> ratioColorScaleMap = new HashMap<>();
    public static Color HIC_MAP_COLOR = Color.RED;

    public PearsonColorScale getPearsonColorScale() {
        return pearsonColorScale;
    }

    public void reset() {
        observedColorScaleMap.clear();
        ratioColorScaleMap.clear();
    }

    public Color getDenseMatrixColor(String key, float score, PearsonColorScale pearsonColorScale, ColorScale genericColorScale) {
        Color color;
        if (Float.isNaN(score) || Float.isInfinite(score)) {
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

    public void setNewDisplayRange(MatrixType displayOption, double min, double max, String key) {

        if (MatrixType.isOEColorScaleType(displayOption)) {

            OEColorScale oeColorScale = ratioColorScaleMap.get(key);
            if (oeColorScale == null) {
                oeColorScale = new OEColorScale(displayOption);
                ratioColorScaleMap.put(key, oeColorScale);
            }
            oeColorScale.setThreshold(max);

        } else {

            ContinuousColorScale observedColorScale = observedColorScaleMap.get(key);
            if (observedColorScale == null) {
                if (HiCGlobals.isDarkulaModeEnabled) {
                    observedColorScale = new ContinuousColorScale(min, max, Color.black, HIC_MAP_COLOR);
                } else {
                    observedColorScale = new ContinuousColorScale(min, max, Color.white, HIC_MAP_COLOR);
                }
                observedColorScaleMap.put(key, observedColorScale);
            }
            observedColorScale.setNegEnd(min);
            observedColorScale.setPosEnd(max);
        }
    }

    public ColorScale getColorScale(String key, MatrixType displayOption, boolean isWholeGenome, List<Block> blocks, List<Block> ctrlBlocks, float max) {
        if (blocks.isEmpty()) {
            return getColorScale(key, displayOption, isWholeGenome, ctrlBlocks, max);
        } else {
            return getColorScale(key, displayOption, isWholeGenome, blocks, max);
        }
    }

    public ColorScale getColorScale(String key, MatrixType displayOption, boolean wholeGenome, List<Block> blocks, float givenMax) {

        if (MatrixType.isOEColorScaleType(displayOption)) {
            OEColorScale oeColorScale = ratioColorScaleMap.get(key);
            if (oeColorScale == null) {
                oeColorScale = new OEColorScale(displayOption);
                ratioColorScaleMap.put(key, oeColorScale);
            }
            return oeColorScale;
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
                    observedColorScale = new ContinuousColorScale(0, max, Color.black, HIC_MAP_COLOR);
                } else {
                    observedColorScale = new ContinuousColorScale(0, max, Color.white, HIC_MAP_COLOR);
                }
                observedColorScaleMap.put(key, observedColorScale);
                //mainWindow.updateColorSlider(0, 2 * max, max);
            }
            return observedColorScale;
        }
    }

    public void updateColorSliderFromColorScale(SuperAdapter superAdapter, MatrixType displayOption, String key) {

        if (MatrixType.isOEColorScaleType(displayOption)) {
            OEColorScale oeColorScale = ratioColorScaleMap.get(key);

            if (oeColorScale == null) {
                oeColorScale = new OEColorScale(displayOption);
                ratioColorScaleMap.put(key, oeColorScale);
            }
            superAdapter.updateRatioColorSlider((int) oeColorScale.getMax(), oeColorScale.getThreshold());
        } else {

            ContinuousColorScale observedColorScale = observedColorScaleMap.get(key);
            if ((observedColorScale != null)) {
                superAdapter.updateColorSlider(observedColorScale.getMinimum(), observedColorScale.getMaximum());
            }
        }
    }

    public float computePercentile(List<Block> blocks, double p) {
        DescriptiveStatistics stats = new DescriptiveStatistics();
        if (blocks != null) {
            for (Block b : blocks) {
                for (int i = 0; i < b.getContactRecords().size(); i += 10) {
                    ContactRecord rec = b.getContactRecords().get(i);
                    if (rec.getBinX() != rec.getBinY()) { // Filter diagonal
                        stats.addValue(rec.getCounts());
                    }
                }
            }
        }
        return stats.getN() == 0 ? 1 : (float) stats.getPercentile(p);
    }

    public float computePercentile(BasicMatrix bm, double p) {
        DescriptiveStatistics stats = new DescriptiveStatistics();
        for (int i = 0; i < bm.getRowDimension(); i++) {
            for (int j = i + 1; j < bm.getColumnDimension(); j++) {
                stats.addValue(bm.getEntry(i, j));
            }
        }

        return stats.getN() == 0 ? 1 : (float) stats.getPercentile( p);
    }

    public float computePercentile(BasicMatrix bm1, BasicMatrix bm2, double percentile) {
        return computePercentile(bm1, percentile) + computePercentile(bm2, percentile);
    }
}
