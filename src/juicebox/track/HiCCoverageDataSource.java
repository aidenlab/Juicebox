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

package juicebox.track;

import juicebox.HiC;
import juicebox.data.Dataset;
import juicebox.data.MatrixZoomData;
import juicebox.data.NormalizationVector;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.stat.StatUtils;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.renderer.DataRange;
import org.broad.igv.track.WindowFunction;

import java.awt.*;
import java.util.ArrayList;
import java.util.Collection;

/**
 * @author jrobinso
 *         Date: 8/1/13
 *         Time: 7:53 PM
 */
public class HiCCoverageDataSource implements HiCDataSource {
    private final HiC hic;
    private final NormalizationType normalizationType;
    private String name;
    private Color color = new Color(97, 184, 209);
    private Color altcolor = color;
    private DataRange dataRange;
    private final boolean isControl;

    public HiCCoverageDataSource(HiC hic, NormalizationType no, boolean isControl) {
        this.name = no.getDescription();
        if (isControl) {
            this.name += " (Control)";
        }

        this.hic = hic;
        this.normalizationType = no;
        this.isControl = isControl;
    }

    private void initDataRange() {
        MatrixZoomData zd;
        try {
            zd = hic.getZd();
        } catch (Exception e) {
            return;
        }

        if (zd != null) {
            int chrIdx = zd.getChr1Idx();
            HiCZoom zoom = zd.getZoom();
            NormalizationVector nv = hic.getDataset().getNormalizationVector(chrIdx, zoom, normalizationType);
            if (nv == null) {
                setDataRange(new DataRange(0, 1));
            } else {
                double max = StatUtils.percentile(nv.getData(), 95);
                setDataRange(new DataRange(0, (float) max));
            }

        }
    }

    public DataRange getDataRange() {
        if (dataRange == null) {
            initDataRange();
        }
        return dataRange;
    }

    public void setDataRange(DataRange dataRange) {
        this.dataRange = dataRange;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Color getPosColor() {
        return color;
    }

    public void setColor(Color color) {
        this.color = color;
    }

    public Color getNegColor() {
        return altcolor;
    }

    public void setNegColor(Color color) {
        this.altcolor = color;
    }

    public boolean isLog() {
        return false;
    }

    public Collection<WindowFunction> getAvailableWindowFunctions() {
        return new ArrayList<>();
    }

    public HiCDataPoint[] getData(Chromosome chr, int startBin, int endBin, HiCGridAxis gridAxis,
                                  double scaleFactor, WindowFunction windowFunction) {

        HiCZoom zoom;
        Dataset dataset;
        try {
            if (isControl) {
                zoom = hic.getControlZd().getZoom();
                dataset = hic.getControlDataset();
            } else {
                zoom = hic.getZd().getZoom();
                dataset = hic.getDataset();
            }
        } catch (Exception e) {
            return null;
        }

        NormalizationVector nv = dataset.getNormalizationVector(chr.getIndex(), zoom, normalizationType);
        if (nv == null) return null;

        double[] data = nv.getData();

        CoverageDataPoint[] dataPoints = new CoverageDataPoint[endBin - startBin + 1];

        for (int b = startBin; b <= endBin; b++) {
            int gStart = gridAxis.getGenomicStart(b);
            int gEnd = gridAxis.getGenomicEnd(b);
            int idx = b - startBin;
            double value = b < data.length ? data[b] : 0;
            dataPoints[idx] = new CoverageDataPoint(b, gStart, gEnd, value);
        }

        return dataPoints;
    }

    public static class CoverageDataPoint implements HiCDataPoint {

        final int binNumber;
        public final int genomicStart;
        public final int genomicEnd;
        public final double value;


        public CoverageDataPoint(int binNumber, int genomicStart, int genomicEnd, double value) {
            this.binNumber = binNumber;
            this.genomicEnd = genomicEnd;
            this.genomicStart = genomicStart;
            this.value = value;
        }

        @Override
        public double getBinNumber() {
            return binNumber;
        }

        @Override
        public double getWithInBins() {
            return 1;
        }

        @Override
        public int getGenomicStart() {
            return genomicStart;
        }

        @Override
        public double getValue(WindowFunction windowFunction) {
            return value;
        }


        @Override
        public int getGenomicEnd() {
            return genomicEnd;
        }

    }
}
