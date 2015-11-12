/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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
import org.broad.igv.feature.Chromosome;
import org.broad.igv.feature.LocusScore;
import org.broad.igv.track.WindowFunction;

import java.util.List;

/**
 * An adapter class to serve as a bridge between an IGV data source and a HiC track.  HiC tracks differ from
 * IGV tracks in that the coordinate system is based on "bins", each of which can correspond to a variable
 * genomic length.
 *
 * @author jrobinso
 *         Date: 9/10/12
 */
public abstract class HiCDataAdapter implements HiCDataSource {

    private static final double log2 = Math.log(2);
    private final HiC hic;

    private LoadedDataInterval loadedDataInterval;

    HiCDataAdapter(HiC hic) {
        this.hic = hic;
    }


    @Override
    public boolean isLog() {
        return getDataRange() != null && getDataRange().isLog();
    }

    @Override
    public HiCDataPoint[] getData(Chromosome chr, int startBin, int endBin, HiCGridAxis gridAxis, double scaleFactor, WindowFunction windowFunction) {

        //String

        String zoom;
        try {
            zoom = hic.getZd().getZoom().getKey();
        } catch (Exception e) {
            zoom = "null";
        }
        String axisType = gridAxis.getClass().getName();

        if (loadedDataInterval != null && loadedDataInterval.contains(zoom, (int) scaleFactor, axisType, windowFunction,
                chr.getName(), startBin, endBin)) {
            return loadedDataInterval.getData();
        } else {

            // Expand starBin and endBin by 50% to facilitate panning
            int f = (endBin - startBin) / 2;
            startBin = Math.max(0, startBin - f);
            endBin = endBin + f;

            int igvZoom = gridAxis.getIGVZoom();
            int subCount = (int) scaleFactor;

            // Increase zoom level for "super-zoom" (=> get higher resolution data
            if (subCount > 1) {
                int z = (int) (Math.log(scaleFactor) / log2);
                igvZoom += (z + 1);
            }

            DataAccumulator[] tmp = new DataAccumulator[(endBin - startBin + 1) * subCount];

            int gStart = gridAxis.getGenomicStart(startBin);
            int gEnd = gridAxis.getGenomicEnd(endBin);

            List<LocusScore> scores = getLocusScores(chr.getName(), gStart, gEnd, igvZoom, windowFunction);
            if (scores == null) return null;

            int nPts = 0;
            for (LocusScore locusScore : scores) {

                int bs = Math.max(startBin, gridAxis.getBinNumberForGenomicPosition(locusScore.getStart()));
                int be = Math.min(endBin, gridAxis.getBinNumberForGenomicPosition(locusScore.getEnd() - 1));


                for (int b = bs; b <= be; b++) {

                    int bStart = gridAxis.getGenomicStart(b);
                    int bEnd = gridAxis.getGenomicEnd(b);
                    double delta = ((double) (bEnd - bStart)) / subCount;

                    int subBin0 = b == bs ? (int) ((locusScore.getStart() - bStart) / delta) : 0;
                    int subBin1 = b == be ? (int) ((locusScore.getEnd() - bStart) / delta) : subCount - 1;

                    for (int subBin = subBin0; subBin <= subBin1; subBin++) {
                        final double subBinWidth = 1.0 / subCount;

                        int idx = (b - startBin) * subCount + subBin;

                        if (idx < 0 || idx >= tmp.length) continue;

                        DataAccumulator dataBin = tmp[idx];
                        if (dataBin == null) {
                            double bPrime = b + ((double) subBin) / subCount;

                            int g0 = (int) (bStart + subBin * delta);
                            int g1 = (int) (bStart + (subBin + 1) * delta);
                            dataBin = new DataAccumulator(bPrime, subBinWidth, g0, g1); // bStart, bEnd);
                            tmp[idx] = dataBin;
                            nPts++;
                        }
                        dataBin.addScore(locusScore);
                    }
                }
            }

            // Copy data, leaving out null values
            DataAccumulator[] data;
            if (nPts == tmp.length) {
                data = tmp;
            } else {
                data = new DataAccumulator[nPts];
                int idx = 0;
                for (DataAccumulator sum : tmp) {
                    if (sum != null) {
                        data[idx++] = sum;
                    }
                }
            }

            loadedDataInterval = new LoadedDataInterval(zoom, (int) scaleFactor, axisType, windowFunction,
                    chr.getName(), startBin, endBin, data);

            return data;
        }
    }

    @Override
    public void setName(String text) {

    }

    protected abstract List<LocusScore> getLocusScores(String chr, int gStart, int gEnd, int zoom, WindowFunction windowFunction);

    public static class DataAccumulator implements HiCDataPoint {

        final double binNumber;
        double width = 1;
        int nPts = 0;
        double weightedSum = 0;
        double max = 0;
        int genomicStart;
        int genomicEnd;


        public DataAccumulator(double binNumber) {
            this.binNumber = binNumber;
        }

        public DataAccumulator(double binNumber, double delta, int genomicStart, int genomicEnd) {
            this.binNumber = binNumber;
            this.width = delta;
            this.genomicStart = genomicStart;
            this.genomicEnd = genomicEnd;
        }

        @Override
        public double getBinNumber() {
            return binNumber;
        }

        @Override
        public double getWithInBins() {
            return width;
        }

        @Override
        public int getGenomicEnd() {
            return genomicEnd;
        }

        @Override
        public int getGenomicStart() {
            return genomicStart;
        }

        @Override
        public double getValue(WindowFunction windowFunction) {
            return windowFunction == WindowFunction.max ? max :
                    (nPts == 0 ? 0 : (float) (weightedSum / nPts));
        }

        void addScore(LocusScore ls) {
//            if (ls.getStart() >= genomicEnd || ls.getEnd() < genomicStart) return;
//            double weight = ((double) (Math.min(genomicEnd, ls.getEnd()) - Math.max(genomicStart, ls.getStart()))) /
//                    (genomicEnd - genomicStart);
            double weight = 1;
            final float score = ls.getScore();
            weightedSum += weight * score;
            nPts++;

            max = score > max ? score : max;
        }


    }


    class LoadedDataInterval {

        final String zoom;
        final int scaleFactor;
        final String axisType;
        final WindowFunction windowFunction;
        final String chr;
        final int startBin;
        final int endBin;
        final DataAccumulator[] data;

        LoadedDataInterval(String zoom, int scaleFactor, String axisType, WindowFunction windowFunction,
                           String chr, int startBin, int endBin, DataAccumulator[] data) {
            this.zoom = zoom;
            this.scaleFactor = scaleFactor;
            this.axisType = axisType;
            this.windowFunction = windowFunction;
            this.chr = chr;
            this.startBin = startBin;
            this.endBin = endBin;
            this.data = data;
        }

        boolean contains(String zoom, int scaleFactor, String axisType, WindowFunction windowFunction,
                         String chr, int startBin, int endBin) {
            return zoom.equals(this.zoom) &&
                    scaleFactor == this.scaleFactor &&
                    axisType.equals(this.axisType) &&
                    windowFunction == this.windowFunction &&
                    chr.equals(this.chr) &&
                    startBin >= this.startBin &&
                    endBin <= this.endBin;
        }

        DataAccumulator[] getData() {
            return data;
        }
    }
}
