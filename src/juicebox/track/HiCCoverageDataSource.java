package juicebox.track;

import juicebox.HiC;
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
//import java.util.Collections;

/**
 * @author jrobinso
 *         Date: 8/1/13
 *         Time: 7:53 PM
 */
public class HiCCoverageDataSource implements HiCDataSource {
    private String name;
    private Color color = new Color(97, 184, 209);
    private Color altcolor = color;
    private DataRange dataRange;
    private final HiC hic;

    private final NormalizationType normalizationType;

    public HiCCoverageDataSource(HiC hic, NormalizationType no) {
        this.name = no.getLabel();
        this.hic = hic;
        this.normalizationType = no;
    }


    public void initDataRange() {
        if (hic.getZd() != null) {
            int chrIdx = hic.getZd().getChr1Idx();
            HiCZoom zoom = hic.getZd().getZoom();
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
        if(dataRange == null) {
            initDataRange();
        }
        return dataRange;
    }

    public void setColor(Color color) {
        this.color = color;
    }

    public void setAltColor(Color color) {
        this.altcolor = color;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getName() { return name;}

    public Color getColor() { return color;}

    public Color getAltColor() {return altcolor;}

    public boolean isLog() {return false;}

    public Collection<WindowFunction> getAvailableWindowFunctions() {
        return new ArrayList<WindowFunction>();
    }

    public HiCDataPoint[] getData(Chromosome chr, int startBin, int endBin, HiCGridAxis gridAxis, double scaleFactor, WindowFunction windowFunction) {

        HiCZoom zoom = hic.getZd().getZoom();

        NormalizationVector nv = hic.getDataset().getNormalizationVector(chr.getIndex(), zoom, normalizationType);
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

    public void setDataRange(DataRange dataRange) {
        this.dataRange = dataRange;
    }

    public static class CoverageDataPoint implements HiCDataPoint {

        final int binNumber;
        final int genomicStart;
        final int genomicEnd;
        final double value;


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
