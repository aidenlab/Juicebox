package juicebox.track;

import org.broad.igv.feature.LocusScore;
import juicebox.HiC;
import org.broad.igv.renderer.DataRange;
import org.broad.igv.track.DataTrack;
import org.broad.igv.track.WindowFunction;

import java.awt.*;
import java.util.Collection;
import java.util.List;

/**
 * @author jrobinso
 *         Date: 11/8/12
 *         Time: 10:16 AM
 */
public class HiCIGVDataAdapter extends HiCDataAdapter {

    final DataTrack igvTrack;


    public HiCIGVDataAdapter(HiC hic, DataTrack igvTrack) {
        super(hic);
        this.igvTrack = igvTrack;
    }

    public double getMax() {
        return igvTrack.getDataRange().getMaximum();
    }

    public String getName() {
        return igvTrack.getName();
    }

    public Color getColor() {
        return igvTrack.getColor();
    }

    public boolean isLogScale() {
        return igvTrack.getDataRange().isLog();
    }

    public Color getAltColor() {
        return igvTrack.getAltColor();
    }

    public DataRange getDataRange() {
        return igvTrack.getDataRange();
    }

    @Override
    public void setDataRange(DataRange dataRange) {
        igvTrack.setDataRange(dataRange);
    }

    @Override
    public void setName(String text) {
        igvTrack.setName(text);
    }

    @Override
    public void setColor(Color selectedColor) {
        igvTrack.setColor(selectedColor);
    }

    @Override
    public void setAltColor(Color selectedColor) {
        igvTrack.setAltColor(selectedColor);
    }

    @Override
    public Collection<WindowFunction> getAvailableWindowFunctions() {
        return igvTrack.getAvailableWindowFunctions();
    }

    protected List<LocusScore> getLocusScores(String chr, int gStart, int gEnd, int zoom, WindowFunction windowFunction) {
        igvTrack.setWindowFunction(windowFunction);
        if (chr.contains("chr"))  return igvTrack.getSummaryScores(chr, gStart, gEnd, zoom);
        else return igvTrack.getSummaryScores("chr" + chr, gStart, gEnd, zoom);
    }
}
