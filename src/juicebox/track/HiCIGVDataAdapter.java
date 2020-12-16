/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
import org.broad.igv.feature.LocusScore;
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

    private final DataTrack igvTrack;


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

    @Override
    public void setName(String text) {
        igvTrack.setName(text);
    }

    public Color getPosColor() {
        return igvTrack.getColor();
    }

    @Override
    public void setColor(Color selectedColor) {
        igvTrack.setColor(selectedColor);
    }

    public boolean isLogScale() {
        return igvTrack.getDataRange().isLog();
    }

    public Color getNegColor() {
        return igvTrack.getAltColor();
    }

    @Override
    public void setNegColor(Color selectedColor) {
        igvTrack.setAltColor(selectedColor);
    }

    public DataRange getDataRange() {
        return igvTrack.getDataRange();
    }

    @Override
    public void setDataRange(DataRange dataRange) {
        igvTrack.setDataRange(dataRange);
    }

    @Override
    public Collection<WindowFunction> getAvailableWindowFunctions() {
        return igvTrack.getAvailableWindowFunctions();
    }

    protected List<LocusScore> getLocusScores(String chr, long gStart, long gEnd, int zoom, WindowFunction windowFunction) {
        igvTrack.setWindowFunction(windowFunction);
        org.broad.igv.track.LoadedDataInterval<List<LocusScore>> scores = igvTrack.getSummaryScores(chr, (int) gStart, (int) gEnd, zoom);
        // Problems with human not having the "chr".  Return scores if not 0, otherwise try adding "chr"
        if (scores.getFeatures().size() > 0) return scores.getFeatures();
        else return igvTrack.getSummaryScores("chr" + chr, (int) gStart, (int) gEnd, zoom).getFeatures();
    }
}
