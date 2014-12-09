/*
 * Copyright (C) 2011-2014 Aiden Lab - All Rights Reserved
 * SOFTWARE COPYRIGHT NOTICE
 * This software and its documentation are the copyright of the Aiden Lab All rights are reserved.
 *
 * This software is supplied without any warranty or guaranteed support whatsoever. Aiden Lab is not responsible for its use, misuse, or functionality.
 *
 * This software is licensed under the terms of the GNU Lesser General Public License (LGPL),
 * Version 2.1 which is available at http://www.opensource.org/licenses/lgpl-2.1.php.
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

    public Color getColor() {
        return igvTrack.getColor();
    }

    @Override
    public void setColor(Color selectedColor) {
        igvTrack.setColor(selectedColor);
    }

    public boolean isLogScale() {
        return igvTrack.getDataRange().isLog();
    }

    public Color getAltColor() {
        return igvTrack.getAltColor();
    }

    @Override
    public void setAltColor(Color selectedColor) {
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

    protected List<LocusScore> getLocusScores(String chr, int gStart, int gEnd, int zoom, WindowFunction windowFunction) {
        igvTrack.setWindowFunction(windowFunction);
        if (chr.contains("chr")) return igvTrack.getSummaryScores(chr, gStart, gEnd, zoom);
        else return igvTrack.getSummaryScores("chr" + chr, gStart, gEnd, zoom);
    }
}
