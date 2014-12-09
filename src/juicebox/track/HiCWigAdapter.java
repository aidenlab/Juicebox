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
import org.broad.igv.data.BasicScore;
import org.broad.igv.data.WiggleDataset;
import org.broad.igv.data.WiggleParser;
import org.broad.igv.feature.LocusScore;
import org.broad.igv.renderer.DataRange;
import org.broad.igv.track.TrackProperties;
import org.broad.igv.track.WindowFunction;
import org.broad.igv.util.ResourceLocator;

import java.awt.*;
import java.util.*;
import java.util.List;

/**
 * @author jrobinso
 *         Date: 11/8/12
 *         Time: 10:30 AM
 */
public class HiCWigAdapter extends HiCDataAdapter {

    private final Map<String, List<LocusScore>> locusScoreMap = new HashMap<String, List<LocusScore>>();
    private WiggleDataset dataset;
    private String trackName;
    private Color color;
    private Color altColor;
    private DataRange dataRange;

    public HiCWigAdapter(HiC hic, String path) {
        super(hic);
        init(path);
    }


    private void init(String path) {

        dataset = (new WiggleParser(new ResourceLocator(path), null)).parse();

        trackName = dataset.getTrackNames()[0];

        TrackProperties properties = dataset.getTrackProperties();

        color = properties.getColor();
        if (color == null) color = Color.blue.darker();
        altColor = properties.getAltColor();

        float min = properties.getMinValue();
        float max = properties.getMaxValue();
        float mid = properties.getMidValue();
        if (Float.isNaN(min) || Float.isNaN(max)) {
            mid = 0;
            min = dataset.getDataMin();
            max = dataset.getDataMax();
            //   min = dataset.getPercent10();
            //   max = dataset.getPercent90();
            if (min > 0 && max > 0) min = 0;
            else if (min < 0 && max < 0) max = 0;


        } else {
            if (Float.isNaN(mid)) {
                if (min >= 0) {
                    mid = Math.max(min, 0);
                } else {
                    mid = Math.min(max, 0);
                }
            }
        }

        dataRange = new DataRange(min, mid, max);
        if (properties.isLogScale()) {
            dataRange.setType(DataRange.Type.LOG);
        }

    }


    protected java.util.List<LocusScore> getLocusScores(String chr, int gStart, int gEnd, int zoom, WindowFunction windowFunction) {
        if (!chr.startsWith("chr")) chr = "chr" + chr;

        List<LocusScore> scores = locusScoreMap.get(chr);
        if (scores == null) {
            int[] startLocations = dataset.getStartLocations(chr);
            int[] endLocations = dataset.getEndLocations(chr);
            float[] values = dataset.getData(trackName, chr);

            if (values == null) return null;

            scores = new ArrayList<LocusScore>(values.length);
            for (int i = 0; i < values.length; i++) {
                BasicScore bs = new BasicScore(startLocations[i], endLocations[i], values[i]);
                scores.add(bs);
            }
            locusScoreMap.put(chr, scores);

        }
        return scores;
    }

    @Override
    public String getName() {
        return trackName;  //To change body of implemented methods use File | Settings | File Templates.
    }

    @Override
    public void setName(String text) {
        this.trackName = text;
    }

    @Override
    public Color getColor() {
        return color;  //To change body of implemented methods use File | Settings | File Templates.
    }

    @Override
    public void setColor(Color selectedColor) {
        this.color = selectedColor;
    }

    @Override
    public Color getAltColor() {
        return altColor;  //To change body of implemented methods use File | Settings | File Templates.
    }

    @Override
    public void setAltColor(Color selectedColor) {
        this.altColor = selectedColor;
    }

    @Override
    public DataRange getDataRange() {
        return dataRange;  //To change body of implemented methods use File | Settings | File Templates.
    }

    @Override
    public void setDataRange(DataRange dataRange) {
        this.dataRange = dataRange;
    }

    @Override
    public Collection<WindowFunction> getAvailableWindowFunctions() {
        return Arrays.asList(WindowFunction.mean, WindowFunction.max);
    }


}
