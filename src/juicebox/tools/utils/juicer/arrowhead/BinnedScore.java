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

package juicebox.tools.utils.juicer.arrowhead;


import com.google.common.primitives.Doubles;
import juicebox.tools.utils.common.ArrayTools;

import java.util.ArrayList;
import java.util.List;

/**
 * Bin scores based on the connected components found in arrowhead
 */
class BinnedScore {

    private final List<Double> scores = new ArrayList<Double>();
    private final List<Double> uVarScores = new ArrayList<Double>();
    private final List<Double> lVarScores = new ArrayList<Double>();
    private final List<Double> upSigns = new ArrayList<Double>();
    private final List<Double> loSigns = new ArrayList<Double>();
    private int minX;
    private int maxX;
    private int minY;
    private int maxY;

    public BinnedScore(HighScore score) {
        minX = score.getI();
        maxX = score.getI();
        minY = score.getJ();
        maxY = score.getJ();
        appendDataValues(score);
    }

    public static List<HighScore> convertBinnedScoresToHighScores(List<BinnedScore> binnedScores) {
        List<HighScore> highScores = new ArrayList<HighScore>();
        for (BinnedScore score : binnedScores) {
            highScores.add(score.convertToHighScore());
        }
        return highScores;
    }

    /**
     * @return true if given data point is spatially proximate to this data bin
     */
    public boolean isNear(HighScore score, int distanceThreshold) {
        return (Math.abs(minX - score.getI()) < distanceThreshold || Math.abs(maxX - score.getI()) < distanceThreshold)
                && (Math.abs(minY - score.getJ()) < distanceThreshold || Math.abs(maxY - score.getJ()) < distanceThreshold);
    }

    /**
     * Add given data point to this bin and update region bounds and scores/signs
     */
    public void addScoreToBin(HighScore score) {
        if (score.getI() < minX)
            minX = score.getI();
        else if (score.getI() > maxX)
            maxX = score.getI();
        if (score.getJ() < minY)
            minY = score.getJ();
        else if (score.getJ() > maxY)
            maxY = score.getJ();
        appendDataValues(score);
    }

    /**
     * Add given data point's scores/signs to cumulative list
     */
    private void appendDataValues(HighScore score) {
        scores.add(score.getScore());
        uVarScores.add(score.getuVarScore());
        lVarScores.add(score.getlVarScore());
        upSigns.add(score.getUpSign());
        loSigns.add(score.getLoSign());
    }

    /**
     * @return statistical summary of data points in this bin
     */
    private HighScore convertToHighScore() {
        return new HighScore(maxX, maxY,
                ArrayTools.mean(Doubles.toArray(scores)),
                ArrayTools.mean(Doubles.toArray(uVarScores)),
                ArrayTools.mean(Doubles.toArray(lVarScores)),
                ArrayTools.mean(Doubles.toArray(upSigns)),
                ArrayTools.mean(Doubles.toArray(loSigns)));
    }
}
