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

import juicebox.track.feature.Feature2D;

import java.awt.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Wrapper for arrowhead blockbuster results
 * Created by muhammadsaadshamim on 6/8/15.
 */
public class HighScore implements Comparable<HighScore> {
    private final double score;
    private final double uVarScore;
    private final double lVarScore;
    private final double upSign;
    private final double loSign;
    private int i;
    private int j;

    public HighScore(int i, int j, double score, double uVarScore, double lVarScore,
                     double upSign, double loSign) {
        this.i = i;
        this.j = j;
        this.score = score;
        this.uVarScore = uVarScore;
        this.lVarScore = lVarScore;
        this.upSign = upSign;
        this.loSign = loSign;
    }

    private static int compare(double x, double y) {
        return (x < y) ? -1 : ((x == y) ? 0 : 1);
    }

    public String toString() {
        return "" + i + "\t" + j + "\t" + score + "\t" + uVarScore + "\t" + lVarScore + "\t" + upSign + "\t" + loSign;
    }

    public void offsetIndex(int offset) {
        this.i += offset;
        this.j += offset;
    }

    public void scaleIndicesByResolution(int resolution) {
        i *= resolution;
        j *= resolution;
    }

    public int getWidth() {
        return Math.abs(j - i);
    }

    public int getI() {
        return i;
    }

    public int getJ() {
        return j;
    }

    public double getLoSign() {
        return loSign;
    }

    public double getScore() {
        return score;
    }

    public double getuVarScore() {
        return uVarScore;
    }

    public double getlVarScore() {
        return lVarScore;
    }

    public double getUpSign() {
        return upSign;
    }

    @Override
    public boolean equals(Object object) {
        if (this == object)
            return true;
        if (object == null)
            return false;
        if (getClass() != object.getClass())
            return false;
        final HighScore o = (HighScore) object;
        return i == o.getI()
                && j == o.getJ()
                && score == o.getScore()
                && uVarScore == o.getuVarScore()
                && lVarScore == o.getlVarScore()
                && upSign == o.getUpSign()
                && loSign == o.getLoSign();
    }

    @Override
    public int hashCode() {
        return 7 * (i + j) * (int) Math.floor(score + uVarScore + lVarScore + upSign + loSign);
    }

    @Override
    public int compareTo(HighScore o) {
        return compare(this.sortValue(), o.sortValue());
    }

    private double sortValue() {
        return uVarScore + lVarScore;
    }

    public Feature2D toFeature2D(String chrName, int res) {
        Map<String, String> attributes = new HashMap<String, String>();
        attributes.put("score", Double.toString(score));
        attributes.put("uVarScore", Double.toString(uVarScore));
        attributes.put("lVarScore", Double.toString(lVarScore));
        attributes.put("upSign", Double.toString(upSign));
        attributes.put("loSign", Double.toString(loSign));
        return new Feature2D(Feature2D.FeatureType.DOMAIN, chrName, i, j, chrName, i, j, Color.yellow, attributes);
    }
}
