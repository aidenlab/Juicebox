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

package juicebox.mapcolorui;

import org.broad.igv.renderer.ColorScale;

import java.awt.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


/**
 * Created by IM on 7/15/15.
 */

class PreDefColorScale implements ColorScale {
    public static final int defaultMaxPreDefVal = 5;
    private static double max;
    private static double min;
    private List<ColorMapEntry> colorList;
    private String name;

    // Private ctor to enforce use of create
    private PreDefColorScale() {
        super();
    }

    /**
     * Creates a color map using the given colors/score values.
     * Both arrays must have same length.
     *
     * @param name   the name of the color map
     * @param colors the array containing the colors
     * @param score  the score values
     * @return the color map
     */
    public PreDefColorScale(String name, Color[] colors, int[] score) {
        updateColors(colors, score);
    }

    private static Color smooth(java.awt.Color c1, java.awt.Color c2, double ratio) {
        double r1 = 1 - ratio;
        // clip
        if (r1 < 0) r1 = 0d;
        if (r1 > 1) r1 = 1d;
        double r2 = 1 - r1;

        int r = (int) Math.round((r1 * c1.getRed()) + (r2 * c2.getRed()));
        int g = (int) Math.round((r1 * c1.getGreen()) + (r2 * c2.getGreen()));
        int b = (int) Math.round((r1 * c1.getBlue()) + (r2 * c2.getBlue()));
        return new Color(r, g, b);
    }

    public static double getMinimum() {
        return min;
    }

    public static double getMaximum() {
        return max;
    }

    public void setPreDefRange(double minCount, double maxCount) {

        if (maxCount > 0) {
            max = Math.log(maxCount);
        } else {
            max = 0;
        }

        if (minCount > 0) {
            min = Math.log(minCount);
        } else {
            min = 0;
        }
        //min = minCount;

        // find score section
        int last = colorList.size() - 1;
        double interval = (max - min) / (colorList.size() - 1);
        double curScore = min;
//        System.out.println("Set score values");
//        System.out.println("List: " + colorList.size() + ", Min: " + min + ", Max: " + max + ", Interval: " + interval + " ");
        for (int i = 0; i <= last; i++) {
            ColorMapEntry curColor = colorList.get(i);
            curColor.setScore(curScore);
//            System.out.print(curScore);
//            System.out.print(" ");
            curScore = curScore + interval;
        }
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    /**
     * find color for given score:
     *
     * @param inScore
     * @return color
     */
    public Color getColor(float inScore) {

        double score = 0;
        if (inScore > 0) {
            score = Math.log(inScore);
        } else {
            score = inScore;
        }

        if (colorList == null || colorList.size() == 0) {
            return Color.white;
        }

        if (score < colorList.get(0).score) {
            return colorList.get(0).getColor();
        }

        int last = colorList.size() - 1;
        if (score > colorList.get(last).score) {
            return colorList.get(last).getColor();
        }

        // find score section
        for (int i = 0; i < last; i++) {
            ColorMapEntry prevColor = colorList.get(i);
            ColorMapEntry nextColor = colorList.get(i + 1);

            if (prevColor.getScore() <= score && nextColor.getScore() >= score) {

                double val = (score - prevColor.getScore()) / (nextColor.getScore() - prevColor.getScore());
                return smooth(prevColor.getColor(), nextColor.getColor(), val);
            }
        }

        throw new RuntimeException("No Color found for given score " + score);
    }

    public void updateColors(Color[] colors, int[] score) {
        if (colors == null) {
            throw new IllegalArgumentException("colors can't be null");
        }
        if (score == null) {
            throw new IllegalArgumentException("score can't be null");
        }


        if (colors.length != score.length) {
            throw new IllegalArgumentException("Arrays colors and score must have same length: " + colors.length + " vs " + score.length);
        }

        colorList = new ArrayList<ColorMapEntry>();

        for (int i = 0; i < score.length; i++) {
            colorList.add(new ColorMapEntry(colors[i], score[i]));
        }

        // sort by score
        Collections.sort(colorList);
    }

    public Color getColor(String symbol) {
        return null;
    }

    public Color getNoDataColor() {
        return null;
    }

    public String asString() {
        return null;
    }

    public boolean isDefault() {
        return false;
    }

    class ColorMapEntry implements Comparable<ColorMapEntry> {
        private final Color color;
        private double score; // limit

        public ColorMapEntry(Color color, double score) {
            super();
            this.color = color;
            this.score = score;
        }

        public double getScore() {
            return score;
        }

        public void setScore(double score) {
            this.score = score;
        }

        public Color getColor() {
            return color;
        }

        @Override
        public int compareTo(ColorMapEntry o) {
            return (int) (this.score - o.score);
        }
    }
}