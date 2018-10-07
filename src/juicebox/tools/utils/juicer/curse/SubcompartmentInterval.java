/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.juicer.curse;

import juicebox.data.feature.Feature;

import java.awt.*;

public class SubcompartmentInterval extends Feature implements Comparable<SubcompartmentInterval> {

    private static Color[] colors = new Color[]{
            new Color(230, 25, 75),
            new Color(60, 180, 75),
            new Color(255, 225, 25),
            new Color(0, 130, 200),
            new Color(245, 130, 48),
            new Color(145, 30, 180),
            new Color(70, 240, 240),
            new Color(240, 50, 230),
            new Color(210, 245, 60),
            new Color(250, 190, 190),
            new Color(0, 128, 128),
            new Color(230, 190, 255),
            new Color(170, 110, 40),
            new Color(255, 250, 200),
            new Color(128, 0, 0),
            new Color(170, 255, 195),
            new Color(128, 128, 0),
            new Color(255, 215, 180),
            new Color(0, 0, 128),
            new Color(128, 128, 128),
            new Color(255, 255, 255),
            new Color(0, 0, 0)
    };
    private final Integer x1;
    private final Integer x2;
    private final Integer chrIndex;
    private final Integer clusterID;
    private String chrName;

    public SubcompartmentInterval(int chrIndex, String chrName, int x1, int x2, Integer clusterID) {
        this.chrIndex = chrIndex;
        this.chrName = chrName;
        this.x1 = x1;
        this.x2 = x2;
        this.clusterID = clusterID;
    }

    @Override
    public int compareTo(SubcompartmentInterval o) {
        int comparison = chrIndex.compareTo(o.chrIndex);
        if (comparison == 0) comparison = x1.compareTo(o.x1);
        if (comparison == 0) comparison = x2.compareTo(o.x2);
        if (comparison == 0) comparison = clusterID.compareTo(o.clusterID);
        return comparison;
    }

    @Override
    public String getKey() {
        return "" + chrIndex;
    }

    @Override
    public Feature deepClone() {
        return new SubcompartmentInterval(chrIndex, chrName, x1, x2, clusterID);
    }

    public Integer getX1() {
        return x1;
    }

    public Integer getX2() {
        return x2;
    }

    public Integer getClusterID() {
        return clusterID;
    }

    public String getChrName() {
        return chrName;
    }

    //    chr19	0	200000	NA	0	.	0	200000	255,255,255
    //    chr19	200000	500000	B1	-1	.	200000	500000	220,20,60
    @Override
    public String toString() {
        return "chr" + chrName + "\t" + x1 + "\t" + x2 + "\t" + clusterID + "\t" + clusterID
                + "\t.\t" + x1 + "\t" + x2 + "\t" + colors[clusterID % colors.length].toString();
    }
}