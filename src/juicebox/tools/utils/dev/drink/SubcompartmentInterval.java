/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.dev.drink;

import juicebox.data.feature.Feature;

import java.awt.*;

public class SubcompartmentInterval extends SimpleInterval {

    private Integer clusterID;
    private double differenceFromControl = 0;

    private static final Color[] colors = new Color[]{
            new Color(255, 0, 0),
            new Color(255, 255, 0),
            new Color(0, 234, 255),
            new Color(170, 0, 255),
            new Color(255, 127, 0),
            new Color(191, 255, 0),
            new Color(0, 149, 255),
            new Color(255, 0, 170),
            new Color(255, 212, 0),
            new Color(106, 255, 0),
            new Color(0, 64, 255),
            new Color(237, 185, 185),
            new Color(185, 215, 237),
            new Color(231, 233, 185),
            new Color(220, 185, 237),
            new Color(185, 237, 224),
            new Color(143, 35, 35),
            new Color(35, 98, 143),
            new Color(143, 106, 35),
            new Color(107, 35, 143),
            new Color(79, 143, 35),
            new Color(0, 0, 0),
            new Color(115, 115, 115),
            new Color(204, 204, 204)
    };

    public SubcompartmentInterval(int chrIndex, String chrName, int x1, int x2, Integer clusterID) {
        super(chrIndex, chrName, x1, x2);
        this.clusterID = clusterID;
    }

    public Integer getClusterID() {
        return clusterID;
    }

    void setClusterID(Integer clusterID) {
        this.clusterID = clusterID;
    }

    public double getDifferenceFromControl() {
        return differenceFromControl;
    }

    public void setDifferenceFromControl(double differenceFromControl) {
        this.differenceFromControl = differenceFromControl;
    }

    public SubcompartmentInterval absorbAndReturnNewInterval(SubcompartmentInterval interval) {
        return new SubcompartmentInterval(getChrIndex(), getChrName(), getX1(), interval.getX2(), clusterID);
    }

    public boolean overlapsWith(SubcompartmentInterval o) {
        return getChrIndex().equals(o.getChrIndex()) && clusterID.equals(o.clusterID) && getX2().equals(o.getX1());
    }

    @Override
    public String toString() {
        Color color = colors[clusterID % colors.length];
        String colorString = color.getRed() + "," + color.getGreen() + "," + color.getBlue();
        return "chr" + getChrName() + "\t" + getX1() + "\t" + getX2() + "\t" + clusterID + "\t" + clusterID
                + "\t.\t" + getX1() + "\t" + getX2() + "\t" + colorString;
    }

    @Override
    public Feature deepClone() {
        return new SubcompartmentInterval(getChrIndex(), getChrName(), getX1(), getX2(), clusterID);
    }
}