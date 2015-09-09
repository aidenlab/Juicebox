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

/**
 * @author jrobinso
 *         Date: 11/11/12
 *         Time: 11:32 PM
 */
class OEColorScale implements ColorScale {

    public static final int defaultMaxOEVal = 5;
    private static double max;

    public OEColorScale() {
        super();
        resetMax();
    }

    public static void resetMax() {
        max = Math.log(defaultMaxOEVal);
    }

    public Color getColor(float score) {
/*
        int R = (int) (255 * Math.min(score/max, 1));
        int G = 0;
        int B = (int) (255 * Math.min(min * (1.0/score), 1));
  */
        double value = Math.log(score);
        int R, G, B;
        if (value > 0) {
            R = 255;
            value = Math.min(value, max);
            G = (int) (255 * (max - value) / max);
            B = (int) (255 * (max - value) / max);
        } else {
            value = -value;
            value = Math.min(value, max);
            B = 255;
            R = (int) (255 * (max - value) / max);
            G = (int) (255 * (max - value) / max);

        }

        return new Color(R, G, B);

    }

    public Color getColor(String symbol) {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public Color getNoDataColor() {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public String asString() {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public boolean isDefault() {
        return false;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public void setMax(double max) {
        OEColorScale.max = Math.log(max);
    }
}

