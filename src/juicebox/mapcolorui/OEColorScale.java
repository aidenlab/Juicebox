/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

import juicebox.HiCGlobals;
import juicebox.windowui.MatrixType;
import org.broad.igv.renderer.ColorScale;

import java.awt.*;

/**
 * @author jrobinso
 *         Date: 11/11/12
 *         Time: 11:32 PM
 */
class OEColorScale implements ColorScale {

    public static final int defaultMaxOEVal = 5;
    private final MatrixType type;
    private double threshold;

    public OEColorScale(MatrixType type) {
        super();
        this.type = type;
        resetThreshold();
    }

    private void resetThreshold() {
        if(type == MatrixType.DIFF) {
            threshold = defaultMaxOEVal;
        }
        else {
            threshold = Math.log(defaultMaxOEVal);
        }
    }

    public Color getColor(float score) {

        double newValue;
        if (MatrixType.isSubtactType(type)) {
            newValue = score;
        } else if (HiCGlobals.HACK_COLORSCALE_LINEAR) {
            if (score < 1) {
                newValue = 1 - (1 / score);
            } else {
                newValue = score - 1;
            }
        } else {
            newValue = Math.log(score);
        }

        int R, G, B;
        if (newValue > 0) {
            R = 255;
            newValue = Math.min(newValue, threshold);
            G = (int) (255 * (threshold - newValue) / threshold);
            B = (int) (255 * (threshold - newValue) / threshold);
        } else {
            newValue = -newValue;
            newValue = Math.min(newValue, threshold);
            B = 255;
            R = (int) (255 * (threshold - newValue) / threshold);
            G = (int) (255 * (threshold - newValue) / threshold);

        }


        if (HiCGlobals.HACK_COLORSCALE) {
            newValue = score;
            if (newValue > (threshold / 2)) {
                R = 255;
                newValue = Math.min(newValue, threshold);
                G = (int) (255 * (threshold - newValue) / (threshold / 2));
                B = (int) (255 * (threshold - newValue) / (threshold / 2));
            } else {
                newValue = Math.max(newValue, 0);
                B = 255;
                R = (int) (255 * (0 + newValue) / (threshold / 2));
                G = (int) (255 * (0 + newValue) / (threshold / 2));
            }
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

    public double getMax() {
        if (MatrixType.isSubtactType(type) || HiCGlobals.HACK_COLORSCALE || HiCGlobals.HACK_COLORSCALE_LINEAR) {
            return 2 * threshold;
        } else {
            return 2 * Math.exp(threshold);
        }
    }

    public float getThreshold() {
        if (MatrixType.isSubtactType(type) || HiCGlobals.HACK_COLORSCALE || HiCGlobals.HACK_COLORSCALE_LINEAR) {
            return (float) threshold;
        } else {
            return (float) Math.exp(threshold);
        }
    }

    public void setThreshold(double max) {
        if (MatrixType.isSubtactType(type) || HiCGlobals.HACK_COLORSCALE || HiCGlobals.HACK_COLORSCALE_LINEAR) {
            threshold = max;
        } else {
            threshold = Math.log(max);
        }
    }
}

