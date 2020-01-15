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


package juicebox.mapcolorui;

import java.awt.*;
import java.util.HashMap;
import java.util.Map;

/**
 * @author Neva Cherniavsky
 * @since 3/22/12
 */
public class PearsonColorScale {

    private final Map<String, Float> posMinMap = new HashMap<>();
    private final Map<String, Float> posMaxMap = new HashMap<>();
    private final Map<String, Float> negMinMap = new HashMap<>();
    private final Map<String, Float> negMaxMap = new HashMap<>();

    public PearsonColorScale() {
    }

    public float getPosMax(String key) {
        return posMaxMap.get(key);
    }

    public float getPosMin(String key) {
        return posMinMap.get(key);
    }

    public float getNegMax(String key) {
        return negMaxMap.get(key);
    }

    public float getNegMin(String key) {
        return negMinMap.get(key);
    }

    public void setMinMax(String key, float min, float max) {
        setMinMax(key, min, 0, 0, max);
    }

    public void setMinMax(String key, float negMin, float negMax, float posMin, float posMax) {
        negMinMap.put(key, negMin);
        negMaxMap.put(key, negMax);
        posMaxMap.put(key, posMax);
        posMinMap.put(key, posMin);

    }

    public Color getColor(String key, float score) {

        if (score > 0) {
            float min = getPosMin(key), max = getPosMax(key);
            score = (score - min) / (max - min);
            if (score > 0) {
                int R = (int) (255 * Math.min(score, 1));
                int G = 0;
                int B = 0;
                return new Color(R, G, B);
            }
        } else if (score < 0) {
            float min = getNegMin(key), max = getNegMax(key);
            score = (score - max) / (min - max);
            if (score > 0) {
                //if (score < 0) score = -score; // this shouldn't happen but seems to be happening.
                int R = 0;
                int G = 0;
                int B = (int) (255 * Math.min(score, 1));
                return new Color(R, G, B);
            }
        }
        return Color.black;
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

    public boolean doesNotContainKey(String key) {
        return !negMinMap.containsKey(key) || !posMaxMap.containsKey(key);
    }

    public void resetValues(String key) {
        negMinMap.remove(key);
        negMaxMap.remove(key);
        posMinMap.remove(key);
        posMaxMap.remove(key);
    }
}