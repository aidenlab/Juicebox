/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2017 Broad Institute, Aiden Lab
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

package juicebox.track.feature;

import java.awt.*;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 4/22/17.
 */
public class Contig2D extends Feature2D {

    private final String initialChr;
    private final int initialStart, initialEnd;
    private boolean isInverted = false;

    public Contig2D(FeatureType featureType, String chr1, int start1, int end1, Color c, Map<String, String> attributes) {
        super(featureType, chr1, start1, end1, chr1, start1, end1, c, attributes);

        initialChr = chr1;
        initialStart = start1;
        initialEnd = end1;
    }

    public void toggleInversion() {
        isInverted = !isInverted;
    }

    public int setNewStart(int newStart) {
        start1 = newStart;
        start2 = newStart;

        int newEnd = newStart + getTrueWidth();
        end1 = newEnd;
        end2 = newEnd;
        return newEnd;
    }

    private int getTrueWidth() {
        return initialEnd - initialStart;
    }

    @Override
    public String tooltipText() {
        attributes.put("origStart", "" + initialStart);
        attributes.put("origEnd", "" + initialEnd);
        attributes.put("Inverted", "" + isInverted);
        return super.tooltipText();
    }
}
