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
import java.util.HashMap;
import java.util.Map;

/**
 * Created by muhammadsaadshamim on 4/22/17.
 */
public class Contig2D extends Feature2D {

    private String initialChr;
    private int initialStart, initialEnd;
    //    private boolean initiallyInverted;
    private boolean isInverted = false;

    public Contig2D(FeatureType featureType, String chr1, int start1, int end1, Color c, Map<String, String> attributes) {
        super(featureType, chr1, start1, end1, chr1, start1, end1, c, attributes);

        initialChr = chr1;
        initialStart = start1;
        initialEnd = end1;
    }

    private static int processInversionPlotting(int pos, int limitStart, int limitEnd) {
        if (pos >= limitStart && pos <= limitEnd) {
            return limitStart + (limitEnd - pos);
        }
        return pos;
    }

    private static int processTranslationPlotting(int pos, int initStart, int initEnd, int newStart) {
        if (pos >= initStart && pos <= initEnd) {
            return pos + (newStart - initStart);
        }
        return pos;
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

    public void setInitialState(String initialChr, int initialStart, int initialEnd) {
        this.initialChr = initialChr;
        this.initialStart = initialStart;
        this.initialEnd = initialEnd;
    }


    public int getInitialStart() {
        return initialStart;
    }

    public int getInitialEnd() {
        return initialEnd;
    }

    public boolean getInitialInvert() {
        return false;
    } //TODO: generalize!

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

    public boolean isInverted() {
        return isInverted;
    }

    public boolean hasSomeOriginalOverlapWith(int pos) {
        return pos >= initialStart && pos <= initialEnd;
    }

    public int getAlteredBinIndex(int binPos, int binSize) {
        int translatedPos = processTranslationPlotting(binPos * binSize, initialStart, initialEnd, start1);
        if (isInverted) {
            translatedPos = processInversionPlotting(translatedPos, start1, end1);
        }
        return translatedPos / binSize;
    }

    public boolean nowContains(int coordinate) {
        Contig2D contig = this;
        return contig.getStart1() < coordinate && contig.getEnd1() >= coordinate;
    }

    public boolean iniContains(int coordinate) {
        Contig2D contig = this;
        return contig.getInitialStart() < coordinate && contig.getInitialEnd() >= coordinate;
    }

    @Override
    public Feature2D deepCopy() {
        Map<String, String> attrClone = new HashMap<>();
        for (String key : attributes.keySet()) {
            attrClone.put(key, attributes.get(key));
        }
        Contig2D clone = new Contig2D(featureType, getChr1(), start1, end1, getColor(), attrClone);
        clone.initialChr = initialChr;
        clone.initialStart = initialStart;
        clone.initialEnd = initialEnd;
        clone.isInverted = isInverted;
        return clone;
    }

    public Contig2D mergeContigs(Contig2D contig) {
        if (isInverted && contig.isInverted()) {
            if (initialChr.equals(contig.initialChr)
                    && withinTolerance(initialStart, contig.initialEnd)
                    && withinTolerance(start1, contig.end1)) {
                Contig2D merger = new Contig2D(featureType, getChr1(), contig.start1, end1, getColor(), new HashMap<String, String>());
                merger.initialChr = initialChr;
                merger.initialStart = contig.initialStart;
                merger.initialEnd = initialEnd;
                merger.isInverted = isInverted;
                return merger;
            }
        } else if ((!isInverted) && (!contig.isInverted())) {
            if (initialChr.equals(contig.initialChr)
                    && withinTolerance(initialEnd, contig.initialStart)
                    && withinTolerance(end1, contig.start1)) {
                Contig2D merger = new Contig2D(featureType, getChr1(), start1, contig.end1, getColor(), new HashMap<String, String>());
                merger.initialChr = initialChr;
                merger.initialStart = initialStart;
                merger.initialEnd = contig.initialEnd;
                merger.isInverted = isInverted;
                return merger;
            }
        }
        return null;
    }

    private boolean withinTolerance(int val1, int val2) {
        return Math.abs(val1 - val2) < 2;
    }

}
