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

package juicebox.assembly;

import juicebox.track.feature.Feature2D;

/**
 * Created by nathanielmusial on 6/30/17.
 */
public class FragmentProperty {

    //invariant properties
    private String name;
    private int indexId;
    private long length;

    //initial state
    private boolean isInitiallyInverted;
    private long initialStart;

    //current state
    private boolean isInvertedVsInitial;
    private long currentStart;

    //2D features (scaled)
    private Feature2D feature2D;

    // formality
    private String initialChr = "assembly";


    // deprecated constructor
    public FragmentProperty(String name, int indexId, long length, boolean isInitiallyInverted) {
        this.name = name;
        this.indexId = indexId;
        this.length = length;
        this.feature2D = null;
        this.isInvertedVsInitial = false;
        this.isInitiallyInverted = isInitiallyInverted;
    }


    public FragmentProperty(String name, int indexId, long length) {
        this.name = name;
        this.indexId = indexId;
        this.length = length;
        this.feature2D = null;
    }

    public FragmentProperty(FragmentProperty fragmentProperty) {
        // invariant properties
        this.name = fragmentProperty.name;
        this.indexId = fragmentProperty.indexId;
        this.length = fragmentProperty.length;

        // initial state
        this.initialStart = fragmentProperty.initialStart;
        this.isInitiallyInverted = fragmentProperty.isInitiallyInverted;

        // current state
        this.currentStart = fragmentProperty.currentStart;
        this.isInvertedVsInitial = fragmentProperty.isInvertedVsInitial;

        // 2D features
        if (this.feature2D != null)
            this.feature2D = fragmentProperty.feature2D.deepCopy();

        // formality
        this.initialChr = fragmentProperty.initialChr;
    }

    // main properties
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getIndexId() {
        return indexId;
    }

    public void setIndexId(int indexId) {
        this.indexId = indexId;
    }

    public int getSignIndexId() {
        if ((!wasInitiallyInverted()) && (!isInvertedVsInitial) ||
                wasInitiallyInverted() && isInvertedVsInitial) {
            return indexId;
        } else {
            return -indexId;
        }
    }

    public long getLength() {
        return length;
    }

    public boolean isDebris() {
        return name.contains(":::debris");
    }

    public String getOriginalContigName() {
        if (name.contains(":::fragment_")) {
            return name.split(":::fragment_")[0];
        } else {
            return name;
        }
    }

    public int getFragmentNumber() {
        if (name.contains(":::fragment_")) {
            String temp = name.split(":::fragment_")[1];
            if (temp.contains(":::debris")) {
                return Integer.parseInt(temp.split(":::debris")[0]);
            } else {
                return Integer.parseInt(temp); //can just parse int from string
            }
        } else {
            return 0;
        }
    }

    public long getInitialStart() {
        return initialStart;
    }

    // initial state related
    public void setInitialStart(long initialStart) {
        this.initialStart = initialStart;
    }

    public long getInitialEnd() {
        return initialStart + length;
    }

    public void setInitiallyInverted(boolean initiallyInverted) {
        this.isInitiallyInverted = initiallyInverted;
    }

    public boolean wasInitiallyInverted() {
        return isInitiallyInverted;
    }

    public long getCurrentStart() {
        return currentStart;
    }

    // current state related
    public void setCurrentStart(long currentStart) {
        this.currentStart = currentStart;
    }

    public long getCurrentEnd() {
        return currentStart + length;
    }

    public boolean isInvertedVsInitial() {
        return isInvertedVsInitial;
    }

    public void setInvertedVsInitial(boolean invertedVsInitial) {
        this.isInvertedVsInitial = invertedVsInitial;
    }

    public void toggleInversion() {
        isInvertedVsInitial = !isInvertedVsInitial;
    }


    // formality
    public String getInitialChr() {
        return initialChr;
    }

    public Feature2D getFeature2D() {
        return feature2D;
    }

    public void setFeature2D(Feature2D feature2D) {
        this.feature2D = feature2D;
    }


    //
    //
    //
    // deprecated

    public void setInitialState(String initialChr, long initialStart, long initialEnd, boolean inverted) {
        this.initialChr = initialChr;
        this.initialStart = initialStart;
        this.isInvertedVsInitial = inverted;
    }

    @Override
    public String toString() {
        return name + " " + indexId + " " + length;
    }


}