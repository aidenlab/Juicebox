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

import juicebox.HiCGlobals;
import juicebox.track.feature.Contig2D;
import juicebox.track.feature.Feature2D;

import java.awt.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by nathanielmusial on 6/30/17.
 */
// change name to ScaffoldStateTracker?
public class FragmentProperty {

    public static final String unsignedScaffoldIdAttributeKey = "Scaffold #";
    public static final String signedScaffoldIdAttributeKey = "Signed scaffold #";
    public static final String scaffoldNameAttributeKey = "Scaffold name";
    public static final String FRAG_TXT = ":::fragment_";
    public static final String DBRS_TXT = ":::debris";


    //invariant properties
    // todo @nathan @olga you've stated that these are invariant, but that means
    // that it should be ok to make these variables final
    // but then why is there a set name method?
    // either this is a bug, or these aren't invariant
    // if invariant, use "private final" modifier
    private String name;
    private int indexId;
    private long length;

    // added by mss
    private boolean isDebris;

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
    private boolean isFragment;


    // todo this is being used, are you trying to deprecate it?
    // deprecated constructor
    public FragmentProperty(String name, int indexId, long length, boolean isInitiallyInverted) {
        setName(name);
        this.indexId = indexId;
        this.length = length;
        this.isInvertedVsInitial = false;
        this.isInitiallyInverted = isInitiallyInverted;
    }

    public FragmentProperty(String name, int indexId, long length) {
        setName(name);
        this.indexId = indexId;
        this.length = length;
    }

    public FragmentProperty(FragmentProperty fragmentProperty) {
        // invariant properties
        // todo again this seems inaccurate; see above
        setName(fragmentProperty.name);
        this.indexId = fragmentProperty.indexId;
        this.length = fragmentProperty.length;

        // initial state
        this.initialStart = fragmentProperty.initialStart;
        this.isInitiallyInverted = fragmentProperty.isInitiallyInverted;

        // current state
        this.currentStart = fragmentProperty.currentStart;
        this.isInvertedVsInitial = fragmentProperty.isInvertedVsInitial;

        // 2D features
        if (fragmentProperty.feature2D != null)
            this.feature2D = fragmentProperty.feature2D.deepCopy();

        // formality
        this.initialChr = fragmentProperty.initialChr;
    }

    // main properties
    public String getName() {
        return name;
    }

    // todo why do you have this public for an invariant property
    public void setName(String name) {
        this.name = name;
        isDebris = name.contains(DBRS_TXT);
        isFragment = name.contains(FRAG_TXT);
    }

    public int getIndexId() {
        return indexId;
    }

    // todo why do you have this public for an invariant property
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
        return isDebris;
    }

    public String getOriginalScaffoldName() {
        if (isFragment) {
            return name.split(FRAG_TXT)[0];
        } else {
            return name;
        }
    }

    public int getFragmentNumber() {
        if (isFragment) {
            String temp = name.split(FRAG_TXT)[1];
            if (temp.contains(DBRS_TXT)) {
                return Integer.parseInt(temp.split(DBRS_TXT)[0]);
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

    @Override
    public String toString() {
        return name + " " + indexId + " " + length;
    }

    public Contig2D convertToContig2D(String chromosomeName) {
        Map<String, String> attributes = new HashMap<>();
        attributes.put(scaffoldNameAttributeKey, getName());
        attributes.put(signedScaffoldIdAttributeKey, String.valueOf(getSignIndexId()));
        attributes.put(unsignedScaffoldIdAttributeKey, String.valueOf(getIndexId()));
        //attributes.put(initiallyInvertedStatus, Boolean.toString(scaffoldProperty.wasInitiallyInverted()));

        Feature2D scaffoldFeature2D = new Feature2D(Feature2D.FeatureType.SCAFFOLD,
                chromosomeName,
                (int) Math.round(getCurrentStart() / HiCGlobals.hicMapScale),
                (int) Math.round((getCurrentEnd()) / HiCGlobals.hicMapScale),
                chromosomeName,
                (int) Math.round(getCurrentStart() / HiCGlobals.hicMapScale),
                (int) Math.round((getCurrentEnd()) / HiCGlobals.hicMapScale),
                Color.GREEN, attributes);

        //TODO: get rid of Contig2D, too much confusion, too much overlap
        Contig2D contig = scaffoldFeature2D.toContig();
        if (isInvertedVsInitial()) {
            contig.toggleInversion(); //assuming initial contig2D inverted = false
        }
        contig.setInitialState(getInitialChr(),
                (int) Math.round(getInitialStart() / HiCGlobals.hicMapScale),
                (int) Math.round(getInitialEnd() / HiCGlobals.hicMapScale),
                wasInitiallyInverted());

        return contig;
    }
}