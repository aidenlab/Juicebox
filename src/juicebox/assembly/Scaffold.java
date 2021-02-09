/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
import juicebox.data.feature.Feature;
import juicebox.track.feature.Feature2D;

import java.awt.*;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Created by dudcha on 10/10/17.
 * todo rename locus or something generic, with motifanchor extending from this
 */
public class Scaffold extends Feature implements Comparable<Scaffold> {

    public static final Comparator<Scaffold> originalStateComparator = new Comparator<Scaffold>() {

        public int compare(Scaffold o1, Scaffold o2) {

//            System.out.println("I am in originalstatscomparator");

            if (o1.getOriginalStart() == o2.getOriginalStart()) {
                return -Long.compare(o1.length, o2.length);
            }
            return Long.compare(o1.getOriginalStart(), o2.getOriginalStart());
        }
    };
    public final String chrName = "assembly";
    //constants
    private final String unsignedScaffoldIdAttributeKey = "Scaffold #";
    private final String signedScaffoldIdAttributeKey = "Signed scaffold #";
    private final String scaffoldNameAttributeKey = "Scaffold name";
    // formality
    private final Integer chrIndex = 1;
    private final Color defaultColor = new Color(0, 255, 0);
    //invariant properties
    public String name;
    private int indexId;
    //initial state
    private long originalStart;
    private boolean isOriginallyInverted;
    //current state
    private boolean isInvertedVsInitial;
    private long currentStart;
    private long length;

    // Main Constructor
    public Scaffold(String name, int indexId, long length) {
        this.name = name;
        this.indexId = indexId;
        this.length = length;
    }


    // Copy Constructor
    public Scaffold(Scaffold scaffold) {
        // invariant properties
        this.name = scaffold.name;
        this.indexId = scaffold.indexId;
        this.length = scaffold.length;

        // initial state
        this.originalStart = scaffold.originalStart;
        this.isOriginallyInverted = scaffold.isOriginallyInverted;

        // current state
        this.currentStart = scaffold.currentStart;
        this.isInvertedVsInitial = scaffold.isInvertedVsInitial;

    }

    // Invariant properties getters and setters
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

    public long getLength() {
        return length;
    }

    public void setLength(long length) {
        this.length = length;
    }

    // Original state getters and setters
    public long getOriginalStart() {
        return originalStart;
    }

    public void setOriginalStart(long originalStart) {
        this.originalStart = originalStart;
    }

    public boolean getOriginallyInverted() {
        return isOriginallyInverted;
    }

    public void setOriginallyInverted(boolean originallyInverted) {
        this.isOriginallyInverted = originallyInverted;
    }

    // Current state getters and setters
    public long getCurrentStart() {
        return currentStart;
    }

    public void setCurrentStart(long currentStart) {
        this.currentStart = currentStart;
    }

    // todo refactor ; always inverted; rename var; what exactly does invertedVsInitial mean?
    public boolean getInvertedVsInitial() {
        return isInvertedVsInitial;
    }

    public void setInvertedVsInitial(boolean invertedVsInitial) {
        this.isInvertedVsInitial = invertedVsInitial;
    }

    // convenience methods
    public int getSignIndexId() {
        if ((!getOriginallyInverted()) && (!isInvertedVsInitial) ||
                getOriginallyInverted() && isInvertedVsInitial) {
            return indexId;
        } else {
            return -indexId;
        }
    }

    public Feature2D getCurrentFeature2D() {
        Map<String, String> attributes = new HashMap<>();
        attributes.put(scaffoldNameAttributeKey, this.getName());
        attributes.put(signedScaffoldIdAttributeKey, String.valueOf(this.getSignIndexId()));
        attributes.put(unsignedScaffoldIdAttributeKey, String.valueOf(this.getIndexId()));
        return new Feature2D(Feature2D.FeatureType.SCAFFOLD,
                chrName,
                scale(this.getCurrentStart()),
                scale(this.getCurrentEnd()),
                chrName,
                scale(this.getCurrentStart()),
                scale(this.getCurrentEnd()),
                defaultColor,
                attributes);
    }

    private int scale(long longCoordinate) {
        return (int) (longCoordinate / HiCGlobals.hicMapScale);
    }

    public boolean isDebris() {
        return name.contains(":::debris");
    }

    public String getOriginalScaffoldName() {
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

    public long getOriginalEnd() {
        return originalStart + length;
    }

    public long getCurrentEnd() {
        return currentStart + length;
    }

    public void toggleInversion() {
        isInvertedVsInitial = !isInvertedVsInitial;
    }

    public Scaffold mergeWith(Scaffold scaffold) {

        if (this.getOriginalEnd() == scaffold.getOriginalStart()
                && scaffold.isInvertedVsInitial == this.isInvertedVsInitial
                && !this.isInvertedVsInitial) {
            this.length = this.length + scaffold.length;
            return this;
        }
        if (scaffold.getOriginalEnd() == this.originalStart
                && scaffold.isInvertedVsInitial == this.isInvertedVsInitial
                && this.isInvertedVsInitial) {
            this.setOriginalStart(scaffold.getOriginalStart());
            this.length = this.length + scaffold.length;
            return this;
        }
        return null;
    }

    @Override
    public String getKey() {
        return "" + chrIndex;
    }

    @Override
    public Feature deepClone() {
        return new Scaffold(name, indexId, length);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        if (this == obj) {
            return true;
        }
        Scaffold o = (Scaffold) obj;
        if (chrIndex.equals(o.chrIndex)) {

            if (length == o.length) {
                return currentStart == o.currentStart;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        return name + " " + indexId + " " + length;
    }

    @Override
    public int hashCode() {
		return Objects.hash(currentStart, length, HiCGlobals.hicMapScale);
    }

    @Override
    public int compareTo(Scaffold o) {

//        System.out.println("I am in standard comparator");

        if (currentStart == o.currentStart) {
            //           System.out.println(-(new Long(length)).compareTo(o.length));
            return -Long.compare(length, o.length);
        }

//        System.out.println((new Long(currentStart)).compareTo(o.currentStart));
        return Long.compare(currentStart, o.currentStart);
    }

    private boolean currentContains(long x) {
        return x >= currentStart && x <= getCurrentEnd();
    }

    private boolean originalContains(long x) {
        return x >= originalStart && x <= getOriginalEnd();
    }
}