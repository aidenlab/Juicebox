/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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

package juicebox.data.anchor;

import juicebox.data.feature.Feature;
import juicebox.track.feature.Feature2D;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class GenericLocus extends Feature implements Comparable<GenericLocus> {
    // critical components of a motif anchor
    protected final String chr;
    // references to original features if applicable
    protected final List<Feature2D> originalFeatures1 = new ArrayList<>();
    protected final List<Feature2D> originalFeatures2 = new ArrayList<>();
    protected long x1, x2;
    protected String name = "";

    /**
     * Inititalize locus given parameters (e.g. from BED file)
     *
     * @param chr
     * @param x1
     * @param x2
     */
    public GenericLocus(String chr, long x1, long x2) {
        this.chr = chr;
        if (x1 <= x2) {
            // x1 < x2
            this.x1 = x1;
            this.x2 = x2;
        } else {
            System.err.println("Improperly formatted Motif file: chr " + chr + " x1 " + x1 + " x2 " + x2);
            // todo throw new InvalidObjectException();
        }
    }

    public GenericLocus(String chr, int x1, int x2, String name) {
        this(chr, x1, x2);
        this.name = name;
    }

    /**
     * Inititalize anchor given parameters (e.g. from feature list)
     *
     * @param chrIndex
     * @param x1
     * @param x2
     * @param originalFeatures1
     * @param originalFeatures2
     */
    public GenericLocus(String chrIndex, long x1, long x2, List<Feature2D> originalFeatures1, List<Feature2D> originalFeatures2) {
        this(chrIndex, x1, x2);
        this.originalFeatures1.addAll(originalFeatures1);
        this.originalFeatures2.addAll(originalFeatures2);
    }

    @Override
    public String getKey() {
        return "" + chr;
    }

    @Override
    public Feature deepClone() {
        GenericLocus clone = new GenericLocus(chr, x1, x2, originalFeatures1, originalFeatures2);
        clone.name = name;
        return clone;
    }

    public Feature cloneToMotifAnchor() {
        MotifAnchor clone = new MotifAnchor(chr, x1, x2, originalFeatures1, originalFeatures2);
        clone.name = name;
        return clone;
    }

    /**
     * @return chromosome name
     */
    public String getChr() {
        return chr;
    }

    /**
     * @return start point
     */
    public long getX1() {
        return x1;
    }

    /**
     * @return end point
     */
    public long getX2() {
        return x2;
    }

    /**
     * @return width of this anchor
     */
    public int getWidth() {
        return (int) (x2 - x1);
    }

    /**
     * Expand this anchor (symmetrically) by the width given
     *
     * @param width
     */
    public void widenMargins(int width) {
        x1 = x1 - width / 2;
        x2 = x2 + width / 2;
    }

    /**
     * @param x
     * @return true if x is within bounds of anchor
     */
    public boolean contains(long x) {
        return x >= x1 && x <= x2;
    }

    /**
     * @param anchor
     * @return true if this is strictly left of given anchor
     */
    public boolean isStrictlyToTheLeftOf(GenericLocus anchor) {
        return x2 < anchor.x1;
    }

    /**
     * @param anchor
     * @return true if this is strictly right of given anchor
     */
    public boolean isStrictlyToTheRightOf(GenericLocus anchor) {
        return anchor.x2 < x1;
    }

    /**
     * @param anchor
     * @return true if given anchor overlaps at either edge with this anchor
     */
    public boolean hasOverlapWith(GenericLocus anchor) {
        return chr.equalsIgnoreCase(anchor.chr)
                && (this.contains(anchor.x1) || this.contains(anchor.x2) || anchor.contains(x1) || anchor.contains(x2));
    }

    public void mergeWith(GenericLocus anchor) {
        if (chr.equalsIgnoreCase(anchor.chr)) {
            x1 = Math.min(x1, anchor.x1);
            x2 = Math.max(x2, anchor.x2);
            addFeatureReferencesFrom(anchor);
        } else {
            System.err.println("Attempted to merge anchors on different chromosomes");
            System.err.println(this + " & " + anchor);
        }
    }

    public void mergeWithTakeSmaller(GenericLocus anchor) {
        if (chr.equalsIgnoreCase(anchor.chr)) {
            if (anchor.x1 >= x1 && anchor.x2 <= x2) {
                x1 = anchor.x1;
                x2 = anchor.x2;
            } else if (x1 >= anchor.x1 && x2 <= anchor.x2) {
                x1 = x1;
                x2 = x2;
            } else {
                x1 = Math.min(x1, anchor.x1);
                x2 = Math.max(x2, anchor.x2);
            }
            addFeatureReferencesFrom(anchor);
        } else {
            System.err.println("Attempted to merge anchors on different chromosomes");
            System.err.println(this + " & " + anchor);
        }
    }

    @Override
    public String toString() {
        String chrString = chr.startsWith("chr") ? chr.substring(3) : chr;
        return "chr" + chrString + "\t" + x1 + "\t" + x2;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof GenericLocus) {
            GenericLocus o = (GenericLocus) obj;
            return chr.equalsIgnoreCase(o.chr) && x1 == o.x1 && x2 == o.x2;
        }
        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x2, chr, x1);
    }

    @Override
    public int compareTo(GenericLocus o) {
        if (chr.equalsIgnoreCase(o.chr)) {
            if (x1 == o.x1) {
                return Long.compare(x2, o.x2);
            }
            return Long.compare(x1, o.x1);
        }
        return chr.compareTo(o.chr);
    }

    public void addFeatureReferencesFrom(GenericLocus anchor) {
        originalFeatures1.addAll(anchor.originalFeatures1);
        originalFeatures2.addAll(anchor.originalFeatures2);
    }

    public void updateOriginalFeatures(String prefix) {
        if ((originalFeatures1.size() > 0 || originalFeatures2.size() > 0)) {
            for (Feature2D feature : originalFeatures1) {
                feature.addStringAttribute(prefix+"_start_1", "" + x1);
                feature.addStringAttribute(prefix+"_end_1", "" + x2);
            }
            for (Feature2D feature : originalFeatures2) {
                feature.addStringAttribute(prefix+"_start_2", "" + x1);
                feature.addStringAttribute(prefix+"_end_2", "" + x2);
            }
        }
    }

    public List<Feature2D> getOriginalFeatures1() {
        return originalFeatures1;
    }

    public List<Feature2D> getOriginalFeatures2() {
        return originalFeatures2;
    }

    public boolean isDirectionalAnchor(boolean direction) {
        if (direction) {
            return originalFeatures1.size() > 0 && originalFeatures2.size() == 0;
        } else {
            return originalFeatures2.size() > 0 && originalFeatures1.size() == 0;
        }
    }

    public String getName() {
        return name;
    }
}
