/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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
import juicebox.track.feature.Feature2DWithMotif;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 9/28/15.
 */
public class MotifAnchor extends Feature implements Comparable<MotifAnchor> {

    public static boolean uniquenessShouldSupercedeConvergentRule = true;
    private static int posCount = 0;
    private static int negCount = 0;
    // critical components of a motif anchor
    private final int chrIndex;
    // references to original features if applicable
    private final List<Feature2DWithMotif> originalFeatures1 = new ArrayList<Feature2DWithMotif>();
    private final List<Feature2DWithMotif> originalFeatures2 = new ArrayList<Feature2DWithMotif>();
    private boolean strand;
    private int x1;
    private int x2;
    // fimo output loaded as attributes
    private boolean fimoAttributesHaveBeenInitialized = false;
    private double score = 0, pValue, qValue;
    private String sequence;

    /**
     * Inititalize anchor given parameters (e.g. from BED file)
     *
     * @param chrIndex
     * @param x1
     * @param x2
     */
    public MotifAnchor(int chrIndex, int x1, int x2) {
        this.chrIndex = chrIndex;
        if (x1 <= x2) {
            // x1 < x2
            this.x1 = x1;
            this.x2 = x2;
        } else {
            // x2 < x1 shouldn't ever happen, but just in case
            System.err.println("Improperly formatted Motif file");
            //this.x1 = x2;
            //this.x2 = x1;
        }
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
    public MotifAnchor(int chrIndex, int x1, int x2, List<Feature2DWithMotif> originalFeatures1, List<Feature2DWithMotif> originalFeatures2) {
        this(chrIndex, x1, x2);
        this.originalFeatures1.addAll(originalFeatures1);
        this.originalFeatures2.addAll(originalFeatures2);
    }

    @Override
    public String getKey() {
        return "" + chrIndex;
    }

    @Override
    public Feature deepClone() {
        MotifAnchor clone = new MotifAnchor(chrIndex, x1, x2, originalFeatures1, originalFeatures2);

        if (fimoAttributesHaveBeenInitialized) {
            clone.setFIMOAttributes(score, pValue, qValue, strand, sequence);
        }

        return clone;
    }

    /**
     * @return chromosome name
     */
    public int getChr() {
        return chrIndex;
    }

    /**
     * @return start point
     */
    public int getX1() {
        return x1;
    }

    /**
     * @return end point
     */
    public int getX2() {
        return x2;
    }

    /**
     * @return width of this anchor
     */
    public int getWidth() {
        return x2 - x1;
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
    private boolean contains(int x) {
        return x >= x1 && x <= x2;
    }

    /**
     * @param anchor
     * @return true if this is strictly left of given anchor
     */
    public boolean isStrictlyToTheLeftOf(MotifAnchor anchor) {
        return x2 < anchor.x1;
    }

    /**
     * @param anchor
     * @return true if this is strictly right of given anchor
     */
    public boolean isStrictlyToTheRightOf(MotifAnchor anchor) {
        return anchor.x2 < x1;
    }

    /**
     * @param anchor
     * @return true if given anchor overlaps at either edge with this anchor
     */
    public boolean hasOverlapWith(MotifAnchor anchor) {
        return chrIndex == anchor.chrIndex && (this.contains(anchor.x1) || this.contains(anchor.x2));
    }

    public void mergeWith(MotifAnchor anchor) {
        if (chrIndex == anchor.chrIndex) {
            x1 = Math.min(x1, anchor.x1);
            x2 = Math.max(x2, anchor.x2);
            addFeatureReferencesFrom(anchor);
        } else {
            System.err.println("Attempted to merge anchors on different chromosomes");
            System.err.println(this + " & " + anchor);
        }
    }

    @Override
    public String toString() {
        return chrIndex + "\t" + x1 + "\t" + x2;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        MotifAnchor o = (MotifAnchor) obj;
        return chrIndex == o.chrIndex && x1 == o.x1 && x2 == o.x2;
    }

    @Override
    public int hashCode() {
        return x2 * chrIndex + x1;
    }

    @Override
    public int compareTo(MotifAnchor o) {
        if (chrIndex == o.chrIndex) {
            if (x1 == o.x1) {
                if (x2 == o.x2 && sequence != null && o.sequence != null) {
                    return sequence.compareTo(o.sequence);
                }
                return (new Integer(x2)).compareTo(o.x2);
            }
            return (new Integer(x1)).compareTo(o.x1);
        }
        return (new Integer(chrIndex)).compareTo(o.chrIndex);
    }

    public void setFIMOAttributes(double score, double pValue, double qValue, boolean strand, String sequence) {
        this.score = score;
        this.pValue = pValue;
        this.qValue = qValue;
        this.strand = strand;
        this.sequence = sequence;

        fimoAttributesHaveBeenInitialized = true;
    }

    public double getScore() {
        return score;
    }

    public boolean hasFIMOAttributes() {
        return fimoAttributesHaveBeenInitialized;
    }

    public void addFIMOAttributesFrom(MotifAnchor anchor) {
        setFIMOAttributes(anchor.score, anchor.pValue, anchor.qValue, anchor.strand, anchor.sequence);
    }

    public void addFeatureReferencesFrom(MotifAnchor anchor) {
        originalFeatures1.addAll(anchor.originalFeatures1);
        originalFeatures2.addAll(anchor.originalFeatures2);
    }

    public void updateOriginalFeatures(boolean uniqueStatus, int specificStatus) {
        if ((originalFeatures1.size() > 0 || originalFeatures2.size() > 0)) {
            if (fimoAttributesHaveBeenInitialized) {
                if (specificStatus == 1) {
                    for (Feature2DWithMotif feature : originalFeatures1) {
                        if (strand || uniqueStatus) {
                            posCount++;
                            feature.updateMotifData(strand, uniqueStatus, sequence, x1, x2, true, score);
                        }
                    }
                } else if (specificStatus == -1) {
                    for (Feature2DWithMotif feature : originalFeatures2) {
                        if (!strand || uniqueStatus) {
                            negCount++;
                            feature.updateMotifData(strand, uniqueStatus, sequence, x1, x2, false, score);
                        }
                    }
                } else {
                    for (Feature2DWithMotif feature : originalFeatures1) {
                        if (strand || uniqueStatus) {
                            posCount++;
                            feature.updateMotifData(strand, uniqueStatus, sequence, x1, x2, true, score);
                        }
                    }
                    for (Feature2DWithMotif feature : originalFeatures2) {
                        if (!strand || uniqueStatus) {
                            negCount++;
                            feature.updateMotifData(strand, uniqueStatus, sequence, x1, x2, false, score);
                        }
                    }
                }


            } else {
                System.err.println("Attempting to assign motifs on incomplete anchor");
            }
        }
    }

    public String getSequence() {
        return sequence;
    }

    public List<Feature2DWithMotif> getOriginalFeatures1() {
        return originalFeatures1;
    }

    public List<Feature2DWithMotif> getOriginalFeatures2() {
        return originalFeatures2;
    }

    public boolean isDirectionalAnchor(boolean direction) {
        if (direction) {
            return originalFeatures1.size() > 0 && originalFeatures2.size() == 0;
        } else {
            return originalFeatures2.size() > 0 && originalFeatures1.size() == 0;
        }
    }

    /**
     * @return true if positive strand, false if negative strand
     */
    public boolean getStrand() {
        return strand;
    }
}