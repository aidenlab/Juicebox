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
import juicebox.track.feature.Feature2DWithMotif;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Created by muhammadsaadshamim on 9/28/15.
 * todo rename locus or something generic, with motifanchor extending from this
 */
public class MotifAnchor extends GenericLocus {
	
	public static boolean uniquenessShouldSupercedeConvergentRule = true;
	private static int posCount = 0;
	private static int negCount = 0;
	// critical components of a motif anchor
	// references to original features if applicable
	private boolean strand;
	// fimo output loaded as attributes
	private boolean fimoAttributesHaveBeenInitialized = false;
	private double score = 0, pValue, qValue;
	private String sequence;

	
	/**
	 * Inititalize anchor given parameters (e.g. from BED file)
	 *
	 * @param chr
	 * @param x1
	 * @param x2
	 */
	public MotifAnchor(String chr, long x1, long x2) {
		super(chr, x1, x2);
	}

    public MotifAnchor(String chr, int x1, int x2, String name) {
        super(chr, x1, x2, name);
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
	public MotifAnchor(String chrIndex, long x1, long x2, List<Feature2D> originalFeatures1, List<Feature2D> originalFeatures2) {
		super(chrIndex, x1, x2, originalFeatures1, originalFeatures2);
	}

    @Override
    public String getKey() {
        return "" + chr;
    }

    @Override
    public Feature deepClone() {
        MotifAnchor clone = new MotifAnchor(chr, x1, x2, originalFeatures1, originalFeatures2);
        clone.name = name;
        if (fimoAttributesHaveBeenInitialized) {
            clone.setFIMOAttributes(score, pValue, qValue, strand, sequence);
        }

        return clone;
    }

    @Override
    public Feature cloneToMotifAnchor() {
	    return deepClone();
    }

    @Override
    public String toString() {
        return "chr" + chr + "\t" + x1 + "\t" + x2;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof MotifAnchor) {
            MotifAnchor o = (MotifAnchor) obj;
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
                if (o instanceof MotifAnchor) {
                    if (x2 == o.x2 && sequence != null && ((MotifAnchor) o).sequence != null) {
                        return sequence.compareTo(((MotifAnchor) o).sequence);
                    }
                }
				return Long.compare(x2, o.x2);
            }
			return Long.compare(x1, o.x1);
        }
        return chr.compareTo(o.chr);
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
                    for (Feature2D feature : originalFeatures1) {
                        if (feature instanceof Feature2DWithMotif) {
                            if (strand || uniqueStatus) {
                                posCount++;
                                ((Feature2DWithMotif) feature).updateMotifData(strand, uniqueStatus, sequence, x1, x2, true, score);
                            }
                        }
                    }
                } else if (specificStatus == -1) {
                    for (Feature2D feature : originalFeatures2) {
                        if (feature instanceof Feature2DWithMotif) {
                            if (!strand || uniqueStatus) {
                                negCount++;
                                ((Feature2DWithMotif) feature).updateMotifData(strand, uniqueStatus, sequence, x1, x2, false, score);
                            }
                        }
                    }
                } else {
                    for (Feature2D feature : originalFeatures1) {
                        if (feature instanceof Feature2DWithMotif) {
                            if (strand || uniqueStatus) {
                                posCount++;
                                ((Feature2DWithMotif) feature).updateMotifData(strand, uniqueStatus, sequence, x1, x2, true, score);
                            }
                        }
                    }
                    for (Feature2D feature : originalFeatures2) {
                        if (feature instanceof Feature2DWithMotif) {
                            if (!strand || uniqueStatus) {
                                negCount++;
                                ((Feature2DWithMotif) feature).updateMotifData(strand, uniqueStatus, sequence, x1, x2, false, score);
                            }
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

    /**
     * @return true if positive strand, false if negative strand
     */
    public boolean getStrand() {
        return strand;
    }
}